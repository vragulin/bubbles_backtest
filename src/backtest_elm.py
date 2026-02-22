"""backtest_elm.py

Daily backtest of simplified "ELM"-style investor allocation rules.

The engine allocates between:
- Equities: a total return index (TRI)
- Safe asset: a T-bill TRI

Data is read from `data/hist_df.csv` (daily). Required columns:
- `us_tri`, `xus_tri`: equity total return indices (nominal)
- `us_pcaey`, `xus_pcaey`: value metric = 1/CAPE (decimal)
- `cpi`: CPI index level
- `tbill_tri`: T-bill total return index
- `tips_10y`: 10y TIPS yield (annual real rate, decimal)

Important conventions used in this backtest:
- Portfolio wealth is tracked in *nominal dollars* using the nominal TRIs.
- Any *historical-return-based* signals (Extrap, STMR, Momentum, and ValMo's
    momentum leg) are computed using *real* (CPI-deflated) equity returns.
    Concretely we form `eq_tri_real = eq_tri / cpi` and compute lookback returns
    off that series.
- The cashflow/value component uses `value_ey = 1/CAPE` and compares it to the
    real rate (`tips_10y`). This is intended as a real earnings-yield vs real-rate
    comparison (assuming the supplied CAPE is based on real earnings, as in
    Shiller-style CAPE).

Region switch:
- `--region US` uses `us_tri` and `us_pcaey`
- `--region XUS` uses `xus_tri` and `xus_pcaey`
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List


EXPLANATION = """\
What this script does
---------------------
This script backtests several stylized allocation rules ("investor types") on
daily historical data. Each strategy chooses an equity weight a_t in [0, 1]
each day, and the remaining wealth is allocated to the T-bill TRI.

Data and derived series
-----------------------
From `hist_df.csv` we read:
- Equity TRI (nominal): `eq_tri`
- T-bill TRI (nominal): `tbill_tri`
- CPI index: `cpi`
- Value metric: `value_ey` = 1/CAPE (decimal)
- Real rate: `real_rate` = `tips_10y` (annual, decimal)

For signal generation we also compute a CPI-deflated equity TRI:
        eq_tri_real[t] = eq_tri[t] / cpi[t]

Portfolio mechanics
-------------------
We track wealth in nominal dollars using nominal TRIs.
- Equity position is held as units of the equity TRI.
- Safe asset position is held as units of `tbill_tri`.

Strategies (high level)
-----------------------
Static
    Constant equity allocation.

Cashflow ("Merton")
    Uses the value metric and the real rate:
            rp = value_ey - real_rate
            alloc = clip( rp / (gamma * sigma^2), 0, 1 )

Extrap
    Computes a weighted average of *past real* annual log returns of equities
    (CPI-deflated), then applies a tanh squeeze around a center risk premium.
    An EWMA smoother is applied in `backtest_strategy`.

STMR
    Uses a short (≈ weekly) *real* lookback return and compares it to the
    expected return implied by the value metric at the start of the lookback.
    The allocation shift is variance-scaled and capped.

Momentum
    Uses a 1-year *real* lookback return (annualized) and compares it to the
    value metric from the start of the lookback window.

ValMo
    Combines a Merton-style value allocation (value_ey vs real_rate) with a
    momentum tilt that is based on the *real* direction of the equity TRI.

Notes
-----
- Strategy signals are computed in real terms where they depend on realized
    historical returns. Wealth accounting remains nominal.
- `tips_10y` is treated as an annual real rate (no conversion to daily).
"""


def print_explanation() -> None:
        print(EXPLANATION)

# ── constants ────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
DEFAULT_WARMUP_YEARS = 5


# ── configuration dataclass ───────────────────────────────────────────────────
@dataclass
class InvestorConfig:
    """Configuration for each investor strategy.

    Parameters
    ----------
    name : str
        Strategy identifier: 'static' | 'cashflow' | 'extrap' |
        'stmr' | 'momentum' | 'valmo'
    baseline : float
        Base equity allocation (0–1).
    lookback_years : float
        Historical window (years) used by momentum / mean-reversion strategies.
    deviation : float
        Max allocation shift from baseline for tactical strategies.
    crra : float
        Coefficient of relative risk aversion (Merton rule).
    stock_vol_assumed : float
        Assumed annual equity volatility.
    momentum_weight : float
        Weight on momentum signal in 'valmo' strategy.
    rp_max : float
        Max risk-premium deviation factor for 'valmo'.
    extrap_weights : tuple
        Per-year lookback weights for extrapolator (most recent first, sums to 1).
    extrap_center : float
        Target risk premium for extrapolator squeeze.
    extrap_max_dev : float
        Max deviation from centre for extrapolator.
    extrap_squeeze : float
        Tanh squeeze scale for extrapolator.
    extrap_pct_monthly : float
        EWMA step size calibrated in monthly terms. A value of 0.5 means the
        extrapolator moves 50% toward the current estimate each month; this is
        converted to a per-period value.
    stmr_lookback_days : int
        Lookback window in trading days for STMR (≈ 5 = 1 week).
    numb_sigmas_max_dev : float
        Deviation threshold for STMR (in sigmas).
    """
    name: str
    baseline: float

    lookback_years: float = 0.0
    deviation: float = 0.0

    crra: float = 3.0
    stock_vol_assumed: float = 0.16

    momentum_weight: float = 0.0
    rp_max: float = 0.0

    extrap_weights: tuple = (0.30, 0.25, 0.20, 0.15, 0.10)
    extrap_center: float = 0.04
    extrap_max_dev: float = 0.03
    extrap_squeeze: float = 0.1
    extrap_pct_monthly: float = 0.5

    stmr_lookback_days: int = 5       # ≈ 1 week
    numb_sigmas_max_dev: float = 2.5


# ── backtest engine ───────────────────────────────────────────────────────────
class ElmBacktest:
    """Daily backtest framework – equities vs T-bills."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        equity_region: str = 'US',       # 'US' or 'XUS'
    ):
        """
        Parameters
        ----------
        data_path : str, optional
            Path to hist_df.csv.  Defaults to ../data/hist_df.csv.
        equity_region : str
            'US'  → us_tri  + us_pcaey
            'XUS' → xus_tri + xus_pcaey
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'data' / 'hist_df.csv'

        self.equity_region = equity_region.upper()
        if self.equity_region == 'US':
            self.equity_col = 'us_tri'
            self.value_col  = 'us_pcaey'
        elif self.equity_region == 'XUS':
            self.equity_col = 'xus_tri'
            self.value_col  = 'xus_pcaey'
        else:
            raise ValueError(f"equity_region must be 'US' or 'XUS', got '{equity_region}'")

        # ── load data ────────────────────────────────────────────────────────
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df.index.name = 'Date'
        df = df.sort_index()

        required = {'us_tri', 'xus_tri', 'us_pcaey', 'xus_pcaey',
                    'cpi', 'tbill_tri', 'tips_10y'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"hist_df.csv is missing columns: {missing}")

        # ── derived daily series ──────────────────────────────────────────────
        df['eq_tri']    = df[self.equity_col]
        df['value_ey']  = df[self.value_col]          # 1/CAPE, already decimal
        df['eq_ret']    = df['eq_tri'].pct_change()
        df['rf_ret']    = df['tbill_tri'].pct_change()  # daily T-bill return

        # Real equity total return index for signal generation (CPI-deflated).
        # Allocation rules in the paper are written in real terms.
        df['eq_tri_real'] = df['eq_tri'] / df['cpi']

        # Real rate: tips_10y is annual → keep as annual for Merton rule
        df['real_rate'] = df['tips_10y'].ffill()

        # Forward-fill value metric and real rate so there are no gaps
        df['value_ey']  = df['value_ey'].ffill()

        # Drop the very first row (NaN returns)
        df = df.iloc[1:].copy()

        self.df = df
        print(
            f"Loaded {len(df):,} daily observations "
            f"({df.index[0].date()} → {df.index[-1].date()})  |  "
            f"equity region: {self.equity_region}"
        )

    # ── public API ────────────────────────────────────────────────────────────
    def backtest_strategy(
        self,
        config: InvestorConfig,
        start_date=None,
        end_date=None,
        initial_wealth: float = 1.0,
    ) -> Dict:
        """
        Backtest a single strategy.

        Returns
        -------
        dict with keys: config, wealth (ndarray), equity_allocation (ndarray),
                        dataframe (DataFrame with results), stats (dict)
        """
        if end_date is None:
            end_date = self.df.index[-1]
        else:
            end_date = pd.Timestamp(end_date)

        if start_date is None:
            # By default, start after a warm-up window so lookback-dependent
            # signals (notably extrapolators) are fully initialized.
            preferred = pd.Timestamp('1997-01-01')
            if self.df.index[0] <= preferred <= end_date:
                start_date = preferred
            else:
                start_date = self.df.index[0] + pd.DateOffset(years=DEFAULT_WARMUP_YEARS)
        else:
            start_date = pd.Timestamp(start_date)

        if start_date >= end_date:
            raise ValueError(
                f"start_date must be earlier than end_date: {start_date.date()} >= {end_date.date()}"
            )

        # Include extra history for lookback-dependent strategies
        max_lookback_years = DEFAULT_WARMUP_YEARS
        lookback_start = start_date - pd.DateOffset(years=max_lookback_years)

        full_df = self.df.loc[
            (self.df.index >= lookback_start) & (self.df.index <= end_date)
        ].copy()

        # Index within full_df where the real backtest starts
        locs = full_df.index.searchsorted(start_date)
        start_idx = int(locs)

        n_backtest = len(full_df) - start_idx
        if n_backtest < TRADING_DAYS_PER_YEAR:
            raise ValueError(
                f"Less than one year of data from {start_date.date()} to {end_date.date()}"
            )

        # For the historical backtest, we can use the observed valuation metric
        # (earnings yield proxy) directly as the expected-return reference for
        # STMR and momentum, rather than initializing an exogenous constant.

        # ── simulate ─────────────────────────────────────────────────────────
        wealth           = np.empty(n_backtest)
        equity_alloc     = np.empty(n_backtest)
        equity_units     = np.empty(n_backtest)   # units of equity TRI

        eq_tri  = full_df['eq_tri'].values
        tb_tri  = full_df['tbill_tri'].values

        # Allocation function (extrap requires state for EWMA smoothing)
        if config.name == 'extrap':
            ppy = TRADING_DAYS_PER_YEAR
            n_years = len(config.extrap_weights)
            pct = 1 - (1 - config.extrap_pct_monthly) ** (12 / ppy)
            prev_ret_box: List[Optional[float]] = [None]

            def alloc_fn(di: int) -> float:
                current_ret = self._extrap_current_ret(config, full_df, di)
                if current_ret is None:
                    return config.baseline

                if prev_ret_box[0] is None:
                    # Initialize from the prior day if possible; otherwise from current.
                    init_di = di - 1
                    init_ret = self._extrap_current_ret(config, full_df, init_di)
                    prev_ret_box[0] = init_ret if init_ret is not None else current_ret

                smoothed_ret = pct * current_ret + (1 - pct) * float(prev_ret_box[0])
                prev_ret_box[0] = smoothed_ret

                real_rate = full_df['real_rate'].iloc[di]
                if pd.isna(real_rate):
                    real_rate = float(full_df['real_rate'].mean())

                rp = smoothed_ret - real_rate
                vol_sq = config.stock_vol_assumed ** 2
                alloc = rp / (config.crra * vol_sq)
                return float(np.clip(alloc, 0.0, 1.0))

        else:
            def alloc_fn(di: int) -> float:
                return self._allocation(config, full_df, di)

        wealth[0]       = initial_wealth
        equity_alloc[0] = alloc_fn(start_idx)
        equity_units[0] = equity_alloc[0] * wealth[0] / eq_tri[start_idx]

        for t in range(1, n_backtest):
            di      = start_idx + t        # index into full_df
            di_prev = di - 1

            # ── update wealth ──────────────────────────────────────────────
            eq_value   = equity_units[t - 1] * eq_tri[di]
            bond_units = (wealth[t - 1] - equity_units[t - 1] * eq_tri[di_prev])
            # bond_units is in "tbill_tri units"
            tb_value   = bond_units * (tb_tri[di] / tb_tri[di_prev])

            wealth[t] = eq_value + tb_value

            # ── new allocation ─────────────────────────────────────────────
            equity_alloc[t] = alloc_fn(di)
            equity_units[t] = equity_alloc[t] * wealth[t] / eq_tri[di]

        # Attach to dataframe slice
        bt_df = full_df.iloc[start_idx:].copy()
        bt_df['Wealth']           = wealth
        bt_df['Equity_Allocation'] = equity_alloc

        stats = self._calc_stats(bt_df, config.name, equity_alloc, wealth)

        return {
            'config':           config,
            'wealth':           wealth,
            'equity_allocation': equity_alloc,
            'dataframe':        bt_df,
            'stats':            stats,
        }

    # ── internal allocation dispatcher ───────────────────────────────────────
    def _allocation(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        name = config.name
        if name == 'static':
            return config.baseline
        elif name == 'cashflow':
            return self._cashflow(config, df, t)
        elif name == 'extrap':
            return self._extrap(config, df, t)
        elif name == 'stmr':
            return self._stmr(config, df, t)
        elif name == 'momentum':
            return self._momentum(config, df, t)
        elif name == 'valmo':
            return self._valmo(config, df, t)
        else:
            return config.baseline

    # ── strategy implementations ──────────────────────────────────────────────
    def _cashflow(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Merton rule: alloc = (CAPE_yield - real_rate) / (gamma * sigma^2)"""
        ey        = df['value_ey'].iloc[t]
        real_rate = df['real_rate'].iloc[t]

        if pd.isna(ey) or pd.isna(real_rate):
            return config.baseline

        rp      = ey - real_rate
        vol_sq  = config.stock_vol_assumed ** 2
        alloc   = rp / (config.crra * vol_sq)
        return float(np.clip(alloc, 0.0, 1.0))

    def _extrap_current_ret(
        self,
        config: InvestorConfig,
        df: pd.DataFrame,
        t: int,
    ) -> Optional[float]:
        """Compute the *current* (unsmoothed) extrapolator expected real return.

        This matches the reference model's sequence:
        1) weighted lookback average return (real, CPI-deflated)
        2) subtract real rate -> raw risk premium
        3) tanh squeeze around a center point
        4) add back real rate

        Smoothing (EWMA) is applied in `backtest_strategy` to keep per-run state.
        """
        days_per_year = TRADING_DAYS_PER_YEAR
        n_years = len(config.extrap_weights)
        if t < n_years * days_per_year:
            return None

        eq = df['eq_tri'].values
        cpi = df['cpi'].values

        avg_ret = 0.0
        for i, w in enumerate(config.extrap_weights):
            yr_end = t - i * days_per_year
            yr_start = t - (i + 1) * days_per_year
            if yr_start < 0:
                return None

            nominal_ret = np.log(eq[yr_end] / eq[yr_start])
            inflation = np.log(cpi[yr_end] / cpi[yr_start])
            avg_ret += w * (nominal_ret - inflation)

        real_rate = df['real_rate'].iloc[t]
        if pd.isna(real_rate):
            real_rate = float(df['real_rate'].mean())

        rp_raw = avg_ret - real_rate
        dev = rp_raw - config.extrap_center
        squeezed = config.extrap_max_dev * np.tanh(dev / config.extrap_squeeze)
        current_ret = config.extrap_center + squeezed + real_rate
        return float(current_ret)

    def _extrap(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Extrapolator: weighted average of past real annual returns, squeezed.

        avg_ret is computed in real terms (nominal equity TRI deflated by CPI)
        to match the paper, which works entirely in real returns. The real rate
        (tips_10y) is then subtracted to form the equity risk premium.
        """
        days_per_year = TRADING_DAYS_PER_YEAR
        n_years       = len(config.extrap_weights)
        if t < n_years * days_per_year:
            return config.baseline

        eq  = df['eq_tri'].values
        cpi = df['cpi'].values
        avg_ret = 0.0
        for i, w in enumerate(config.extrap_weights):
            yr_end   = t - i * days_per_year
            yr_start = t - (i + 1) * days_per_year
            if yr_start >= 0:
                nominal_ret = np.log(eq[yr_end]  / eq[yr_start])
                inflation   = np.log(cpi[yr_end] / cpi[yr_start])   # annual CPI growth
                avg_ret    += w * (nominal_ret - inflation)          # real return

        real_rate = float(df['real_rate'].iloc[t])
        if pd.isna(real_rate):
            real_rate = float(df['real_rate'].mean())

        rp_raw = avg_ret - real_rate
        dev    = rp_raw - config.extrap_center
        squeezed = config.extrap_max_dev * np.tanh(dev / config.extrap_squeeze)
        exp_ret  = config.extrap_center + squeezed + real_rate

        rp     = exp_ret - real_rate
        vol_sq = config.stock_vol_assumed ** 2
        alloc  = rp / (config.crra * vol_sq)
        return float(np.clip(alloc, 0.0, 1.0))

    def _stmr(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Short-term mean reversion: fade recent excess moves."""
        lb = config.stmr_lookback_days
        if t < lb:
            return config.baseline

        eq_real = df['eq_tri_real'].values
        ret_lb  = eq_real[t] / eq_real[t - lb] - 1.0

        # Expected return over lb days using the valuation metric available at
        # the start of the lookback window.
        ey_ref = float(df['value_ey'].iloc[t - lb])
        if not np.isfinite(ey_ref):
            ey_ref = float(df['value_ey'].iloc[t])
        if not np.isfinite(ey_ref):
            return config.baseline
        exp_ret_lb = (1 + ey_ref) ** (lb / TRADING_DAYS_PER_YEAR) - 1

        excess  = ret_lb - exp_ret_lb
        var     = excess ** 2
        lb_time = lb / TRADING_DAYS_PER_YEAR
        max_dev_v = lb_time * (config.numb_sigmas_max_dev * config.stock_vol_assumed) ** 2
        dev_used  = min(config.deviation, config.deviation * var / max_dev_v)

        alloc = config.baseline + (-dev_used if excess > 0 else dev_used)
        return float(np.clip(alloc, 0.0, 1.0))

    def _momentum(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Momentum: follow 1-year equity direction."""
        lb = int(config.lookback_years * TRADING_DAYS_PER_YEAR)
        if t < lb:
            return config.baseline

        eq_real = df['eq_tri_real'].values
        # Annualize the lookback return (matches the docx formula and the
        # reference implementation when lookback != 1 year).
        ret_pa  = (eq_real[t] / eq_real[t - lb]) ** (TRADING_DAYS_PER_YEAR / lb) - 1.0
        mu_ref = float(df['value_ey'].iloc[t - lb])
        if not np.isfinite(mu_ref):
            mu_ref = float(df['value_ey'].iloc[t])
        if not np.isfinite(mu_ref):
            return config.baseline

        alloc = config.baseline + (config.deviation if ret_pa > mu_ref else -config.deviation)
        return float(np.clip(alloc, 0.0, 1.0))

    def _valmo(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Value + Momentum hybrid."""
        lb = int(config.lookback_years * TRADING_DAYS_PER_YEAR)
        if t < lb:
            return config.baseline

        eq_real  = df['eq_tri_real'].values
        momentum = 1 if eq_real[t] > eq_real[t - lb] else -1
        mom_frac = config.momentum_weight * config.baseline * momentum

        ey        = df['value_ey'].iloc[t]
        real_rate = df['real_rate'].iloc[t]
        if pd.isna(ey) or pd.isna(real_rate):
            return config.baseline

        rp     = ey - real_rate
        vol_sq = config.stock_vol_assumed ** 2
        opt    = rp / (config.crra * vol_sq)
        opt    = np.clip(
            opt,
            (1 - config.rp_max) * config.baseline,
            (1 + config.rp_max) * config.baseline,
        )

        alloc = opt + mom_frac
        return float(np.clip(alloc, 0.0, 1.0))

    # ── statistics ────────────────────────────────────────────────────────────
    def _calc_stats(
        self,
        df: pd.DataFrame,
        name: str,
        equity_alloc: np.ndarray,
        wealth: np.ndarray,
    ) -> Dict:
        daily_ret = np.diff(wealth) / wealth[:-1]
        daily_ret = daily_ret[np.isfinite(daily_ret)]

        # Annualisation
        n_days = len(df)
        years  = n_days / TRADING_DAYS_PER_YEAR

        cagr     = (wealth[-1] / wealth[0]) ** (1 / years) - 1 if years > 0 else 0.0
        daily_vol = np.std(daily_ret)
        ann_vol   = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe vs T-bill
        rf_daily = df['rf_ret'].iloc[1:].to_numpy(dtype=float)
        n_min    = min(len(rf_daily), len(daily_ret))
        excess   = daily_ret[:n_min] - rf_daily[:n_min]
        sharpe   = (np.mean(excess) / daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
                    if daily_vol > 0 else 0.0)

        # Max drawdown
        cum = np.cumprod(1 + daily_ret)
        cum = np.concatenate(([1.0], cum))
        peak = np.maximum.accumulate(cum)
        dd   = (cum - peak) / peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        return {
            'name':                  name,
            'cagr':                  cagr,
            'annual_vol':            ann_vol,
            'sharpe_ratio':          sharpe,
            'avg_equity_allocation': float(np.mean(equity_alloc)),
            'max_drawdown':          max_dd,
            'final_wealth':          float(wealth[-1]),
        }


# ── default configs ───────────────────────────────────────────────────────────
def create_configs() -> Dict[str, InvestorConfig]:
    return {
        'benchmark': InvestorConfig(name='static',   baseline=1.0),  # 100% equities
        'static':    InvestorConfig(name='static',   baseline=0.5),
        'cashflow':  InvestorConfig(name='cashflow', baseline=0.5,
                                    crra=3.0, stock_vol_assumed=0.16),
        'extrap':    InvestorConfig(name='extrap',   baseline=0.5,
                                    crra=3.0, stock_vol_assumed=0.16),
        'stmr':      InvestorConfig(name='stmr',     baseline=0.5,
                                    deviation=1/6,
                                    stmr_lookback_days=5,
                                    numb_sigmas_max_dev=2.5,
                                    stock_vol_assumed=0.16),
        'momentum':  InvestorConfig(name='momentum', baseline=0.5,
                                    lookback_years=1.0, deviation=1/6),
        'valmo':     InvestorConfig(name='valmo',    baseline=0.5,
                                    lookback_years=1.0,
                                    momentum_weight=1/3,
                                    rp_max=2/3,
                                    crra=3.0, stock_vol_assumed=0.16),
    }


# ── main runner ───────────────────────────────────────────────────────────────
def run_backtest(
    equity_region: str = 'US',
    start_date=None,
    end_date=None,
    data_path: Optional[str] = None,
) -> Dict:
    """
    Run all strategies and print a summary table.

    Parameters
    ----------
    equity_region : 'US' or 'XUS'
    start_date, end_date : str or Timestamp (default: full available history)
    data_path : path to hist_df.csv  (default: ../data/hist_df.csv)
    """
    bt      = ElmBacktest(data_path=data_path, equity_region=equity_region)
    configs = create_configs()

    if end_date is None:
        end_date = bt.df.index[-1]
    if start_date is None:
        start_date = pd.Timestamp('1997-01-01')  # 5-year warm-up from 1992 for extrap

    end_date   = pd.Timestamp(end_date)
    start_date = pd.Timestamp(start_date)

    print(f"\n{'='*80}")
    print(f"ELM BACKTEST  |  region: {equity_region}")
    print(f"Period: {start_date.date()} → {end_date.date()}")
    print(f"{'='*80}\n")

    results = {}
    for label, cfg in configs.items():
        print(f"  Running {label} …")
        r = bt.backtest_strategy(cfg, start_date=start_date, end_date=end_date)
        results[label] = r

    # ── summary table ──────────────────────────────────────────────────────
    rows = []
    for label, r in results.items():
        s = r['stats']
        rows.append({
            'Strategy':    label.upper(),
            'CAGR':        f"{s['cagr']:.2%}",
            'Ann Vol':     f"{s['annual_vol']:.2%}",
            'Sharpe':      f"{s['sharpe_ratio']:.3f}",
            'Avg Equity':  f"{s['avg_equity_allocation']:.1%}",
            'Max DD':      f"{s['max_drawdown']:.2%}",
            'Final $':     f"{s['final_wealth']:.2f}",
        })

    summary = pd.DataFrame(rows)
    print(f"\n{'-'*80}")
    print(summary.to_string(index=False))
    print(f"{'-'*80}\n")

    # ── save results ───────────────────────────────────────────────────────
    out = Path(__file__).parent.parent / 'results'
    out.mkdir(exist_ok=True)
    region_tag = equity_region.lower()

    summary.to_csv(out / f'elm_summary_{region_tag}.csv', index=False)
    for label, r in results.items():
        r['dataframe'].to_csv(out / f'elm_{region_tag}_{label}.csv')

    print(f"Results saved to: {out}")
    return results


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ELM investor backtest')
    parser.add_argument('--region',     default='US',
                        choices=['US', 'XUS'],
                        help="Equity universe: US or XUS (default: US)")
    parser.add_argument('--start',      default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',        default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--data',       default=None, help='Path to hist_df.csv')
    parser.add_argument('--explain',    action='store_true',
                        help='Print an explanation of the model/backtest and exit')
    args = parser.parse_args()

    if args.explain:
        print_explanation()
        raise SystemExit(0)

    run_backtest(
        equity_region=args.region,
        start_date=args.start,
        end_date=args.end,
        data_path=args.data,
    )
