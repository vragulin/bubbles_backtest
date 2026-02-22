# `backtest_elm.py` — Structure and Main Components

This document is a narrative code review of `src/backtest_elm.py`.

## What the script is

`backtest_elm.py` is a **self-contained daily backtest engine** for simplified “ELM-style” investor allocation rules that allocate between:

- **Equities**: a region-specific equity total return index (TRI)
- **Safe asset**: a T-bill TRI

It runs multiple stylized investor types (static, cashflow/Merton, extrapolator, STMR, momentum, valmo), prints a summary table, and saves per-strategy daily outputs to `results/`.

A defining convention of this backtest is that it uses **real (CPI-deflated) equity returns** when a signal depends on *past realized returns*, but it tracks portfolio wealth in **nominal dollars** using **nominal** TRIs.

## High-level layout

The file is arranged as a small library with a CLI wrapper:

1. **Module specification** (top docstring + `EXPLANATION` string)
2. **Constants** (`TRADING_DAYS_PER_YEAR`, `DEFAULT_WARMUP_YEARS`)
3. **Configuration** (`InvestorConfig` dataclass)
4. **Backtest engine** (`ElmBacktest` class)
5. **Default strategy configs** (`create_configs()`)
6. **Orchestration** (`run_backtest()`)
7. **CLI entry point** (`argparse` in `__main__`)

## 1) Module spec and “paper conventions”

At the top, a long module docstring and the `EXPLANATION` block act as an executable spec:

- Required columns expected in `data/hist_df.csv`.
- A **region switch** that maps `--region US` to `us_tri`/`us_pcaey` and `--region XUS` to `xus_tri`/`xus_pcaey`.
- Accounting conventions:
  - Wealth is tracked using nominal `eq_tri` and `tbill_tri`.
  - “Return-history-based” signals are computed from `eq_tri_real = eq_tri / cpi`.
  - The value/cashflow signal uses `value_ey = 1/CAPE` and compares it to `tips_10y` (treated as an annual real rate).

`print_explanation()` exists only to print this block when `--explain` is passed.

## 2) Configuration: `InvestorConfig`

`InvestorConfig` is a dataclass that centralizes parameters across strategies. The intent is:

- Keep the backtest engine generic (`backtest_strategy(config, ...)`).
- Keep each strategy’s math in a dedicated method.

Key fields:

- `name`: strategy identifier (`static`, `cashflow`, `extrap`, `stmr`, `momentum`, `valmo`).
- `baseline`: baseline equity allocation.
- `lookback_years`, `deviation`: tactical strategy knobs.
- `crra`, `stock_vol_assumed`: Merton sizing parameters.
- `extrap_*`: extrapolator’s lookback weights and squeeze parameters.
- `stmr_lookback_days`, `numb_sigmas_max_dev`: STMR controls.

## 3) Engine: `ElmBacktest`

`ElmBacktest` owns the dataset and provides `backtest_strategy()`.

### Data load and derived series (`__init__`)

`__init__`:

- Resolves the default `hist_df.csv` path.
- Chooses region-specific columns (`us_*` vs `xus_*`) and creates a standardized set:
  - `eq_tri` — equity TRI (nominal)
  - `value_ey` — the valuation signal (1/CAPE)
  - `tbill_tri` is used directly for the safe asset
- Computes convenience return series:
  - `eq_ret = eq_tri.pct_change()`
  - `rf_ret = tbill_tri.pct_change()`
- Computes the **real equity TRI** used for signal generation:
  - `eq_tri_real = eq_tri / cpi`
- Forward-fills `real_rate` (from `tips_10y`) and `value_ey`.

The engine drops the first NaN-return row and prints a brief “loaded N observations” message.

### Simulation loop (`backtest_strategy`)

`backtest_strategy()` is the main routine:

- Determines `start_date`/`end_date`, with a default start designed to allow a warm-up.
- Creates `full_df`, a slice that includes **extra history** before `start_date` so lookback signals can be computed.
- Finds the index where the reported backtest begins (`start_idx`).

The portfolio simulation uses **units accounting**:

- Equity is held as `equity_units` (units of `eq_tri`).
- The safe asset is held implicitly as “units of `tbill_tri`” derived from leftover wealth.

Each day:

1. Update equity value using the new `eq_tri`.
2. Update T-bill value using the `tbill_tri` ratio.
3. Sum to get new wealth.
4. Compute the day’s target equity weight `a_t` and rebalance by setting new `equity_units`.

#### Extrapolator special case

For `config.name == 'extrap'`, `backtest_strategy()` defines an `alloc_fn` closure that applies an **EWMA smoother** to the extrapolator’s current expected return estimate. This requires per-run state (previous smoothed estimate), which is why it’s implemented as a closure rather than in the pure `_extrap()` method.

## 4) Strategy dispatcher and implementations

### Dispatcher (`_allocation`)

`_allocation(config, df, t)` routes to the corresponding strategy method based on `config.name`.

### Cashflow / “Merton” (`_cashflow`)

Implements a clipped Merton-style allocation:

- Risk premium: `rp = value_ey - real_rate`
- Allocation: `alloc = rp / (gamma * sigma^2)`
- Clipped to `[0, 1]`

### Extrapolator (`_extrap_current_ret`, `_extrap`)

The extrapolator computes a weighted average of **past real annual log returns** of equities:

- For each lookback year, compute nominal log return of `eq_tri` and subtract CPI inflation.
- Subtract the real rate to form a raw risk premium.
- Apply a **tanh squeeze** around `extrap_center` with maximum deviation `extrap_max_dev`.

`_extrap_current_ret()` returns the *unsmoothed* expected real return (used by the EWMA in `backtest_strategy()`). `_extrap()` returns the allocation implied by the resulting risk premium.

### STMR: short-term mean reversion (`_stmr`)

STMR compares a short-horizon realized return (computed on `eq_tri_real`) to an “expected” return implied by the valuation signal at the start of the window. It then adjusts allocation up/down by a deviation that is **variance-scaled** and capped.

### Momentum (`_momentum`)

Momentum uses a real lookback (CPI-deflated TRI) and annualizes it for general lookback lengths. It compares that annualized realized return to a valuation reference and shifts allocation by `± deviation`.

### ValMo (`_valmo`)

ValMo combines:

- A clipped Merton/value allocation around `baseline` (bounded by `rp_max`), and
- A directional momentum tilt (`momentum_weight * baseline` times sign).

## 5) Statistics (`_calc_stats`)

`_calc_stats()` computes summary metrics from the wealth path:

- CAGR
- Annualized volatility
- Sharpe ratio (vs the observed daily T-bill return series `rf_ret`)
- Max drawdown
- Average equity allocation
- Final wealth

These metrics are returned as a dict and used for the console summary table and outputs.

## 6) Defaults and orchestration

### Default configs (`create_configs`)

`create_configs()` returns a dict of labeled `InvestorConfig` instances:

- `benchmark`: 100% equities
- `static`: 50% equities
- `cashflow`, `extrap`, `stmr`, `momentum`, `valmo`: parameterized versions of each investor type

### Runner (`run_backtest`)

`run_backtest()`:

- Instantiates `ElmBacktest` for a chosen region.
- Iterates through all configs, calling `backtest_strategy()`.
- Prints a summary table.
- Writes outputs to `results/`:
  - `elm_summary_{region}.csv`
  - `elm_{region}_{strategy}.csv` for each strategy

## 7) CLI entry point

The `__main__` block exposes:

- `--region {US,XUS}`
- `--start YYYY-MM-DD`
- `--end YYYY-MM-DD`
- `--data path/to/hist_df.csv`
- `--explain` to print the spec

Example:

```bash
python src/backtest_elm.py --region XUS
```
