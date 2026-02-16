"""
Backtest of 6 Core Investor Classes on Historical S&P Data
Uses actual S&P 500 price and earnings data from Shiller dataset
No equilibrium price formation - just strategy-based portfolio allocation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


@dataclass
class InvestorConfig:
    """Configuration for each investor type
    
    Parameters:
        name: Strategy identifier (e.g., 'cashflow', 'extrap', 'momentum')
        baseline: Base equity allocation (0.0 to 1.0)
        lookback_years: Historical window for momentum/mean reversion strategies
        deviation: Allocation adjustment magnitude for tactical strategies
        crra: Coefficient of relative risk aversion (for Merton allocation)
        stock_vol_assumed: Assumed annual stock volatility (default 16%)
        momentum_weight: Weight on momentum signal in valmo strategy
        rp_max: Maximum risk premium deviation for valmo (relative to baseline)
        use_cape: If True, uses CAPE yield (10-year smoothed); if False, uses trailing earnings yield.
                  CAPE yield is more stable and less cyclical than trailing earnings.
                  Default is True (recommended for long-term strategies).
        use_lagged_earnings: If True, uses earnings from prior period scaled by CPI growth.
                             This is more realistic since current earnings are not available in real-time.
                             Default is True (recommended for realistic backtesting).
        earnings_lag_months: Number of months to lag earnings data (default 1).
                             Earnings at time T are assumed to be from T-lag, scaled by CPI growth.
        extrap_weights: Lookback weights for extrapolator (5 years)
        extrap_center: Target expected return for extrapolator squeeze
        extrap_max_dev: Maximum deviation from center for extrapolator
        extrap_squeeze: Squeeze function parameter for extrapolator
        stmr_lookback_months: Lookback window for short-term mean reversion
        numb_sigmas_max_dev: Maximum deviation threshold for STMR (in sigmas)
    """
    name: str
    baseline: float  # baseline equity allocation
    lookback_years: float = 0.0  # for momentum/mean reversion strategies
    deviation: float = 0.0  # allocation adjustment magnitude
    
    # Strategy-specific parameters
    crra: float = 3.0  # coefficient of relative risk aversion
    stock_vol_assumed: float = 0.16
    momentum_weight: float = 0.0  # for valmo
    rp_max: float = 0.0  # for valmo
    use_cape: bool = True  # use CAPE yield instead of trailing earnings yield
    use_lagged_earnings: bool = True  # use lagged earnings scaled by CPI (more realistic)
    earnings_lag_months: int = 1  # how many months to lag earnings data
    
    # Extrap-specific
    extrap_weights: tuple = (0.30, 0.25, 0.20, 0.15, 0.10)
    extrap_center: float = 0.04
    extrap_max_dev: float = 0.03
    extrap_squeeze: float = 0.1
    
    # STMR-specific
    stmr_lookback_months: int = 1  # 1 month lookback
    numb_sigmas_max_dev: float = 2.5


class InvestorBacktest:
    """Backtest framework for investor strategies on historical data"""
    
    def __init__(self, data_path: str = None, rf_data_path: str = None):
        """Initialize with historical data"""
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'results' / 'shiller_data_processed.csv'
        
        # Load main data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Load risk-free rate data
        if rf_data_path is None:
            # Try to load FEDFUNDS data first, then fallback
            rf_file = Path(__file__).parent.parent / 'data' / 'FEDFUNDS.csv'
            fallback_file = Path(__file__).parent.parent / 'data' / 'TB3MS_fallback.csv'
            
            if rf_file.exists():
                # Load FEDFUNDS data
                rf_df_raw = pd.read_csv(rf_file)
                rf_df_raw['observation_date'] = pd.to_datetime(rf_df_raw['observation_date'])
                
                # Convert to end-of-month dates to match main data
                rf_df_raw['observation_date'] = rf_df_raw['observation_date'] + pd.offsets.MonthEnd(0)
                
                rf_df_raw = rf_df_raw.set_index('observation_date')
                rf_df_raw.index.name = 'Date'
                
                # Convert annual rate (in percent) to monthly decimal return
                # FEDFUNDS is in percent per year
                rf_df = pd.DataFrame(index=rf_df_raw.index)
                rf_df['RF_Monthly'] = (1 + rf_df_raw['FEDFUNDS'] / 100) ** (1/12) - 1
                
            elif fallback_file.exists():
                rf_df = pd.read_csv(fallback_file, index_col=0, parse_dates=True)
            else:
                # Create fallback on the fly
                print("Risk-free rate data not found. Creating fallback...")
                from download_rf_rate import create_fallback_rf_rate
                rf_df = create_fallback_rf_rate()
        else:
            rf_df = pd.read_csv(rf_data_path, index_col=0, parse_dates=True)
        
        # Merge RF rate with main data (forward fill for any missing dates)
        self.df = self.df.merge(rf_df[['RF_Monthly']], left_index=True, right_index=True, how='left')
        self.df['RF_Monthly'] = self.df['RF_Monthly'].ffill().bfill()
        
        # Calculate returns
        self.df['Price_Return'] = self.df['SPX'].pct_change()
        self.df['TR_Return'] = self.df['SPX_TR'].pct_change()
        
        # Forward fill earnings to handle NaN values
        self.df['E'] = self.df['E'].ffill()
        self.df['Earnings_Yield'] = self.df['E'] / self.df['SPX']
        
        # Calculate CAPE yield (inverse of CAPE ratio)
        # CAPE is Cyclically Adjusted P/E using 10-year average real earnings
        self.df['CAPE_Yield'] = 1.0 / self.df['CAPE']
        
        # Real rate: RF rate minus inflation for Merton allocation
        # Approximate inflation as 12-month CPI change
        self.df['Inflation'] = self.df['CPI'].pct_change(12)
        # Annualize the monthly RF rate for comparison with annual inflation
        self.df['RF_Annual'] = (1 + self.df['RF_Monthly']) ** 12 - 1
        self.df['Real_Rate'] = self.df['RF_Annual'] - self.df['Inflation']
        
        # For missing real rate, use long-term average
        self.df['Real_Rate'] = self.df['Real_Rate'].fillna(self.df['Real_Rate'].mean())
        
        # Bond returns: use actual 10-year bond total return for portfolio allocation
        # Bond_10y_TR_monthly is already a monthly return multiplier, convert to return
        self.df['Bond_Return'] = self.df['Bond_10y_TR_monthly'].fillna(1.0) - 1.0
        
        self.n_periods = len(self.df)
        
    def get_default_dates(self, years_back: int = 40):
        """Get default start and end dates for backtesting"""
        end_date = self.df.index[-1]
        start_date = end_date - pd.DateOffset(years=years_back)
        return start_date, end_date
        
    def backtest_strategy(self, config: InvestorConfig, start_date=None, end_date=None,
                         initial_wealth: float = 1.0) -> Dict:
        """
        Backtest a single investor strategy
        
        Args:
            config: Investor configuration
            start_date: Start date for backtest (default: 40 years before end_date)
            end_date: End date for backtest (default: last date in dataset)
            initial_wealth: Starting wealth
        
        Returns:
            Dictionary with wealth series, allocation series, and performance stats
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = self.df.index[-1]
        else:
            end_date = pd.Timestamp(end_date)
            
        if start_date is None:
            start_date = end_date - pd.DateOffset(years=40)
        else:
            start_date = pd.Timestamp(start_date)
        
        # Include lookback history before start_date (max lookback is 5 years for extrap)
        lookback_start = start_date - pd.DateOffset(years=5)
        
        # Get full data including lookback period
        full_df = self.df[(self.df.index >= lookback_start) & (self.df.index <= end_date)].copy()
        
        # Find the index where backtest actually starts (for wealth tracking)
        start_idx = full_df.index.get_loc(start_date) if start_date in full_df.index else full_df.index.searchsorted(start_date)
        
        # Number of periods in backtest (wealth tracking period)
        n_backtest = len(full_df) - start_idx
        
        if n_backtest < 12:
            raise ValueError(f"Not enough data for backtesting from {start_date}")
        
        # Calculate initial earnings yield from historical data (before backtest starts)
        # This serves as the benchmark for STMR and Momentum strategies
        if start_idx >= 12:
            # Use the 12 months immediately before backtest start
            historical_ey = full_df['Earnings_Yield'].iloc[start_idx-12:start_idx].mean()
        else:
            # Not enough pre-backtest data, use all available historical data
            historical_ey = full_df['Earnings_Yield'].iloc[:start_idx].mean()
        
        # Store for use in allocation calculations
        self._backtest_init_ey = historical_ey if not pd.isna(historical_ey) else 0.05
        self._backtest_start_idx = start_idx
        
        # Initialize arrays (only for backtest period)
        wealth = np.zeros(n_backtest)
        equity_allocation = np.zeros(n_backtest)
        equity_shares = np.zeros(n_backtest)
        
        wealth[0] = initial_wealth
        
        # Calculate initial allocation using actual historical lookback data
        # t=start_idx in full_df corresponds to t=0 in wealth arrays
        equity_allocation[0] = self._calculate_allocation(config, full_df, start_idx, wealth[0])
        
        equity_shares[0] = equity_allocation[0] * wealth[0] / full_df['SPX'].iloc[start_idx]
        
        # Simulate forward period by period
        for t in range(1, n_backtest):
            # Map backtest index t to full_df index
            df_idx = start_idx + t
            df_idx_prev = start_idx + t - 1
            
            prev_wealth = wealth[t-1]
            prev_shares = equity_shares[t-1]
            prev_price = full_df['SPX'].iloc[df_idx_prev]
            curr_price = full_df['SPX'].iloc[df_idx]
            
            # Calculate wealth from price changes and bond returns
            stock_value = prev_shares * curr_price
            bond_value = (prev_wealth - prev_shares * prev_price) * (1 + full_df['Bond_Return'].iloc[df_idx])
            
            wealth[t] = stock_value + bond_value
            
            # Calculate new allocation based on strategy (using full_df with lookback history)
            equity_allocation[t] = self._calculate_allocation(
                config, full_df, df_idx, wealth[t]
            )
            
            # Update shares based on new allocation
            equity_shares[t] = equity_allocation[t] * wealth[t] / curr_price
        
        # Create result dataframe (only backtest period)
        backtest_df = full_df.iloc[start_idx:].copy()
        backtest_df['Wealth'] = wealth
        backtest_df['Equity_Allocation'] = equity_allocation
        
        stats = self._calculate_statistics(
            backtest_df, config.name, equity_allocation, wealth
        )
        
        return {
            'config': config,
            'wealth': wealth,
            'equity_allocation': equity_allocation,
            'dataframe': backtest_df,
            'stats': stats
        }
    
    def _get_earnings_yield(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Get earnings yield, optionally lagged and scaled by CPI growth
        
        Args:
            config: Investor configuration
            df: DataFrame with market data
            t: Current time index
            
        Returns:
            Earnings yield (or CAPE yield) for allocation calculation, or None if invalid
        """
        yield_col = 'CAPE_Yield' if config.use_cape else 'Earnings_Yield'
        
        if config.use_lagged_earnings and t >= config.earnings_lag_months:
            # Use lagged earnings scaled by CPI growth to approximate current earnings
            lag = config.earnings_lag_months
            
            if config.use_cape:
                # For CAPE yield: scale the yield by inverse CPI growth
                # CAPE_Yield[t] ≈ CAPE_Yield[t-lag] × (CPI[t-lag] / CPI[t])
                lagged_yield = df[yield_col].iloc[t - lag]
                cpi_t = df['CPI'].iloc[t]
                cpi_lagged = df['CPI'].iloc[t - lag]
                if not pd.isna(cpi_t) and not pd.isna(cpi_lagged) and cpi_t > 0:
                    ey = lagged_yield * (cpi_lagged / cpi_t)
                else:
                    ey = lagged_yield
            else:
                # For trailing earnings: scale earnings by CPI growth rate
                # E[t] ≈ E[t-lag] × (CPI[t] / CPI[t-lag])
                lagged_earnings = df['E'].iloc[t - lag]
                current_price = df['SPX'].iloc[t]
                cpi_t = df['CPI'].iloc[t]
                cpi_lagged = df['CPI'].iloc[t - lag]
                if not pd.isna(cpi_t) and not pd.isna(cpi_lagged) and cpi_lagged > 0:
                    scaled_earnings = lagged_earnings * (cpi_t / cpi_lagged)
                    ey = scaled_earnings / current_price
                else:
                    ey = lagged_earnings / current_price
        else:
            # Use current period earnings (unrealistic but for comparison)
            ey = df[yield_col].iloc[t]
        
        # Fallback to historical average if NaN or non-positive
        if pd.isna(ey) or ey <= 0:
            ey = df[yield_col].iloc[:t].mean()
            if pd.isna(ey) or ey <= 0:
                return None
        
        return ey
    
    def _calculate_allocation(self, config: InvestorConfig, df: pd.DataFrame,
                             t: int, wealth: float) -> float:
        """Calculate equity allocation for current period based on strategy"""
        
        if config.name == 'static':
            return config.baseline
        
        elif config.name == 'cashflow':
            # Earnings yield based allocation (Merton)
            # Use CAPE yield (10-year smoothed) or trailing earnings yield
            # Optionally lag earnings by 1 month scaled by CPI (more realistic)
            ey = self._get_earnings_yield(config, df, t)
            if ey is None:
                return config.baseline
            
            real_rate = df['Real_Rate'].iloc[t]
            if pd.isna(real_rate):
                real_rate = df['Real_Rate'].mean()
            
            rp = ey - real_rate
            vol_sq = config.stock_vol_assumed ** 2
            alloc = rp / (config.crra * vol_sq)
            return np.clip(alloc, 0.0, 1.0)
        
        elif config.name == 'extrap':
            # Extrapolator: weighted lookback returns with squeeze
            return self._extrap_allocation(config, df, t)
        
        elif config.name == 'stmr':
            # Short-term mean reversion
            return self._stmr_allocation(config, df, t)
        
        elif config.name == 'momentum':
            # Momentum based on 1-year return
            return self._momentum_allocation(config, df, t)
        
        elif config.name == 'valmo':
            # Value + Momentum hybrid
            return self._valmo_allocation(config, df, t)
        
        else:
            return config.baseline
    
    def _extrap_allocation(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Extrapolator allocation with weighted lookback and squeeze"""
        if t < 60:  # Need 5 years of monthly data
            return config.baseline
        
        weights = config.extrap_weights
        
        # Calculate weighted average return over past 5 consecutive annual periods
        # Each weight applies to a separate 1-year period (not overlapping cumulative returns)
        avg_ret = 0.0
        for i, w in enumerate(weights):
            year_end = t - 12 * i  # End of this year's period
            year_start = t - 12 * (i + 1)  # Start of this year's period
            if year_start >= 0:
                ret = np.log(df['SPX_TR'].iloc[year_end] / df['SPX_TR'].iloc[year_start])
                avg_ret += w * ret
        
        # Squeeze around center
        real_rate = df['Real_Rate'].iloc[t]
        rp_raw = avg_ret - real_rate
        dev = rp_raw - config.extrap_center
        squeezed = config.extrap_max_dev * np.tanh(dev / config.extrap_squeeze)
        expected_return = config.extrap_center + squeezed + real_rate
        
        # Merton allocation
        rp = expected_return - real_rate
        vol_sq = config.stock_vol_assumed ** 2
        alloc = rp / (config.crra * vol_sq)
        return np.clip(alloc, 0.0, 1.0)
    
    def _stmr_allocation(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Short-term mean reversion allocation with variance-adjusted deviation"""
        lookback = config.stmr_lookback_months
        if t < lookback:
            return config.baseline
        
        # Period return over lookback
        return_period = df['SPX_TR'].iloc[t] / df['SPX_TR'].iloc[t - lookback] - 1
        
        # Expected return using historical earnings yield as fixed benchmark
        # (calculated from pre-backtest period)
        init_ey = getattr(self, '_backtest_init_ey', 0.05)
        expected_return_period = (1 + init_ey) ** (lookback / 12) - 1
        
        # Excess return and its variance
        excess_return = return_period - expected_return_period
        variance = excess_return ** 2
        
        # Variance-adjusted deviation (matches who_killed_rw.py logic)
        lookback_time = lookback / 12  # convert months to years
        max_dev_v = lookback_time * (config.numb_sigmas_max_dev * config.stock_vol_assumed) ** 2
        deviation_used = min(config.deviation, config.deviation * variance / max_dev_v)
        
        # Adjust allocation: negative excess → higher allocation
        if excess_return < 0:
            alloc = config.baseline + deviation_used
        else:
            alloc = config.baseline - deviation_used
        
        return np.clip(alloc, 0.0, 1.0)
    
    def _momentum_allocation(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Momentum allocation based on 1-year return"""
        lookback = int(config.lookback_years * 12)
        if t < lookback:
            return config.baseline
        
        # 1-year return
        ret_annual = df['SPX_TR'].iloc[t] / df['SPX_TR'].iloc[t - lookback] - 1
        
        # Compare to historical earnings yield as fixed benchmark
        # (calculated from pre-backtest period)
        init_ey = getattr(self, '_backtest_init_ey', 0.05)
        
        if ret_annual > init_ey:
            alloc = config.baseline + config.deviation
        else:
            alloc = config.baseline - config.deviation
        
        return np.clip(alloc, 0.0, 1.0)
    
    def _valmo_allocation(self, config: InvestorConfig, df: pd.DataFrame, t: int) -> float:
        """Value + Momentum hybrid allocation"""
        lookback = int(config.lookback_years * 12)
        if t < lookback:
            return config.baseline
        
        # Momentum component
        tr_change = df['SPX_TR'].iloc[t] - df['SPX_TR'].iloc[t - lookback]
        if tr_change > 0:
            momentum_frac = config.momentum_weight * config.baseline
        else:
            momentum_frac = -config.momentum_weight * config.baseline
        
        # Value component (earnings yield)
        # Use CAPE yield (10-year smoothed) or trailing earnings yield
        # Optionally lag earnings by 1 month scaled by CPI (more realistic)
        ey = self._get_earnings_yield(config, df, t)
        if ey is None:
            return config.baseline
        
        real_rate = df['Real_Rate'].iloc[t]
        if pd.isna(real_rate):
            real_rate = df['Real_Rate'].mean()
            
        rp_val = ey - real_rate
        vol_sq = config.stock_vol_assumed ** 2
        
        opt = rp_val / (config.crra * vol_sq)
        opt = np.clip(
            opt,
            (1 - config.rp_max) * config.baseline,
            (1 + config.rp_max) * config.baseline
        )
        
        alloc = opt + momentum_frac
        return np.clip(alloc, 0.0, 1.0)
    
    def _calculate_statistics(self, df: pd.DataFrame, name: str,
                            equity_allocation: np.ndarray, wealth: np.ndarray) -> Dict:
        """Calculate performance statistics"""
        
        # Wealth returns
        wealth_returns = np.diff(wealth) / wealth[:-1]
        
        # Remove infinite/nan returns
        wealth_returns = wealth_returns[np.isfinite(wealth_returns)]
        
        # Years for annualization
        years = len(df) / 12
        
        # CAGR
        cagr = (wealth[-1] / wealth[0]) ** (1 / years) - 1 if years > 0 else 0.0
        
        # Annualized volatility
        monthly_vol = np.std(wealth_returns)
        annual_vol = monthly_vol * np.sqrt(12)
        
        # Sharpe ratio (using risk-free rate as benchmark, not bond returns)
        rf_monthly = df['RF_Monthly'].iloc[1:].values
        if len(rf_monthly) > len(wealth_returns):
            rf_monthly = rf_monthly[:len(wealth_returns)]
        elif len(rf_monthly) < len(wealth_returns):
            wealth_returns = wealth_returns[:len(rf_monthly)]
            
        excess_return = np.mean(wealth_returns) - np.mean(rf_monthly)
        sharpe = (excess_return / monthly_vol * np.sqrt(12)) if monthly_vol > 0 else 0.0
        
        # Average equity allocation
        avg_equity = np.mean(equity_allocation)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + wealth_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'name': name,
            'cagr': cagr,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe,
            'avg_equity_allocation': avg_equity,
            'max_drawdown': max_drawdown,
            'final_wealth': wealth[-1]
        }


def create_investor_configs() -> Dict[str, InvestorConfig]:
    """Create configurations for the 6 core investor types plus SPX benchmark"""
    
    configs = {
        'spx': InvestorConfig(
            name='spx',
            baseline=1.0  # 100% equity
        ),
        
        'static': InvestorConfig(
            name='static',
            baseline=0.5
        ),
        
        'cashflow': InvestorConfig(
            name='cashflow',
            baseline=0.5,  # not used, calculated via Merton
            crra=3.0,
            stock_vol_assumed=0.16
        ),
        
        'extrap': InvestorConfig(
            name='extrap',
            baseline=0.5,
            crra=3.0,
            stock_vol_assumed=0.16,
            extrap_weights=(0.30, 0.25, 0.20, 0.15, 0.10),
            extrap_center=0.04,
            extrap_max_dev=0.03,
            extrap_squeeze=0.1
        ),
        
        'stmr': InvestorConfig(
            name='stmr',
            baseline=0.5,
            deviation=1/6,
            stmr_lookback_months=1,
            numb_sigmas_max_dev=2.5,
            stock_vol_assumed=0.16  # needed for variance adjustment
        ),
        
        'momentum': InvestorConfig(
            name='momentum',
            baseline=0.5,
            lookback_years=1.0,
            deviation=1/6
        ),
        
        'valmo': InvestorConfig(
            name='valmo',
            baseline=0.6,
            lookback_years=1.0,
            momentum_weight=1/3,
            rp_max=2/3,
            crra=3.0,
            stock_vol_assumed=0.16
        )
    }
    
    return configs


def run_backtest_comparison(start_date=None, end_date=None):
    """Run backtest for all 6 investor types and compare results
    
    Args:
        start_date: Start date for backtest (default: 40 years before end_date)
        end_date: End date for backtest (default: last date in dataset)
    """
    # Initialize backtest framework
    backtest = InvestorBacktest()
    
    # Set default dates if not provided
    if end_date is None:
        end_date = backtest.df.index[-1]
    else:
        end_date = pd.Timestamp(end_date)
    
    if start_date is None:
        start_date = end_date - pd.DateOffset(years=40)
    else:
        start_date = pd.Timestamp(start_date)
    
    print(f"\n{'='*80}")
    print(f"INVESTOR STRATEGY BACKTEST - Historical S&P 500 Data")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}\n")
    
    # Create investor configurations
    configs = create_investor_configs()
    
    # Run backtests
    results = {}
    for name, config in configs.items():
        print(f"Backtesting {name.upper()} strategy...")
        result = backtest.backtest_strategy(config, start_date=start_date, end_date=end_date)
        results[name] = result
    
    # Display results
    print(f"\n{'-'*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'-'*80}\n")
    
    # Create summary DataFrame
    summary_data = []
    for name, result in results.items():
        stats = result['stats']
        summary_data.append({
            'Strategy': name.upper(),
            'CAGR': f"{stats['cagr']:.2%}",
            'Annual Vol': f"{stats['annual_vol']:.2%}",
            'Sharpe Ratio': f"{stats['sharpe_ratio']:.3f}",
            'Avg Equity %': f"{stats['avg_equity_allocation']:.1%}",
            'Max Drawdown': f"{stats['max_drawdown']:.2%}",
            'Final Wealth': f"${stats['final_wealth']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()
    
    # Save results
    results_path = Path(__file__).parent.parent / 'results'
    results_path.mkdir(exist_ok=True)
    
    summary_df.to_csv(results_path / 'investor_backtest_summary.csv', index=False)
    
    # Save detailed results for each investor
    for name, result in results.items():
        result['dataframe'].to_csv(results_path / f'investor_backtest_{name}.csv')
    
    print(f"Results saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    # Run backtest comparison
    # Default: last 40 years of data
    results = run_backtest_comparison(start_date='1986-02-01', end_date='2026-02-01')
