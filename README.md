# Bubbles Backtest: Historical Analysis of Investor Strategies

This repository contains code for backtesting six investor strategies on historical S&P 500 data, inspired by the methodology in the "Who Killed the Random Walk?" paper ([Haghani, Ragulin, Rosenbluth, White](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5072895)).

## Overview

The `backtest_investors.py` script applies the investor strategies from the multi-agent simulation model to actual historical market data, allowing us to evaluate how these different investment approaches would have performed in real markets rather than in simulation.

## Reference Implementation

The file `who_killed_rw.py` contains the original multi-agent simulation model from the paper, sourced from:
- **Repository**: https://github.com/vhaghani/multiagent-sim
- **Original file**: `multi_7_agent_model_with_sims_public 20260214 updated.py`
- **Paper**: "Who Killed the Random Walk? How Extrapolators Create Booms, Busts, Trends, and Opportunity"

This reference implementation provides the theoretical foundation for the investor strategies tested in our historical backtest.

## Main Script: `backtest_investors.py`

### What It Does

The script backtests six investor strategies on historical S&P 500 data:

1. **SPX** - 100% equity benchmark (buy-and-hold)
2. **Static** - Constant 50% equity / 50% bond allocation
3. **Cashflow** - Merton optimal allocation based on current earnings yield
4. **Extrap** - Extrapolator with 5-year weighted lookback returns and squeeze function
5. **STMR** - Short-term mean reversion (1-month lookback)
6. **Momentum** - 1-year momentum-based allocation
7. **Valmo** - Hybrid value + momentum strategy

### Key Features

- Uses actual historical S&P 500 prices and earnings from the Shiller dataset
- Implements actual 10-year Treasury bond returns for portfolio allocation
- Uses Federal Funds rate for Sharpe ratio calculations and Merton allocations
- Includes 5 years of lookback data before backtest start for proper initialization
- Calculates performance metrics: CAGR, volatility, Sharpe ratio, max drawdown

### How to Run

**Basic usage** (backtests last 40 years of data):
```bash
python src/backtest_investors.py
```

**Custom date range**:
```python
from src.backtest_investors import run_backtest_comparison

# Backtest from 1986 to 2026
results = run_backtest_comparison(start_date='1986-02-01', end_date='2026-02-01')
```

**Output**: 
- Console summary table with performance statistics
- Detailed CSV files saved to `results/` directory:
  - `investor_backtest_summary.csv` - Performance summary for all strategies
  - `investor_backtest_{strategy}.csv` - Period-by-period wealth and allocation for each strategy

## Data Files

The repository includes essential data files for replication:

- `data/ie_data_only.csv` - Shiller S&P 500 dataset (prices, earnings, dividends, CPI)
- `data/FEDFUNDS.csv` - Federal Funds rate (risk-free rate proxy)
- `data/TB3MS_fallback.csv` - Fallback risk-free rate data
- `results/shiller_data_processed.csv` - Processed dataset with calculated returns

## Other Scripts

### `load_shiller_data.py`

Processes raw Shiller data to create the analysis-ready dataset:
- Calculates bond returns using 10-year Treasury yields
- Computes S&P 500 total return index
- Handles missing data and forward-fills earnings

**Usage**:
```bash
python src/load_shiller_data.py
```

### `download_rf_rate.py`

Downloads risk-free rate data from FRED (Federal Reserve Economic Data):
- Primary: 3-month Treasury Bill rate (TB3MS)
- Fallback: Approximates from 10-year Treasury yield (GS10)

**Usage**:
```bash
python src/download_rf_rate.py
```

## Tests

The `tests/` directory contains pytest unit tests:

**Run all tests**:
```bash
pytest tests/
```

**Test coverage includes**:
- Bond pricing calculations (semi-annual compounding, fractional maturities)
- Par bonds, premium/discount bonds, zero-coupon bonds
- Edge cases and known analytical solutions

## Requirements

```bash
pip install numpy pandas
```

Optional (for downloading data):
```bash
pip install urllib3
```

For running tests:
```bash
pip install pytest
```

## Key Differences from Simulation

Unlike the multi-agent simulation in `who_killed_rw.py`, this backtest:

1. **Uses actual historical data** - Real S&P 500 prices and earnings from 1871-2026
2. **No equilibrium price formation** - Markets determine prices, not agent demand/supply clearing
3. **Actual bond returns** - Uses 10-year Treasury total returns instead of risk-free rate
4. **Historical initialization** - Strategies start with actual 5-year lookback data, not simulated history

This approach allows us to evaluate whether the insights from the theoretical model hold up when tested against real market behavior.

## Results Interpretation

The backtest reveals how different investor behaviors would have fared historically:
- **Extrap** strategies tend to increase equity allocation after strong markets
- **STMR** strategies buy dips and reduce after rallies
- **Momentum** strategies chase trends
- **Cashflow** strategies adjust based on valuation (earnings yield)
- **Valmo** combines value and momentum signals

Performance metrics show the risk-return tradeoffs and the historical effectiveness of each approach.

## License

This code is provided for research and educational purposes. Please cite the original "Who Killed the Random Walk?" paper when using these methodologies.

## References

Haghani, V., Ragulin, V., Rosenbluth, J., & White, J. "Who Killed the Random Walk? How Extrapolators Create Booms, Busts, Trends, and Opportunity" (2025).
