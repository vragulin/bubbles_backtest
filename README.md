# Bubbles Backtest (daily ELM backtest + return histograms)

This repo’s **primary deliverables** are:

1) A simplified daily “ELM-style” allocation backtest: `src/backtest_elm.py`
2) A script to plot simulated-vs-real daily return histograms (with optional KDE overlays): `src/plot_return_histograms.py`

Everything else in the repository should be considered as **drafts/supporting material**.

## 1) Run the daily ELM backtest (`src/backtest_elm.py`)

### Install

Minimum dependencies:

```bash
pip install numpy pandas
```

### US backtest

```bash
python src/backtest_elm.py --region US
```

### Non-US backtest

```bash
python src/backtest_elm.py --region XUS
```

### Optional arguments

- `--start YYYY-MM-DD` and `--end YYYY-MM-DD` to bound the sample
- `--data path/to/hist_df.csv` to point at a different dataset
- `--explain` to print the model/backtest conventions used by the script

### Outputs

The script prints a summary table and writes CSVs to `results/`:

- `results/elm_summary_us.csv` and/or `results/elm_summary_xus.csv`
- Per-strategy time series: `results/elm_{region}_{strategy}.csv`

### Code structure notes

A short narrative code review describing the structure and main components is in:

- `docs/backtest_elm_structure.md`

## 2) Generate return histograms (with KDE overlays)

This figure compares **simulated** per-period total-return “daily” returns from the random-walk-killer simulation to **real** daily returns from `data/hist_df.csv`.

### Install

```bash
pip install numpy pandas matplotlib
```

### Step A — export simulated “daily” TR returns

Generate `results/rw_daily_tr_returns.csv`:

```bash
python src/who_killed_rw_dist.py --n-sims 200 --out results/rw_daily_tr_returns.csv
```

### Step B — plot histograms + KDE

```bash
python src/plot_return_histograms.py \
  --sim results/rw_daily_tr_returns.csv \
  --hist data/hist_df.csv \
  --out results/return_histograms.png \
  --kde
```

Notes:

- The plotted variable is the daily total **log return** `log(1+r)`.
- This also writes distribution-moment stats to `results/return_distribution_stats.csv` (computed on `log(1+r)` increments).
- By default, out-of-range observations are clipped into the edge histogram bins; use `--drop-tails` to drop them instead.

## Appendix (supporting scripts)

Other scripts include the paper’s reference simulation (`src/who_killed_rw.py`), a monthly historical backtest draft (`src/backtest_investors.py`), data-processing utilities (e.g., `src/load_shiller_data.py`, `src/download_rf_rate.py`), and helper plotting scripts (e.g., `tests/plot_tri_comparison.py`). These are useful for validation and experimentation but are not the main “run this first” entry points.

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
