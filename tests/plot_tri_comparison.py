"""
Compare US equity total return indices from two data sources:
  - hist_df.csv      (daily)   → us_tri          used by backtest_elm.py
  - shiller_data_processed.csv (monthly) → SPX_TR  used by backtest_investors.py

Both are normalised to 1.0 on START_DATE so growth paths can be compared directly.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

START_DATE = '1997-01-01'

root = Path(__file__).parent.parent

# ── load hist_df.csv (daily) ─────────────────────────────────────────────────
elm_df = pd.read_csv(root / 'data' / 'hist_df.csv', index_col=0, parse_dates=True)
elm_df = elm_df.sort_index()
elm_tri = elm_df['us_tri']

# ── load shiller_data_processed.csv (monthly) ──────────────────────────────
shl_df = pd.read_csv(root / 'results' / 'shiller_data_processed.csv',
                     index_col=0, parse_dates=True)
shl_df = shl_df.sort_index()
shl_tri = shl_df['SPX_TR']

# ── filter to START_DATE → end of overlap ───────────────────────────────────
start = pd.Timestamp(START_DATE)
end   = min(elm_tri.index[-1], shl_tri.index[-1])

elm_tri = elm_tri.loc[start:end]
shl_tri = shl_tri.loc[start:end]

# ── normalise to 1.0 at START_DATE ──────────────────────────────────────────
elm_tri = elm_tri / elm_tri.iloc[0]
shl_tri = shl_tri / shl_tri.iloc[0]

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# --- top panel: TRI levels ---
ax = axes[0]
ax.plot(elm_tri.index, elm_tri.values, label='hist_df  us_tri  (daily)',
        color='steelblue', linewidth=1.2)
ax.plot(shl_tri.index, shl_tri.values, label='shiller  SPX_TR  (monthly)',
        color='darkorange', linewidth=1.4, linestyle='--')
ax.set_title(f'US Equity Total Return Index  (normalised to 1.0 on {START_DATE})')
ax.set_ylabel('Cumulative Total Return')
ax.legend()
ax.grid(True, alpha=0.3)

# --- bottom panel: ratio (elm / shiller) to show divergence -----------------
# Align to monthly for ratio (use shiller dates as spine)
elm_monthly = elm_tri.resample('ME').last()
common_idx  = elm_monthly.index.intersection(shl_tri.index)
ratio = elm_monthly.loc[common_idx] / shl_tri.loc[common_idx]

ax2 = axes[1]
ax2.plot(ratio.index, ratio.values, color='seagreen', linewidth=1.2)
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_title('Ratio: hist_df us_tri  /  shiller SPX_TR  (monthly, both re-normalised to 1.0)')
ax2.set_ylabel('Ratio')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)

# annotate final ratio
final_ratio = ratio.iloc[-1]
ax2.annotate(f'Final ratio: {final_ratio:.4f}',
             xy=(ratio.index[-1], final_ratio),
             xytext=(-120, 20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)

plt.tight_layout()

out_path = root / 'results' / 'tri_comparison.png'
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")

# ── print summary stats ──────────────────────────────────────────────────────
years = (end - start).days / 365.25
elm_cagr = elm_tri.iloc[-1] ** (1 / years) - 1
shl_cagr = shl_tri.iloc[-1] ** (1 / years) - 1

print(f"\nPeriod : {START_DATE} to {end.date()}")
print(f"hist_df  us_tri  CAGR : {elm_cagr:.2%}")
print(f"shiller  SPX_TR  CAGR : {shl_cagr:.2%}")
print(f"Difference             : {elm_cagr - shl_cagr:+.2%}")
print(f"Final ratio (elm/shl)  : {final_ratio:.4f}")

plt.show()
