"""Shock-vs-price attribution analysis for the Who Killed the Random Walk? simulation.

Investigates whether large price moves in the simulation are driven by large
exogenous shocks (earnings, interest rates, investor flows) or instead emerge
from the fragile equilibrium among heterogeneous investors even when shocks
are ordinary.

Runs N Monte Carlo simulations, pools all observations, and produces three
complementary analyses for each horizon (daily / weekly / monthly):

    A) Conditional-mean table — average shock size within quantile bins of
       price return, showing whether extreme moves coincide with extreme shocks.

    B) R² regression — OLS of price return on the 5 shock variables, reporting
       coefficients, t-stats, and R² to quantify explained variance.

    C) Scatter plots — ΔP vs each shock variable, with tail observations
       (top/bottom 0.01%) highlighted in colour.

Usage
-----
    python src/shock_vs_price_analysis.py [--n-sims 30] [--out-dir results/big_p_moves]

The script inherits the default SimulationParams from who_killed_rw.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Ensure `src/` is on sys.path so we can import the simulation engine.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from who_killed_rw import SimulationParams, run_simulation  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data extraction
# ═══════════════════════════════════════════════════════════════════════════

# Horizons: (label, number_of_periods)
HORIZONS = [
    ("daily", 1),
    ("weekly", 5),
    ("monthly", 20),
]

# The 5 right-hand-side variables and how each is aggregated over a window.
#   "log_diff"  → log(level[t]) − log(level[t-h])    (for GBM-type: earnings)
#   "diff"      → level[t] − level[t-h]               (for OU-type: real rate)
#   "sum_shock" → sum of raw shocks over [t-h+1 .. t]  (for i.i.d. flows)
RHS_SPEC = {
    "d_log_earn": ("log_diff", "earnings"),
    "d_rate":     ("diff",     "real_rate"),
    "cf_shock":   ("sum_shock", "cashflow_flow"),
    "static_shock": ("sum_shock", "static_flow"),
    "extrap_shock": ("sum_shock", "extrap_flow"),
}

RHS_LABELS = {
    "d_log_earn":    "Δ log(Earnings)",
    "d_rate":        "Δ Real Rate",
    "cf_shock":      "Cashflow Flow Shock",
    "static_shock":  "Static Flow Shock",
    "extrap_shock":  "Extrap Flow Shock",
}

RHS_SHORT = {
    "d_log_earn":    "Δ ln E",
    "d_rate":        "Δ r",
    "cf_shock":      "CF shock",
    "static_shock":  "Static shock",
    "extrap_shock":  "Extrap shock",
}


def extract_sim_data(
    params: SimulationParams,
    n_sims: int,
    base_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run simulations and build one pooled DataFrame per horizon.

    Each DataFrame has columns:  sim, t, dp, d_log_earn, d_rate,
    cf_shock, static_shock, extrap_shock.
    """
    ppy = params.periods_per_year
    n_per = params.n_periods  # usable indices: 0 .. n_per

    # Collect raw arrays first (faster than building DataFrames in the loop)
    raw: dict[str, list[np.ndarray]] = {label: [] for label, _ in HORIZONS}

    t0 = time.perf_counter()
    for i in range(n_sims):
        seed = base_seed + i
        market, _inv, shocks = run_simulation(params, seed=seed)

        # Level arrays (length n_per + 1, index 0 .. n_per)
        price = np.asarray(market.price, dtype=np.float64)
        earnings = np.asarray(market.earnings, dtype=np.float64)
        real_rate = np.asarray(market.real_rate, dtype=np.float64)

        # Shock arrays (same length)
        cf_flow = np.asarray(shocks.cashflow_flow, dtype=np.float64)
        st_flow = np.asarray(shocks.static_flow, dtype=np.float64)
        ex_flow = np.asarray(shocks.extrap_flow, dtype=np.float64)

        for label, h in HORIZONS:
            # Indices: t runs from h to n_per  (non-overlapping would be
            # t = h, 2h, 3h, ... but that throws away data; we use
            # overlapping windows since observations are pooled across
            # independent simulations, limiting cross-sim dependence.)
            t_end = np.arange(h, n_per + 1)
            t_start = t_end - h

            dp = price[t_end] / price[t_start] - 1.0
            d_log_e = np.log(earnings[t_end]) - np.log(earnings[t_start])
            d_r = real_rate[t_end] - real_rate[t_start]

            # Flow shocks: sum over the window [t_start+1 .. t_end]
            # Use cumsum trick for O(n) computation of all rolling sums.
            cum_cf = np.concatenate([[0.0], np.cumsum(cf_flow)])
            cum_st = np.concatenate([[0.0], np.cumsum(st_flow)])
            cum_ex = np.concatenate([[0.0], np.cumsum(ex_flow)])

            sum_cf = cum_cf[t_end + 1] - cum_cf[t_start + 1]
            sum_st = cum_st[t_end + 1] - cum_st[t_start + 1]
            sum_ex = cum_ex[t_end + 1] - cum_ex[t_start + 1]

            block = np.column_stack([
                np.full(len(t_end), i, dtype=np.float32),  # sim
                t_end.astype(np.float32),                   # t
                dp, d_log_e, d_r, sum_cf, sum_st, sum_ex,
            ])
            raw[label].append(block)

        elapsed = time.perf_counter() - t0
        print(f"\r  Sim {i+1}/{n_sims}  ({elapsed:.1f}s)", end="", flush=True)
    print()

    cols = ["sim", "t", "dp", "d_log_earn", "d_rate",
            "cf_shock", "static_shock", "extrap_shock"]

    dfs = {}
    for label, _ in HORIZONS:
        arr = np.vstack(raw[label])
        dfs[label] = pd.DataFrame(arr, columns=cols)
        dfs[label]["sim"] = dfs[label]["sim"].astype(int)
        dfs[label]["t"] = dfs[label]["t"].astype(int)

    return dfs


# ═══════════════════════════════════════════════════════════════════════════
# 2. Option A – Conditional-mean table
# ═══════════════════════════════════════════════════════════════════════════

QUANTILE_BINS = [0.0, 0.0001, 0.01, 0.05, 0.25, 0.75, 0.95, 0.99, 0.9999, 1.0]
BIN_LABELS = [
    "Bot 0.01%", "0.01–1%", "1–5%", "5–25%", "25–75%",
    "75–95%", "95–99%", "99–99.99%", "Top 0.01%"
]

RHS_COLS = ["d_log_earn", "d_rate", "cf_shock", "static_shock", "extrap_shock"]


def conditional_mean_table(df: pd.DataFrame, horizon_label: str) -> pd.DataFrame:
    """Build the quantile-bin conditional-mean table for one horizon."""
    dp = df["dp"]
    edges = np.quantile(dp, QUANTILE_BINS)
    # pd.cut with quantile edges
    df = df.copy()
    df["bin"] = pd.cut(
        dp, bins=edges, labels=BIN_LABELS, include_lowest=True
    )

    agg = df.groupby("bin", observed=True).agg(
        n=("dp", "size"),
        avg_dp=("dp", "mean"),
        std_dp=("dp", "std"),
        avg_d_log_earn=("d_log_earn", "mean"),
        avg_d_rate=("d_rate", "mean"),
        avg_cf_shock=("cf_shock", "mean"),
        avg_static_shock=("static_shock", "mean"),
        avg_extrap_shock=("extrap_shock", "mean"),
    )

    # Add unconditional row
    unc = pd.DataFrame({
        "n": [len(df)],
        "avg_dp": [dp.mean()],
        "std_dp": [dp.std()],
        "avg_d_log_earn": [df["d_log_earn"].mean()],
        "avg_d_rate": [df["d_rate"].mean()],
        "avg_cf_shock": [df["cf_shock"].mean()],
        "avg_static_shock": [df["static_shock"].mean()],
        "avg_extrap_shock": [df["extrap_shock"].mean()],
    }, index=pd.CategoricalIndex(["All"], name="bin"))

    agg = pd.concat([agg, unc])
    agg.index.name = "bin"
    return agg


def print_conditional_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Pretty-print all conditional-mean tables."""
    for horizon, tbl in tables.items():
        print(f"\n{'='*90}")
        print(f"  Option A — Conditional Mean Table : {horizon}")
        print(f"{'='*90}")

        # Header
        header = (
            f"{'Bin':>12s} {'n':>8s} {'avg ΔP':>10s} {'std ΔP':>10s}"
            f" {'avg Δln E':>10s} {'avg Δr':>10s}"
            f" {'avg CF':>10s} {'avg Stat':>10s} {'avg Ext':>10s}"
        )
        print(header)
        print("-" * len(header))

        for idx, row in tbl.iterrows():
            label = str(idx)
            print(
                f"{label:>12s} {int(row['n']):>8d}"
                f" {row['avg_dp']:>10.5f} {row['std_dp']:>10.5f}"
                f" {row['avg_d_log_earn']:>10.6f} {row['avg_d_rate']:>10.6f}"
                f" {row['avg_cf_shock']:>10.4f} {row['avg_static_shock']:>10.4f}"
                f" {row['avg_extrap_shock']:>10.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Option B – OLS R² regression
# ═══════════════════════════════════════════════════════════════════════════

def _ols_core(y: np.ndarray, X_raw: np.ndarray) -> dict:
    """Bare-bones OLS returning beta, se, t_stat, R², adj_R², y_hat, resid."""
    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])
    k_full = k + 1

    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    y_hat = X @ beta
    resid = y - y_hat
    ss_res = resid @ resid
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (ss_res / (n - k_full)) / (ss_tot / (n - 1)) if n > k_full else 0.0

    s2 = ss_res / max(n - k_full, 1)
    var_beta = s2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se

    return {
        "beta": beta, "se": se, "t_stat": t_stats,
        "r2": r2, "adj_r2": adj_r2, "y_hat": y_hat, "resid": resid,
    }


from scipy.stats import kurtosis as _scipy_kurtosis  # noqa: E402


def ols_regression(df: pd.DataFrame) -> dict:
    """OLS: dp ~ const + d_log_earn + d_rate + cf_shock + static_shock + extrap_shock.

    Additionally computes:
      - kurtosis of actual ΔP, predicted ΔP, and residuals
      - R² restricted to tail observations (top/bottom 5%) vs middle 90%
    """
    y = df["dp"].values
    X_raw = df[RHS_COLS].values
    full = _ols_core(y, X_raw)

    # --- Kurtosis comparison (excess kurtosis) ---
    kurt_actual = float(_scipy_kurtosis(y, fisher=True))
    kurt_predicted = float(_scipy_kurtosis(full["y_hat"], fisher=True))
    kurt_resid = float(_scipy_kurtosis(full["resid"], fisher=True))

    # --- Tail-conditional R² ---
    q05, q95 = np.quantile(y, [0.05, 0.95])
    tail_mask = (y <= q05) | (y >= q95)
    mid_mask = ~tail_mask

    # R² on tail subset (using full-sample betas)
    y_tail = y[tail_mask]
    yhat_tail = full["y_hat"][tail_mask]
    ss_res_tail = np.sum((y_tail - yhat_tail) ** 2)
    ss_tot_tail = np.sum((y_tail - y_tail.mean()) ** 2)
    r2_tail = 1 - ss_res_tail / ss_tot_tail if ss_tot_tail > 0 else 0.0

    # R² on middle subset
    y_mid = y[mid_mask]
    yhat_mid = full["y_hat"][mid_mask]
    ss_res_mid = np.sum((y_mid - yhat_mid) ** 2)
    ss_tot_mid = np.sum((y_mid - y_mid.mean()) ** 2)
    r2_mid = 1 - ss_res_mid / ss_tot_mid if ss_tot_mid > 0 else 0.0

    names = ["const"] + RHS_COLS
    return {
        "names": names,
        "beta": full["beta"],
        "se": full["se"],
        "t_stat": full["t_stat"],
        "r2": full["r2"],
        "adj_r2": full["adj_r2"],
        "n_obs": len(y),
        # New diagnostics
        "kurt_actual": kurt_actual,
        "kurt_predicted": kurt_predicted,
        "kurt_resid": kurt_resid,
        "r2_tail": r2_tail,
        "r2_mid": r2_mid,
        "n_tail": int(tail_mask.sum()),
        "n_mid": int(mid_mask.sum()),
    }


def print_regression_results(results: dict[str, dict]) -> None:
    """Pretty-print OLS results for all horizons."""
    for horizon, res in results.items():
        print(f"\n{'='*80}")
        print(f"  Option B — OLS Regression : {horizon}   (n = {res['n_obs']:,})")
        print(f"  R² = {res['r2']:.6f}    Adj R² = {res['adj_r2']:.6f}")
        print(f"{'='*80}")

        header = f"{'Variable':>16s} {'Coeff':>12s} {'Std Err':>12s} {'t-stat':>10s}"
        print(header)
        print("-" * len(header))
        for name, b, s, t in zip(
            res["names"], res["beta"], res["se"], res["t_stat"]
        ):
            print(f"{name:>16s} {b:>12.6f} {s:>12.6f} {t:>10.2f}")

        print(f"\n  Tail diagnostics:")
        print(f"    Kurtosis (excess):  actual = {res['kurt_actual']:.2f}")
        print(f"                      predicted = {res['kurt_predicted']:.2f}")
        print(f"                       residual = {res['kurt_resid']:.2f}")
        print(f"    R² on tails (top/bot 5%): {res['r2_tail']:.4f}  (n={res['n_tail']:,})")
        print(f"    R² on middle 90%:         {res['r2_mid']:.4f}  (n={res['n_mid']:,})")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Option C – Scatter plots
# ═══════════════════════════════════════════════════════════════════════════

def make_scatter_plots(
    dfs: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """For each horizon, create a panel of scatter plots: ΔP vs each RHS variable.

    Tail observations (top/bottom 1% of ΔP) are highlighted in red/blue.
    A random subsample is plotted to keep file sizes manageable.
    """
    max_points = 20_000  # subsample for scatter visibility

    for horizon_label, df in dfs.items():
        n = len(df)
        dp = df["dp"].values

        # Identify tails (top/bottom 0.01%)
        q_lo, q_hi = np.quantile(dp, [0.0001, 0.9999])
        tail_lo = dp <= q_lo
        tail_hi = dp >= q_hi
        middle = ~(tail_lo | tail_hi)

        # Subsample the middle band for plotting
        mid_idx = np.where(middle)[0]
        if len(mid_idx) > max_points:
            rng = np.random.default_rng(0)
            mid_idx = rng.choice(mid_idx, size=max_points, replace=False)

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
        fig.suptitle(
            f"Option C — Price Return vs Shocks  ({horizon_label},  "
            f"n = {n:,},  tails = top/bottom 0.01%)",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for ax, rhs_col in zip(axes, RHS_COLS):
            x = df[rhs_col].values

            # Middle (grey, small, subsampled)
            ax.scatter(
                x[mid_idx], dp[mid_idx],
                s=0.3, alpha=0.15, c="grey", rasterized=True,
            )
            # Bottom tail (blue)
            ax.scatter(
                x[tail_lo], dp[tail_lo],
                s=2, alpha=0.5, c="tab:blue", label="Bottom 0.01%",
                rasterized=True,
            )
            # Top tail (red)
            ax.scatter(
                x[tail_hi], dp[tail_hi],
                s=2, alpha=0.5, c="tab:red", label="Top 0.01%",
                rasterized=True,
            )

            ax.set_xlabel(RHS_SHORT[rhs_col], fontsize=10)
            ax.axhline(0, lw=0.5, c="k", ls="--")
            ax.axvline(0, lw=0.5, c="k", ls="--")

            # Light trend line
            try:
                z = np.polyfit(x, dp, 1)
                x_range = np.array([x.min(), x.max()])
                ax.plot(x_range, np.polyval(z, x_range), c="orange", lw=1.5,
                        ls="--", label=f"OLS slope")
            except Exception:
                pass

        axes[0].set_ylabel(f"ΔP  ({horizon_label})", fontsize=10)
        axes[-1].legend(fontsize=8, loc="upper left", markerscale=3)

        fig.tight_layout()
        fname = out_dir / f"scatter_{horizon_label}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Summary R² bar chart
# ═══════════════════════════════════════════════════════════════════════════

def make_r2_summary_chart(
    regression_results: dict[str, dict],
    out_dir: Path,
) -> None:
    """Bar chart of R² by horizon — one-glance summary of Option B."""
    horizons = list(regression_results.keys())
    r2_vals = [regression_results[h]["r2"] for h in horizons]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(horizons, r2_vals, color=["#4c72b0", "#55a868", "#c44e52"],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Price Return Variance Explained by Shocks", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, max(r2_vals) * 1.4 if max(r2_vals) > 0 else 0.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.2%}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    fig.tight_layout()
    fname = out_dir / "r2_summary.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Conditional-mean visualisation (heatmap-style)
# ═══════════════════════════════════════════════════════════════════════════

def make_conditional_mean_chart(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """For each horizon, plot the conditional means of each shock variable
    across quantile bins, as grouped bar charts.  This is the visual companion
    to the Option A table.
    """
    shock_cols = [
        ("avg_d_log_earn", "% Chg Earnings"),
        ("avg_d_rate", "Δ r (bps)"),
        ("avg_cf_shock", "CF shock"),
        ("avg_static_shock", "Static shock"),
        ("avg_extrap_shock", "Extrap shock"),
    ]

    for horizon_label, tbl in tables.items():
        # Drop the "All" row for the chart
        tbl_plot = tbl.drop("All", errors="ignore")
        bin_labels = [str(b) for b in tbl_plot.index]
        n_bins = len(bin_labels)

        fig, axes = plt.subplots(1, len(shock_cols), figsize=(22, 4.5),
                                 sharey=False)
        fig.suptitle(
            f"Option A — Conditional Means by ΔP Quantile  ({horizon_label})",
            fontsize=13, fontweight="bold", y=1.02,
        )

        colours = plt.cm.RdBu_r(np.linspace(0.1, 0.9, n_bins))

        for ax, (col, label) in zip(axes, shock_cols):
            vals = tbl_plot[col].values.copy()
            # Convert log-earnings change to percentage for readability
            if col == "avg_d_log_earn":
                vals = (np.exp(vals) - 1) * 100  # Δln E → % change
            # Convert rate change to basis points for readability
            if col == "avg_d_rate":
                vals = vals * 10_000

            ax.barh(
                np.arange(n_bins), vals, color=colours, edgecolor="black",
                linewidth=0.4,
            )
            ax.set_yticks(np.arange(n_bins))
            ax.set_yticklabels(bin_labels, fontsize=8)
            ax.axvline(0, c="k", lw=0.7)
            ax.set_xlabel(label, fontsize=10)

        fig.tight_layout()
        fname = out_dir / f"cond_mean_{horizon_label}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shock-vs-price attribution analysis"
    )
    parser.add_argument("--n-sims", type=int, default=30,
                        help="Number of simulations to pool (default 30)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed (sim i uses seed = base_seed + i)")
    parser.add_argument("--out-dir", type=str,
                        default=str(Path("results") / "big_p_moves"),
                        help="Output directory for tables and plots")
    parser.add_argument("--max-move", type=float, default=0.20,
                        help="Max per-period price move (default 0.20)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Simulation parameters ---
    mpm = args.max_move
    mgm = min(24 * mpm, 0.98)
    params = SimulationParams().with_grid(max_price_move=mpm, max_grid_move=mgm)

    ppy = params.periods_per_year
    years = params.years
    print(
        f"Running {args.n_sims} simulations  "
        f"(ppy={ppy}, years={years}, n_periods={params.n_periods})"
    )

    # ── 1. Extract data ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    dfs = extract_sim_data(params, n_sims=args.n_sims, base_seed=args.base_seed)
    elapsed = time.perf_counter() - t0
    print(f"Data extraction: {elapsed:.1f}s")
    for label, df in dfs.items():
        print(f"  {label}: {len(df):,} observations")

    # ── 2. Option A — Conditional-mean tables ────────────────────────────
    cond_tables: dict[str, pd.DataFrame] = {}
    for label, df in dfs.items():
        cond_tables[label] = conditional_mean_table(df, label)

    print_conditional_tables(cond_tables)

    # Save to CSV
    for label, tbl in cond_tables.items():
        fname = out_dir / f"cond_mean_{label}.csv"
        tbl.to_csv(fname, float_format="%.6f")
    print(f"\n  Conditional-mean CSVs saved to {out_dir}/")

    # ── 3. Option B — OLS regressions ────────────────────────────────────
    reg_results: dict[str, dict] = {}
    for label, df in dfs.items():
        reg_results[label] = ols_regression(df)

    print_regression_results(reg_results)

    # Save regression summary to CSV
    rows = []
    for label, res in reg_results.items():
        for name, b, se, t in zip(res["names"], res["beta"], res["se"], res["t_stat"]):
            rows.append({
                "horizon": label,
                "variable": name,
                "coeff": b,
                "std_err": se,
                "t_stat": t,
            })
        for extra_key in ["R²", "R²_tail", "R²_mid",
                          "kurt_actual", "kurt_predicted", "kurt_resid"]:
            val_map = {
                "R²": res["r2"], "R²_tail": res["r2_tail"],
                "R²_mid": res["r2_mid"], "kurt_actual": res["kurt_actual"],
                "kurt_predicted": res["kurt_predicted"],
                "kurt_resid": res["kurt_resid"],
            }
            rows.append({
                "horizon": label,
                "variable": extra_key,
                "coeff": val_map[extra_key],
                "std_err": np.nan,
                "t_stat": np.nan,
            })
    reg_df = pd.DataFrame(rows)
    reg_fname = out_dir / "ols_regression_summary.csv"
    reg_df.to_csv(reg_fname, index=False, float_format="%.6f")
    print(f"\n  Regression CSV saved to {reg_fname}")

    # ── 4. Option C — Scatter plots ──────────────────────────────────────
    print("\nGenerating scatter plots...")
    make_scatter_plots(dfs, out_dir)

    # ── 5. Summary charts ────────────────────────────────────────────────
    print("\nGenerating summary charts...")
    make_r2_summary_chart(reg_results, out_dir)
    make_conditional_mean_chart(cond_tables, out_dir)

    total = time.perf_counter() - t0
    print(f"\nDone. Total time: {total:.1f}s")


if __name__ == "__main__":
    main()
