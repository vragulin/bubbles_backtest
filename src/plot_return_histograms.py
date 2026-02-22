"""Plot histogram of simulated vs real daily equity returns.

Inputs
------
- results/rw_daily_tr_returns.csv  (from src/who_killed_rw_dist.py)
- data/hist_df.csv                 (real US/XUS TRI)

Outputs
-------
- A PNG plot saved to results/return_histograms.png (by default)
- Printed summary statistics table for each distribution

Usage
-----
python src/plot_return_histograms.py \
  --sim results/rw_daily_tr_returns.csv \
  --hist data/hist_df.csv \
  --out results/return_histograms.png \
  --kde

Notes
-----
- Simulation periods-per-year is 240 by default in who_killed_rw.
- hist_df daily data is treated as 252 trading days/year for annualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_true_daily_start(
    simple_returns: pd.Series,
    *,
    window: int = 30,
    min_nonzero_frac: float = 0.80,
    eps: float = 1e-12,
) -> pd.Timestamp:
    """Detect the start of a "true daily" segment.

    Some early segments of hist_df are constructed by forward-filling monthly total
    returns onto a daily grid, producing mostly-zero daily returns with a jump at
    month-end. That heavily distorts histograms.

    Heuristic: find the earliest date where a rolling window has a sufficiently
    high fraction of non-zero returns.
    """
    r = simple_returns.dropna().astype(float)
    if r.empty:
        raise ValueError("Empty return series")

    nonzero = (r.abs() > eps).astype(float)
    frac = nonzero.rolling(window=window, min_periods=window).mean()
    ok = frac >= float(min_nonzero_frac)
    if not bool(ok.any()):
        return pd.Timestamp(r.index[0])

    return pd.Timestamp(ok.idxmax())


def _annualized_moments_from_returns(
    simple_returns: pd.Series,
    *,
    periods_per_year: int,
    use_log_returns: bool = True,
) -> dict[str, float]:
    """Compute annualized mean/std/skew/kurtosis for a return series.

    Implementation notes
    --------------------
    - We compute moments on (log) period returns.
    - Mean and stdev are annualized via: mean * ppy, std * sqrt(ppy).
    - Skew and excess kurtosis are scaled assuming aggregation of iid increments:
      skew_ann = skew / sqrt(ppy), ex_kurt_ann = ex_kurt / ppy.
    """
    r = simple_returns.dropna().astype(float)
    r = r[np.isfinite(r)]
    if r.empty:
        return {
            "n": 0,
            "mean_ann": float("nan"),
            "std_ann": float("nan"),
            "skew_ann": float("nan"),
            "kurtosis_ann": float("nan"),
        }

    x = np.log1p(r.to_numpy()) if use_log_returns else r.to_numpy()
    x = x[np.isfinite(x)]

    mean_p = float(np.mean(x))
    std_p = float(np.std(x, ddof=0))
    xc = x - mean_p
    m2 = float(np.mean(xc**2))
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    skew = float(m3 / (m2 ** 1.5)) if m2 > 0 else float("nan")
    ppy = float(periods_per_year)
    ex_kurt = float(m4 / (m2**2) - 3.0) if m2 > 0 else float("nan")

    ex_kurt_ann = ex_kurt / ppy if np.isfinite(ex_kurt) else float("nan")
    kurtosis_ann = ex_kurt_ann + 3.0 if np.isfinite(ex_kurt_ann) else float("nan")
    return {
        "n": int(x.size),
        "mean_ann": mean_p * ppy,
        "std_ann": std_p * np.sqrt(ppy),
        "skew_ann": skew / np.sqrt(ppy) if np.isfinite(skew) else float("nan"),
        "kurtosis_ann": kurtosis_ann,
    }


def _moments_row(name: str, r: pd.Series, periods_per_year: int) -> pd.Series:
    out = _annualized_moments_from_returns(r, periods_per_year=periods_per_year, use_log_returns=True)
    out["periods_per_year"] = int(periods_per_year)
    return pd.Series(out, name=name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simulated vs real daily return histograms")
    parser.add_argument("--sim", default=str(Path("results") / "rw_daily_tr_returns.csv"), help="Simulation CSV path")
    parser.add_argument("--hist", default=str(Path("data") / "hist_df.csv"), help="hist_df.csv path")
    parser.add_argument("--out", default=str(Path("results") / "return_histograms.png"), help="Output PNG path")
    parser.add_argument(
        "--stats-out",
        default=str(Path("results") / "return_distribution_stats.csv"),
        help="Output CSV for annualized distribution moments",
    )
    parser.add_argument("--bins", type=int, default=160, help="Number of histogram bins")
    parser.add_argument(
        "--clip",
        type=float,
        default=0.002,
        help="Clip both tails at this quantile for bin-range selection (e.g., 0.002 = 0.2%% tails)",
    )
    parser.add_argument(
        "--keep-zero-actual",
        action="store_true",
        help="Keep days where US/XUS return is exactly 0.0 (default is to drop them)",
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        help="Overlay a fitted density line (KDE) on top of each histogram",
    )
    args = parser.parse_args()

    sim_path = Path(args.sim)
    hist_path = Path(args.hist)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim = pd.read_csv(sim_path)
    if "tr_return" not in sim.columns:
        raise ValueError(f"Expected column 'tr_return' in {sim_path}")
    sim_r = pd.to_numeric(sim["tr_return"], errors="coerce").dropna()

    hist_df = pd.read_csv(hist_path, index_col=0, parse_dates=True).sort_index()
    for col in ("us_tri", "xus_tri"):
        if col not in hist_df.columns:
            raise ValueError(f"hist_df.csv missing required column: {col}")

    us_r = hist_df["us_tri"].pct_change().dropna()
    xus_r = hist_df["xus_tri"].pct_change().dropna()

    us_start = _detect_true_daily_start(us_r)
    xus_start = _detect_true_daily_start(xus_r)
    real_start = max(us_start, xus_start)
    us_r = us_r.loc[us_r.index >= real_start]
    xus_r = xus_r.loc[xus_r.index >= real_start]

    if not args.keep_zero_actual:
        us_r = us_r.loc[us_r != 0.0]
        xus_r = xus_r.loc[xus_r != 0.0]

    plot_lo, plot_hi = -0.03, 0.03
    bins = np.linspace(plot_lo, plot_hi, int(args.bins) + 1)

    sim_r_plot = sim_r.loc[(sim_r >= plot_lo) & (sim_r <= plot_hi)]
    us_r_plot = us_r.loc[(us_r >= plot_lo) & (us_r <= plot_hi)]
    xus_r_plot = xus_r.loc[(xus_r >= plot_lo) & (xus_r <= plot_hi)]

    sim_ppy = 240
    hist_ppy = 252
    # Keep legend consistent with the printed table below.
    # Both should report the *annualized mean of log(1+r)* (a continuously-compounded mean).
    # Converting via expm1(...) produces a slightly higher simple-return number and will not
    # match the table, so we do NOT apply expm1 here.
    sim_ann_mean = float(
        _annualized_moments_from_returns(sim_r, periods_per_year=sim_ppy, use_log_returns=True)["mean_ann"]
    )
    us_ann_mean = float(
        _annualized_moments_from_returns(us_r, periods_per_year=hist_ppy, use_log_returns=True)["mean_ann"]
    )
    xus_ann_mean = float(
        _annualized_moments_from_returns(xus_r, periods_per_year=hist_ppy, use_log_returns=True)["mean_ann"]
    )

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for this script. Install it or run with an environment that has it.\n"
            "For example (if you have it): C:/Users/vragu/anaconda3/envs/bubbles_vh/python.exe src/plot_return_histograms.py"
        ) from e

    from matplotlib.ticker import FuncFormatter
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def darken(color: str, factor: float = 0.7):
        r, g, b = mcolors.to_rgb(color)
        return (r * factor, g * factor, b * factor)

    c_sim = "tab:blue"
    c_us = "tab:orange"
    c_xus = "tab:green"

    ls_sim = "-"
    ls_us = "--"
    ls_xus = ":"

    title_fs = 18
    label_fs = 16
    tick_fs = 13
    legend_fs = 13

    plt.figure(figsize=(12, 7))
    plt.hist(sim_r_plot, bins=bins, density=True, alpha=0.45, color=c_sim, label="_nolegend_")
    plt.hist(us_r_plot, bins=bins, density=True, alpha=0.45, color=c_us, label="_nolegend_")
    plt.hist(xus_r_plot, bins=bins, density=True, alpha=0.45, color=c_xus, label="_nolegend_")

    if args.kde:
        def _kde_values(data: pd.Series, x_grid: np.ndarray) -> np.ndarray:
            arr = np.asarray(data.dropna().to_numpy(dtype=float))
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.full_like(x_grid, np.nan, dtype=float)

            max_n = 50_000
            if arr.size > max_n:
                rng = np.random.RandomState(0)
                arr = rng.choice(arr, size=max_n, replace=False)

            try:
                from scipy.stats import gaussian_kde  # type: ignore

                kde = gaussian_kde(arr)
                return kde.evaluate(x_grid)
            except Exception:
                n = arr.size
                sd = float(np.std(arr, ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    return np.full_like(x_grid, np.nan, dtype=float)

                h = 1.06 * sd * (n ** (-1 / 5))
                if not np.isfinite(h) or h <= 0:
                    return np.full_like(x_grid, np.nan, dtype=float)

                inv = 1.0 / (n * h * np.sqrt(2 * np.pi))
                z = (x_grid[:, None] - arr[None, :]) / h
                out = np.empty(x_grid.shape[0], dtype=float)
                step = 200
                for i0 in range(0, z.shape[0], step):
                    i1 = min(i0 + step, z.shape[0])
                    out[i0:i1] = inv * np.sum(np.exp(-0.5 * z[i0:i1] ** 2), axis=1)
                return out

        x_grid = np.linspace(plot_lo, plot_hi, 800)
        plt.plot(
            x_grid,
            _kde_values(sim_r_plot, x_grid),
            color=darken(c_sim),
            linewidth=2.0,
            linestyle=ls_sim,
            label="_nolegend_",
        )
        plt.plot(
            x_grid,
            _kde_values(us_r_plot, x_grid),
            color=darken(c_us),
            linewidth=2.0,
            linestyle=ls_us,
            label="_nolegend_",
        )
        plt.plot(
            x_grid,
            _kde_values(xus_r_plot, x_grid),
            color=darken(c_xus),
            linewidth=2.0,
            linestyle=ls_xus,
            label="_nolegend_",
        )

    ax = plt.gca()

    plt.title(
        "Daily Total Return Distributions: Simulation vs. Actual US and Non-US (XUS) Stocks",
        fontsize=title_fs,
    )
    plt.xlabel("Daily Return (simple)", fontsize=label_fs)
    plt.ylabel("Density", fontsize=label_fs)

    if args.kde:
        handles = [
            Line2D([0], [0], color=darken(c_sim), linestyle=ls_sim, linewidth=2.5),
            Line2D([0], [0], color=darken(c_us), linestyle=ls_us, linewidth=2.5),
            Line2D([0], [0], color=darken(c_xus), linestyle=ls_xus, linewidth=2.5),
        ]
    else:
        handles = [
            Patch(facecolor=c_sim, edgecolor="none", alpha=0.45),
            Patch(facecolor=c_us, edgecolor="none", alpha=0.45),
            Patch(facecolor=c_xus, edgecolor="none", alpha=0.45),
        ]

    labels = [
        f"Simulation (ann. mean log ret. = {sim_ann_mean * 100:.1f}%)",
        f"US (ann. mean log ret. = {us_ann_mean * 100:.1f}%)",
        f"XUS (ann. mean log ret. = {xus_ann_mean * 100:.1f}%)",
    ]
    plt.legend(handles, labels, fontsize=legend_fs)
    plt.grid(True, alpha=0.2)

    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x * 100:.1f}%"))
    plt.xlim(plot_lo, plot_hi)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    stats = pd.DataFrame(
        [
            _moments_row("sim_tr_return", sim_r, periods_per_year=sim_ppy),
            _moments_row("us_tr_return", us_r, periods_per_year=hist_ppy),
            _moments_row("xus_tr_return", xus_r, periods_per_year=hist_ppy),
        ]
    )

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_out, index=True)

    with pd.option_context("display.width", 160, "display.max_columns", None):
        print("\nNominal-return start date used (hist_df):", real_start.date())
        print("Histogram saved to:", out_path)
        print("Stats saved to:", stats_out)
        print("\nAnnualized distribution moments (computed on log1p returns):")
        print(stats)


if __name__ == "__main__":
    main()
