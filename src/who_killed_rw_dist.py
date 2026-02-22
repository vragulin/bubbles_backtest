"""Monte Carlo distribution export for Who Killed the Random Walk? simulation.

Runs the same simulation engine as `who_killed_rw.py`, but writes per-period
("daily") total stock returns to a CSV so you can build histograms or other
distributional plots.

Notes
-----
- The model period is `SimulationParams.periods_per_year` (default 240). We treat
  each period as a "day" for export purposes.
- We export simple returns from the model's total-return price index:
    TR[t]/TR[t-1] - 1.

The simulation does not model inflation, so these are nominal (and also real,
trivially).

Example
-------
python src/who_killed_rw_dist.py --n-sims 50 --out results/rw_daily_tr_returns.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from who_killed_rw import SimulationParams, run_simulation, periodic_returns


def simulate_daily_tr_returns(
    params: SimulationParams,
    n_sims: int,
    base_seed: int | None = 42,
) -> pd.DataFrame:
    """Run Monte Carlo and return a long DataFrame of per-period TR returns."""

    rows: list[pd.DataFrame] = []

    for i in range(n_sims):
        seed = base_seed + i if base_seed is not None else None
        market, _investors, _shocks = run_simulation(params, seed=seed)

        tr = np.asarray(market.tr_price, dtype=float)
        tr_ret = periodic_returns(tr)

        # periods are 1..n_periods (return t uses (t-1 -> t))
        df_i = pd.DataFrame(
            {
                "sim": i,
                "t": np.arange(1, len(tr), dtype=int),
                "tr_return": tr_ret,
                "log_tr_return": np.log1p(tr_ret),
            }
        )
        rows.append(df_i)

        if (i + 1) % max(1, min(10, n_sims)) == 0:
            print(f"  Sim {i+1}/{n_sims} complete")

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-period (daily) total-return distribution from who_killed_rw simulation"
    )
    parser.add_argument("--n-sims", type=int, default=200, help="Number of simulations to run")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed; sim i uses seed=base_seed+i (set to -1 for no seeding)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "rw_daily_tr_returns.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.n_sims <= 0:
        raise SystemExit("--n-sims must be positive")

    base_seed = None if args.base_seed == -1 else int(args.base_seed)

    params = SimulationParams()  # same defaults as who_killed_rw.py

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {args.n_sims} simulations "
        f"(ppy={params.periods_per_year}, years={params.years}) → {out_path}"
    )

    df = simulate_daily_tr_returns(params=params, n_sims=args.n_sims, base_seed=base_seed)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} rows")


if __name__ == "__main__":
    main()
