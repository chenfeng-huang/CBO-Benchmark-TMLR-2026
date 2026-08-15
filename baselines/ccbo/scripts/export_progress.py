"""Export the most recent ccbo experiment CSV into the BO/CBO progress format.

ccbo/ccbo/experiments/run_optimization.py writes per-run results to
``ccbo/ccbo/experiment_results/<label>_<model_name>_<timestamp>.csv`` with
columns ``trial,best_so_far,y,current_optimal``.

This script picks the most recent file matching ``<label>_<model>_*.csv``,
truncates it to the requested trial count, and writes a copy at
``CBO_Benchmark/results/<METHOD>/<label>/<METHOD>_<label>_<n>-trials_progress.csv``
with columns ``trial_number,current_optimal`` -- the schema accepted by
``CBO_Benchmark/scripts/evaluation/calculate_gap_metrics.py``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent  # the ccbo/ folder (sibling of CBO_Benchmark)


def _find_experiment_results_dir() -> Path:
    """Locate ccbo's experiment_results directory, regardless of CWD."""
    candidates = [
        _REPO_ROOT / "ccbo" / "experiment_results",
        _REPO_ROOT / "experiment_results",
        Path.cwd() / "experiment_results",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the canonical one (will fail clearly downstream).
    return _REPO_ROOT / "ccbo" / "experiment_results"


def _find_cbo_benchmark_root() -> Path:
    """Locate CBO_Benchmark sibling project."""
    candidates = [
        _REPO_ROOT.parent / "CBO_Benchmark",
        Path.cwd().parent / "CBO_Benchmark",
        Path.cwd() / "CBO_Benchmark",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate CBO_Benchmark (looked at "
        + ", ".join(str(c) for c in candidates) + ")"
    )


def _newest_matching(results_dir: Path, label: str, model: str) -> Path:
    pattern = str(results_dir / f"{label}_{model}_*.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No experiment CSVs match {pattern}. "
            f"Did `python -m ccbo.experiments.run_optimization` run successfully?"
        )
    return Path(matches[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True,
                        help="results_label used by the ccbo config "
                             "(e.g. 'protein').")
    parser.add_argument("--method", default="CCBO",
                        help="Method name used in the output path "
                             "(default: CCBO).")
    parser.add_argument("--model", default="ccbo_single_task",
                        help="ccbo model_name suffix in the source filename.")
    parser.add_argument("--task", choices=["min", "max"], default="min",
                        help="Optimization task (only used for re-deriving the "
                             "running best from `y` if `current_optimal` is "
                             "missing in the source CSV).")
    parser.add_argument("--num_trials", type=int, default=100,
                        help="Number of trials to include in the output CSV.")
    parser.add_argument("--source", type=Path, default=None,
                        help="Override the source CSV path (skips auto-pick).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override the output CSV path.")
    args = parser.parse_args()

    src = args.source or _newest_matching(
        _find_experiment_results_dir(), args.label, args.model
    )
    print(f"[export_progress] source: {src}")

    df = pd.read_csv(src)
    if "current_optimal" in df.columns:
        values = df["current_optimal"].tolist()
    elif "best_so_far" in df.columns:
        values = df["best_so_far"].tolist()
    elif "y" in df.columns:
        # Fall back to deriving the running best from per-trial y values.
        ys = df["y"].tolist()
        values = []
        running = ys[0] if ys else None
        for y in ys:
            if running is None:
                running = y
            else:
                running = min(running, y) if args.task == "min" else max(running, y)
            values.append(running)
    else:
        sys.exit(
            f"[export_progress] source CSV has none of "
            f"current_optimal/best_so_far/y columns: {list(df.columns)}"
        )

    # ccbo's run_optimization.py writes a leading sentinel row with `inf`
    # (its "before any intervention" placeholder). Drop any leading
    # non-finite values so trial 0 in the output corresponds to the first
    # real observation, matching the BO/CBO progress convention.
    import math
    finite_start = 0
    while finite_start < len(values) and (
        values[finite_start] is None
        or (isinstance(values[finite_start], float) and not math.isfinite(values[finite_start]))
    ):
        finite_start += 1
    if finite_start > 0:
        print(f"[export_progress] dropped {finite_start} leading non-finite "
              f"sentinel row(s)")
    values = values[finite_start:]

    n = min(args.num_trials, len(values))
    if n < args.num_trials:
        print(f"[export_progress] warning: only {n} trials available "
              f"(requested {args.num_trials}); writing what we have")
    out_df = pd.DataFrame({
        "trial_number": list(range(n)),
        "current_optimal": values[:n],
    })

    out_path = args.out or (
        _find_cbo_benchmark_root()
        / "results" / args.method / args.label
        / f"{args.method}_{args.label}_{args.num_trials}-trials_progress.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[export_progress] wrote {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
