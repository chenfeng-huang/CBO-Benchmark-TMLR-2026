"""Convert DCBO/MCBO/HCBO per-trial output into the BO/CBO progress CSV schema.

The schema accepted by ``CBO_Benchmark/scripts/evaluation/calculate_gap_metrics.py`` is::

    trial_number,current_optimal
    0,<best after trial 0>
    1,<best after trial 1>
    ...

Usage
-----
DCBO::

    python scripts/evaluation/export_progress.py dcbo \\
        --src baselines/DCBO/results/dcbo_best_so_far_seed1_T1_trials100_reps1.csv \\
        --dataset protein --task min --num_trials 100

MCBO::

    python scripts/evaluation/export_progress.py mcbo \\
        --src baselines/mcbo/trial_results_MCBO_protein_42.csv \\
        --dataset protein --task min --num_trials 100

HCBO::

    python scripts/evaluation/export_progress.py hcbo \\
        --src baselines/HCBO/result/ProteinBenchmark/Baseline/Causal/HCBO \\
        --dataset protein --task min --num_trials 100

ACBO::

    python scripts/evaluation/export_progress.py acbo \\
        --src baselines/acbo/trial_results_ACBO_Ackley-Perturb_42.csv \\
        --dataset ackley --task max --num_trials 100
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


_HERE = Path(__file__).resolve().parent
_BENCHMARK_ROOT = _HERE.parent.parent  # CBO_Benchmark/


def _output_path(method: str, dataset: str, num_trials: int,
                 override: Optional[Path]) -> Path:
    if override is not None:
        return override
    return (_BENCHMARK_ROOT / "results" / method / dataset
            / f"{method}_{dataset}_{num_trials}-trials_progress.csv")


def _running_best(values: List[float], task: str) -> List[float]:
    out: List[float] = []
    running: Optional[float] = None
    for v in values:
        if running is None:
            running = v
        else:
            running = min(running, v) if task == "min" else max(running, v)
        out.append(running)
    return out


def _drop_leading_sentinels(values: List[float],
                            sentinel_threshold: float = 1e6) -> List[float]:
    """Drop leading non-finite or sentinel-magnitude rows.

    DCBO writes ``10000000.0`` for the trial-0 sentinel; ccbo writes ``inf``.
    Either way we strip them so the output's trial 0 corresponds to a real
    iteration (matching the BO/CBO progress convention).
    """
    i = 0
    n_dropped = 0
    while i < len(values) and (
        values[i] is None
        or (isinstance(values[i], float) and not math.isfinite(values[i]))
        or abs(values[i]) >= sentinel_threshold
    ):
        i += 1
        n_dropped += 1
    if n_dropped:
        print(f"[export_progress] dropped {n_dropped} leading sentinel row(s)")
    return values[i:]


def _write_progress(values: List[float], num_trials: int,
                    out_path: Path) -> None:
    n = min(num_trials, len(values))
    if n < num_trials:
        print(f"[export_progress] warning: only {n} trials available "
              f"(requested {num_trials}); writing what we have")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "trial_number": list(range(n)),
        "current_optimal": values[:n],
    }).to_csv(out_path, index=False)
    print(f"[export_progress] wrote {n} rows -> {out_path}")


# ---------------------------------------------------------------------------
# DCBO
# ---------------------------------------------------------------------------
def cmd_dcbo(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.src)
    if "best_so_far_value" not in df.columns:
        sys.exit(f"DCBO source CSV missing 'best_so_far_value': {list(df.columns)}")
    # Filter to single replicate / time-index
    if "replicate" in df.columns:
        df = df[df["replicate"] == args.replicate]
    if "time_index" in df.columns:
        df = df[df["time_index"] == args.time_index]
    df = df.sort_values("trial_index") if "trial_index" in df.columns else df
    values = _drop_leading_sentinels(df["best_so_far_value"].tolist())
    out_path = _output_path("DCBO", args.dataset, args.num_trials, args.out)
    _write_progress(values, args.num_trials, out_path)


# ---------------------------------------------------------------------------
# MCBO
# ---------------------------------------------------------------------------
def cmd_mcbo(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.src)
    if "current_optimal" not in df.columns:
        sys.exit(f"MCBO source CSV missing 'current_optimal': {list(df.columns)}")
    if "trial_number" in df.columns:
        df = df.sort_values("trial_number")
    raw = df["current_optimal"].tolist()
    # MCBO maximizes a (possibly negated) target; flip back when task=='min'
    # since the env's evaluate() for min-tasks negates Y. We only need to flip
    # if the user asks for min and the env negated internally; the per-CSV
    # contract in mcbo_trial keeps `current_optimal` in the maximized space.
    if args.task == "min":
        values = [-float(v) for v in raw]
        # _drop_leading_sentinels treats |x|>=1e6 as a sentinel; MCBO doesn't
        # emit those, so this is a no-op for normal runs.
    else:
        values = [float(v) for v in raw]
    out_path = _output_path("MCBO", args.dataset, args.num_trials, args.out)
    _write_progress(values, args.num_trials, out_path)


# ---------------------------------------------------------------------------
# ACBO
# ---------------------------------------------------------------------------
def cmd_acbo(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.src)
    if "current_optimal" not in df.columns:
        sys.exit(f"ACBO source CSV missing 'current_optimal': {list(df.columns)}")
    if "trial_number" in df.columns:
        df = df.sort_values("trial_number")
    raw = [float(v) for v in df["current_optimal"].tolist()]
    # ACBO's raw CSV labels this column `current_optimal`, but the upstream
    # trial loop can emit a per-step value that decreases after the initial
    # design. Normalize it here to the benchmark best-so-far contract.
    values = _running_best(raw, args.task)
    out_path = _output_path("ACBO", args.dataset, args.num_trials, args.out)
    _write_progress(values, args.num_trials, out_path)


# ---------------------------------------------------------------------------
# HCBO
# ---------------------------------------------------------------------------
def _hcbo_extract_y_list(payload):
    """Best-effort extract of result_y_list from various HCBO pickle shapes."""
    # Common shape (from co_analysis.py): tuple
    # (result_x_list, result_y_list, visit_IS_list, issf_info)
    if isinstance(payload, tuple) and len(payload) >= 2:
        return list(payload[1])
    if isinstance(payload, dict):
        for key in ("result_y_list", "y_list", "y"):
            if key in payload:
                return list(payload[key])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unrecognized HCBO pickle payload type: {type(payload)}")


def _flatten_floats(seq) -> List[float]:
    """Flatten 1-elem tensors / arrays / nested lists into Python floats."""
    out: List[float] = []
    for x in seq:
        try:
            v = float(x)
        except Exception:
            try:
                v = float(getattr(x, "item")())
            except Exception:
                try:
                    v = float(x[-1])
                except Exception:
                    continue
        out.append(v)
    return out


def cmd_hcbo(args: argparse.Namespace) -> None:
    src = Path(args.src)
    if src.is_dir():
        candidates = sorted(src.glob("*.pkl"))
        if not candidates:
            sys.exit(f"HCBO src directory has no .pkl files: {src}")
        # Pick the first replicate's pickle deterministically.
        pkl_path = candidates[args.replicate if args.replicate < len(candidates) else 0]
    else:
        pkl_path = src
    print(f"[export_progress] HCBO source pkl: {pkl_path}")
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)

    y_list = _hcbo_extract_y_list(payload)
    values = _flatten_floats(y_list)
    values = _drop_leading_sentinels(values)
    if not values:
        sys.exit("[export_progress] HCBO payload yielded zero usable y values")

    # HCBO records raw target evaluations (one entry per intervention call), not
    # cumulative best -- so we accumulate here.
    values = _running_best(values, args.task)

    out_path = _output_path("HCBO", args.dataset, args.num_trials, args.out)
    _write_progress(values, args.num_trials, out_path)


# ---------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--src", required=True, help="Source CSV / dir / pickle path.")
    p.add_argument("--dataset", default="protein",
                   help="Dataset label used in the output path (default: protein).")
    p.add_argument("--task", choices=["min", "max"], default="min",
                   help="Optimization task (default: min).")
    p.add_argument("--num_trials", type=int, default=100,
                   help="Number of trials in the output CSV (default: 100).")
    p.add_argument("--out", type=Path, default=None,
                   help="Override the output CSV path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dcbo = sub.add_parser("dcbo")
    _add_common(p_dcbo)
    p_dcbo.add_argument("--replicate", type=int, default=0)
    p_dcbo.add_argument("--time_index", type=int, default=0)
    p_dcbo.set_defaults(func=cmd_dcbo)

    p_mcbo = sub.add_parser("mcbo")
    _add_common(p_mcbo)
    p_mcbo.set_defaults(func=cmd_mcbo)

    p_acbo = sub.add_parser("acbo")
    _add_common(p_acbo)
    p_acbo.set_defaults(func=cmd_acbo)

    p_hcbo = sub.add_parser("hcbo")
    _add_common(p_hcbo)
    p_hcbo.add_argument("--replicate", type=int, default=0,
                        help="Index of the replicate pkl to use when --src is a dir.")
    p_hcbo.set_defaults(func=cmd_hcbo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
