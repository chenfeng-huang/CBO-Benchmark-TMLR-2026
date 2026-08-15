"""Run vendored CEO baseline on CBO_Benchmark datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("MPLBACKEND", "Agg")

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]
if not hasattr(np, "PINF"):
    np.PINF = np.inf  # type: ignore[attr-defined]
if not hasattr(np, "NINF"):
    np.NINF = -np.inf  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

_CEO_DIR = Path(__file__).resolve().parent
_BENCHMARK_ROOT = _CEO_DIR.parents[1]
sys.path.insert(0, str(_BENCHMARK_ROOT))
sys.dont_write_bytecode = True

from baselines.CEO.benchmark.adapter import load_ceo_benchmark_inputs  # noqa: E402
from baselines.CEO.benchmark.runner import run_benchmark_ceo  # noqa: E402


STANDARD_DATASETS = [
    "synthetic",
    "toyGraph",
    "synthetic_2",
    "protein",
    "healthcare",
    "ecology",
    "epidemiology",
    "chain",
]


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    print(f"[CEO] seed set to {seed}")


def _write_progress(values: Iterable[float], out_path: Path, num_trials: int) -> List[float]:
    clean_values = [float(np.asarray(v).squeeze()) for v in values]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(num_trials, len(clean_values))
    pd.DataFrame({
        "trial_number": list(range(n)),
        "current_optimal": clean_values[:n],
    }).to_csv(out_path, index=False)
    print(f"[CEO] wrote progress CSV to {out_path}")
    return clean_values[:n]


def _maybe_emit_gap_metrics(global_opt: List[float], task: str, dataset: str, num_trials: int) -> None:
    try:
        from scripts.evaluation.run_cbo import display_algorithm_metrics  # type: ignore
    except Exception as exc:
        print(f"[CEO] skipping GAP metrics: {exc}")
        return

    try:
        display_algorithm_metrics(global_opt, task, "CEO", dataset, num_trials)
    except Exception as exc:
        print(f"[CEO] GAP metric computation failed: {exc}")


def run_dataset(args: argparse.Namespace) -> None:
    _set_seeds(args.seed)
    os.chdir(_BENCHMARK_ROOT)

    inputs = load_ceo_benchmark_inputs(
        args.dataset,
        _BENCHMARK_ROOT,
        seed=args.seed,
        seeds_replicate=args.seeds_replicate,
        num_graph_candidates=args.num_graph_candidates,
    )

    out_dir = Path(args.output_dir) if args.output_dir else _BENCHMARK_ROOT / "results" / "CEO" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_log_file = out_dir / f"CEO_{args.dataset}_{args.num_trials}-trials_progress.csv"
    print(f"[CEO] writing progress to {csv_log_file}")
    print(f"[CEO] exploration sets: {inputs.exploration_sets}")
    print(f"[CEO] graph candidates: {len(inputs.graphs)}")

    start = time.time()
    if args.use_original_ceo:
        raw_progress = _run_original_ceo(args, inputs)
    else:
        result = run_benchmark_ceo(
            inputs,
            num_trials=args.num_trials,
            seed=args.seed,
            n_anchor_points=args.n_anchor_points,
        )
        raw_progress = result.progress
    elapsed = time.time() - start

    progress = _write_progress(raw_progress, csv_log_file, args.num_trials)
    summary = {
        "dataset": args.dataset,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "n_anchor_points": args.n_anchor_points,
        "seeds_replicate": list(args.seeds_replicate),
        "num_graph_candidates": len(inputs.graphs),
        "elapsed_seconds": elapsed,
        "final_best": progress[-1] if progress else None,
        "progress_csv": str(csv_log_file),
    }
    with (out_dir / f"CEO_{args.dataset}_{args.num_trials}-trials_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[CEO] summary: {json.dumps(summary, indent=2)}")

    _maybe_emit_gap_metrics(progress, inputs.config["task"], args.dataset, args.num_trials)


def _run_original_ceo(args: argparse.Namespace, inputs) -> List[float]:
    sys.path.insert(0, str(_CEO_DIR))
    sys.path.insert(0, str(_CEO_DIR / "src"))
    import src  # type: ignore[import-not-found] # noqa: F401
    from src.methods.ceo import CEO  # type: ignore[import-not-found]
    from src.utils.sem_utils.sem_estimate import build_sem_hat  # type: ignore[import-not-found]

    ceo = CEO(
        graphs=inputs.graphs,
        init_posterior=inputs.init_posterior,
        sem=inputs.sem_class,
        make_sem_estimator=build_sem_hat,
        observation_samples=inputs.observation_samples,
        intervention_domain=inputs.intervention_domain,
        intervention_samples=inputs.intervention_samples,
        intervention_samples_noiseless=inputs.intervention_samples_noiseless,
        exploration_sets=inputs.exploration_sets,
        number_of_trials=args.num_trials,
        base_target_variable=inputs.config["target"],
        task=inputs.config["task"],
        sample_anchor_points=True,
        seed_anchor_points=args.seed,
        num_anchor_points=args.n_anchor_points,
        random_state=np.random.RandomState(args.seed),
        manipulative_variables=list(inputs.config["intervention"]),
        debug_mode=False,
    )
    ceo.run()
    return ceo.optimal_outcome_values_during_trials[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vendored CEO on a CBO_Benchmark dataset")
    parser.add_argument("--dataset", required=True, choices=STANDARD_DATASETS)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_anchor_points", type=int, default=5)
    parser.add_argument("--seeds_replicate", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--num_graph_candidates", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--use_original_ceo",
        action="store_true",
        help="Use the vendored CEO research package directly.",
    )
    args = parser.parse_args()
    run_dataset(args)


if __name__ == "__main__":
    main()

