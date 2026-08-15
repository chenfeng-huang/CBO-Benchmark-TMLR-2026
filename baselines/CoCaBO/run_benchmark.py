"""Run vendored CoCaBO on a CBO_Benchmark dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_CoCaBO_DIR = Path(__file__).resolve().parent
_BENCHMARK_ROOT = _CoCaBO_DIR.parents[1]
sys.path.insert(0, str(_BENCHMARK_ROOT))
sys.path.insert(0, str(_CoCaBO_DIR))
sys.path.insert(0, str(_CoCaBO_DIR / "scmmab"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = pd.DataFrame._append  # type: ignore[attr-defined]

import pomps.hebo_adapted as _hebo_adapted  # noqa: E402

_OrigAdHEBO_init = _hebo_adapted.AdHEBO.__init__


def _patched_adhebo_init(self, space, model_name="gp", rand_sample=None,
                         acq_cls=None, es="nsga2", model_config=None):
    if model_name == "gpy":
        model_name = "gp"
    return _OrigAdHEBO_init(
        self,
        space,
        model_name=model_name,
        rand_sample=rand_sample,
        acq_cls=acq_cls,
        es=es,
        model_config=model_config,
    )


_hebo_adapted.AdHEBO.__init__ = _patched_adhebo_init  # type: ignore[assignment]

from baselines.CoCaBO.benchmark.runner import run_cocabo  # noqa: E402
from baselines.CoCaBO.benchmark.sem_to_fcm import supported_datasets  # noqa: E402


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[CoCaBO] seed set to {seed}")


def _maybe_emit_gap_metrics(global_opt, task, dataset, num_trials):
    try:
        from scripts.evaluation.run_cbo import display_algorithm_metrics  # type: ignore
    except Exception as exc:
        print(f"[CoCaBO] skipping GAP metrics: {exc}")
        return

    try:
        display_algorithm_metrics(global_opt, task, "CoCaBO", dataset, num_trials)
    except Exception as exc:
        print(f"[CoCaBO] GAP metric computation failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run vendored CoCaBO on a CBO_Benchmark dataset")
    parser.add_argument("--dataset", required=True, choices=supported_datasets())
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    _set_seeds(args.seed)
    dataset = args.dataset
    out_dir = Path(args.output_dir) if args.output_dir else _BENCHMARK_ROOT / "results" / "CoCaBO" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_log_file = out_dir / f"CoCaBO_{dataset}_{args.num_trials}-trials_progress.csv"
    print(f"[CoCaBO] writing progress to {csv_log_file}")

    os.chdir(_BENCHMARK_ROOT)
    global_opt, elapsed = run_cocabo(
        dataset=dataset,
        num_trials=args.num_trials,
        seed=args.seed,
        csv_log_file=str(csv_log_file),
    )

    summary = {
        "dataset": dataset,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "final_best": global_opt[-1] if global_opt else None,
        "progress_csv": str(csv_log_file),
    }
    with (out_dir / f"CoCaBO_{dataset}_{args.num_trials}-trials_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[CoCaBO] summary: {json.dumps(summary, indent=2)}")

    import yaml
    with (_BENCHMARK_ROOT / "configs" / f"{dataset}.yaml").open("r") as f:
        task = yaml.safe_load(f)["task"]
    _maybe_emit_gap_metrics(global_opt, task, dataset, args.num_trials)


if __name__ == "__main__":
    main()

