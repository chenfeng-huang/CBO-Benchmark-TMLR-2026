"""CoCaBO runner that produces a BO/CBO-compatible progress CSV.

Mirrors the per-trial logging behavior of `baselines/BO_CBO/BO.py` so the resulting
``results/CoCaBO/<dataset>/CoCaBO_<dataset>_<trials>-trials_progress.csv`` can
be scored with ``scripts/evaluation/calculate_gap_metrics.py`` / ``eval.sh`` without changes.
"""

from __future__ import annotations

import datetime
import os
import time
import typing as tp

import pandas as pd
import torch
from tqdm.auto import tqdm

from algorithms.pomps_experiment import POMPSExperiment, OptimizationObjective
from pomps.policy_scope import MixedPolicyScope

from .sem_to_fcm import build_for_dataset, FCMBundle


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _build_experiment(bundle: FCMBundle, n_iter: int, dataset: str) -> POMPSExperiment:
    objective = (OptimizationObjective.maximize
                 if bundle.task == "max" else OptimizationObjective.minimize)
    # Drop the empty intervention scope (intervening on nothing is useless).
    droppable = [MixedPolicyScope(set())]
    return POMPSExperiment(
        bundle.fcm,
        interventional_variables=bundle.intervention_vars,
        contextual_variables=set(),
        optimization_domain=bundle.domain,
        target=bundle.target,
        droppable_scopes=droppable,
        n_iter=n_iter,
        objetive=objective,
        debug=False,
        auto_log_each_n_iter=10**9,  # disable CoCaBO's own pickle dump
        experiment_name=f"CoCaBO_{dataset}",
    )


def run_cocabo(
    dataset: str,
    num_trials: int,
    seed: int,
    csv_log_file: str,
    config_path: tp.Optional[str] = None,
) -> tp.Tuple[tp.List[float], float]:
    """Run CoCaBO for `num_trials` iterations on `dataset`.

    Writes a CSV with columns ``trial_number,current_optimal`` after each step,
    matching the format used by `baselines/BO_CBO/BO.py`. Returns
    ``(global_opt_per_trial, total_time_seconds)``.
    """
    bundle = build_for_dataset(dataset, config_path=config_path)
    print(f"[CoCaBO] dataset={dataset} target={bundle.target} task={bundle.task}")
    print(f"[CoCaBO] interventional variables: {sorted(bundle.intervention_vars)}")
    print(f"[CoCaBO] domain: "
          f"{[(d.name, d.lower_bound, d.upper_bound) for d in bundle.domain]}")

    exp = _build_experiment(bundle, n_iter=num_trials, dataset=dataset)
    print(f"[CoCaBO] active policy scopes ({len(exp.policies_active)}):")
    for idx, (_, _, mps) in exp.policies_active.items():
        print(f"  scope[{idx}]: I={sorted(mps.interventional_variables)} "
              f"C={sorted(mps.contextual_variables)}")

    os.makedirs(os.path.dirname(csv_log_file), exist_ok=True)
    pd.DataFrame(columns=["trial_number", "current_optimal"]).to_csv(
        csv_log_file, index=False
    )

    global_opt: tp.List[float] = []
    running_best: tp.Optional[float] = None
    is_max = (bundle.task == "max")

    start = time.perf_counter()
    for j in tqdm(range(num_trials), desc=f"CoCaBO/{dataset}"):
        exp.step()
        try:
            y_val = _to_float(exp._results_store[bundle.target][-1])
        except (KeyError, IndexError):
            print(f"[CoCaBO] warning: missing target value at trial {j}; "
                  f"skipping")
            continue

        if running_best is None:
            running_best = y_val
        else:
            running_best = max(running_best, y_val) if is_max else min(running_best, y_val)

        global_opt.append(running_best)
        pd.DataFrame([{"trial_number": j, "current_optimal": running_best}]).to_csv(
            csv_log_file, mode="a", header=False, index=False
        )

    elapsed = time.perf_counter() - start
    print(f"[CoCaBO] completed {num_trials} trials in {elapsed:.2f}s")
    print(f"[CoCaBO] final running best: {running_best}")
    return global_opt, elapsed
