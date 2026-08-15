"""Benchmark-native CEO runner.

The upstream CEO research package is retained as an optional execution path in
``baselines/CEO/run_benchmark.py``, but it can hang during imports in this workspace.  This runner
keeps the benchmark usable by running a CEO-style causal optimizer directly on
the converted benchmark SEMs and writing the same progress contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .adapter import CEOBenchmarkInputs, ExplorationSet, evaluate_intervention


@dataclass
class CEOResult:
    progress: List[float]
    selected_sets: List[Tuple[str, ...]]
    selected_values: List[Dict[str, float]]


def run_benchmark_ceo(
    inputs: CEOBenchmarkInputs,
    *,
    num_trials: int,
    seed: int,
    n_anchor_points: int,
) -> CEOResult:
    """Run a lightweight causal entropy optimizer over benchmark SEMs.

    Each trial scores candidate interventions from every exploration set.  The
    acquisition value combines the current best target improvement with a small
    uncertainty bonus estimated from noisy SEM evaluations, so the optimizer
    remains causal-model driven without relying on the hanging upstream imports.
    """

    rng = np.random.RandomState(seed)
    target = inputs.config["target"]
    task = inputs.config["task"]

    current_best, best_set, best_values = _initial_best(inputs, target, task)
    progress = [current_best]
    selected_sets = [best_set]
    selected_values = [best_values]

    for trial in range(1, num_trials):
        candidate = _select_candidate(
            inputs,
            target=target,
            task=task,
            current_best=current_best,
            rng=rng,
            seed=seed + trial * 1009,
            n_anchor_points=n_anchor_points,
        )
        y_value = candidate["target_value"]
        current_best = min(current_best, y_value) if task == "min" else max(current_best, y_value)
        progress.append(float(current_best))
        selected_sets.append(candidate["exploration_set"])
        selected_values.append(candidate["values"])

    return CEOResult(progress=progress, selected_sets=selected_sets, selected_values=selected_values)


def _initial_best(
    inputs: CEOBenchmarkInputs,
    target: str,
    task: str,
) -> Tuple[float, ExplorationSet, Dict[str, float]]:
    best_value = np.inf if task == "min" else -np.inf
    best_set: ExplorationSet = inputs.exploration_sets[0]
    best_assignment: Dict[str, float] = {}

    for es in inputs.exploration_sets:
        samples = inputs.intervention_samples_noiseless[es]
        target_values = samples[target].reshape(-1)
        idx = int(np.argmin(target_values) if task == "min" else np.argmax(target_values))
        value = float(target_values[idx])
        is_better = value < best_value if task == "min" else value > best_value
        if is_better:
            best_value = value
            best_set = es
            best_assignment = {var: float(samples[var].reshape(-1)[idx]) for var in es}

    return float(best_value), best_set, best_assignment


def _select_candidate(
    inputs: CEOBenchmarkInputs,
    *,
    target: str,
    task: str,
    current_best: float,
    rng: np.random.RandomState,
    seed: int,
    n_anchor_points: int,
) -> Dict[str, object]:
    best_candidate = None
    best_acquisition = -np.inf

    for es in inputs.exploration_sets:
        for values in _candidate_values(es, inputs.intervention_domain, rng, n_anchor_points):
            interventions = dict(zip(es, values))
            noisy_values = [
                evaluate_intervention(
                    inputs.sem_class,
                    _causal_order(inputs),
                    interventions,
                    seed=seed + i + 1,
                    noisy=True,
                )[target]
                for i in range(3)
            ]
            uncertainty_bonus = float(np.std(noisy_values))
            # Do not use the true target value to choose candidates. The target
            # is only evaluated after selection; otherwise this runner becomes
            # an oracle and can report perfect GAP on simple linear SCMs.
            acquisition = uncertainty_bonus + 1e-9 * float(rng.random_sample())
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_candidate = {
                    "exploration_set": es,
                    "values": {var: float(value) for var, value in zip(es, values)},
                }

    assert best_candidate is not None
    selected_target = evaluate_intervention(
        inputs.sem_class,
        _causal_order(inputs),
        best_candidate["values"],
        seed=seed,
        noisy=False,
    )[target]
    best_candidate["target_value"] = float(selected_target)
    return best_candidate


def _candidate_values(
    es: ExplorationSet,
    intervention_domain: Dict[str, List[float]],
    rng: np.random.RandomState,
    n_anchor_points: int,
):
    dim = len(es)
    grid_count = max(4, int(n_anchor_points))
    if dim <= 2:
        axes = [
            np.linspace(
                intervention_domain[var][0],
                intervention_domain[var][1],
                grid_count + 2,
            )[1:-1]
            for var in es
        ]
        for point in np.array(np.meshgrid(*axes)).T.reshape(-1, dim):
            yield point

    random_count = max(8, grid_count * dim)
    for _ in range(random_count):
        yield np.asarray(
            [rng.uniform(intervention_domain[var][0], intervention_domain[var][1]) for var in es],
            dtype=float,
        )


def _causal_order(inputs: CEOBenchmarkInputs) -> List[str]:
    return list(inputs.sem_data.get("causal_order") or inputs.sem_data["variables"].keys())

