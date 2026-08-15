"""Bridge CBO_Benchmark datasets into the sibling CEO implementation.

The CEO package expects static DBN-style SEM callables, NetworkX graphs whose
nodes are suffixed by a time index, observational samples as numpy arrays, and
initial interventional samples keyed by exploration set.  CBO_Benchmark stores
the same information as YAML configs, SEM JSON files, and CSV datasets; this
module performs that conversion for the benchmark runners.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml


ExplorationSet = Tuple[str, ...]


@dataclass
class CEOBenchmarkInputs:
    """Container passed by the vendored CEO benchmark runner."""

    dataset: str
    config: dict
    sem_data: dict
    sem_class: type
    graphs: List[nx.MultiDiGraph]
    init_posterior: List[float]
    observation_samples: Dict[str, np.ndarray]
    intervention_domain: Dict[str, List[float]]
    exploration_sets: List[ExplorationSet]
    intervention_samples: Dict[ExplorationSet, Dict[str, np.ndarray]]
    intervention_samples_noiseless: Dict[ExplorationSet, Dict[str, np.ndarray]]


def load_ceo_benchmark_inputs(
    dataset: str,
    benchmark_root: Path,
    *,
    seed: int,
    seeds_replicate: Sequence[int],
    num_graph_candidates: int,
) -> CEOBenchmarkInputs:
    """Load and convert one CBO_Benchmark dataset for CEO."""

    config_path = benchmark_root / "configs" / f"{dataset}.yaml"
    sem_path = benchmark_root / "sem" / f"{dataset}_sem_equations.json"
    obs_path = benchmark_root / "observational_datasets" / f"{dataset}.csv"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    with sem_path.open("r") as f:
        sem_data = json.load(f)

    df = pd.read_csv(obs_path)
    cols_to_drop = config.get("cols_to_drop") or []
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    if config.get("num_observations"):
        df = df.sample(
            n=min(int(config["num_observations"]), len(df)),
            random_state=seed,
            replace=False,
        )

    causal_order = _causal_order(sem_data)
    sem_class = make_sem_class(sem_data, causal_order)
    observation_samples = _observation_dict(df, causal_order)
    intervention_domain = {
        var: [float(bounds[0]), float(bounds[1])]
        for var, bounds in config["interventional_domain"].items()
    }
    intervention_vars = [str(v) for v in config["intervention"]]
    exploration_sets = _all_exploration_sets(intervention_vars)
    graphs = _build_graph_candidates(
        sem_data,
        causal_order,
        num_graph_candidates=max(1, int(num_graph_candidates)),
    )
    init_posterior = [1.0 / len(graphs)] * len(graphs)

    intervention_samples, intervention_samples_noiseless = _initial_interventions(
        sem_class,
        causal_order,
        exploration_sets,
        intervention_domain,
        seeds_replicate,
    )

    return CEOBenchmarkInputs(
        dataset=dataset,
        config=config,
        sem_data=sem_data,
        sem_class=sem_class,
        graphs=graphs,
        init_posterior=init_posterior,
        observation_samples=observation_samples,
        intervention_domain=intervention_domain,
        exploration_sets=exploration_sets,
        intervention_samples=intervention_samples,
        intervention_samples_noiseless=intervention_samples_noiseless,
    )


def make_sem_class(sem_data: dict, causal_order: Sequence[str]) -> type:
    """Create a CEO-compatible SEM class from benchmark SEM JSON."""

    variables = sem_data["variables"]
    functions = OrderedDict(
        (var, _make_node_function(var, variables[var])) for var in causal_order
    )

    class BenchmarkSEM:
        @staticmethod
        def static() -> OrderedDict:
            return functions.copy()

        @staticmethod
        def dynamic():
            return None

    return BenchmarkSEM


def _causal_order(sem_data: dict) -> List[str]:
    order = sem_data.get("causal_order")
    if order:
        return [str(v) for v in order]
    return list(sem_data["variables"].keys())


def _observation_dict(df: pd.DataFrame, causal_order: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for var in causal_order:
        if var not in df.columns:
            raise KeyError(f"Observational dataset missing SEM variable '{var}'")
        out[var] = df[var].to_numpy(dtype=float).reshape(-1, 1)
    return out


def _all_exploration_sets(intervention_vars: Sequence[str]) -> List[ExplorationSet]:
    return [
        tuple(combo)
        for size in range(1, len(intervention_vars) + 1)
        for combo in itertools.combinations(intervention_vars, size)
    ]


def _build_true_graph(sem_data: dict, causal_order: Sequence[str]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for var in causal_order:
        graph.add_node(f"{var}_0")
    for child, node_info in sem_data["variables"].items():
        for parent in node_info.get("dependencies", []):
            if parent in sem_data["variables"]:
                graph.add_edge(f"{parent}_0", f"{child}_0")
    graph.T = 1
    return graph


def _build_graph_candidates(
    sem_data: dict,
    causal_order: Sequence[str],
    *,
    num_graph_candidates: int,
) -> List[nx.MultiDiGraph]:
    true_graph = _build_true_graph(sem_data, causal_order)
    graphs = [true_graph]
    edges = list(true_graph.edges(keys=True))

    for edge in edges:
        if len(graphs) >= num_graph_candidates:
            break
        candidate = true_graph.copy()
        candidate.remove_edge(*edge)
        candidate.T = 1
        if nx.is_directed_acyclic_graph(candidate):
            graphs.append(candidate)

    # Keep the posterior shape requested even for tiny graphs with few edges.
    while len(graphs) < num_graph_candidates:
        duplicate = true_graph.copy()
        duplicate.T = 1
        graphs.append(duplicate)

    return graphs


def _initial_interventions(
    sem_class: type,
    causal_order: Sequence[str],
    exploration_sets: Sequence[ExplorationSet],
    intervention_domain: Mapping[str, Sequence[float]],
    seeds_replicate: Sequence[int],
) -> Tuple[
    Dict[ExplorationSet, Dict[str, np.ndarray]],
    Dict[ExplorationSet, Dict[str, np.ndarray]],
]:
    noisy = {es: {var: [] for var in causal_order} for es in exploration_sets}
    noiseless = {es: {var: [] for var in causal_order} for es in exploration_sets}

    for es in exploration_sets:
        for seed in seeds_replicate:
            rng = np.random.RandomState(seed)
            levels = {
                var: float(rng.uniform(intervention_domain[var][0], intervention_domain[var][1]))
                for var in es
            }
            noisy_sample = _sample_sem_once(sem_class, causal_order, levels, rng, noisy=True)
            noiseless_sample = _sample_sem_once(
                sem_class, causal_order, levels, rng, noisy=False
            )
            for var in causal_order:
                noisy[es][var].append(float(noisy_sample[var]))
                noiseless[es][var].append(float(noiseless_sample[var]))

    return _stack_samples(noisy), _stack_samples(noiseless)


def _stack_samples(
    samples: Dict[ExplorationSet, Dict[str, List[float]]]
) -> Dict[ExplorationSet, Dict[str, np.ndarray]]:
    return {
        es: {var: np.asarray(values, dtype=float).reshape(-1, 1) for var, values in data.items()}
        for es, data in samples.items()
    }


def _sample_sem_once(
    sem_class: type,
    causal_order: Sequence[str],
    interventions: Mapping[str, float],
    rng: np.random.RandomState,
    *,
    noisy: bool,
) -> Dict[str, float]:
    sem = sem_class.static()
    sample = OrderedDict((var, np.zeros(1, dtype=float)) for var in causal_order)
    for var, fn in sem.items():
        if var in interventions:
            value = interventions[var]
        else:
            epsilon = float(rng.normal()) if noisy else 0.0
            value = fn(epsilon, 0, sample)
        sample[var][0] = float(np.asarray(value).squeeze())
    return {var: float(sample[var][0]) for var in causal_order}


def evaluate_intervention(
    sem_class: type,
    causal_order: Sequence[str],
    interventions: Mapping[str, float],
    *,
    seed: int,
    noisy: bool = False,
) -> Dict[str, float]:
    """Evaluate one intervention assignment through the adapted benchmark SEM."""

    rng = np.random.RandomState(seed)
    return _sample_sem_once(sem_class, causal_order, interventions, rng, noisy=noisy)


def _make_node_function(var: str, node_info: dict) -> Callable:
    rel_type = node_info.get("relationship_type", "linear")
    deps = list(node_info.get("dependencies", []))
    coeffs = node_info.get("coefficients", {})
    params = node_info.get("relationship_params", {})
    intercept = float(node_info.get("intercept", 0.0))

    def node_function(epsilon=0.0, t=0, sample=None):
        values = {
            dep: float(np.asarray(sample[dep][t]).squeeze())
            for dep in deps
            if sample is not None and dep in sample
        }
        return _evaluate_relationship(
            var,
            rel_type,
            deps,
            coeffs,
            params,
            intercept,
            values,
            float(np.asarray(epsilon).squeeze()),
        )

    return node_function


def _evaluate_relationship(
    var: str,
    rel_type: str,
    deps: Sequence[str],
    coeffs: Mapping[str, float],
    params: Mapping[str, object],
    intercept: float,
    values: Mapping[str, float],
    epsilon: float,
) -> float:
    noise_std = float(params.get("noise_std", 0.0) or 0.0)
    noise = noise_std * epsilon

    if rel_type == "uniform":
        lo = float(params.get("min_val", intercept - 1.0))
        hi = float(params.get("max_val", intercept + 1.0))
        if abs(epsilon) < 1e-12:
            return 0.5 * (lo + hi)
        # Probability integral transform: Phi(eps) is exactly U(0,1) for
        # eps ~ N(0,1), so the node is exactly Uniform(lo, hi).
        return lo + (hi - lo) * 0.5 * (1.0 + math.erf(epsilon / math.sqrt(2.0)))

    if rel_type == "normal":
        return intercept + noise

    if rel_type == "linear":
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "sigmoid":
        return _sigmoid(_linear(deps, coeffs, intercept, values)) + noise

    if rel_type == "binary_sigmoid":
        threshold = float(params.get("threshold", 0.5))
        return float(_sigmoid(_linear(deps, coeffs, intercept, values)) + noise > threshold)

    if rel_type in {"exponential", "exponential_additive"}:
        if len(deps) == 1 and deps[0] in values:
            return math.exp(-values[deps[0]]) + noise
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "exponential_linear":
        if len(deps) == 1 and deps[0] in values:
            coef = float(coeffs.get(deps[0], 0.5))
            return math.exp(coef * values[deps[0]] + noise)
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "interaction_linear":
        result = _linear(deps, coeffs, intercept, values)
        interactions = params.get("interactions", []) or []
        if interactions and all(dep in values for dep in deps):
            for interaction in interactions:
                product = 1.0
                for name in interaction.get("variables", []):
                    product *= values[name]
                result += float(interaction.get("coefficient", 1.0)) * product
        return result + noise

    if rel_type in {"trigonometric", "trigonometric_exponential_additive"}:
        if len(deps) == 1 and deps[0] in values:
            z_val = values[deps[0]]
            return math.cos(z_val) - math.exp(-z_val / 20.0) + noise
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "quadratic_latent":
        if len(deps) == 1 and deps[0] in values:
            return values[deps[0]] ** 2 + noise
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "latent":
        return noise

    if rel_type == "scaled_exponential":
        if len(deps) == 1 and deps[0] in values:
            return math.exp(-values[deps[0]]) * float(params.get("scale_factor", 0.1)) + noise
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "trigonometric_mixed":
        if len(deps) >= 2 and all(dep in values for dep in deps[:2]):
            return math.cos(values[deps[0]]) + values[deps[1]] / 10.0 + noise
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "complex_trigonometric_latent":
        if len(deps) >= 2 and all(dep in values for dep in deps[:2]):
            return math.cos(values[deps[0]]) + math.sin(values[deps[1]]) + noise
        return _linear(deps, coeffs, intercept, values) + noise

    if rel_type == "complex_trigonometric":
        if {"T", "L", "R", "B"}.issubset(values):
            return (
                intercept
                + math.cos(4.0 * values["T"])
                + math.sin(-values["L"] + 2.0 * values["R"])
                + values["B"]
                + noise
            )
        return _linear(deps, coeffs, intercept, values) + noise

    return _linear(deps, coeffs, intercept, values) + noise


def _linear(
    deps: Iterable[str],
    coeffs: Mapping[str, float],
    intercept: float,
    values: Mapping[str, float],
) -> float:
    result = intercept
    for dep in deps:
        if dep in values:
            result += float(coeffs.get(dep, 0.0)) * values[dep]
    return result


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

