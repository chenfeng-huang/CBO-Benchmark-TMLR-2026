#!/usr/bin/env python3
"""
Generate interventional data for the protein signaling dataset.

Produces BO and CBO pickle files consumed by scripts/evaluation/run_cbo.py.
"""

import numpy as np
import pandas as pd
import json
import yaml
import pickle
from pathlib import Path
from baselines.BO_CBO.graph import setup_optimization_from_discovery


DATASET = "protein"


def load_config():
    """Load all protein dataset artifacts."""
    with open(f"configs/{DATASET}.yaml") as f:
        config = yaml.safe_load(f)

    with open(f"observational_datasets/{DATASET}_feature_params.json") as f:
        feature_params = json.load(f)

    df = pd.read_csv(f"observational_datasets/{DATASET}.csv")

    with open(f"sem/{DATASET}_sem_equations.json") as f:
        sem_equations = json.load(f)

    model_results = np.load(
        f"sem/{DATASET}_sem_model.pkl", allow_pickle=True
    )

    return config, feature_params, df, sem_equations, model_results


def generate_bo_data(graph, manipulative_variables, feature_params, target, n_samples=1000):
    """[data_x, data_y] for the BO algorithm."""
    print(f"Generating BO interventional data ({n_samples} samples)...")

    ranges_map = graph.get_interventional_ranges()
    ranges = [(ranges_map[v][0], ranges_map[v][1]) for v in manipulative_variables]
    data_x = np.column_stack(
        [np.random.uniform(lo, hi, n_samples) for lo, hi in ranges]
    )

    data_y = np.empty(n_samples)
    intervention_dict_indexed = {v: i for i, v in enumerate(manipulative_variables)}
    target_function, _ = graph.intervention_function(intervention_dict_indexed)

    for i in range(n_samples):
        data_y[i] = target_function(data_x[i : i + 1])[0, 0]
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_samples}")

    print(f"  BO data: x={data_x.shape}, y=[{data_y.min():.4f}, {data_y.max():.4f}]")
    return [data_x, data_y]


def generate_cbo_data(graph, feature_params, target, n_samples=500):
    """List of [n_vars, intervention_set, data_x, data_y] for the CBO algorithm."""
    cbo_vars, _, _ = graph.get_sets()
    print(f"Generating CBO interventional data ({len(cbo_vars)} sets, {n_samples} samples each)...")

    ranges_map = graph.get_interventional_ranges()
    cbo_data = []
    for idx, intervention_set in enumerate(cbo_vars):
        n_vars = len(intervention_set)
        ranges = [(ranges_map[v][0], ranges_map[v][1]) for v in intervention_set]

        cols = [np.random.uniform(lo, hi, n_samples) for lo, hi in ranges]
        if n_vars == 1:
            data_x = cols[0]
        else:
            data_x = np.column_stack(cols)

        data_y = np.empty(n_samples)
        intervention_dict_indexed = {v: i for i, v in enumerate(intervention_set)}
        target_function, _ = graph.intervention_function(intervention_dict_indexed)

        for i in range(n_samples):
            if n_vars == 1:
                x_row = np.array([[data_x[i]]])
            else:
                x_row = data_x[i : i + 1]
            data_y[i] = target_function(x_row)[0, 0]

        cbo_data.append([n_vars, intervention_set, data_x, data_y])
        print(
            f"  Set {idx + 1}/{len(cbo_vars)} {intervention_set}: "
            f"y=[{data_y.min():.4f}, {data_y.max():.4f}]"
        )

    return cbo_data


def save(bo_data, cbo_data):
    """Write pickle files."""
    out = Path("interventional_datasets")
    out.mkdir(exist_ok=True)

    bo_path = out / f"{DATASET}_interventional_BO.pkl"
    with open(bo_path, "wb") as f:
        pickle.dump(bo_data, f)
    print(f"Saved {bo_path}")

    cbo_path = out / f"{DATASET}_interventional_CBO.pkl"
    with open(cbo_path, "wb") as f:
        pickle.dump(cbo_data, f)
    print(f"Saved {cbo_path}")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("Generating Interventional Data for Protein Dataset")
    print("=" * 60)

    config, feature_params, df, sem_equations, model_results = load_config()

    graph, obs_samples, functions = setup_optimization_from_discovery(
        model_results, sem_equations, df, config["target"], config, feature_params
    )

    manipulative_vars = list(config["intervention"])
    target = config["target"]
    print(f"Intervention variables: {manipulative_vars}")
    print(f"Target: {target}, Task: {config['task']}")

    bo_data = generate_bo_data(
        graph, manipulative_vars, feature_params, target, n_samples=1000
    )
    cbo_data = generate_cbo_data(graph, feature_params, target, n_samples=500)

    save(bo_data, cbo_data)

    print("\nDone. Interventional data ready for scripts/evaluation/run_cbo.py --dataset protein")
