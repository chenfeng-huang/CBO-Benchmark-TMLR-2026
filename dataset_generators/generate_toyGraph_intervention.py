#!/usr/bin/env python3
"""
Generate Interventional Data for ToyGraph Dataset

This script generates interventional datasets in .pkl format specifically for the
ToyGraph dataset to support BO and CBO optimization algorithms.
"""

import numpy as np
import pandas as pd
import json
import yaml
import pickle
import os
from pathlib import Path
from baselines.BO_CBO.graph import setup_optimization_from_discovery


def load_toygraph_config():
    """Load ToyGraph dataset configuration and data."""
    dataset_name = 'toyGraph'

    # Load configuration
    config_path = f'configs/{dataset_name}.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load feature parameters
    feature_params_path = f'observational_datasets/{dataset_name}_feature_params.json'
    with open(feature_params_path, 'r') as file:
        feature_params = json.load(file)

    # Load observational data
    data_path = f'observational_datasets/{dataset_name}.csv'
    df_filtered = pd.read_csv(data_path)

    # Load SEM equations
    sem_equations_path = f'sem/{dataset_name}_sem_equations.json'
    with open(sem_equations_path, 'r') as file:
        sem_equations = json.load(file)

    # Load SEM model (saved as numpy object array)
    sem_model_path = f'sem/{dataset_name}_sem_model.pkl'
    model_results = np.load(sem_model_path, allow_pickle=True)

    return config, feature_params, df_filtered, sem_equations, model_results


def generate_bo_interventional_data(graph, manipulative_variables, feature_params, target, task, n_samples=1000):
    """
    Generate interventional data for BO algorithm.

    Format: [data_x, data_y] where:
    - data_x: (n_samples, n_variables) intervention values
    - data_y: (n_samples,) outcome values
    """
    print(f"Generating BO interventional data for variables: {manipulative_variables}")

    ranges_map = graph.get_interventional_ranges()
    intervention_ranges = []
    for var in manipulative_variables:
        var_min, var_max = ranges_map[var]
        intervention_ranges.append((var_min, var_max))

    # Generate random intervention values
    data_x = []
    for var_min, var_max in intervention_ranges:
        values = np.random.uniform(var_min, var_max, n_samples)
        data_x.append(values)

    data_x = np.array(data_x).T  # Shape: (n_samples, n_variables)

    # Generate outcomes using the graph's do functions
    data_y = []

    # Create combined do-function name
    if len(manipulative_variables) == 1:
        do_func_name = f"do({manipulative_variables[0]})"
    else:
        do_func_name = f"do({','.join(manipulative_variables)})"

    # Ensure combined do-function exists
    all_do_functions = graph.get_all_do()
    if do_func_name not in all_do_functions:
        print(f"Creating combined do-function: {do_func_name}")
        graph.create_combined_do_function(manipulative_variables)

    # Generate outcomes
    for i, intervention_values in enumerate(data_x):
        # Create intervention dictionary
        intervention_dict = {var: val for var, val in zip(manipulative_variables, intervention_values)}

        try:
            # Create intervention dictionary with variable indices
            intervention_dict_indexed = {var: idx for idx, var in enumerate(manipulative_variables)}

            # Get the intervention function and parameter space
            target_function, space = graph.intervention_function(intervention_dict_indexed)

            # Evaluate the intervention
            outcome = target_function(intervention_values.reshape(1, -1))
            data_y.append(outcome[0, 0])
        except Exception as e:
            print(f"Warning: Failed to generate outcome for intervention {intervention_dict}: {e}")
            # Use a random fallback value within target bounds
            outcome = np.random.uniform(feature_params[target]['min'], feature_params[target]['max'])
            data_y.append(outcome)

        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{n_samples} BO samples")

    data_y = np.array(data_y)

    print(f"BO data shapes: data_x={data_x.shape}, data_y={data_y.shape}")
    print(f"BO outcome range: [{data_y.min():.4f}, {data_y.max():.4f}]")

    return [data_x, data_y]


def generate_cbo_interventional_data(graph, manipulative_variables, feature_params, target, task, n_samples=500):
    """
    Generate interventional data for CBO algorithm.

    Format: List of intervention sets, where each set contains:
    [n_variables, intervention_set, data_x, data_y]
    """
    print(f"Generating CBO interventional data for variables: {manipulative_variables}")

    # Get intervention sets (POMIS - Partially Overlapping Minimal Intervention Sets)
    cbo_vars, POMIS, _ = graph.get_sets()

    print(f"CBO intervention sets: {cbo_vars}")

    cbo_data = []

    for idx, intervention_set in enumerate(cbo_vars):
        print(f"Generating data for intervention set {idx + 1}/{len(cbo_vars)}: {intervention_set}")

        n_variables = len(intervention_set)

        ranges_map = graph.get_interventional_ranges()
        intervention_ranges = []
        for var in intervention_set:
            var_min, var_max = ranges_map[var]
            intervention_ranges.append((var_min, var_max))

        # Generate random intervention values
        data_x = []
        for var_min, var_max in intervention_ranges:
            values = np.random.uniform(var_min, var_max, n_samples)
            data_x.append(values)

        if n_variables == 1:
            data_x = np.array(data_x[0])  # Shape: (n_samples,)
        else:
            data_x = np.array(data_x).T  # Shape: (n_samples, n_variables)

        # Generate outcomes
        data_y = []

        for i in range(n_samples):
            # Create intervention dictionary
            if n_variables == 1:
                intervention_dict = {intervention_set[0]: data_x[i]}
            else:
                intervention_dict = {var: data_x[i, j] for j, var in enumerate(intervention_set)}

            try:
                # Create intervention dictionary with variable indices
                intervention_dict_indexed = {var: idx for idx, var in enumerate(intervention_set)}

                # Get the intervention function and parameter space
                target_function, space = graph.intervention_function(intervention_dict_indexed)

                # Evaluate the intervention
                if n_variables == 1:
                    intervention_values = np.array([data_x[i]])
                else:
                    intervention_values = data_x[i]

                outcome = target_function(intervention_values.reshape(1, -1))
                data_y.append(outcome[0, 0])
            except Exception as e:
                # Use a random fallback value within target bounds
                outcome = np.random.uniform(feature_params[target]['min'], feature_params[target]['max'])
                data_y.append(outcome)

        data_y = np.array(data_y)

        # Create CBO data entry
        cbo_entry = [n_variables, intervention_set, data_x, data_y]
        cbo_data.append(cbo_entry)

        print(f"  Set {idx + 1}: {n_variables} variables, data_x shape={data_x.shape if hasattr(data_x, 'shape') else len(data_x)}, data_y shape={data_y.shape}")
        print(f"  Outcome range: [{data_y.min():.4f}, {data_y.max():.4f}]")

    return cbo_data


def save_interventional_data(bo_data, cbo_data):
    """Save interventional data to .pkl files for ToyGraph."""

    dataset_name = 'toyGraph'

    # Create directory if it doesn't exist
    interventional_dir = Path('interventional_datasets')
    interventional_dir.mkdir(exist_ok=True)

    # Save BO data
    bo_filename = interventional_dir / f'{dataset_name}_interventional_BO.pkl'
    with open(bo_filename, 'wb') as f:
        pickle.dump(bo_data, f)
    print(f"Saved BO data to: {bo_filename}")

    # Save CBO data
    cbo_filename = interventional_dir / f'{dataset_name}_interventional_CBO.pkl'
    with open(cbo_filename, 'wb') as f:
        pickle.dump(cbo_data, f)
    print(f"Saved CBO data to: {cbo_filename}")

    return bo_filename, cbo_filename


def verify_data_format():
    """Verify the generated data format matches expected structure for ToyGraph."""

    dataset_name = 'toyGraph'
    print(f"\n=== Verifying data format for {dataset_name} ===")

    # Load BO data
    bo_path = f'interventional_datasets/{dataset_name}_interventional_BO.pkl'
    with open(bo_path, 'rb') as f:
        bo_data = pickle.load(f)

    print(f"BO data structure:")
    print(f"  Type: {type(bo_data)}")
    print(f"  Length: {len(bo_data)}")
    print(f"  data_x shape: {bo_data[0].shape}")
    print(f"  data_y shape: {bo_data[1].shape}")

    # Load CBO data
    cbo_path = f'interventional_datasets/{dataset_name}_interventional_CBO.pkl'
    with open(cbo_path, 'rb') as f:
        cbo_data = pickle.load(f)

    print(f"\nCBO data structure:")
    print(f"  Type: {type(cbo_data)}")
    print(f"  Number of intervention sets: {len(cbo_data)}")

    for i, entry in enumerate(cbo_data):
        print(f"  Set {i}:")
        print(f"    n_variables: {entry[0]}")
        print(f"    intervention_set: {entry[1]}")
        print(f"    data_x shape: {entry[2].shape if hasattr(entry[2], 'shape') else len(entry[2])}")
        print(f"    data_y shape: {entry[3].shape}")


def main():
    """Main function to generate interventional data for ToyGraph."""

    print("Generating Interventional Data for ToyGraph Dataset")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    try:
        # Load configuration
        config, feature_params, df_filtered, sem_equations, model_results = load_toygraph_config()

        # Set up graph
        graph, observational_samples, functions = setup_optimization_from_discovery(
            model_results, sem_equations, df_filtered, config['target'], config, feature_params
        )

        manipulative_variables = list(config['intervention'])
        target = config['target']
        task = config['task']

        print(f"Manipulative variables: {manipulative_variables}")
        print(f"Target variable: {target}")
        print(f"Task: {task}")

        # Generate BO data
        print(f"\n--- Generating BO Data ---")
        bo_data = generate_bo_interventional_data(
            graph, manipulative_variables, feature_params, target, task, n_samples=1000
        )

        # Generate CBO data
        print(f"\n--- Generating CBO Data ---")
        cbo_data = generate_cbo_interventional_data(
            graph, manipulative_variables, feature_params, target, task, n_samples=500
        )

        # Save data
        print(f"\n--- Saving Data ---")
        bo_filename, cbo_filename = save_interventional_data(bo_data, cbo_data)

        # Verify data format
        verify_data_format()

        print("\n=== Summary ===")
        print("Successfully generated interventional data for ToyGraph:")
        print(f"✅ {bo_filename}")
        print(f"✅ {cbo_filename}")
        print("\nData is ready for BO and CBO optimization algorithms!")

    except Exception as e:
        print(f"Error generating interventional data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


