#!/usr/bin/env python3
"""
Generate Interventional Data for Epidemiology Dataset

SCM (Structural Causal Model):
B = U[-1, 1]                                    (16)
T = U[4, 8]                                     (17) 
L = exp(0.5 * T + U)                           (18)
R = 4 + L * T                                  (19)
 Y = 0.5 + cos(4 * T) + sin(-L + 2 * R) + B + ε  (20)

Interventions allowed on L and B.
Saves:
- interventional_datasets/epidemiology_interventional_BO.pkl
- interventional_datasets/epidemiology_interventional_CBO.pkl  
"""

import numpy as np
import pandas as pd
import json
import yaml
import pickle
import os
from pathlib import Path
from baselines.BO_CBO.graph import setup_optimization_from_discovery


def load_epidemiology_config():
    dataset_name = 'epidemiology'

    # Config
    config_path = f'configs/{dataset_name}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Feature params
    fp_path = f'observational_datasets/{dataset_name}_feature_params.json'
    with open(fp_path, 'r') as f:
        feature_params = json.load(f)

    # Data
    data_path = f'observational_datasets/{dataset_name}.csv'
    df = pd.read_csv(data_path)

    # SEM
    sem_eq_path = f'sem/{dataset_name}_sem_equations.json'
    with open(sem_eq_path, 'r') as f:
        sem_equations = json.load(f)

    sem_model_path = f'sem/{dataset_name}_sem_model.pkl'
    model_results = np.load(sem_model_path, allow_pickle=True)

    return config, feature_params, df, sem_equations, model_results


def generate_bo_interventional_data(graph, manipulative_variables, feature_params, target, task, n_samples=1000):
    print(f"Generating BO interventional data for variables: {manipulative_variables}")
    ranges_map = graph.get_interventional_ranges()
    ranges = [(ranges_map[v][0], ranges_map[v][1]) for v in manipulative_variables]
    data_x = np.column_stack([
        np.random.uniform(vmin, vmax, n_samples) for vmin, vmax in ranges
    ])

    data_y = []
    all_do = graph.get_all_do()
    do_name = f"do({','.join(manipulative_variables)})" if len(manipulative_variables) > 1 else f"do({manipulative_variables[0]})"
    if do_name not in all_do:
        graph.create_combined_do_function(manipulative_variables)

    for i in range(n_samples):
        try:
            idx_map = {v: j for j, v in enumerate(manipulative_variables)}
            target_fn, space = graph.intervention_function(idx_map)
            y = target_fn(data_x[i].reshape(1, -1))
            data_y.append(y[0, 0])
        except Exception as e:
            fallback = np.random.uniform(feature_params[target]['min'], feature_params[target]['max'])
            data_y.append(fallback)
            if i < 10:  # Only print first few warnings
                print(f"Warning: Used fallback for sample {i}: {e}")

    data_y = np.asarray(data_y)
    return [data_x, data_y]


def generate_cbo_interventional_data(graph, manipulative_variables, feature_params, target, task, n_samples=500):
    print(f"Generating CBO interventional data for variables: {manipulative_variables}")
    cbo_vars, POMIS, _ = graph.get_sets()
    ranges_map = graph.get_interventional_ranges()
    out = []
    for intervention_set in cbo_vars:
        ranges = [(ranges_map[v][0], ranges_map[v][1]) for v in intervention_set]
        if len(intervention_set) == 1:
            data_x = np.random.uniform(ranges[0][0], ranges[0][1], n_samples)
        else:
            data_x = np.column_stack([
                np.random.uniform(vmin, vmax, n_samples) for vmin, vmax in ranges
            ])
        data_y = []
        for i in range(n_samples):
            try:
                idx_map = {v: j for j, v in enumerate(intervention_set)}
                target_fn, space = graph.intervention_function(idx_map)
                x_row = data_x[i] if hasattr(data_x, 'shape') else np.array([data_x[i]])
                y = target_fn(np.asarray(x_row).reshape(1, -1))
                data_y.append(y[0, 0])
            except Exception as e:
                data_y.append(np.random.uniform(feature_params[target]['min'], feature_params[target]['max']))
                if i < 5:  # Only print first few warnings per set
                    print(f"Warning: Used fallback for CBO sample {i}: {e}")
        
        data_y = np.asarray(data_y)
        out.append([data_x, data_y])
    return out



def main():
    print("Generating Epidemiology Interventional Datasets")
    print("=" * 50)
    print("SCM Equations:")
    print("  B = U[-1, 1]")
    print("  T = U[4, 8]")
    print("  L = exp(0.5 * T + U)")
    print("  R = 4 + L * T")
    print("  Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε")
    print("=" * 50)
    
    # Load configuration and data
    config, feature_params, df, sem_equations, model_results = load_epidemiology_config()
    
    target = config['target']
    manipulative_variables = config['intervention']
    task = config['task']
    
    print(f"Target variable: {target}")
    print(f"Manipulative variables: {manipulative_variables}")
    print(f"Task: {task}")
    
    # Display intervention ranges
    print(f"\nIntervention ranges:")
    for var in manipulative_variables:
        min_val, max_val = feature_params[var]['min'], feature_params[var]['max']
        print(f"  {var}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Setup graph
    print("\nSetting up causal graph...")
    graph, observational_samples, functions = setup_optimization_from_discovery(
        model_results, sem_equations, df, target, config, feature_params
    )
    
    # Create output directories
    os.makedirs('interventional_datasets', exist_ok=True)
    os.makedirs('observational_datasets', exist_ok=True)
    
    # Generate BO interventional data
    print("\n" + "="*30)
    print("GENERATING BO DATA")
    print("="*30)
    bo_data = generate_bo_interventional_data(
        graph, manipulative_variables, feature_params, target, task, n_samples=1000
    )
    
    bo_path = 'interventional_datasets/epidemiology_interventional_BO.pkl'
    with open(bo_path, 'wb') as f:
        pickle.dump(bo_data, f)
    print(f"✅ Saved BO data: {bo_path}")
    print(f"   Shape: X={bo_data[0].shape}, Y={bo_data[1].shape}")
    
    # Generate CBO interventional data
    print("\n" + "="*30)
    print("GENERATING CBO DATA")
    print("="*30)
    cbo_data = generate_cbo_interventional_data(
        graph, manipulative_variables, feature_params, target, task, n_samples=500
    )
    
    cbo_path = 'interventional_datasets/epidemiology_interventional_CBO.pkl'
    with open(cbo_path, 'wb') as f:
        pickle.dump(cbo_data, f)
    print(f"✅ Saved CBO data: {cbo_path}")
    print(f"   Number of intervention sets: {len(cbo_data)}")
    for i, (x, y) in enumerate(cbo_data):
        print(f"   Set {i}: X={x.shape if hasattr(x, 'shape') else len(x)}, Y={y.shape}")
    
    
    print("\n" + "="*50)
    print("EPIDEMIOLOGY INTERVENTIONAL DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Files generated:")
    print(f"  - {bo_path}")
    print(f"  - {cbo_path}")


if __name__ == '__main__':
    main()
