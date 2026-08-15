#!/usr/bin/env python3
"""
Generate Interventional Data for Synthetic_2 Dataset

This script generates interventional datasets in .pkl format specifically for the 
Synthetic_2 dataset (Fig. 2b SCM) to support BO and CBO optimization algorithms.

SCM:
- X = U_X ~ N(0,1)
- Z = exp(-X) + U_Z ~ N(0,1)  
- Y = cos(Z) - exp(-Z/20) + U_Y ~ N(0,1)

Interventional domains: D(X) = [-3, 2], D(Z) = [-1, 1]
"""

import numpy as np
import pandas as pd
import json
import yaml
import pickle
import os
from pathlib import Path
from baselines.BO_CBO.graph import setup_optimization_from_discovery


class Synthetic2InterventionalGenerator:
    """Generate interventional data for Synthetic_2 dataset."""
    
    def __init__(self, random_seed=42):
        """Initialize the generator with SCM functions."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Interventional domains as specified in Fig. 2b
        self.interventional_domains = {
            'X': [-3, 2],
            'Z': [-1, 1]
        }
    
    def scm_x(self, n_samples=1):
        """Generate X = U_X ~ N(0,1)"""
        return np.random.normal(0, 1, n_samples)
    
    def scm_z_from_x(self, X):
        """Generate Z = exp(-X) + U_Z where U_Z ~ N(0,1)"""
        U_Z = np.random.normal(0, 1, len(X))
        return np.exp(-X) + U_Z
    
    def scm_y_from_z(self, Z):
        """Generate Y = cos(Z) - exp(-Z/20) + U_Y where U_Y ~ N(0,1)"""
        U_Y = np.random.normal(0, 1, len(Z))
        return np.cos(Z) - np.exp(-Z/20) + U_Y
    
    def intervene_on_x(self, x_values):
        """Perform intervention do(X=x) and generate resulting Y values."""
        # When intervening on X, Z follows: Z = exp(-X) + U_Z
        Z = self.scm_z_from_x(x_values)
        # Y follows: Y = cos(Z) - exp(-Z/20) + U_Y
        Y = self.scm_y_from_z(Z)
        return Y
    
    def intervene_on_z(self, z_values):
        """Perform intervention do(Z=z) and generate resulting Y values."""
        # When intervening on Z directly, Y follows: Y = cos(Z) - exp(-Z/20) + U_Y
        Y = self.scm_y_from_z(z_values)
        return Y
    
    def intervene_on_x_and_z(self, x_values, z_values):
        """Perform joint intervention do(X=x, Z=z) and generate resulting Y values."""
        # When intervening on both X and Z, only Y follows its structural equation
        # Z is set directly, ignoring the causal path from X
        Y = self.scm_y_from_z(z_values)
        return Y

def load_synthetic2_config():
    """Load Synthetic_2 dataset configuration and data."""
    
    # Load configuration
    config_path = 'configs/synthetic_2.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load feature parameters
    feature_params_path = 'observational_datasets/synthetic_2_feature_params.json'
    with open(feature_params_path, 'r') as file:
        feature_params = json.load(file)
    
    # Load observational data
    data_path = 'observational_datasets/synthetic_2.csv'
    df_filtered = pd.read_csv(data_path)
    
    # Load SEM equations
    sem_equations_path = 'sem/synthetic_2_sem_equations.json'
    with open(sem_equations_path, 'r') as file:
        sem_equations = json.load(file)
    
    # Load SEM model
    sem_model_path = 'sem/synthetic_2_sem_model.pkl'
    with open(sem_model_path, 'rb') as file:
        model_results = pickle.load(file)
    
    return config, feature_params, df_filtered, sem_equations, model_results

def generate_bo_interventional_data_synthetic2(generator, manipulative_variables, feature_params, n_samples=1000):
    """
    Generate interventional data for BO algorithm using Synthetic_2 SCM.
    
    Format: [data_x, data_y] where:
    - data_x: (n_samples, n_variables) intervention values
    - data_y: (n_samples,) outcome values
    """
    print(f"Generating BO interventional data for variables: {manipulative_variables}")
    
    # Generate random intervention values within domains
    data_x = []
    for var in manipulative_variables:
        if var in generator.interventional_domains:
            var_min, var_max = generator.interventional_domains[var]
        else:
            var_min = feature_params[var]['min']
            var_max = feature_params[var]['max']
        
        values = np.random.uniform(var_min, var_max, n_samples)
        data_x.append(values)
    
    data_x = np.array(data_x).T  # Shape: (n_samples, n_variables)
    
    # Generate outcomes using SCM interventions
    data_y = []
    
    for i, intervention_values in enumerate(data_x):
        if len(manipulative_variables) == 1:
            if manipulative_variables[0] == 'X':
                # Intervene on X only
                y_outcome = generator.intervene_on_x(np.array([intervention_values[0]]))
            elif manipulative_variables[0] == 'Z':
                # Intervene on Z only
                y_outcome = generator.intervene_on_z(np.array([intervention_values[0]]))
            data_y.append(y_outcome[0])
            
        elif len(manipulative_variables) == 2:
            # Joint intervention on X and Z
            x_val = intervention_values[0] if manipulative_variables[0] == 'X' else intervention_values[1]
            z_val = intervention_values[1] if manipulative_variables[1] == 'Z' else intervention_values[0]
            y_outcome = generator.intervene_on_x_and_z(np.array([x_val]), np.array([z_val]))
            data_y.append(y_outcome[0])
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{n_samples} BO samples")
    
    data_y = np.array(data_y)
    
    print(f"BO data shapes: data_x={data_x.shape}, data_y={data_y.shape}")
    print(f"BO outcome range: [{data_y.min():.4f}, {data_y.max():.4f}]")
    
    return [data_x, data_y]

def generate_cbo_interventional_data_synthetic2(generator, manipulative_variables, feature_params, n_samples=500):
    """
    Generate interventional data for CBO algorithm using Synthetic_2 SCM.
    
    For Synthetic_2, the intervention sets are:
    - Set 1: {X} (single variable intervention)
    - Set 2: {Z} (single variable intervention)  
    - Set 3: {X, Z} (joint intervention, if needed)
    
    Format: List of intervention sets, where each set contains:
    [n_variables, intervention_set, data_x, data_y]
    """
    print(f"Generating CBO interventional data for variables: {manipulative_variables}")
    
    # Define intervention sets for Synthetic_2 (POMIS)
    # For the simple chain X -> Z -> Y, we have:
    intervention_sets = [
        ['X'],      # Intervene on X only
        ['Z'],      # Intervene on Z only
        ['X', 'Z']  # Joint intervention (if both are manipulative)
    ]
    
    # Filter to only include sets with variables in manipulative_variables
    valid_sets = []
    for int_set in intervention_sets:
        if all(var in manipulative_variables for var in int_set):
            valid_sets.append(int_set)
    
    print(f"CBO intervention sets: {valid_sets}")
    
    cbo_data = []
    
    for idx, intervention_set in enumerate(valid_sets):
        print(f"Generating data for intervention set {idx + 1}/{len(valid_sets)}: {intervention_set}")
        
        n_variables = len(intervention_set)
        
        # Generate random intervention values
        data_x = []
        for var in intervention_set:
            if var in generator.interventional_domains:
                var_min, var_max = generator.interventional_domains[var]
            else:
                var_min = feature_params[var]['min']
                var_max = feature_params[var]['max']
            
            values = np.random.uniform(var_min, var_max, n_samples)
            data_x.append(values)
        
        if n_variables == 1:
            data_x = np.array(data_x[0])  # Shape: (n_samples,)
        else:
            data_x = np.array(data_x).T  # Shape: (n_samples, n_variables)
        
        # Generate outcomes using SCM
        data_y = []
        
        for i in range(n_samples):
            if n_variables == 1:
                var_name = intervention_set[0]
                var_value = data_x[i]
                
                if var_name == 'X':
                    y_outcome = generator.intervene_on_x(np.array([var_value]))
                elif var_name == 'Z':
                    y_outcome = generator.intervene_on_z(np.array([var_value]))
                
                data_y.append(y_outcome[0])
                
            else:  # n_variables == 2
                # Joint intervention
                x_val = data_x[i, 0] if intervention_set[0] == 'X' else data_x[i, 1]
                z_val = data_x[i, 1] if intervention_set[1] == 'Z' else data_x[i, 0]
                y_outcome = generator.intervene_on_x_and_z(np.array([x_val]), np.array([z_val]))
                data_y.append(y_outcome[0])
        
        data_y = np.array(data_y)
        
        # Create CBO data entry
        cbo_entry = [n_variables, intervention_set, data_x, data_y]
        cbo_data.append(cbo_entry)
        
        print(f"  Set {idx + 1}: {n_variables} variables, data_x shape={data_x.shape if hasattr(data_x, 'shape') else len(data_x)}, data_y shape={data_y.shape}")
        print(f"  Outcome range: [{data_y.min():.4f}, {data_y.max():.4f}]")
    
    return cbo_data

def save_interventional_data_synthetic2(bo_data, cbo_data):
    """Save interventional data to .pkl files."""
    
    # Create directory if it doesn't exist
    interventional_dir = Path('interventional_datasets')
    interventional_dir.mkdir(exist_ok=True)
    
    # Save BO data
    bo_filename = interventional_dir / 'synthetic_2_interventional_BO.pkl'
    with open(bo_filename, 'wb') as f:
        pickle.dump(bo_data, f)
    print(f"Saved BO data to: {bo_filename}")
    
    # Save CBO data
    cbo_filename = interventional_dir / 'synthetic_2_interventional_CBO.pkl'
    with open(cbo_filename, 'wb') as f:
        pickle.dump(cbo_data, f)
    print(f"Saved CBO data to: {cbo_filename}")
    
    return bo_filename, cbo_filename


def verify_data_format_synthetic2():
    """Verify the generated data format matches expected structure."""
    
    print(f"\n=== Verifying data format for synthetic_2 ===")
    
    # Load BO data
    bo_path = 'interventional_datasets/synthetic_2_interventional_BO.pkl'
    with open(bo_path, 'rb') as f:
        bo_data = pickle.load(f)
    
    print(f"BO data structure:")
    print(f"  Type: {type(bo_data)}")
    print(f"  Length: {len(bo_data)}")
    print(f"  data_x shape: {bo_data[0].shape}")
    print(f"  data_y shape: {bo_data[1].shape}")
    print(f"  data_x range: [{bo_data[0].min():.3f}, {bo_data[0].max():.3f}]")
    print(f"  data_y range: [{bo_data[1].min():.3f}, {bo_data[1].max():.3f}]")
    
    # Load CBO data
    cbo_path = 'interventional_datasets/synthetic_2_interventional_CBO.pkl'
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
        if hasattr(entry[2], 'shape'):
            print(f"    data_x range: [{entry[2].min():.3f}, {entry[2].max():.3f}]")
        print(f"    data_y range: [{entry[3].min():.3f}, {entry[3].max():.3f}]")

def analyze_optimal_interventions(generator):
    """Analyze optimal intervention values for Synthetic_2."""
    
    print("\n=== Analyzing Optimal Interventions ===")
    
    # Test intervention on X
    x_test = np.linspace(generator.interventional_domains['X'][0], 
                        generator.interventional_domains['X'][1], 100)
    y_from_x = []
    for x_val in x_test:
        # Multiple samples to account for noise
        y_samples = [generator.intervene_on_x(np.array([x_val]))[0] for _ in range(10)]
        y_from_x.append(np.mean(y_samples))
    
    best_x_idx = np.argmin(y_from_x)
    best_x = x_test[best_x_idx]
    best_y_from_x = y_from_x[best_x_idx]
    
    print(f"Optimal X intervention:")
    print(f"  X* = {best_x:.4f}")
    print(f"  E[Y | do(X={best_x:.4f})] ≈ {best_y_from_x:.4f}")
    
    # Test intervention on Z
    z_test = np.linspace(generator.interventional_domains['Z'][0], 
                        generator.interventional_domains['Z'][1], 100)
    y_from_z = []
    for z_val in z_test:
        # Multiple samples to account for noise
        y_samples = [generator.intervene_on_z(np.array([z_val]))[0] for _ in range(10)]
        y_from_z.append(np.mean(y_samples))
    
    best_z_idx = np.argmin(y_from_z)
    best_z = z_test[best_z_idx]
    best_y_from_z = y_from_z[best_z_idx]
    
    print(f"Optimal Z intervention:")
    print(f"  Z* = {best_z:.4f}")
    print(f"  E[Y | do(Z={best_z:.4f})] ≈ {best_y_from_z:.4f}")
    
    # Determine globally optimal intervention
    if best_y_from_x < best_y_from_z:
        print(f"Global optimum: Intervene on X = {best_x:.4f} (Y ≈ {best_y_from_x:.4f})")
    else:
        print(f"Global optimum: Intervene on Z = {best_z:.4f} (Y ≈ {best_y_from_z:.4f})")
    
    return {
        'optimal_x': {'value': best_x, 'expected_y': best_y_from_x},
        'optimal_z': {'value': best_z, 'expected_y': best_y_from_z}
    }

def main():
    """Main function to generate Synthetic_2 interventional data."""
    
    print("Generating Interventional Data for Synthetic_2 Dataset (Fig. 2b SCM)")
    print("=" * 70)
    print("SCM:")
    print("  X = U_X ~ N(0,1)")
    print("  Z = exp(-X) + U_Z ~ N(0,1)")
    print("  Y = cos(Z) - exp(-Z/20) + U_Y ~ N(0,1)")
    print("Interventional domains: D(X) = [-3, 2], D(Z) = [-1, 1]")
    print("=" * 70)
    
    try:
        # Load configuration
        config, feature_params, df_filtered, sem_equations, model_results = load_synthetic2_config()
        
        # Initialize generator
        generator = Synthetic2InterventionalGenerator(random_seed=42)
        
        manipulative_variables = list(config['intervention'])
        target = config['target']
        task = config['task']
        
        print(f"Manipulative variables: {manipulative_variables}")
        print(f"Target variable: {target}")
        print(f"Task: {task}")
        
        # Generate BO data
        print(f"\n--- Generating BO Data ---")
        bo_data = generate_bo_interventional_data_synthetic2(
            generator, manipulative_variables, feature_params, n_samples=1000
        )
        
        # Generate CBO data
        print(f"\n--- Generating CBO Data ---")
        cbo_data = generate_cbo_interventional_data_synthetic2(
            generator, manipulative_variables, feature_params, n_samples=500
        )
        
        # Save data
        print(f"\n--- Saving Data ---")
        bo_filename, cbo_filename = save_interventional_data_synthetic2(bo_data, cbo_data)
        
        # Verify data format
        verify_data_format_synthetic2()
        
        # Analyze optimal interventions
        optimal_results = analyze_optimal_interventions(generator)
        
        print("\n=== Summary ===")
        print("Successfully generated interventional data for Synthetic_2:")
        print(f"✅ {bo_filename}")
        print(f"✅ {cbo_filename}")
        print("\nData is ready for BO and CBO optimization algorithms!")
        
        # Save optimal results for reference
        optimal_path = 'observational_datasets/synthetic_2_optimal_interventions.json'
        with open(optimal_path, 'w') as f:
            json.dump(optimal_results, f, indent=2)
        print(f"✅ Saved optimal intervention analysis: {optimal_path}")
        
    except Exception as e:
        print(f"Error generating interventional data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
