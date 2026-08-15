#!/usr/bin/env python3
"""
Generate Synthetic_2 Dataset based on Fig. 2(b) SCM

This script generates a synthetic dataset with the causal structure from Fig. 2(b):
X -> Z -> Y where:
- X = U_X (exogenous, U_X ~ N(0,1))
- Z = exp(-X) + U_Z (U_Z ~ N(0,1))
- Y = cos(Z) - exp(-Z/20) + U_Y (U_Y ~ N(0,1))

Interventional variables: I = C = {X, Z}
Interventional domains: D(X) = [-3, 2], D(Z) = [-1, 1]
"""

import numpy as np
import pandas as pd
import json
import yaml
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

class Synthetic2Generator:
    """Generate Synthetic_2 dataset based on Fig. 2(b) SCM"""
    
    def __init__(self, n_samples=2000, random_seed=42):
        """
        Initialize Synthetic_2 generator.
        
        Args:
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define interventional domains as specified
        self.interventional_domains = {
            'X': [-3, 2],
            'Z': [-1, 1]
        }
        
        # Define causal structure
        self.causal_structure = {
            'X': {'parents': [], 'type': 'exogenous'},
            'Z': {'parents': ['X'], 'type': 'endogenous'},
            'Y': {'parents': ['Z'], 'type': 'endogenous'}
        }
        
    def generate_x(self):
        """Generate exogenous variable X = U_X ~ N(0,1)"""
        # Generate from standard normal as specified in the SCM
        U_X = np.random.normal(0, 1, self.n_samples)
        return U_X
    
    def generate_z_from_x(self, X):
        """Generate Z = exp(-X) + U_Z where U_Z ~ N(0,1)"""
        # Generate noise term
        U_Z = np.random.normal(0, 1, self.n_samples)
        
        # Apply the SCM equation: Z = exp(-X) + U_Z
        Z = np.exp(-X) + U_Z
        
        return Z
    
    def generate_y_from_z(self, Z):
        """Generate Y = cos(Z) - exp(-Z/20) + U_Y where U_Y ~ N(0,1)"""
        # Generate noise term
        U_Y = np.random.normal(0, 1, self.n_samples)
        
        # Apply the SCM equation: Y = cos(Z) - exp(-Z/20) + U_Y
        Y = np.cos(Z) - np.exp(-Z/20) + U_Y
        
        return Y
    
    def generate_data(self):
        """Generate complete dataset following the SCM"""
        print(f"Generating Synthetic_2 dataset with {self.n_samples} samples...")
        print("SCM: X = U_X, Z = exp(-X) + U_Z, Y = cos(Z) - exp(-Z/20) + U_Y")
        
        # Generate variables following causal order
        X = self.generate_x()
        Z = self.generate_z_from_x(X)
        Y = self.generate_y_from_z(Z)
        
        # Create dataframe
        data = pd.DataFrame({
            'X': X,
            'Z': Z,
            'Y': Y
        })
        
        print(f"Generated data shapes:")
        print(f"  X: {X.shape}, range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Z: {Z.shape}, range: [{Z.min():.3f}, {Z.max():.3f}]")
        print(f"  Y: {Y.shape}, range: [{Y.min():.3f}, {Y.max():.3f}]")
        
        return data
    
    def create_feature_params(self, data):
        """Create feature parameters JSON with interventional domains"""
        feature_params = {}
        
        for col in data.columns:
            if col in self.interventional_domains:
                # Use interventional domain for intervention variables
                min_val, max_val = self.interventional_domains[col]
            else:
                # Use data range for target variable
                min_val, max_val = data[col].min(), data[col].max()
                
            feature_params[col] = {
                'min': float(min_val),
                'max': float(max_val),
                'mean': float(data[col].mean()),
                'std': float(data[col].std())
            }
        
        return feature_params
    
    def create_config(self):
        """Create dataset configuration YAML"""
        config = {
            'target': 'Y',
            'num_intervention': 1,
            'intervention': ['X', 'Z'],
            'num_observations': self.n_samples,
            'task': 'min',
            'interventional_domain': self.interventional_domains
        }
        
        return config
    
    def create_sem_equations(self):
        """Create SEM equations JSON based on the Fig. 2(b) SCM"""
        sem_equations = {
            'variables': {
                'X': {
                    'type': 'exogenous',
                    'dependencies': [],
                    'intercept': 0.0,
                    'relationship_type': 'normal',
                    'relationship_params': {
                        'noise_std': 1.0,
                        'function': 'U_X ~ N(0,1)'
                    }
                },
                'Z': {
                    'type': 'endogenous',
                    'dependencies': ['X'],
                    'intercept': 0.0,
                    'coefficients': {
                        'X': 1.0  # This will be overridden by custom function
                    },
                    'relationship_type': 'exponential_additive',
                    'relationship_params': {
                        'noise_std': 1.0,
                        'function': 'exp(-X) + U_Z',
                        'noise_distribution': 'N(0,1)'
                    }
                },
                'Y': {
                    'type': 'endogenous',
                    'dependencies': ['Z'],
                    'intercept': 0.0,
                    'coefficients': {
                        'Z': 1.0  # This will be overridden by custom function
                    },
                    'relationship_type': 'trigonometric_exponential_additive',
                    'relationship_params': {
                        'noise_std': 1.0,
                        'function': 'cos(Z) - exp(-Z/20) + U_Y',
                        'noise_distribution': 'N(0,1)'
                    }
                }
            },
            'causal_order': ['X', 'Z', 'Y'],
            'interventional_variables': ['X', 'Z'],
            'target_variable': 'Y'
        }
        
        return sem_equations
    
    def create_sem_model(self, data):
        """Create SEM model pickle"""
        
        # Create adjacency matrix
        node_names = ['X', 'Z', 'Y']
        n = len(node_names)
        adjacency_matrix = np.zeros((n, n))
        
        # Define edges: X -> Z -> Y
        edges = [
            ('X', 'Z'),  # X -> Z
            ('Z', 'Y')   # Z -> Y
        ]
        
        # Fill adjacency matrix
        for parent, child in edges:
            parent_idx = node_names.index(parent)
            child_idx = node_names.index(child)
            adjacency_matrix[child_idx, parent_idx] = 1
        
        # Create model results structure
        model_results = {
            'exogenous_variables': ['X'],
            'endogenous_variables': ['Z', 'Y'],
            'intercepts': {
                'X': 0.0,
                'Z': 0.0,
                'Y': 0.0
            },
            'adjacency_matrix': adjacency_matrix,
            'node_names': node_names,
            'causal_order': ['X', 'Z', 'Y'],
            'interventional_domains': self.interventional_domains,
            'scm_equations': {
                'X': 'U_X ~ N(0,1)',
                'Z': 'exp(-X) + U_Z where U_Z ~ N(0,1)',
                'Y': 'cos(Z) - exp(-Z/20) + U_Y where U_Y ~ N(0,1)'
            }
        }
        
        return model_results
    

    def visualize_data(self, data, save_path=None):
        """Create visualization of the generated data"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Synthetic_2 Dataset (Fig. 2b SCM) Visualization', fontsize=16, fontweight='bold')
        
        # Plot 1: X distribution
        ax1.hist(data['X'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(self.interventional_domains['X'][0], color='red', linestyle='--', alpha=0.7, label='Intervention bounds')
        ax1.axvline(self.interventional_domains['X'][1], color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Frequency')
        ax1.set_title('X Distribution (X = U_X ~ N(0,1))')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z vs X relationship
        ax2.scatter(data['X'], data['Z'], alpha=0.4, s=15, color='green')
        x_range = np.linspace(data['X'].min(), data['X'].max(), 100)
        z_theoretical = np.exp(-x_range)
        ax2.plot(x_range, z_theoretical, 'r-', linewidth=2, label='Z = exp(-X) (no noise)')
        ax2.axhline(self.interventional_domains['Z'][0], color='orange', linestyle='--', alpha=0.7, label='Z intervention bounds')
        ax2.axhline(self.interventional_domains['Z'][1], color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Causal Relationship: X -> Z\n(Z = exp(-X) + U_Z)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Y vs Z relationship
        ax3.scatter(data['Z'], data['Y'], alpha=0.4, s=15, color='orange')
        z_range = np.linspace(data['Z'].min(), data['Z'].max(), 100)
        y_theoretical = np.cos(z_range) - np.exp(-z_range/20)
        ax3.plot(z_range, y_theoretical, 'r-', linewidth=2, label='Y = cos(Z) - exp(-Z/20) (no noise)')
        ax3.set_xlabel('Z')
        ax3.set_ylabel('Y')
        ax3.set_title('Causal Relationship: Z -> Y\n(Y = cos(Z) - exp(-Z/20) + U_Y)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Y distribution
        ax4.hist(data['Y'], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Y Distribution (Target Variable)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization: {save_path}")
        
        plt.close()
    
    def save_all_files(self, data):
        """Save all required files for Synthetic_2 dataset"""
        
        print("\n=== Saving Synthetic_2 Dataset Files ===")
        
        # Create directories
        os.makedirs('observational_datasets', exist_ok=True)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('sem', exist_ok=True)
        os.makedirs('graphs', exist_ok=True)
        
        # 1. Save main dataset
        data_path = 'observational_datasets/synthetic_2.csv'
        data.to_csv(data_path, index=False)
        print(f"✅ Saved dataset: {data_path}")
        
        # Data saved as main dataset
        
        # 3. Save feature parameters
        feature_params = self.create_feature_params(data)
        feature_params_path = 'observational_datasets/synthetic_2_feature_params.json'
        with open(feature_params_path, 'w') as f:
            json.dump(feature_params, f, indent=2)
        print(f"✅ Saved feature parameters: {feature_params_path}")
        
        # 4. Save configuration
        config = self.create_config()
        config_path = 'configs/synthetic_2.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ Saved configuration: {config_path}")
        
        # 5. Save SEM equations
        sem_equations = self.create_sem_equations()
        sem_equations_path = 'sem/synthetic_2_sem_equations.json'
        with open(sem_equations_path, 'w') as f:
            json.dump(sem_equations, f, indent=2)
        print(f"✅ Saved SEM equations: {sem_equations_path}")
        
        # 6. Save SEM model
        sem_model = self.create_sem_model(data)
        sem_model_path = 'sem/synthetic_2_sem_model.pkl'
        with open(sem_model_path, 'wb') as f:
            pickle.dump(sem_model, f)
        print(f"✅ Saved SEM model: {sem_model_path}")
        
        # 7. Create empty categorical encodings (not needed for Synthetic_2)
        categorical_encodings = {}
        categorical_path = 'observational_datasets/synthetic_2_categorical_encodings.json'
        with open(categorical_path, 'w') as f:
            json.dump(categorical_encodings, f, indent=2)
        print(f"✅ Saved categorical encodings: {categorical_path}")
        
        # 8. Theoretical best file is maintained manually
        theoretical_path = 'observational_datasets/synthetic_2_theoretical_best.json'
        
        # 9. Save SEM equations text file
        text_content = f"""Synthetic_2 SEM Equations (Fig. 2b):

Structural Causal Model:
X = U_X,    where U_X ~ N(0,1)
Z = exp(-X) + U_Z,    where U_Z ~ N(0,1)  
Y = cos(Z) - exp(-Z/20) + U_Y,    where U_Y ~ N(0,1)

Causal Structure:
X -> Z -> Y

Interventional Variables: I = C = {{X, Z}}
Interventional Domains: 
- D(X) = [-3, 2]
- D(Z) = [-1, 1]

Target Variable: Y (to be minimized)

Variable Types:
- X: Exogenous (standard normal)
- Z: Endogenous (exponential transformation + noise)
- Y: Endogenous (trigonometric-exponential combination + noise)

Noise Terms:
- All U_X, U_Z, U_Y ~ N(0,1) (independent)
"""
        
        text_path = 'sem/synthetic_2_sem_equations.txt'
        with open(text_path, 'w') as f:
            f.write(text_content)
        print(f"✅ Saved SEM equations text: {text_path}")
        
        # 10. Save visualization
        viz_path = 'graphs/synthetic_2.png'
        self.visualize_data(data, save_path=viz_path)
        
        print(f"\n🎉 All Synthetic_2 dataset files created successfully!")
        
        return {
            'data': data_path,
            'feature_params': feature_params_path,
            'config': config_path,
            'sem_equations': sem_equations_path,
            'sem_model': sem_model_path,
            'categorical': categorical_path,
            'theoretical_best': theoretical_path,
            'sem_text': text_path,
            'visualization': viz_path
        }

def main():
    """Main function to generate Synthetic_2 dataset"""
    
    print("Synthetic_2 Dataset Generator (Fig. 2b SCM)")
    print("=" * 50)
    print("Structural Causal Model:")
    print("  X = U_X ~ N(0,1)")
    print("  Z = exp(-X) + U_Z ~ N(0,1)")
    print("  Y = cos(Z) - exp(-Z/20) + U_Y ~ N(0,1)")
    print("Causal Structure: X -> Z -> Y")
    print("Interventional Variables: I = C = {X, Z}")
    print("Interventional Domains: D(X) = [-3, 2], D(Z) = [-1, 1]")
    print("=" * 50)
    
    # Create generator
    generator = Synthetic2Generator(n_samples=2000, random_seed=42)
    
    # Generate data
    data = generator.generate_data()
    
    # Display summary statistics
    print(f"\nDataset Summary:")
    print(data.describe())
    
    # Save raw CSV and feature parameters
    os.makedirs('observational_datasets', exist_ok=True)
    data_path = 'observational_datasets/synthetic_2.csv'
    data.to_csv(data_path, index=False)
    print(f"\n✅ Saved dataset: {data_path}")
    feature_params = generator.create_feature_params(data)
    feature_params_path = 'observational_datasets/synthetic_2_feature_params.json'
    with open(feature_params_path, 'w') as f:
        json.dump(feature_params, f, indent=2)
    print(f"✅ Saved feature parameters: {feature_params_path}")

    # Save SEM artifacts (required by interventional generation)
    os.makedirs('sem', exist_ok=True)
    sem_equations = generator.create_sem_equations()
    sem_equations_path = 'sem/synthetic_2_sem_equations.json'
    with open(sem_equations_path, 'w') as f:
        json.dump(sem_equations, f, indent=2)
    print(f"✅ Saved SEM equations: {sem_equations_path}")

    sem_model = generator.create_sem_model(data)
    sem_model_path = 'sem/synthetic_2_sem_model.pkl'
    with open(sem_model_path, 'wb') as f:
        pickle.dump(sem_model, f)
    print(f"✅ Saved SEM model: {sem_model_path}")
    
    # Verify SCM relationships
    print(f"\nSCM Relationship Verification:")
    
    # Check X -> Z relationship
    x_sample = data['X'].iloc[:10].values
    z_expected_mean = np.exp(-x_sample)  # Expected mean of Z given X
    z_actual = data['Z'].iloc[:10].values
    print(f"X -> Z relationship check (first 10 samples):")
    print(f"  X values: {x_sample}")
    print(f"  Z expected mean (exp(-X)): {z_expected_mean}")
    print(f"  Z actual: {z_actual}")
    print(f"  Mean deviation from expected: {np.mean(z_actual - z_expected_mean):.4f}")
    
    # Check Z -> Y relationship
    z_sample = data['Z'].iloc[:10].values
    y_expected_mean = np.cos(z_sample) - np.exp(-z_sample/20)  # Expected mean of Y given Z
    y_actual = data['Y'].iloc[:10].values
    print(f"\nZ -> Y relationship check (first 10 samples):")
    print(f"  Z values: {z_sample}")
    print(f"  Y expected mean (cos(Z) - exp(-Z/20)): {y_expected_mean}")
    print(f"  Y actual: {y_actual}")
    print(f"  Mean deviation from expected: {np.mean(y_actual - y_expected_mean):.4f}")
    
    print(f"\n✅ Synthetic_2 dataset generation completed successfully!")
    print(f"   Dataset follows Fig. 2(b) SCM with interventional domains D(X)=[-3,2], D(Z)=[-1,1]")
    
    return data, { 'data': data_path, 'feature_params': feature_params_path, 'sem_equations': sem_equations_path, 'sem_model': sem_model_path }

if __name__ == "__main__":
    main()
