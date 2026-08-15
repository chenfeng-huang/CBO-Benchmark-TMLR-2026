#!/usr/bin/env python3
"""
Generate ToyGraph Dataset

This script generates the ToyGraph dataset with the causal structure:
X -> Z -> Y where:
- X is exogenous (independent variable)
- Z = exp(-X) (exponential relationship)  
- Y = cos(Z) - exp(-Z/20) (trigonometric relationship)

This creates a simple but non-linear causal chain for testing optimization algorithms.
"""

import numpy as np
import pandas as pd
import json
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

class ToyGraphGenerator:
    """Generate ToyGraph synthetic dataset with causal structure X -> Z -> Y"""
    
    def __init__(self, n_samples=1000, random_seed=42):
        """
        Initialize ToyGraph generator.
        
        Args:
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define causal structure
        self.causal_structure = {
            'X': {'parents': [], 'type': 'exogenous'},
            'Z': {'parents': ['X'], 'type': 'endogenous'},  
            'Y': {'parents': ['Z'], 'type': 'endogenous'}
        }
        
    def generate_x(self):
        """Generate exogenous variable X from linspace(-5, 5) with constrained noise"""
        # Use linspace like the reference implementation
        X_base = np.linspace(-5, 5, self.n_samples)
        # Add noise as in reference (noise_stdev=1) but clip to stay within bounds
        noise = np.random.normal(0, 0.5, self.n_samples)  # Reduced noise to stay within bounds
        X = X_base + noise
        # Clip to ensure values stay within the expected range to avoid extreme Z values
        X = np.clip(X, -5, 5)
        return X
    
    def generate_z_from_x(self, X):
        """Generate Z = exp(-X) + noise"""
        # Exponential relationship: Z = exp(-X)
        Z_base = np.exp(-X)
        
        # Add noise as in reference, but scaled relative to the base value to avoid extreme outliers
        noise = np.random.normal(0, 0.1, self.n_samples)  # Reduced noise to prevent extreme values
        Z = Z_base + noise
        
        # Ensure Z stays within reasonable bounds for the interventional domain
        Z = np.clip(Z, -5, 20)
        
        return Z
    
    def generate_y_from_z(self, Z):
        """Generate Y = cos(Z) - exp(-Z/20) + noise"""
        # Trigonometric relationship with decay
        Y_base = np.cos(Z) - np.exp(-Z/20)
        
        # Add noise as in reference, but scaled appropriately
        noise = np.random.normal(0, 0.1, self.n_samples)  # Reduced noise for stability
        Y = Y_base + noise
        
        return Y
    
    def generate_data(self):
        """Generate complete dataset following causal structure"""
        print(f"Generating ToyGraph dataset with {self.n_samples} samples...")
        
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
        """Create feature parameters JSON with reference interventional domains"""
        feature_params = {}
        
        # Use reference interventional domains for X and Z
        reference_domains = {
            'X': [-5, 5],
            'Z': [-5, 20],
            'Y': [data['Y'].min(), data['Y'].max()]  # Keep Y as derived from data
        }
        
        for col in data.columns:
            if col in reference_domains:
                min_val, max_val = reference_domains[col]
            else:
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
            'interventional_domain': {
                'X': [-5, 5],
                'Z': [-5, 20]
            }
        }
        
        return config
    
    def create_sem_equations(self):
        """Create SEM equations JSON"""
        sem_equations = {
            'variables': {
                'X': {
                    'type': 'exogenous',
                    'dependencies': [],
                    'intercept': 0.0,
                    'relationship_type': 'uniform',
                    'relationship_params': {
                        'min_val': -5.0,
                        'max_val': 5.0
                    }
                },
                'Z': {
                    'type': 'endogenous',
                    'dependencies': ['X'],
                    'intercept': 0.0,
                    'coefficients': {
                        'X': 1.0
                    },
                    'relationship_type': 'exponential',
                    'relationship_params': {
                        'noise_std': 0.1,
                        'function': 'exp(-X)'
                    }
                },
                'Y': {
                    'type': 'endogenous',
                    'dependencies': ['Z'],
                    'intercept': 0.0,
                    'coefficients': {
                        'Z': 1.0
                    },
                    'relationship_type': 'trigonometric',
                    'relationship_params': {
                        'noise_std': 0.1,
                        'function': 'cos(Z) - exp(-Z/20)'
                    }
                }
            },
            'causal_order': ['X', 'Z', 'Y']
        }
        
        return sem_equations
    
    def create_sem_model(self, data):
        """Create SEM model pickle (simplified)"""
        
        # Simple linear model coefficients for compatibility
        # This is a simplified representation for the optimization algorithms
        sem_model = {
            'causal_order': ['X', 'Z', 'Y'],
            'fitted_models': {
                'Z': {
                    'coefficients': {'X': -1.0, 'intercept': 0.0},
                    'residuals': np.random.normal(0, 0.1, 100)
                },
                'Y': {
                    'coefficients': {'Z': 1.0, 'intercept': 0.0},
                    'residuals': np.random.normal(0, 0.05, 100)
                }
            },
            # graph.py reads adjacency as A[child, parent] != 0
            'adjacency_matrix': np.array([
                [0, 0, 0],  # X has no parents
                [1, 0, 0],  # Z <- X
                [0, 1, 0]   # Y <- Z
            ]),
            'node_names': ['X', 'Z', 'Y'],
            'intervention_ranges': {
                'X': (data['X'].min(), data['X'].max()),
                'Z': (data['Z'].min(), data['Z'].max())
            }
        }
        
        return sem_model
    
    def visualize_data(self, data, save_path=None):
        """Create visualization of the generated data"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ToyGraph Dataset Visualization', fontsize=16, fontweight='bold')
        
        # Plot 1: X distribution
        ax1.hist(data['X'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Frequency')
        ax1.set_title('X Distribution (Exogenous)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z vs X relationship
        ax2.scatter(data['X'], data['Z'], alpha=0.6, s=20, color='green')
        x_range = np.linspace(data['X'].min(), data['X'].max(), 100)
        z_theoretical = np.exp(-x_range)
        ax2.plot(x_range, z_theoretical, 'r-', linewidth=2, label='Z = exp(-X)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Causal Relationship: X -> Z')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Y vs Z relationship
        ax3.scatter(data['Z'], data['Y'], alpha=0.6, s=20, color='orange')
        z_range = np.linspace(data['Z'].min(), data['Z'].max(), 100)
        y_theoretical = np.cos(z_range) - np.exp(-z_range/20)
        ax3.plot(z_range, y_theoretical, 'r-', linewidth=2, label='Y = cos(Z) - exp(-Z/20)')
        ax3.set_xlabel('Z')
        ax3.set_ylabel('Y')
        ax3.set_title('Causal Relationship: Z -> Y')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Y distribution
        ax4.hist(data['Y'], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Y Distribution (Target)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization: {save_path}")
        
        plt.close()  # Close the figure to prevent display in headless mode
    
    def save_raw_and_feature_params(self, data):
        """Save raw CSV and feature parameters for ToyGraph dataset"""
        os.makedirs('observational_datasets', exist_ok=True)
        data_path = 'observational_datasets/toyGraph.csv'
        data.to_csv(data_path, index=False)
        print(f"✅ Saved dataset: {data_path}")
        feature_params = self.create_feature_params(data)
        feature_params_path = 'observational_datasets/toyGraph_feature_params.json'
        with open(feature_params_path, 'w') as f:
            json.dump(feature_params, f, indent=2)
        print(f"✅ Saved feature parameters: {feature_params_path}")
        
        # Save SEM artifacts
        os.makedirs('sem', exist_ok=True)
        sem_equations = self.create_sem_equations()
        sem_equations_path = 'sem/toyGraph_sem_equations.json'
        with open(sem_equations_path, 'w') as f:
            json.dump(sem_equations, f, indent=2)
        print(f"✅ Saved SEM equations: {sem_equations_path}")
        
        sem_model = self.create_sem_model(data)
        sem_model_path = 'sem/toyGraph_sem_model.pkl'
        with open(sem_model_path, 'wb') as f:
            pickle.dump(sem_model, f)
        print(f"✅ Saved SEM model: {sem_model_path}")
        return {
            'data': data_path,
            'feature_params': feature_params_path,
            'sem_equations': sem_equations_path,
            'sem_model': sem_model_path
        }

def main():
    """Main function to generate ToyGraph dataset"""
    
    print("ToyGraph Dataset Generator")
    print("=" * 40)
    print("Causal Structure: X -> Z -> Y")
    print("X: Exogenous (Normal distribution)")
    print("Z: exp(-X) + noise")
    print("Y: cos(Z) - exp(-Z/20) + noise")
    print("=" * 40)
    
    # Create generator
    generator = ToyGraphGenerator(n_samples=1000, random_seed=42)
    
    # Generate data
    data = generator.generate_data()
    
    # Display summary statistics
    print(f"\nDataset Summary:")
    print(data.describe())
    
    # Save raw CSV and feature parameters
    file_paths = generator.save_raw_and_feature_params(data)
    
    print(f"\nGenerated Files:")
    for file_type, path in file_paths.items():
        print(f"  {file_type}: {path}")
    
    # Verify causal relationships
    print(f"\nCausal Relationship Verification:")
    
    # Check X -> Z relationship
    x_sample = data['X'].iloc[:10].values
    z_expected = np.exp(-x_sample)
    z_actual = data['Z'].iloc[:10].values
    print(f"X -> Z relationship check (first 10 samples):")
    print(f"  X values: {x_sample}")
    print(f"  Z expected (exp(-X)): {z_expected}")
    print(f"  Z actual: {z_actual}")
    print(f"  Mean absolute error: {np.mean(np.abs(z_expected - z_actual)):.4f}")
    
    # Check Z -> Y relationship  
    z_sample = data['Z'].iloc[:10].values
    y_expected = np.cos(z_sample) - np.exp(-z_sample/20)
    y_actual = data['Y'].iloc[:10].values
    print(f"\nZ -> Y relationship check (first 10 samples):")
    print(f"  Z values: {z_sample}")
    print(f"  Y expected (cos(Z) - exp(-Z/20)): {y_expected}")
    print(f"  Y actual: {y_actual}")
    print(f"  Mean absolute error: {np.mean(np.abs(y_expected - y_actual)):.4f}")
    
    print(f"\n✅ ToyGraph dataset and feature parameters saved successfully!")
    
    return data, file_paths

if __name__ == "__main__":
    main()