#!/usr/bin/env python3
"""
Generate Epidemiology Observational Dataset

SCM (Structural Causal Model):
B = U[-1, 1]                                    (16)
T = U[4, 8]                                     (17) 
L = exp(0.5 * T + U)                           (18)
R = 4 + L * T                                  (19)
Y = 0.5 + cos(4 * T) + sin(-L + 2 * R) + B + ε  (20)

Where U ~ N(0,1) for noise terms and ε ~ N(0,1)

Causal Structure: B (exogenous), T (exogenous) → L → R → Y
Interventional Variables: L, B
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class EpidemiologyGenerator:
    def __init__(self, n_samples=2000, random_seed=42):
        """Initialize epidemiology dataset generator"""
        self.n_samples = n_samples
        np.random.seed(random_seed)
        
        # Define causal relationships
        self.causal_relationships = {
            'B': [],  # Exogenous uniform [-1, 1]
            'T': [],  # Exogenous uniform [4, 8]
            'L': ['T'],  # L = exp(0.5 * T + U)
            'R': ['L', 'T'],  # R = 4 + L * T
            'Y': ['T', 'L', 'R', 'B']  # Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε
        }
        
        self.causal_order = ['B', 'T', 'L', 'R', 'Y']
        self.target = 'Y'
        self.interventional_variables = ['L', 'B']
        
        print(f"Epidemiology Dataset Generator initialized")
        print(f"Causal order: {' → '.join(self.causal_order)}")
        print(f"Interventional variables: {self.interventional_variables}")
        print(f"Target variable: {self.target}")
    
    def generate_data(self):
        """Generate epidemiology dataset following the SCM"""
        print(f"\nGenerating epidemiology dataset with {self.n_samples} samples...")
        
        # Generate exogenous variables
        B = np.random.uniform(-1, 1, self.n_samples)  # B = U[-1, 1]
        T = np.random.uniform(4, 8, self.n_samples)   # T = U[4, 8]
        
        # Generate L = exp(0.5 * T + U) where U ~ N(0,1)
        U_L = np.random.normal(0, 1, self.n_samples)
        L = np.exp(0.5 * T + U_L)
        
        # Generate R = 4 + L * T
        R = 4 + L * T
        
        # Generate Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε
        epsilon = np.random.normal(0, 1, self.n_samples)
        Y = 0.5 + np.cos(4 * T) + np.sin(-L + 2 * R) + B + epsilon
        
        # Create dataframe
        data = pd.DataFrame({
            'B': B,
            'T': T,
            'L': L,
            'R': R,
            'Y': Y
        })
        
        print(f"Generated data shapes:")
        for col in data.columns:
            print(f"  {col}: range=[{data[col].min():.3f}, {data[col].max():.3f}], mean={data[col].mean():.3f}")
        
        return data
    
    def create_feature_params(self, data):
        """Create feature parameters JSON"""
        feature_params = {}
        for col in data.columns:
            var_stats = data[col].describe()
            feature_params[col] = {
                'min': float(var_stats['min']),
                'max': float(var_stats['max']),
                'mean': float(var_stats['mean']),
                'std': float(var_stats['std'])
            }
        return feature_params
    
    def create_sem_equations(self):
        """Create SEM equations JSON file"""
        sem_equations = {
            "variables": {
                "B": {
                    "type": "exogenous",
                    "dependencies": [],
                    "intercept": 0.0,
                    "relationship_type": "uniform",
                    "relationship_params": {
                        "min_val": -1.0,
                        "max_val": 1.0,
                        "function": "B = U[-1, 1]"
                    }
                },
                "T": {
                    "type": "exogenous",
                    "dependencies": [],
                    "intercept": 0.0,
                    "relationship_type": "uniform",
                    "relationship_params": {
                        "min_val": 4.0,
                        "max_val": 8.0,
                        "function": "T = U[4, 8]"
                    }
                },
                "L": {
                    "type": "endogenous",
                    "dependencies": ["T"],
                    "intercept": 0.0,
                    "coefficients": {
                        "T": 0.5
                    },
                    "relationship_type": "exponential_linear",
                    "relationship_params": {
                        "noise_std": 1.0,
                        "function": "L = exp(0.5 * T + U) where U ~ N(0,1)"
                    }
                },
                "R": {
                    "type": "endogenous",
                    "dependencies": ["L", "T"],
                    "intercept": 4.0,
                    "coefficients": {
                        "L": 0.0,
                        "T": 0.0
                    },
                    "relationship_type": "interaction_linear",
                    "relationship_params": {
                        "noise_std": 0.0,
                        "interactions": [
                            {
                                "variables": ["L", "T"],
                                "coefficient": 1.0
                            }
                        ],
                        "function": "R = 4 + L * T"
                    }
                },
                "Y": {
                    "type": "endogenous",
                    "dependencies": ["T", "L", "R", "B"],
                    "intercept": 0.5,
                    "coefficients": {
                        "T": 1.0,
                        "L": 1.0,
                        "R": 2.0,
                        "B": 1.0
                    },
                    "relationship_type": "complex_trigonometric",
                    "relationship_params": {
                        "noise_std": 1.0,
                        "function": "Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε where ε ~ N(0,1)"
                    }
                }
            },
            "causal_order": self.causal_order,
            "interventional_variables": self.interventional_variables,
            "target_variable": self.target
        }
        
        return sem_equations
    
    def create_sem_model(self, data):
        """Create SEM model pickle file"""
        # Create adjacency matrix
        n_vars = len(self.causal_order)
        adj_matrix = np.zeros((n_vars, n_vars))
        var_to_idx = {var: idx for idx, var in enumerate(self.causal_order)}
        
        # Define edges based on causal relationships
        edges = {
            ('T', 'L'): 0.5,  # Coefficient in exp(0.5 * T + U)
            ('L', 'R'): 1.0,  # Coefficient in R = 4 + L * T
            ('T', 'R'): 1.0,  # Coefficient in R = 4 + L * T
            ('T', 'Y'): 4.0,  # Coefficient in cos(4*T)
            ('L', 'Y'): -1.0, # Coefficient in sin(-L + 2*R)
            ('R', 'Y'): 2.0,  # Coefficient in sin(-L + 2*R)
            ('B', 'Y'): 1.0   # Additive effect of B on Y
        }
        
        for (parent, child), coef in edges.items():
            parent_idx = var_to_idx[parent]
            child_idx = var_to_idx[child]
            # graph.py reads adjacency as A[child, parent] != 0
            adj_matrix[child_idx, parent_idx] = coef
        
        sem_model = {
            'node_names': self.causal_order,
            'adjacency_matrix': adj_matrix,
            'causal_order': self.causal_order,
            'data_stats': {var: {
                'mean': float(data[var].mean()),
                'std': float(data[var].std()),
                'min': float(data[var].min()),
                'max': float(data[var].max())
            } for var in data.columns},
            'causal_relationships': self.causal_relationships,
            'target': self.target,
            'interventional_variables': self.interventional_variables
        }
        
        return sem_model
    
    def save_all_files(self, data):
        """Save all generated files"""
        print("\nSaving files...")
        
        # Create directories
        os.makedirs('sem', exist_ok=True)
        os.makedirs('observational_datasets', exist_ok=True)
        
        # Save dataset
        data_path = 'observational_datasets/epidemiology.csv'
        data.to_csv(data_path, index=False)
        print(f"✅ Saved dataset: {data_path}")
        
        # Save feature parameters
        feature_params = self.create_feature_params(data)
        fp_path = 'observational_datasets/epidemiology_feature_params.json'
        with open(fp_path, 'w') as f:
            json.dump(feature_params, f, indent=2)
        print(f"✅ Saved feature parameters: {fp_path}")
        
        # Save SEM equations
        sem_equations = self.create_sem_equations()
        sem_eq_path = 'sem/epidemiology_sem_equations.json'
        with open(sem_eq_path, 'w') as f:
            json.dump(sem_equations, f, indent=2)
        print(f"✅ Saved SEM equations: {sem_eq_path}")
        
        # Save SEM model
        sem_model = self.create_sem_model(data)
        sem_model_path = 'sem/epidemiology_sem_model.pkl'
        with open(sem_model_path, 'wb') as f:
            pickle.dump(sem_model, f)
        print(f"✅ Saved SEM model: {sem_model_path}")
        
        return {
            'data': data_path,
            'feature_params': fp_path,
            'sem_equations': sem_eq_path,
            'sem_model': sem_model_path
        }
    
    def print_summary(self, data):
        """Print summary of the epidemiology dataset"""
        print("\n" + "="*60)
        print("EPIDEMIOLOGY DATASET SUMMARY")
        print("="*60)
        print(f"Dataset shape: {data.shape}")
        print(f"Variables: {', '.join(data.columns)}")
        print(f"Target variable: {self.target}")
        print(f"Interventional variables: {', '.join(self.interventional_variables)}")
        print(f"Causal order: {' → '.join(self.causal_order)}")
        
        print(f"\nSCM Equations:")
        print("  B = U[-1, 1]")
        print("  T = U[4, 8]") 
        print("  L = exp(0.5 * T + U)")
        print("  R = 4 + L * T")
        print("  Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε")
        
        print(f"\nData statistics:")
        print(data.describe())
        
        print(f"\nCausal relationships:")
        for var, parents in self.causal_relationships.items():
            if parents:
                print(f"  {var} ← {', '.join(parents)}")
            else:
                print(f"  {var} (exogenous)")
        
        print("="*60)


def main():
    """Main function to generate epidemiology dataset"""
    print("Epidemiology Dataset Generator")
    print("="*50)
    print("SCM Equations:")
    print("  B = U[-1, 1]")
    print("  T = U[4, 8]")
    print("  L = exp(0.5 * T + U)")
    print("  R = 4 + L * T")
    print("  Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε")
    print("="*50)
    
    # Create generator
    generator = EpidemiologyGenerator(n_samples=2000, random_seed=42)
    
    # Generate data
    data = generator.generate_data()
    
    # Save all files
    files = generator.save_all_files(data)
    
    # Print summary
    generator.print_summary(data)
    
    print(f"\n✅ Epidemiology dataset generation complete!")
    print("Files generated:")
    for file_type, path in files.items():
        print(f"  - {path}")


if __name__ == '__main__':
    main()
