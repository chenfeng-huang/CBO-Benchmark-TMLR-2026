#!/usr/bin/env python3
"""
Generate Synthetic dataset (CompleteGraph-style SCM) and related files
"""

import torch
import pandas as pd
import numpy as np
import json
import pickle
from pandas import DataFrame
import os

# CompleteGraph class implementation
class CompleteGraph(object):
    @staticmethod
    def generate_exogenous_variables(num_samples):
        """Generate all exogenous/latent variables"""
        # Latent variables
        U1 = torch.normal(0, 1, (num_samples, 1))  # ε_YA
        U2 = torch.normal(0, 1, (num_samples, 1))  # ε_YB
        F = torch.normal(0, 1, (num_samples, 1))   # ε_F
        
        return U1, U2, F
    
    @staticmethod
    def A(F, U1, noise_mean=0, noise_stdev=0):
        """A = F² + U₁ + ε_A"""
        noise = torch.normal(noise_mean, noise_stdev, F.shape)
        return F**2 + U1 + noise
    
    @staticmethod
    def B(U2, noise_mean=0, noise_stdev=0):
        """B = U₂ + ε_B"""
        noise = torch.normal(noise_mean, noise_stdev, U2.shape)
        return U2 + noise
    
    @staticmethod
    def C(B, noise_mean=0, noise_stdev=0):
        """C = exp(-B) + ε_C"""
        noise = torch.normal(noise_mean, noise_stdev, B.shape)
        return torch.exp(-B) + noise
    
    @staticmethod
    def D(C, noise_mean=0, noise_stdev=0):
        """D = exp(-C)/10 + ε_D"""
        noise = torch.normal(noise_mean, noise_stdev, C.shape)
        return torch.exp(-C) / 10 + noise
    
    @staticmethod
    def E(A, C, noise_mean=0, noise_stdev=0):
        """E = cos(A) + C/10 + ε_E"""
        noise = torch.normal(noise_mean, noise_stdev, A.shape)
        return torch.cos(A) + C/10 + noise
    
    @staticmethod
    def Y(D, E, U1, U2, noise_mean=0, noise_stdev=0):
        """Y = cos(D) + sin(E) + U₁ + U₂·ε_y"""
        epsilon_y = torch.normal(noise_mean, noise_stdev, D.shape)
        return torch.cos(D) + torch.sin(E) + U1 + U2 * epsilon_y

def generate_completegraph_dataset(num_samples=2000):
    """Generate Synthetic dataset using the CompleteGraph functions"""
    
    print(f"Generating {num_samples} samples...")
    
    # Generate exogenous variables
    U1, U2, F = CompleteGraph.generate_exogenous_variables(num_samples)
    
    # Generate endogenous variables following causal order
    A_data = CompleteGraph.A(F, U1, noise_mean=0, noise_stdev=0)
    B_data = CompleteGraph.B(U2, noise_mean=0, noise_stdev=0)
    C_data = CompleteGraph.C(B_data, noise_mean=0, noise_stdev=0)
    D_data = CompleteGraph.D(C_data, noise_mean=0, noise_stdev=0)
    E_data = CompleteGraph.E(A_data, C_data, noise_mean=0, noise_stdev=0)
    # epsilon_y ~ N(0, 0.1), per sem/synthetic_sem_equations.json noise_std
    Y_data = CompleteGraph.Y(D_data, E_data, U1, U2, noise_mean=0, noise_stdev=0.1)
    
    # Create DataFrame (only observable variables)
    df = DataFrame()
    df['F'] = torch.flatten(F).tolist()
    df['A'] = torch.flatten(A_data).tolist()
    df['B'] = torch.flatten(B_data).tolist()
    df['C'] = torch.flatten(C_data).tolist()
    df['D'] = torch.flatten(D_data).tolist()
    df['E'] = torch.flatten(E_data).tolist()
    df['Y'] = torch.flatten(Y_data).tolist()
    
    print(f"Generated {len(df)} samples")
    
    return df

def create_feature_params(df):
    """Create feature parameters JSON file"""
    feature_params = {}
    
    for col in df.columns:
        feature_params[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    
    return feature_params

def create_sem_equations():
    """Create SEM equations JSON based on CompleteGraph functions"""
    
    sem_equations = {
        "variables": {
            "F": {
                "type": "exogenous",
                "dependencies": [],
                "intercept": 0.0,
                "relationship_type": "normal",
                "relationship_params": {
                    "noise_std": 1.0,
                    "function": "U_F ~ N(0,1)"
                }
            },
            "A": {
                "type": "endogenous",
                "dependencies": ["F"],
                "intercept": 0.0,
                "coefficients": {
                    "F": 1.0  # This will be overridden by custom function
                },
                "relationship_type": "quadratic_latent",
                "relationship_params": {
                    "noise_std": 0.0,
                    "function": "F^2 + U1",
                    "latent_variables": ["U1"]
                }
            },
            "B": {
                "type": "endogenous",
                "dependencies": [],
                "intercept": 0.0,
                "coefficients": {},
                "relationship_type": "latent",
                "relationship_params": {
                    "noise_std": 0.0,
                    "function": "U2",
                    "latent_variables": ["U2"]
                }
            },
            "C": {
                "type": "endogenous",
                "dependencies": ["B"],
                "intercept": 0.0,
                "coefficients": {
                    "B": 1.0  # This will be overridden by custom function
                },
                "relationship_type": "exponential",
                "relationship_params": {
                    "noise_std": 0.0,
                    "function": "exp(-B)"
                }
            },
            "D": {
                "type": "endogenous",
                "dependencies": ["C"],
                "intercept": 0.0,
                "coefficients": {
                    "C": 1.0  # This will be overridden by custom function
                },
                "relationship_type": "scaled_exponential",
                "relationship_params": {
                    "noise_std": 0.0,
                    "function": "exp(-C)/10",
                    "scale_factor": 0.1
                }
            },
            "E": {
                "type": "endogenous",
                "dependencies": ["A", "C"],
                "intercept": 0.0,
                "coefficients": {
                    "A": 1.0,  # This will be overridden by custom function
                    "C": 0.1
                },
                "relationship_type": "trigonometric_mixed",
                "relationship_params": {
                    "noise_std": 0.0,
                    "function": "cos(A) + C/10"
                }
            },
            "Y": {
                "type": "endogenous",
                "dependencies": ["D", "E"],
                "intercept": 0.0,
                "coefficients": {
                    "D": 1.0,  # This will be overridden by custom function
                    "E": 1.0
                },
                "relationship_type": "complex_trigonometric_latent",
                "relationship_params": {
                    "noise_std": 0.1,
                    "function": "cos(D) + sin(E) + U1 + U2*ε_y, ε_y ~ N(0, 0.1)",
                    "latent_variables": ["U1", "U2"]
                }
            }
        },
        "causal_order": ["F", "A", "B", "C", "D", "E", "Y"]
    }
    
    return sem_equations

def create_sem_model(df):
    """Create SEM model pickle file structure"""
    
    # Create adjacency matrix based on dependencies
    node_names = ["F", "A", "B", "C", "D", "E", "Y"]
    n = len(node_names)
    adjacency_matrix = np.zeros((n, n))
    
    # Define edges based on the causal model
    edges = [
        ("F", "A"),  # F -> A
        ("B", "C"),  # B -> C
        ("C", "D"),  # C -> D
        ("A", "E"),  # A -> E
        ("C", "E"),  # C -> E
        ("D", "Y"),  # D -> Y
        ("E", "Y")   # E -> Y
    ]
    
    # Fill adjacency matrix
    for parent, child in edges:
        parent_idx = node_names.index(parent)
        child_idx = node_names.index(child)
        adjacency_matrix[child_idx, parent_idx] = 1  # Note: child as row, parent as column
    
    # Create model results structure
    model_results = {
        'exogenous_variables': ['F'],
        'endogenous_variables': ['A', 'B', 'C', 'D', 'E', 'Y'],
        'intercepts': {
            'F': 0.0,
            'A': 0.0,
            'B': 0.0,
            'C': 0.0,
            'D': 0.0,
            'E': 0.0,
            'Y': 0.0
        },
        'adjacency_matrix': adjacency_matrix,
        'node_names': node_names,
        'causal_order': ["F", "A", "B", "C", "D", "E", "Y"]
    }
    
    return model_results

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic dataset...")
    df = generate_completegraph_dataset(2000)
    
    # Save raw CSV and feature parameters
    os.makedirs('observational_datasets', exist_ok=True)
    df.to_csv('observational_datasets/synthetic.csv', index=False)
    print("Saved synthetic.csv")
    print("Creating feature parameters...")
    feature_params = create_feature_params(df)
    with open('observational_datasets/synthetic_feature_params.json', 'w') as f:
        json.dump(feature_params, f, indent=2)
    print("Saved synthetic_feature_params.json")
    
    # Save SEM artifacts
    os.makedirs('sem', exist_ok=True)
    sem_equations = create_sem_equations()
    with open('sem/synthetic_sem_equations.json', 'w') as f:
        json.dump(sem_equations, f, indent=2)
    print("Saved synthetic_sem_equations.json")
    
    model_results = create_sem_model(df)
    with open('sem/synthetic_sem_model.pkl', 'wb') as f:
        pickle.dump(model_results, f)
    print("Saved synthetic_sem_model.pkl")