#!/usr/bin/env python3
"""
Generate healthcare dataset artifacts from PSAGraph class.
"""

import torch
import pandas as pd
import numpy as np
import json
import pickle
from pandas import DataFrame
import os

# PSAGraph class implementation
class PSAGraph(object):
    # U(55, 75)
    @staticmethod
    def age(num_data_points):
        # For now, keep the manual implementation since we don't have functions_utils
        return (75 - 55) * torch.rand(num_data_points, 1) + 55
    
    # N(27.0 − 0.01 × age, 0.7)
    @staticmethod
    def bmi(age):
        return torch.normal(27.0 - 0.01 * age, 0.7)

    # σ(−8.0 + 0.10 × age + 0.03 × bmi)
    @staticmethod
    def aspirin(age, bmi):
        return torch.sigmoid(-8.0 + 0.10 * age + 0.03 * bmi)
        
    # σ(−13.0 + 0.10 × age + 0.20 × bmi)
    @staticmethod
    def statin(age, bmi):
        return torch.sigmoid(-13.0 + 0.10 * age + 0.20 * bmi)
    
    # σ(2.2 − 0.05 × age + 0.01 × bmi − 0.04 × statin + 0.02 × aspirin)
    @staticmethod
    def cancer(age, bmi, aspirin, statin):
        return torch.sigmoid(2.2 - 0.05 * age + 0.01 * bmi - 0.04 * statin + 0.02 * aspirin)
    
    # N (6.8 + 0.04 × age − 0.15 × bmi − 0.60 × statin + 0.55 × aspirin + 1.00 × cancer, 0.4)
    @staticmethod
    def psa(age, bmi, aspirin, statin, cancer):
        return torch.normal(
            6.8 + 0.04 * age - 0.15 * bmi - 0.60 * statin + 0.55 * aspirin + 1.00 * cancer, 0.4
        )

def generate_psa_dataset(num_samples=2000):
    """Generate PSA dataset using the PSAGraph functions"""
    
    # Generate data following causal order
    print(f"Generating {num_samples} samples...")
    
    # Generate age (exogenous) - no noise
    age_data = PSAGraph.age(num_samples)
    
    # Generate BMI (depends on age) - with default noise (0.7)
    bmi_data = PSAGraph.bmi(age_data)
    
    # Generate aspirin (depends on age, bmi) - no additional noise
    aspirin_data = PSAGraph.aspirin(age_data, bmi_data)
    
    # Generate statin (depends on age, bmi) - no additional noise
    statin_data = PSAGraph.statin(age_data, bmi_data)
    
    # Generate cancer (depends on age, bmi, statin, aspirin) - no additional noise
    cancer_data = PSAGraph.cancer(age_data, bmi_data, aspirin_data, statin_data)
    
    # Generate PSA (depends on all previous variables) - with default noise (0.4)
    psa_data = PSAGraph.psa(age_data, bmi_data, aspirin_data, statin_data, cancer_data)
    
    # Create DataFrame
    df = DataFrame()
    df['Age'] = torch.flatten(age_data).tolist()
    df['BMI'] = torch.flatten(bmi_data).tolist()
    df['Aspirin'] = torch.flatten(aspirin_data).tolist()
    df['Statin'] = torch.flatten(statin_data).tolist()
    df['cancer'] = torch.flatten(cancer_data).tolist()
    df['PSA'] = torch.flatten(psa_data).tolist()
    
    # Remove invalid samples (PSA <= 0)
    print(f"Samples before filtering: {len(df)}")
    df = df.drop(df[df['PSA'] <= 0].index)
    print(f"Samples after filtering: {len(df)}")
    
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
    """Create SEM equations JSON based on PSAGraph functions"""
    
    sem_equations = {
        "variables": {
            "Age": {
                "type": "exogenous",
                "dependencies": [],
                "intercept": 65.0,  # Mean of uniform(55, 75)
                "relationship_type": "uniform",
                "relationship_params": {
                    "min_val": 55,
                    "max_val": 75
                }
            },
            "BMI": {
                "type": "endogenous",
                "dependencies": ["Age"],
                "intercept": 27.0,
                "coefficients": {
                    "Age": -0.01
                },
                "relationship_type": "linear",
                "relationship_params": {
                    "noise_std": 0.7
                }
            },
            "Aspirin": {
                "type": "endogenous",
                "dependencies": ["Age", "BMI"],
                "intercept": -8.0,
                "coefficients": {
                    "Age": 0.10,
                    "BMI": 0.03
                },
                "relationship_type": "sigmoid",
                "relationship_params": {
                    "noise_std": 0.0
                }
            },
            "Statin": {
                "type": "endogenous",
                "dependencies": ["Age", "BMI"],
                "intercept": -13.0,
                "coefficients": {
                    "Age": 0.10,
                    "BMI": 0.20
                },
                "relationship_type": "sigmoid",
                "relationship_params": {
                    "noise_std": 0.0
                }
            },
            "cancer": {
                "type": "endogenous",
                "dependencies": ["Age", "BMI", "Statin", "Aspirin"],
                "intercept": 2.2,
                "coefficients": {
                    "Age": -0.05,
                    "BMI": 0.01,
                    "Statin": -0.04,
                    "Aspirin": 0.02
                },
                "relationship_type": "sigmoid",
                "relationship_params": {
                    "noise_std": 0.0
                }
            },
            "PSA": {
                "type": "endogenous",
                "dependencies": ["Age", "BMI", "Statin", "Aspirin", "cancer"],
                "intercept": 6.8,
                "coefficients": {
                    "Age": 0.04,
                    "BMI": -0.15,
                    "Statin": -0.60,
                    "Aspirin": 0.55,
                    "cancer": 1.00
                },
                "relationship_type": "linear",
                "relationship_params": {
                    "noise_std": 0.4
                }
            }
        },
        "causal_order": ["Age", "BMI", "Aspirin", "Statin", "cancer", "PSA"]
    }
    
    return sem_equations

def create_sem_model(df):
    """Create SEM model pickle file structure"""
    
    # Create adjacency matrix based on dependencies
    node_names = ["Age", "BMI", "Aspirin", "Statin", "cancer", "PSA"]
    n = len(node_names)
    adjacency_matrix = np.zeros((n, n))
    
    # Define edges based on PSAGraph structure
    edges = [
        ("Age", "BMI"),
        ("Age", "Aspirin"), 
        ("Age", "Statin"),
        ("Age", "cancer"),
        ("Age", "PSA"),
        ("BMI", "Aspirin"),
        ("BMI", "Statin"), 
        ("BMI", "cancer"),
        ("BMI", "PSA"),
        ("Aspirin", "cancer"),
        ("Aspirin", "PSA"),
        ("Statin", "cancer"),
        ("Statin", "PSA"),
        ("cancer", "PSA")
    ]
    
    # Fill adjacency matrix
    for parent, child in edges:
        parent_idx = node_names.index(parent)
        child_idx = node_names.index(child)
        adjacency_matrix[child_idx, parent_idx] = 1  # Note: child as row, parent as column
    
    # Create model results structure
    model_results = {
        'exogenous_variables': ['Age'],
        'endogenous_variables': ['BMI', 'Aspirin', 'Statin', 'cancer', 'PSA'],
        'intercepts': {
            'Age': 65.0,
            'BMI': 27.0,
            'Aspirin': -8.0,
            'Statin': -13.0,
            'cancer': 2.2,
            'PSA': 6.8
        },
        'adjacency_matrix': adjacency_matrix,
        'node_names': node_names,
        'causal_order': ["Age", "BMI", "Aspirin", "Statin", "cancer", "PSA"]
    }
    
    return model_results

if __name__ == "__main__":
    # Generate dataset
    print("Generating healthcare dataset...")
    df = generate_psa_dataset(2000)
    
    # Save raw CSV and feature parameters
    os.makedirs('observational_datasets', exist_ok=True)
    data_path = 'observational_datasets/healthcare.csv'
    df.to_csv(data_path, index=False)
    print("Saved healthcare.csv")
    print("Creating feature parameters...")
    feature_params = create_feature_params(df)
    with open('observational_datasets/healthcare_feature_params.json', 'w') as f:
        json.dump(feature_params, f, indent=2)
    print("Saved healthcare_feature_params.json")
    
    # Save SEM artifacts
    os.makedirs('sem', exist_ok=True)
    sem_equations = create_sem_equations()
    with open('sem/healthcare_sem_equations.json', 'w') as f:
        json.dump(sem_equations, f, indent=2)
    print("Saved healthcare_sem_equations.json")
    
    model_results = create_sem_model(df)
    with open('sem/healthcare_sem_model.pkl', 'wb') as f:
        pickle.dump(model_results, f)
    print("Saved healthcare_sem_model.pkl")