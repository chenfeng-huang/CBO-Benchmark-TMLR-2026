#!/usr/bin/env python3
"""
Generate Chain Dataset (X -> W -> Z -> Y) with linear-Gaussian SCM

SCM (as per provided description):
X = U_X
W = U_W
Z = -0.5 X + U_Z
Y = - W - 3 Z X + U_Y

U_X, U_W, U_Z, U_Y ~ N(0,1)

Intervention domains (example): Z in [-1, 1], W in [-1, 1]
"""

import numpy as np
import pandas as pd
import json
import os
import pickle


class ChainGenerator:
    def __init__(self, n_samples=2000, random_seed=42):
        self.n_samples = n_samples
        np.random.seed(random_seed)

    def generate_data(self):
        X = np.random.normal(0, 1, self.n_samples)
        W = np.random.normal(0, 1, self.n_samples)
        Z = -0.5 * X + np.random.normal(0, 1, self.n_samples)
        Y = - W - 3.0 * Z * X + np.random.normal(0, 1, self.n_samples)

        df = pd.DataFrame({
            'X': X,
            'W': W,
            'Z': Z,
            'Y': Y
        })
        return df

    def create_feature_params(self, df):
        feature_params = {}
        for col in df.columns:
            feature_params[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        return feature_params

    def create_sem_equations(self):
        sem = {
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
                'W': {
                    'type': 'exogenous',
                    'dependencies': [],
                    'intercept': 0.0,
                    'relationship_type': 'normal',
                    'relationship_params': {
                        'noise_std': 1.0,
                        'function': 'U_W ~ N(0,1)'
                    }
                },
                'Z': {
                    'type': 'endogenous',
                    'dependencies': ['X'],
                    'intercept': 0.0,
                    'coefficients': {
                        'X': -0.5
                    },
                    'relationship_type': 'linear',
                    'relationship_params': {
                        'noise_std': 1.0,
                        'function': 'Z = -0.5*X + U_Z'
                    }
                },
                'Y': {
                    'type': 'endogenous',
                    'dependencies': ['X', 'W', 'Z'],
                    'intercept': 0.0,
                    'coefficients': {
                        'W': -1.0,
                        'X': 0.0,
                        'Z': 0.0
                    },
                    'relationship_type': 'interaction_linear',
                    'relationship_params': {
                        'noise_std': 1.0,
                        'interactions': [
                            {'variables': ['Z', 'X'], 'coefficient': -3.0}
                        ],
                        'function': 'Y = -W - 3*Z*X + U_Y'
                    }
                }
            },
            'causal_order': ['X', 'W', 'Z', 'Y'],
            'interventional_variables': ['Z', 'W'],
            'target_variable': 'Y'
        }
        return sem

    def create_sem_model(self, df):
        node_names = ['X','W','Z','Y']
        adjacency = np.zeros((4,4))
        # X -> Z, W -> Y, Z -> Y, X -> Y (through interaction, keep simple edge X->Y)
        adjacency[node_names.index('Z'), node_names.index('X')] = 1
        adjacency[node_names.index('Y'), node_names.index('W')] = 1
        adjacency[node_names.index('Y'), node_names.index('Z')] = 1
        adjacency[node_names.index('Y'), node_names.index('X')] = 1
        return {
            'node_names': node_names,
            'adjacency_matrix': adjacency,
            'causal_order': ['X','W','Z','Y']
        }


def main():
    print('Generating Chain dataset...')
    gen = ChainGenerator(n_samples=2000, random_seed=42)
    df = gen.generate_data()

    os.makedirs('observational_datasets', exist_ok=True)
    os.makedirs('sem', exist_ok=True)

    data_path = 'observational_datasets/chain.csv'
    df.to_csv(data_path, index=False)
    print(f'Saved {data_path}')

    feature_params = gen.create_feature_params(df)
    fp_path = 'observational_datasets/chain_feature_params.json'
    with open(fp_path, 'w') as f:
        json.dump(feature_params, f, indent=2)
    print(f'Saved {fp_path}')

    sem_eq = gen.create_sem_equations()
    sem_eq_path = 'sem/chain_sem_equations.json'
    with open(sem_eq_path, 'w') as f:
        json.dump(sem_eq, f, indent=2)
    print(f'Saved {sem_eq_path}')

    sem_model = gen.create_sem_model(df)
    sem_model_path = 'sem/chain_sem_model.pkl'
    with open(sem_model_path, 'wb') as f:
        pickle.dump(sem_model, f)
    print(f'Saved {sem_model_path}')

    print('Chain dataset generation complete.')


if __name__ == '__main__':
    main()


