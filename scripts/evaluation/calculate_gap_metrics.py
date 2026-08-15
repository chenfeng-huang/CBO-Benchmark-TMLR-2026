#!/usr/bin/env python3
import argparse
"""
Script to calculate GAP and PA_GAP metrics for different trial counts.
This script processes optimization results and computes metrics for 100, 50, and 20 trials.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.GAP import GAP
from metrics.PA_GAP import PA_GAP


def load_theoretical_best(dataset_name: str) -> float:
    """Load the theoretical best value for a given dataset."""
    theoretical_best_file = f"observational_datasets/{dataset_name}_theoretical_best.json"
    
    if not os.path.exists(theoretical_best_file):
        raise FileNotFoundError(f"Theoretical best file not found: {theoretical_best_file}")
    
    with open(theoretical_best_file, 'r') as f:
        data = json.load(f)
    
    return data['global_best_value']


def calculate_metrics_for_trials(best_so_far_values: list, theoretical_best: float, 
                                trial_counts: list = [100, 50, 20], task: str = 'min'):
    """
    Calculate GAP and PA_GAP metrics for different trial counts.
    
    Args:
        best_so_far_values: List of best-so-far values from optimization
        theoretical_best: Theoretical optimal value
        trial_counts: List of trial counts to evaluate
        task: 'min' for minimization, 'max' for maximization
    
    Returns:
        Dictionary with results for each trial count
    """
    results = {}
    
    for trial_count in trial_counts:
        if trial_count > len(best_so_far_values):
            print(f"Warning: Requested {trial_count} trials but only {len(best_so_far_values)} available")
            trial_count = len(best_so_far_values)
        
        # Extract the first trial_count values
        truncated_values = best_so_far_values[:trial_count]
        
        # Calculate GAP metric
        gap_calculator = GAP(theoretical_best)
        gap_calculator.calculate_GAP(truncated_values, task)
        
        # Calculate PA_GAP metric
        pa_gap_calculator = PA_GAP(theoretical_best)
        pa_gap_calculator.calculate_PA_GAP(truncated_values, task)
        
        results[trial_count] = {
            'GAP': {
                'value': gap_calculator.GAP_value,
                'optimal_trial': gap_calculator.optimal_trial,
                'optimal_value': gap_calculator.optimal_value,
                'initial_value': gap_calculator.initial_value,
                'total_trials': gap_calculator.total_trials
            },
            'PA_GAP': {
                'value': pa_gap_calculator.PA_GAP_value,
                'std': np.std(pa_gap_calculator.PA_GAP_record) if len(pa_gap_calculator.PA_GAP_record) > 0 else 0.0,
                'optimal_trial': pa_gap_calculator.optimal_trial,
                'final_value': pa_gap_calculator.final_value,
                'initial_value': pa_gap_calculator.initial_value,
                'total_trials': pa_gap_calculator.total_trials,
                'improvement_achieved': pa_gap_calculator.improvement_achieved,
                'improvement_possible': pa_gap_calculator.improvement_possible,
                'pa_gap_per_trial': pa_gap_calculator.PA_GAP_record
            }
        }
    
    return results


def process_csv_file(
    csv_file_path: str,
    dataset_name: str,
    trial_counts: list = [100, 50, 20],
    task: str = "min",
):
    """
    Process a CSV file and calculate GAP and PA_GAP metrics.
    
    Args:
        csv_file_path: Path to the CSV file with results
        dataset_name: Name of the dataset (for loading theoretical best)
        trial_counts: List of trial counts to evaluate
    
    Returns:
        Dictionary with calculated metrics
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract the best-so-far values column. The BO/CBO runners write this
    # column as `current_optimal`, but older runs may have used
    # `best_so_far_value`, so accept either for backward compatibility.
    if 'current_optimal' in df.columns:
        value_column = 'current_optimal'
    elif 'best_so_far_value' in df.columns:
        value_column = 'best_so_far_value'
    else:
        raise ValueError(
            f"Neither 'current_optimal' nor 'best_so_far_value' column found in {csv_file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    best_so_far_values = df[value_column].tolist()
    
    # Load theoretical best value
    theoretical_best = load_theoretical_best(dataset_name)
    
    # Calculate metrics
    results = calculate_metrics_for_trials(
        best_so_far_values,
        theoretical_best,
        trial_counts,
        task=task,
    )
    
    return results


def print_results(results: dict, method_name: str, dataset_name: str, theoretical_best: float = None):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Results for {method_name} on {dataset_name}")
    if theoretical_best is not None:
        print(f"Theoretical Best Value: {theoretical_best:.6f}")
    print(f"{'='*60}")
    
    for trial_count, metrics in results.items():
        print(f"\n--- {trial_count} Trials ---")
        
        # GAP results
        gap = metrics['GAP']
        print(f"GAP Metric:")
        print(f"  Value: {gap['value']:.6f}")
        print(f"  Optimal found at trial: {gap['optimal_trial']}")
        print(f"  Optimal value: {gap['optimal_value']:.6f}")
        print(f"  Initial value: {gap['initial_value']:.6f}")
        
        # PA_GAP results
        pa_gap = metrics['PA_GAP']
        print(f"PA_GAP Metric:")
        print(f"  Value (Average): {pa_gap['value']:.6f}")
        print(f"  Standard Deviation: {pa_gap['std']:.6f}")
        print(f"  Optimal found at trial: {pa_gap['optimal_trial']}")
        print(f"  Final value: {pa_gap['final_value']:.6f}")
        print(f"  Initial value: {pa_gap['initial_value']:.6f}")
        
        # Show PA_GAP progression for each trial
        if 'pa_gap_per_trial' in pa_gap and len(pa_gap['pa_gap_per_trial']) > 0:
            print(f"  PA_GAP per trial (first 10 trials):")
            for i, trial_pa_gap in enumerate(pa_gap['pa_gap_per_trial'][:10], 1):
                print(f"    Trial {i}: {trial_pa_gap:.6f}")
            
            if len(pa_gap['pa_gap_per_trial']) > 10:
                print(f"    ... (showing first 10 of {len(pa_gap['pa_gap_per_trial'])} trials)")
                print(f"    Last trial ({len(pa_gap['pa_gap_per_trial'])}): {pa_gap['pa_gap_per_trial'][-1]:.6f}")
            
            # Additional statistics
            pa_gap_values = pa_gap['pa_gap_per_trial']
            print(f"  PA_GAP Statistics:")
            print(f"    Min: {min(pa_gap_values):.6f}")
            print(f"    Max: {max(pa_gap_values):.6f}")
            print(f"    Median: {np.median(pa_gap_values):.6f}")


def save_results_to_csv(results: dict, method_name: str, dataset_name: str, output_file: str = None):
    """Save results to CSV file."""
    os.makedirs(f"eval_results/{method_name}/{dataset_name}", exist_ok=True)
    if output_file is None:
        output_file = f"eval_results/{method_name}/{dataset_name}/gap_metrics_{method_name}_{dataset_name}.csv"
    
    # Prepare data for CSV
    csv_data = []
    for trial_count, metrics in results.items():
        row = {
            'method': method_name,
            'dataset': dataset_name,
            'trial_count': trial_count,
            'GAP_value': metrics['GAP']['value'],
            'GAP_optimal_trial': metrics['GAP']['optimal_trial'],
            'GAP_optimal_value': metrics['GAP']['optimal_value'],
            'GAP_initial_value': metrics['GAP']['initial_value'],
            'PA_GAP_value': metrics['PA_GAP']['value'],
            'PA_GAP_std': metrics['PA_GAP']['std'],
            'PA_GAP_optimal_trial': metrics['PA_GAP']['optimal_trial'],
            'PA_GAP_final_value': metrics['PA_GAP']['final_value'],
            'PA_GAP_initial_value': metrics['PA_GAP']['initial_value']
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def save_detailed_pa_gap_to_csv(results: dict, method_name: str, dataset_name: str, output_file: str = None):
    """Save detailed PA_GAP per trial data to CSV file."""
    if output_file is None:
        output_file = f"eval_results/{method_name}/{dataset_name}/detailed_pa_gap_{method_name}_{dataset_name}.csv"
    
    # Prepare detailed data for CSV
    detailed_data = []
    for trial_count, metrics in results.items():
        pa_gap_per_trial = metrics['PA_GAP']['pa_gap_per_trial']
        for trial_idx, pa_gap_value in enumerate(pa_gap_per_trial, 1):
            row = {
                'method': method_name,
                'dataset': dataset_name,
                'trial_count_setting': trial_count,
                'trial_number': trial_idx,
                'PA_GAP_value': pa_gap_value
            }
            detailed_data.append(row)
    
    df = pd.DataFrame(detailed_data)
    df.to_csv(output_file, index=False)
    print(f"Detailed PA_GAP per trial saved to: {output_file}")


def main(args):
    """Main function to demonstrate usage."""
    # Example usage with the provided DCBO data
    csv_file = f"results/{args.method}/{args.dataset}/{args.method}_{args.dataset}_100-trials_progress.csv"
    dataset_name = args.dataset    
    method_name = args.method

    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    try:
        # Load theoretical best value
        theoretical_best = load_theoretical_best(dataset_name)
        
        # Calculate metrics
        results = process_csv_file(
            csv_file,
            dataset_name,
            trial_counts=[100, 50, 20],
            task=args.task,
        )
        
        # Print results
        print_results(results, method_name, dataset_name, theoretical_best)
        
        # Save results to CSV
        save_results_to_csv(results, method_name, dataset_name)
        
        # Save detailed PA_GAP per trial data
        save_detailed_pa_gap_to_csv(results, method_name, dataset_name)
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate GAP and PA_GAP metrics')
    parser.add_argument('--dataset', type=str, default='chain', help='Dataset name')
    parser.add_argument('--method', type=str, default='MCBO', help='Method name')
    parser.add_argument(
        "--task",
        type=str,
        choices=["min", "max"],
        default="min",
        help="Optimization task type: minimization (min) or maximization (max)",
    )
    args = parser.parse_args()
    main(args)
