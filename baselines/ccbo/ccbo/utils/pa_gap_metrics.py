# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PA_GAP metrics for optimization algorithm evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any


class PA_GAP:
    """
    Unified PA_GAP metrics calculator for optimization algorithms.
    
    This class provides robust PA_GAP metric calculations with multiple formulations,
    error handling, validation, and comprehensive result reporting.
    """
    
    def __init__(self, best_theoretical: float):
        """
        Initialize PA_GAP calculator.
        
        Args:
            best_theoretical: The theoretical best/optimal value for the optimization problem
        """
        self.total_trials = 0
        self.initial_value = 0
        self.final_value = 0
        self.best_theoretical = best_theoretical
        self.PA_GAP_value = 0
        self.improvement_achieved = 0
        self.improvement_possible = 0
        self.optimal_trial = 0
        self.PA_GAP_record = []

    def calculate_PA_GAP(self, 
                        outcomes: List[float],
                        task: str = 'min') -> Dict[str, Any]:
        """
        Calculate PA_GAP metrics from optimization outcomes.
        
        Args:
            outcomes: List of objective function values over trials
            task: 'min' for minimization, 'max' for maximization
            
        Returns:
            Dictionary containing PA_GAP metrics and summary statistics
        """
        outcomes_array = np.array(outcomes)
        self.initial_value = outcomes_array[0]
        self.total_trials = len(outcomes_array) - 1  # Exclude initial value
        total_PA_GAP = []
        
        # Track the best value found so far at each trial
        current_best = self.initial_value
        for i in range(1, len(outcomes_array)):
            # Update best value found so far
            if task == 'min':
                if outcomes_array[i] < current_best:
                    current_best = outcomes_array[i]
            else:
                if outcomes_array[i] > current_best:
                    current_best = outcomes_array[i]
            
            # Calculate PA_GAP for the best value found so far
            total_PA_GAP.append(self._calculate_PA_GAP(current_best, i, task))

        # Basic metrics - use the best value found in all trials
        self.final_value = min(outcomes_array) if task == 'min' else max(outcomes_array)
        
        # Averaging PA_GAP
        self.PA_GAP_record = total_PA_GAP
        self.PA_GAP_value = sum(total_PA_GAP) / len(total_PA_GAP) if len(total_PA_GAP) > 0 else 0.0

        # Find optimal trial (when best value was first reached) 
        self.optimal_trial = len(outcomes_array)  # Default to last trial
        for i, outcome in enumerate(outcomes_array):
            if abs(outcome - self.final_value) < 1e-10:
                self.optimal_trial = i
                break
        
        # Calculate final improvement metrics
        if task == 'max':
            self.improvement_achieved = max(0, self.final_value - self.initial_value) 
            self.improvement_possible = max(0, self.best_theoretical - self.initial_value)
        else:
            self.improvement_achieved = max(0, self.initial_value - self.final_value)
            self.improvement_possible = max(0, self.initial_value - self.best_theoretical)
        
        # Return comprehensive results
        return self.get_results_summary()

    def _calculate_PA_GAP(self, current_best_value: float, trial: int, task: str) -> float:
        """
        Calculate PA_GAP for a single trial.
        
        Args:
            current_best_value: Best value found up to this trial
            trial: Current trial number (1-indexed)
            task: 'min' or 'max'
            
        Returns:
            PA_GAP value for this trial
        """
        # Calculate improvement correctly for the task type
        if task == 'max':
            # For maximization
            improvement = max(0, current_best_value - self.initial_value) 
            improvement_possible = max(0, self.best_theoretical - self.initial_value)
        else:
            # For minimization (default)
            improvement = max(0, self.initial_value - current_best_value)
            improvement_possible = max(0, self.initial_value - self.best_theoretical)
        
        # Avoid division by zero
        if improvement_possible == 0:
            improvement_ratio = 1.0 if improvement == 0 else 0.0
        else:
            improvement_ratio = improvement / improvement_possible
        
        # Trial efficiency (earlier trials get higher weight)
        trial_efficiency = (self.total_trials - (trial - 1)) / self.total_trials
        
        # Combine improvement ratio and trial efficiency
        PA_GAP_trial = improvement_ratio * trial_efficiency
        
        return PA_GAP_trial
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive PA_GAP results summary.
        
        Returns:
            Dictionary with all PA_GAP metrics and statistics
        """
        return {
            'PA_GAP_mean': self.PA_GAP_value,
            'PA_GAP_final': self.PA_GAP_record[-1] if self.PA_GAP_record else 0.0,
            'PA_GAP_max': max(self.PA_GAP_record) if self.PA_GAP_record else 0.0,
            'PA_GAP_std': np.std(self.PA_GAP_record) if self.PA_GAP_record else 0.0,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'best_theoretical': self.best_theoretical,
            'improvement_achieved': self.improvement_achieved,
            'improvement_possible': self.improvement_possible,
            'improvement_ratio': (self.improvement_achieved / self.improvement_possible 
                                if self.improvement_possible > 0 else 0.0),
            'optimal_trial': self.optimal_trial,
            'total_trials': self.total_trials,
            'PA_GAP_record': self.PA_GAP_record.copy()
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export PA_GAP data to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with trial-by-trial PA_GAP values
        """
        return pd.DataFrame({
            'trial': list(range(1, len(self.PA_GAP_record) + 1)),
            'PA_GAP': self.PA_GAP_record
        })

    def print_summary(self) -> None:
        """Print a formatted summary of PA_GAP results."""
        results = self.get_results_summary()
        
        print("\n" + "="*60)
        print("PA_GAP METRICS SUMMARY")
        print("="*60)
        print(f"Average PA_GAP:           {results['PA_GAP_mean']:.6f}")
        print(f"Final PA_GAP:             {results['PA_GAP_final']:.6f}")
        print(f"Max PA_GAP:               {results['PA_GAP_max']:.6f}")
        print(f"PA_GAP Std Dev:           {results['PA_GAP_std']:.6f}")
        print("-"*60)
        print(f"Initial Value:            {results['initial_value']:.6f}")
        print(f"Final Value:              {results['final_value']:.6f}")
        print(f"Theoretical Best:         {results['best_theoretical']:.6f}")
        print(f"Improvement Achieved:     {results['improvement_achieved']:.6f}")
        print(f"Improvement Possible:     {results['improvement_possible']:.6f}")
        print(f"Improvement Ratio:        {results['improvement_ratio']:.4f}")
        print(f"Optimal Found at Trial:   {results['optimal_trial']}")
        print(f"Total Trials:             {results['total_trials']}")
        print("="*60)


def calculate_theoretical_best(ground_truth_values: Dict[Tuple[str, ...], Any], 
                              task: str = 'min') -> float:
    """
    Calculate theoretical best value from ground truth.
    
    Args:
        ground_truth_values: Dictionary of ground truth values for different intervention sets
        task: 'min' for minimization, 'max' for maximization
        
    Returns:
        Theoretical best value
    """
    if not ground_truth_values:
        raise ValueError("Ground truth values cannot be empty")
    
    # Extract all values from the ground truth dictionary
    all_values = []
    for key, value in ground_truth_values.items():
        if isinstance(value, (list, np.ndarray)):
            all_values.extend(value)
        else:
            all_values.append(value)
    
    if task == 'min':
        return min(all_values)
    else:
        return max(all_values)
