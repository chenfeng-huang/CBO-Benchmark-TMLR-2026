import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any


class GapMetricsCalculator:
    """
    Unified GAP metrics calculator for optimization algorithms.
    
    This class provides robust GAP metric calculations with multiple formulations,
    error handling, validation, and comprehensive result reporting.
    """
    
    def __init__(self):
        self.total_trials = 0
        self.initial_value = 0
        self.final_value = 0
        self.best_theoretical = 0
        self.gap_value = 0
        self.improvement_achieved = 0
        self.improvement_possible = 0
        self.optimal_trial = 0
        self.gap_record = []


    def calculate_gap_metric(self, 
                           outcomes: List[float],
                           best_theoretical: float,
                           task: str = 'max'):

        outcomes_array = np.array(outcomes)
        self.initial_value = outcomes_array[0]
        self.best_theoretical = best_theoretical
        self.total_trials = len(outcomes_array) - 1  # Exclude initial value
        total_gap = []
        
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
            
            # Calculate GAP for the best value found so far
            total_gap.append(self._calculate_gap(current_best, i, task))

        # Basic metrics - use the best value found in all trials
        self.final_value = min(outcomes_array) if task == 'min' else max(outcomes_array)
        
        # Calculate improvements correctly for the task type
        if task == 'min':
            # For minimization: improvement is reduction from initial
            self.improvement_achieved = max(0, self.initial_value - self.final_value)
            self.improvement_possible = max(0, self.initial_value - best_theoretical)
        else:
            # For maximization: improvement is increase from initial  
            self.improvement_achieved = max(0, self.final_value - self.initial_value)
            self.improvement_possible = max(0, best_theoretical - self.initial_value)
    
        # Normalize gap
        self.gap_record = total_gap
        self.gap_value = sum(total_gap) / len(total_gap) if len(total_gap) > 0 else 0.0

        # Find optimal trial (when best value was first reached) 
        self.optimal_trial = len(outcomes_array)  # Default to last trial
        for i, outcome in enumerate(outcomes_array):
            if abs(outcome - self.final_value) < 1e-10:
                self.optimal_trial = i
                break
            
        return
    
    def _calculate_gap(self, current_best_value, trial, task):
        # Calculate improvement correctly for the task type
        if task == 'max':
            # For maximization
            improvement = max(0, current_best_value - self.initial_value) 
            improvement_possible = max(0, self.best_theoretical - self.initial_value)
        else:
            # For minimization (default)
            improvement = max(0, self.initial_value - current_best_value)
            improvement_possible = max(0, self.initial_value - self.best_theoretical)
        
        # Calculate improvement ratio
        if improvement_possible > 1e-10:
            improvement_ratio = improvement / improvement_possible
        else:
            improvement_ratio = 1.0 if improvement > 1e-10 else 0.0
        
        # Trial efficiency: reward finding good solutions early
        # Higher values for earlier trials when improvement is achieved
        if improvement_ratio > 0.0:
            trial_efficiency = max(0.0, 1.0 - (trial - 1) / self.total_trials)
        else:
            trial_efficiency = 0.0

        # Combine improvement ratio and trial efficiency
        gap_trial = improvement_ratio + 0.5 * trial_efficiency  # Weight efficiency less than improvement
        
        # Normalize by maximum possible gap (improvement_ratio=1 + efficiency=1)
        gap_trial = gap_trial / 1.5
        
        return min(1.0, max(0.0, gap_trial))  # Ensure [0,1] range
        
    def get_metrics_dict(self) -> Dict[str, float]:
        """
        Returns a dictionary with all calculated metrics for easy logging.
        """
        return {
            'gap_metric': self.gap_value,
            'improvement_achieved': self.improvement_achieved,
            'improvement_possible': self.improvement_possible,
            'improvement_ratio': self.improvement_achieved / self.improvement_possible if self.improvement_possible > 1e-10 else 0.0,
            'optimal_trial': self.optimal_trial,
            'total_trials': self.total_trials,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'best_theoretical': self.best_theoretical
        } 