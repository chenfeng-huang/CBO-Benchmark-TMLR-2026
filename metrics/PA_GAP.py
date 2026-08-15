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
        self.total_trials = 0
        self.initial_value = 0
        self.final_value = 0
        self.best_theoretical = best_theoretical
        self.PA_GAP_value = 0
        self.PA_GAP_std = 0
        self.improvement_achieved = 0
        self.improvement_possible = 0
        self.optimal_trial = 0
        self.PA_GAP_record = []


    def calculate_PA_GAP(self, 
                           outcomes: List[float],
                           task: str = 'min'):

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
        self.PA_GAP_std = np.std(total_PA_GAP) if len(total_PA_GAP) > 0 else 0.0
        # Find optimal trial (when best value was first reached) 
        self.optimal_trial = len(outcomes_array)  # Default to last trial
        for i, outcome in enumerate(outcomes_array):
            if abs(outcome - self.final_value) < 1e-10:
                self.optimal_trial = i
                break
            
        return

        
    
    def _calculate_PA_GAP(self, current_best_value, trial, task):
        # Calculate improvement correctly for the task type
        if task == 'max':
            # For maximization
            improvement = max(0, current_best_value - self.initial_value)
            improvement_possible = max(0, self.best_theoretical - self.initial_value)
        else:
            # For minimization (default)
            improvement = max(0, self.initial_value - current_best_value)
            improvement_possible = max(0, self.initial_value - self.best_theoretical)

        # Guard against the (improvement >= improvement_possible) edge case that
        # arises when the noisy observed-best surpasses the recorded "theoretical
        # best" (often an expected-value optimum). Treat such runs as having
        # captured the full possible improvement at this trial.
        if improvement_possible <= 0:
            improvement_ratio = 1.0 if improvement > 0 else 0.0
        else:
            improvement_ratio = min(improvement / improvement_possible, 1.0)

        if self.total_trials > 0:
            trial_efficiency = (self.total_trials - (trial - 1)) / self.total_trials
        else:
            trial_efficiency = 0.0

        # Combine improvement ratio and trial efficiency
        PA_GAP_trial = improvement_ratio * trial_efficiency  # Weight efficiency less than improvement
        
  
        
        return PA_GAP_trial
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export PA_GAP data to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with trial-by-trial PA_GAP
        """
        return pd.DataFrame({
            'trial': list(range(1, len(self.PA_GAP_record) + 1)),
            'PA_GAP': self.PA_GAP_record
        })

        
      
   