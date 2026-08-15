import numpy as np
from typing import Dict, List, Optional, Any
from .PA_GAP import PA_GAP


class GAP:
    """
    GAP metric for optimization algorithms.
    
    GAP only considers the final result and focuses on:
    - The trial when the optimal was found
    - The optimal value achieved
    """
    
    def __init__(self, best_theoretical: float):
        self.optimal_trial = 0
        self.optimal_value = 0.0
        self.best_theoretical = best_theoretical
        self.total_trials = 0
        self.initial_value = 0.0
        self.GAP_value = 0.0
        
    def calculate_GAP(self, outcomes: List[float], task: str = 'min'):
        """
        Calculate GAP metric based on final optimal result.
        
        Args:
            outcomes: List of best-so-far values from all trials
            task: 'min' for minimization, 'max' for maximization
        """
        outcomes_array = np.array(outcomes)
        self.total_trials = len(outcomes_array) - 1
        self.initial_value = outcomes_array[0]
        
        if outcomes_array[-1] == outcomes_array[0]:
            self.optimal_trial = self.total_trials
            self.optimal_value = outcomes_array[0]
        # Find the optimal value achieved
        elif task == 'min':
            self.optimal_value = np.min(outcomes_array)
            # Find when this optimal was first achieved
            self.optimal_trial = np.argmin(outcomes_array)  
        else:
            self.optimal_value = np.max(outcomes_array)
            # Find when this optimal was first achieved
            self.optimal_trial = np.argmax(outcomes_array) 
        
        # Calculate GAP based on final optimal result and efficiency
        self._calculate_gap_value(task)
    
    def _calculate_gap_value(self, task: str):
        """
        Calculate the GAP value considering:
        1. Quality of final result (how close to theoretical best)
        2. Efficiency (how early the optimal was found)
        """
        # Calculate improvement achieved vs possible
        if task == 'min':
            improvement_achieved = max(0, self.initial_value - self.optimal_value)
            improvement_possible = max(0, self.initial_value - self.best_theoretical)
        else:
            improvement_achieved = max(0, self.optimal_value - self.initial_value)
            improvement_possible = max(0, self.best_theoretical - self.initial_value)

        # Guard against the (achieved >= possible) edge case that arises when the
        # noisy observed-best surpasses the recorded "theoretical best" (which is
        # often the expected-value optimum, e.g. protein's high-noise Erk).
        # Treat such runs as having captured the full possible improvement.
        if improvement_possible <= 0:
            improvement_ratio = 1.0 if improvement_achieved > 0 else 0.0
        else:
            improvement_ratio = min(improvement_achieved / improvement_possible, 1.0)

        if self.total_trials > 0:
            efficiency_ratio = (self.total_trials - self.optimal_trial) / self.total_trials
        else:
            efficiency_ratio = 0.0

        if self.total_trials > 0:
            normalizer = 1 + (self.total_trials - 1) / self.total_trials
        else:
            normalizer = 1.0
        self.GAP_value = (improvement_ratio + efficiency_ratio) / normalizer
    


    
    


