## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns


from emukit.core.acquisition import Acquisition

class Cost(Acquisition):
    def __init__(self, costs_functions, evaluated_set):
        self.costs_functions = costs_functions
        self.evaluated_set = evaluated_set

       

    def evaluate(self, x):
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        num_vars = len(self.evaluated_set)
        # Initialize cost vector
        cost = np.zeros((x.shape[0], 1))
        for i in range(num_vars):
            var_name = self.evaluated_set[i]
            var_cost_fn = self.costs_functions[var_name]
            if callable(var_cost_fn):
                # Select appropriate column for this variable if available
                if x.shape[1] > i:
                    x_col = x[:, i]
                else:
                    # Fallback to the first column
                    x_col = x[:, 0]
                contrib = var_cost_fn(x_col)
                contrib = np.asarray(contrib)
                if contrib.ndim == 1:
                    contrib = contrib.reshape(-1, 1)
                elif contrib.ndim == 0:
                    contrib = np.ones((x.shape[0], 1)) * float(contrib)
            else:
                # Constant cost
                contrib = np.ones((x.shape[0], 1)) * float(var_cost_fn)
            cost += contrib
        return cost
    
    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)



def total_cost(intervention_variables, costs, x_new_dict):
  total_cost = 0.
  for i in range(len(intervention_variables)):
    total_cost += costs[intervention_variables[i]](x_new_dict[intervention_variables[i]])
  return total_cost
