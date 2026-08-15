import time
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools

import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer

from baselines.BO_CBO.utils import *
from baselines.BO_CBO.kernels import CausalRBF

def NonCausal_BO(num_trials, graph, dict_ranges, interventional_data_x, interventional_data_y, costs, 
            observational_samples, functions, initial_intervention_value, initial_y, intervention_variables, 
            Causal_prior=False, scaler_x=None, task='min', csv_log_file=None):
    """
    Standard Bayesian Optimization algorithm with optional causal prior.
    
    Args:
        num_trials: Number of optimization trials
        graph: Graph structure object implementing GraphStructure interface
        dict_ranges: Dictionary of intervention variable ranges
        interventional_data_x: Initial interventional data (features)
        interventional_data_y: Initial interventional data (target)
        costs: Dictionary of costs for each intervention variable
        observational_samples: DataFrame of observational data
        functions: Dictionary of fitted functions from the graph
        initial_intervention_value: Initial best intervention value
        initial_y: Initial best target value
        intervention_variables: List of variables to intervene on
        Causal_prior: Whether to use causal prior in the GP model
        scaler_x: Scaler for normalizing input data
        task: 'min' for minimization or 'max' for maximization
        csv_log_file: Optional path to CSV file for logging trial progress
        
    Returns:
        Tuple of (current_cost, current_best_x, current_best_y, total_time)
    """
    function_name = get_do_function_name(intervention_variables)
    do_function = graph.get_all_do()[function_name]

    input_space = len(intervention_variables)

    current_best_x = np.zeros((num_trials + 2, input_space))
    current_best_y = np.zeros((num_trials + 2, 1))
    current_cost = np.zeros((num_trials + 2, 1))
    
    global_opt = []

    mean_function_do, var_function_do = mean_var_do_functions(do_function, observational_samples, functions)

    data_x = interventional_data_x.copy()
    data_y = interventional_data_y.copy()
    
    # For maximization, negate the y values to convert to minimization
    if task == 'max':
        data_y = -data_y
        initial_y = -initial_y

    current_cost[0] = 0.
    current_best_y[0] = initial_y
    current_best_x[0] = initial_intervention_value
    cumulative_cost = 0.
    # global_opt stores true-scale values; initial_y was negated above for max tasks
    global_opt.append(-initial_y if task == 'max' else initial_y)
    target_variable = graph.target

    target_function, space_parameters = graph.intervention_function(get_interventional_dict(intervention_variables))

    if Causal_prior == False:
        # Define the model without Causal prior
        gpy_model = GPy.models.GPRegression(data_x, data_y, GPy.kern.RBF(input_space, lengthscale=1., variance=1.), noise_var=1e-10)
        emukit_model = GPyModelWrapper(gpy_model)
    else:
        # Define the model with Causal prior
        mf = GPy.core.Mapping(input_space, 1)
        if task == 'max':
            # For maximization, negate the mean function
            mf.f = lambda x: -mean_function_do(x)
        else:
            mf.f = lambda x: mean_function_do(x)
        mf.update_gradients = lambda a, b: None
        kernel = CausalRBF(input_space, variance_adjustment=var_function_do, lengthscale=1., variance=1., rescale_variance=1., ARD=False)
        gpy_model = GPy.models.GPRegression(data_x, data_y, kernel, noise_var=1e-10, mean_function=mf)
        emukit_model = GPyModelWrapper(gpy_model)

    if csv_log_file:
        trial_data = []
        df_header = pd.DataFrame(columns=['trial_number', 'current_optimal'])
        df_header.to_csv(csv_log_file, index=False)

    start_time = time.perf_counter()
    for j in range(num_trials):
        print('Iteration', j)
        emukit_model.optimize()
        
        # Always use standard ExpectedImprovement (which is for minimization)
        acquisition = ExpectedImprovement(emukit_model)
        
        optimizer = GradientAcquisitionOptimizer(space_parameters)
        x_new, _ = optimizer.optimize(acquisition)
        y_new = target_function(x_new)
        
        # Reduce y_new shape from (1, 1, 1) to (1, 1)
        if y_new.ndim > 2:
            y_new = y_new.reshape(-1, 1)
            
        # For maximization, negate the new y value
        if task == 'max':
            y_new = -y_new
            
        data_x = np.append(data_x, x_new, axis=0)
        data_y = np.append(data_y, y_new, axis=0)
        emukit_model.set_data(data_x, data_y)

        # Compute cost - need to use original space for cost calculation if scaler is used
        if scaler_x is not None:
            x_new_original = scaler_x.inverse_transform(x_new)
            x_new_dict = get_new_dict_x(x_new_original, intervention_variables)
        else:
            x_new_dict = get_new_dict_x(x_new, intervention_variables)
            
        cumulative_cost += total_cost(intervention_variables, costs, x_new_dict)
        current_cost[j + 1] = cumulative_cost
        
        results = np.concatenate((emukit_model.X, emukit_model.Y), axis=1)
        
        # Find the best value (minimum of potentially negated values)
        best_y_idx = np.argmin(results[:, input_space])
        best_y_value = results[best_y_idx, input_space]
        
        # Store the best y value (un-negate if maximization)
        if task == 'max':
            outcome_value = -best_y_value
            current_best_y[j + 1] = outcome_value
        else:
            outcome_value = best_y_value
            current_best_y[j + 1] = outcome_value
            
        best_x = results[best_y_idx, :input_space]
        
        # Store optimum in scaled space
        current_best_x[j + 1] = best_x
        
        trial_cost = current_cost[j + 1] - current_cost[j]
        intervention_set_str = '_'.join(intervention_variables) if len(intervention_variables) > 1 else intervention_variables[0]
        

        # Print the actual best value (not negated)
        if task == 'max':
            print(f'Current best Y (max): {-best_y_value}')
            current_optimal = -best_y_value
        else:
            print(f'Current best Y (min): {best_y_value}')
            current_optimal = best_y_value
        global_opt.append(current_optimal)
        if csv_log_file:
            trial_data.append({'trial_number': j, 'current_optimal': current_optimal})
            df_row = pd.DataFrame([{'trial_number': j, 'current_optimal': current_optimal}])
            df_row.to_csv(csv_log_file, mode='a', header=False, index=False)

    total_time = time.perf_counter() - start_time

    return (current_cost, current_best_x, current_best_y, total_time, global_opt) 