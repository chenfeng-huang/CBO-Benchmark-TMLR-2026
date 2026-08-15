import time
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools

from baselines.BO_CBO.utils import *

def CBO(num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list, best_intervention_value, opt_y, 
        best_variable, dict_ranges, functions, observational_samples, coverage_total, graph, 
        num_additional_observations, costs, full_observational_samples, task='min', max_N=200, 
        initial_num_obs_samples=100, num_interventions=10, Causal_prior=False, scaler_x=None, csv_log_file=None):
    """
    Causal Bayesian Optimization algorithm.
    
    Args:
        num_trials: Number of optimization trials
        exploration_set: List of sets of variables to explore (POMIS)
        manipulative_variables: List of manipulative variables
        data_x_list: List of initial interventional data x for each exploration set
        data_y_list: List of initial interventional data y for each exploration set
        best_intervention_value: Initial best intervention value
        opt_y: Initial best target value
        best_variable: Initial best variable
        dict_ranges: Dictionary of intervention ranges
        functions: Dictionary of fitted functions
        observational_samples: DataFrame of observational data
        coverage_total: Total coverage of observational data
        graph: Graph structure object implementing GraphStructure interface
        num_additional_observations: Number of additional observations to collect
        costs: Dictionary of costs for each intervention
        full_observational_samples: Complete observational dataset for sampling
        task: Optimization task ('min' or 'max')
        max_N: Maximum number of observations
        initial_num_obs_samples: Initial number of observational samples
        num_interventions: Number of interventions
        Causal_prior: Whether to use causal prior in GP models
        scaler_x: Scaler for normalizing input data
        
    Returns:
        Tuple of (current_cost, current_best_x, current_best_y, global_opt, observed, total_time)
    """
    current_cost = []
    global_opt = []
    current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set, task)
    
    
    # Handle initial best intervention value properly for multi-variable interventions
    # Check if best_variable represents a multi-variable intervention by looking for underscore
    if '_' in best_variable:
        # Multi-variable intervention - best_intervention_value should be a list/array
        if isinstance(best_intervention_value, (list, np.ndarray)) and len(best_intervention_value) > 1:
            current_best_x[best_variable].append(best_intervention_value.tolist() if isinstance(best_intervention_value, np.ndarray) else best_intervention_value)
        else:
            # Single value for what should be multi-variable - create list
            current_best_x[best_variable].append([best_intervention_value])
    else:
        # Single variable intervention - store as clean scalar
        if isinstance(best_intervention_value, (list, np.ndarray)):
            if hasattr(best_intervention_value, 'item'):
                current_best_x[best_variable].append(float(best_intervention_value.item()))
            else:
                current_best_x[best_variable].append(float(best_intervention_value[0]))
        else:
            current_best_x[best_variable].append(float(best_intervention_value))
    
    current_best_y[best_variable].append(opt_y)
    global_opt.append(opt_y)
    current_cost.append(0.)

    observed = 0
    cumulative_cost = 0.
            
    target_function_list = [None] * len(exploration_set)
    space_list = [None] * len(exploration_set)
    model_list = [None] * len(exploration_set)
    type_trial = []

    # Define intervention functions for each exploration set
    for s in range(len(exploration_set)):
        target_function_list[s], space_list[s] = graph.intervention_function(get_interventional_dict(exploration_set[s]))

    if csv_log_file:
        csv_columns = ['trial_number', 'current_optimal'] + manipulative_variables
        df_header = pd.DataFrame(columns=csv_columns)
        df_header.to_csv(csv_log_file, index=False)

    start_time = time.perf_counter()
    for i in range(num_trials+1):
        print(f'Optimization step {i}')
        # Decide to observe or intervene and then recompute the obs coverage
        coverage_obs = update_hull(observational_samples, manipulative_variables)
        rescale = observational_samples.shape[0] / max_N
        epsilon_coverage = (coverage_obs / coverage_total) / rescale
        uniform = np.random.uniform(0., 1.)
        # At least observe and intervene once
        if i == 0:
            uniform = 0.
        if i == 1:
            uniform = 1.

        if i < epsilon_coverage:
            # Collect observations
            observed += 1
            type_trial.append(0)
            
            new_observational_samples = observe(
                num_observation=num_additional_observations, 
                complete_dataset=full_observational_samples, 
                initial_num_obs_samples=initial_num_obs_samples
            )

            observational_samples = pd.concat([observational_samples, new_observational_samples], ignore_index=True)
            
            # Refit the models for the conditional distributions 
            functions = graph.refit_models(observational_samples)
            
            # Update the mean functions and var functions given the current set of observational data
            mean_functions_list, var_functions_list = update_all_do_functions(
                graph, exploration_set, functions, dict_interventions, 
                observational_samples, x_dict_mean, x_dict_var
            )
            
            global_opt.append(global_opt[i])
            current_cost.append(current_cost[i])

            print(f'Current best Y {global_opt[i]}')
            
            # Log to CSV if file path provided (observation step)
            if csv_log_file:
                csv_row = {'trial_number': i, 'current_optimal': global_opt[i]}
                for var in manipulative_variables:
                    csv_row[var] = None
                df_row = pd.DataFrame([csv_row])
                df_row.to_csv(csv_log_file, mode='a', header=False, index=False)
        else:
            # Perform intervention
            type_trial.append(1)
   
            y_acquisition_list = [None] * len(exploration_set)
            x_new_list = [None] * len(exploration_set)
            
            # This is the global opt from previous iteration
            current_global = find_current_global(current_best_y, dict_interventions, task)

            # Update models based on previous trial type
            if type_trial[i-1] == 0:
                # If previous trial was observation, update all models
                for s in range(len(exploration_set)):
                    model_list[s] = update_BO_models(
                        mean_functions_list[s], 
                        var_functions_list[s], 
                        data_x_list[s], 
                        data_y_list[s], 
                        Causal_prior
                    )
            else:
                # If previous trial was intervention, update only the model that was used
                model_list[index] = update_BO_models(
                    mean_functions_list[index], 
                    var_functions_list[index], 
                    data_x_list[index], 
                    data_y_list[index], 
                    Causal_prior
                )
            
            # Compute acquisition function for each intervention set
            for s in range(len(exploration_set)): 
                
                y_acquisition_list[s], x_new_list[s] = find_next_y_point(
                    space_list[s], 
                    model_list[s], 
                    current_global, 
                    exploration_set[s], 
                    costs, 
                    task=task
                )
            print(f"Iteration {i}: Acquisition values:")
            for s in range(len(exploration_set)):
                # Handle case where y_acquisition_list[s] might be a numpy array
                if hasattr(y_acquisition_list[s], 'item'):
                    acq_value = y_acquisition_list[s].item()
                elif isinstance(y_acquisition_list[s], np.ndarray):
                    acq_value = float(y_acquisition_list[s].flat[0])
                else:
                    acq_value = float(y_acquisition_list[s])
                print(f"  {exploration_set[s]}: {acq_value:.6f}")
        
            # Selecting the variable to intervene based on acquisition function values
            index = np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]
            var_to_intervene = exploration_set[index]
            y_new = target_function_list[index](x_new_list[index])
            print(y_new)
            # Append the new data - ensure dimensions match
            # Make sure x_new is 2D
            if x_new_list[index].ndim == 1:
                x_new_list[index] = x_new_list[index].reshape(1, -1)
            
            # Make sure y_new is 2D (not 3D)
            if y_new.ndim == 3:
                y_new = y_new.reshape(y_new.shape[0], y_new.shape[2])
            elif y_new.ndim == 1:
                y_new = y_new.reshape(1, -1)
            
            # Handle dimension matching for x data
            if data_x_list[index].shape[1] != x_new_list[index].shape[1]:
                expanded_data_x = np.zeros((data_x_list[index].shape[0], x_new_list[index].shape[1]))
                for i in range(min(data_x_list[index].shape[1], x_new_list[index].shape[1])):
                    expanded_data_x[:, i] = data_x_list[index][:, i]
                data_x_list[index] = expanded_data_x
            
            # Handle dimension matching for y data
            if data_y_list[index].ndim != y_new.ndim:
                if data_y_list[index].ndim == 1:
                    data_y_list[index] = data_y_list[index].reshape(-1, 1)
                elif y_new.ndim == 1:
                    y_new = y_new.reshape(1, -1)
            
            data_x_list[index] = np.vstack((data_x_list[index], x_new_list[index]))
            data_y_list[index] = np.vstack((data_y_list[index], y_new))
            
            data_x, data_y_x = add_data(
                [data_x_list[index], data_y_list[index]], 
                [x_new_list[index], y_new]
            )
            model_list[index].set_data(data_x, data_y_x)
        
            # Compute cost - need to use original space for cost calculation if scaler is used
            if scaler_x is not None:
                x_new_original = scaler_x.inverse_transform(x_new_list[index])
                x_new_dict = get_new_dict_x(x_new_original, dict_interventions[index])
            else:
                x_new_dict = get_new_dict_x(x_new_list[index], dict_interventions[index])
                
            trial_cost = total_cost(var_to_intervene, costs, x_new_dict)
            cumulative_cost += trial_cost
            var_to_intervene = dict_interventions[index]
            current_cost.append(cumulative_cost)

            # Convert var_to_intervene to a string key for the dictionaries
            if isinstance(var_to_intervene, list):
                var_to_intervene_str = '_'.join(var_to_intervene)
            else:
                var_to_intervene_str = var_to_intervene

            # Update the dict storing the current optimal solution
            # Check if this is a multi-variable intervention based on original exploration_set
            original_var_set = exploration_set[index]
            is_multi_variable = isinstance(original_var_set, list) and len(original_var_set) > 1
            
            if is_multi_variable:
                # For multi-variable intervention, append the full array as a list
                current_best_x[var_to_intervene_str].append(x_new_list[index][0].tolist())
            else:
                # For single variable, store as clean scalar
                if isinstance(x_new_list[index][0], np.ndarray):
                    current_best_x[var_to_intervene_str].append(float(x_new_list[index][0].item()))
                else:
                    current_best_x[var_to_intervene_str].append(float(x_new_list[index][0]))
            
            if y_new.ndim > 2:
                outcome_value = y_new[0, 0]
                current_best_y[var_to_intervene_str].append(outcome_value)
            elif y_new.ndim == 2:
                outcome_value = y_new[0, 0]
                current_best_y[var_to_intervene_str].append(outcome_value)
            else:
                outcome_value = y_new[0]
                current_best_y[var_to_intervene_str].append(outcome_value)
            
            current_global = find_current_global(current_best_y, dict_interventions, task)
            global_opt.append(current_global)
            print(f'Current best Y: {current_global:.4f}')
            
            # Display intervention values in clean format
            if is_multi_variable:
                intervention_values = x_new_list[index][0].tolist()
            else:
                if isinstance(x_new_list[index][0], np.ndarray):
                    intervention_values = float(x_new_list[index][0].item())
                else:
                    intervention_values = float(x_new_list[index][0])
            
            if csv_log_file:
                csv_row = {'trial_number': i, 'current_optimal': current_global}
                for var in manipulative_variables:
                    csv_row[var] = None
                
                # Populate the specific variables that were intervened upon
                intervention_vars = exploration_set[index]
                if isinstance(intervention_vars, list):
                    # Multi-variable intervention
                    if isinstance(intervention_values, list):
                        for j, var in enumerate(intervention_vars):
                            if j < len(intervention_values):
                                csv_row[var] = intervention_values[j]
                    else:
                        # Single value for multiple variables (shouldn't happen but handle gracefully)
                        csv_row[intervention_vars[0]] = intervention_values
                else:
                    # Single variable intervention
                    csv_row[intervention_vars] = intervention_values
                
                df_row = pd.DataFrame([csv_row])
                df_row.to_csv(csv_log_file, mode='a', header=False, index=False)
            
  
            
            print(f"Iteration {i+1}, Selected intervention: {var_to_intervene}, Values: {intervention_values}")
        
            model_list[index].optimize()

    total_time = time.perf_counter() - start_time

    # Find the best intervention values for the final best value
    best_global_value = find_current_global(current_best_y, dict_interventions, task)
    best_intervention_set = None
    best_intervention_value = None
    
    # Find which intervention set gave the best result
    for s in range(len(dict_interventions)):
        var_str = dict_interventions[s]
        if var_str in current_best_y and len(current_best_y[var_str]) > 0:
            for i, value in enumerate(current_best_y[var_str]):
                if (task == 'min' and value == best_global_value) or (task == 'max' and value == best_global_value):
                    best_intervention_set = var_str
                    # Get the corresponding x value and clean it up
                    if i < len(current_best_x[var_str]):
                        raw_value = current_best_x[var_str][i]
                        # Clean up the format for display
                        if '_' in var_str:
                            # Multi-variable intervention
                            if isinstance(raw_value, list):
                                # Clean up any nested arrays in the list
                                cleaned_value = []
                                for val in raw_value:
                                    if isinstance(val, np.ndarray):
                                        if val.size == 1:
                                            cleaned_value.append(float(val.item()))
                                        else:
                                            cleaned_value.extend(val.tolist())
                                    else:
                                        cleaned_value.append(float(val))
                                best_intervention_value = cleaned_value
                            else:
                                best_intervention_value = [raw_value]
                        else:
                            # Single variable intervention - ensure clean scalar
                            def extract_scalar(value):
                                """Recursively extract scalar from nested structures"""
                                if isinstance(value, (list, tuple)):
                                    if len(value) == 1:
                                        return extract_scalar(value[0])
                                    else:
                                        # If multiple values, take the first one
                                        return extract_scalar(value[0])
                                elif isinstance(value, np.ndarray):
                                    if value.size == 1:
                                        return float(value.item())
                                    else:
                                        return float(value.flat[0])
                                else:
                                    return float(value)
                            
                            best_intervention_value = extract_scalar(raw_value)
                    break
            if best_intervention_set is not None:
                break
    
    # Create a cleaner result for the best intervention
    if best_intervention_set is not None:
        # Find the original variable list by looking up in the exploration_set and dict_interventions mapping
        intervention_variables = None
        for s in range(len(dict_interventions)):
            if dict_interventions[s] == best_intervention_set:
                intervention_variables = exploration_set[s]
                break
        
        # If we found the mapping, use it; otherwise fall back to splitting on underscores
        if intervention_variables is not None:
            if isinstance(intervention_variables, list):
                intervention_set = intervention_variables
            else:
                intervention_set = [intervention_variables]
        else:
            # Fallback: try to split on underscores (less reliable for variables with underscores)
            if '_' in best_intervention_set:
                intervention_set = best_intervention_set.split('_')
            else:
                intervention_set = [best_intervention_set]
        
        # Create a structured result that clearly shows the intervention set and values
        best_intervention = {
            'intervention_set': intervention_set,
            'intervention_values': best_intervention_value,
            'original_key': best_intervention_set  # Keep original for compatibility
        }
    else:
        best_intervention = current_best_x

    return (current_cost, best_intervention, current_best_y, global_opt, observed, total_time) 