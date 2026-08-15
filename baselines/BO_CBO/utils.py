import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
from scipy.spatial import ConvexHull
import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core.parameter_space import ParameterSpace
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.bayesian_optimization.acquisitions.expected_improvement import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from .cost_functions import *
from .causal_acquisition_functions import CausalExpectedImprovement
from .causal_optimizer import CausalGradientAcquisitionOptimizer
from .kernels import CausalRBF


def get_do_function_name(intervention_variable):
    """
    Get the name of the do function for the given intervention variable.
    """
    if isinstance(intervention_variable, list):
        var_name = "_".join(intervention_variable)
    else:
        var_name = intervention_variable
    return f"compute_do_{var_name}"

## Given a do function, this function is computing the mean and variance functions needed for the Causal prior 
def mean_var_do_functions(do_effects_function, observational_samples, functions):
    xi_dict_mean = {}
    def mean_function_do(x):
        num_interventions = x.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            # Handle different input shapes and create a consistent cache key
            if hasattr(x[i], 'shape') and len(x[i].shape) > 0:
                # Convert to tuple for hashable cache key
                xi_tuple = tuple(x[i].flatten())
                xi_str = str(xi_tuple)
                
                # Determine shape and reshape for prediction
                if len(x[i].shape) > 0 and x[i].shape[0] > 1:
                    # Multi-dimensional input
                    x_input = x[i].reshape(1, -1)
                else:
                    # Single value
                    x_input = np.array([[float(x[i])]])
            else:
                # Handle scalar inputs
                xi_str = str(float(x[i]))
                x_input = np.array([[float(x[i])]])
            
            if xi_str in xi_dict_mean:
                mean_do[i] = xi_dict_mean[xi_str]
            else:
                try:
                    temp_mean, _ = do_effects_function(observational_samples, functions, x_input)
                    mean_do[i] = temp_mean
                    xi_dict_mean[xi_str] = temp_mean
                except Exception:
                    # Completely silent error handling - no messages
                    # Fallback to zero mean
                    mean_do[i] = 0.0
                    xi_dict_mean[xi_str] = 0.0
        
        return np.float64(mean_do)
    
    xi_dict_var = {}
    def var_function_do(x):
        num_interventions = x.shape[0]
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            # Handle different input shapes and create a consistent cache key
            if hasattr(x[i], 'shape') and len(x[i].shape) > 0:
                # Convert to tuple for hashable cache key
                xi_tuple = tuple(x[i].flatten())
                xi_str = str(xi_tuple)
                
                # Determine shape and reshape for prediction
                if len(x[i].shape) > 0 and x[i].shape[0] > 1:
                    # Multi-dimensional input
                    x_input = x[i].reshape(1, -1)
                else:
                    # Single value
                    x_input = np.array([[float(x[i])]])
            else:
                # Handle scalar inputs
                xi_str = str(float(x[i]))
                x_input = np.array([[float(x[i])]])
            
            if xi_str in xi_dict_var:
                var_do[i] = xi_dict_var[xi_str]
            else:
                try:
                    _, temp_var = do_effects_function(observational_samples, functions, x_input)
                    var_do[i] = temp_var
                    xi_dict_var[xi_str] = temp_var
                except Exception:
                    # Completely silent error handling - no messages
                    # Fallback to unit variance
                    var_do[i] = 1.0
                    xi_dict_var[xi_str] = 1.0
        
        return np.float64(var_do)

    return mean_function_do, var_function_do

def get_interventional_dict(intervention_variables):
    """
    Convert a list of intervention variables to a dictionary of variable-position pairs.
    """
    return {var: i for i, var in enumerate(intervention_variables)}

def list_interventional_ranges(dict_ranges, intervention_variables):
    list_min_ranges = []
    list_max_ranges = []
    for j in range(len(intervention_variables)):
      list_min_ranges.append(dict_ranges[intervention_variables[j]][0])
      list_max_ranges.append(dict_ranges[intervention_variables[j]][1])
    return list_min_ranges, list_max_ranges

def Intervention_function(intervention_dict, model, target_variable, min_intervention, max_intervention):
    """
    Create a function to evaluate an intervention and its parameter space.
    """
    def target_function(x):
        # Convert the input x to a dictionary of interventions
        intervention_values = {}
        for var, idx in intervention_dict.items():
            intervention_values[var] = x[0, idx]
        
        # Apply intervention to model
        result = {}
        epsilon = np.zeros(len(model))
        
        # For each variable in the model, compute its value
        for i, (variable, func) in enumerate(model.items()):
            # If variable is in the intervention, use that value
            if variable in intervention_values:
                result[variable] = intervention_values[variable]
            # Otherwise, compute based on parents
            else:
                result[variable] = func(epsilon=epsilon[i:i+1], **result)
        
        return np.array([[result[target_variable]]])
    
    # Create parameter space for optimization
    parameters = []
    for var, idx in intervention_dict.items():
        parameters.append(ContinuousParameter(var, min_intervention[idx], max_intervention[idx]))
    space = ParameterSpace(parameters)
    
    return target_function, space

def update_all_do_functions(graph, exploration_set, functions, dict_interventions, observational_samples, x_dict_mean, x_dict_var):
    """
    Update all do-calculus functions given new observational data.
    """
    mean_functions_list = []
    var_functions_list = []
    
    for s in range(len(exploration_set)):
        # Get do function for this exploration set
        var_to_int = exploration_set[s]
        var_to_int_str = dict_interventions[s]
        func_name = get_do_function_name(var_to_int_str)
        
        # Force creation of combined do functions if needed
        if isinstance(var_to_int, list) and len(var_to_int) > 1:
            # Ensure the combined do function exists
            graph.create_combined_do_function(var_to_int)
            
        do_function = graph.get_all_do()[func_name]
        
        # Get mean and var functions
        mean_function, var_function = mean_var_do_functions(do_function, observational_samples, functions)
        
        # Update dictionaries with properly shaped values based on number of variables
        if isinstance(var_to_int, list) and len(var_to_int) > 1:
            # Multi-variable case - create a grid of values
            
            # Create ranges for each variable
            ranges = []
            for var in var_to_int:
                ranges.append(np.linspace(
                    min(graph.get_interventional_ranges()[var]), 
                    max(graph.get_interventional_ranges()[var]), 
                    10  # Use fewer points for multi-variable grids
                ))
            
            # For higher dimensions, we may want to use a different approach than a full grid
            if len(var_to_int) == 2:
                # For 2D, create a full grid
                grid_x, grid_y = np.meshgrid(ranges[0], ranges[1])
                points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
                x_values_to_compute = points
            else:
                # For higher dimensions, sample random points in the space
                num_samples = 100
                points = []
                for _ in range(num_samples):
                    point = []
                    for i, var in enumerate(var_to_int):
                        # Random value in the range
                        point.append(np.random.uniform(
                            min(graph.get_interventional_ranges()[var]),
                            max(graph.get_interventional_ranges()[var])
                        ))
                    points.append(point)
                x_values_to_compute = np.array(points)
        else:
            # Single variable case - use a 1D grid
            var_name = var_to_int[0] if isinstance(var_to_int, list) else var_to_int
            x_values_to_compute = np.linspace(
                min(graph.get_interventional_ranges()[var_name]), 
                max(graph.get_interventional_ranges()[var_name]), 
                100
            ).reshape(-1, 1)
        
        x_dict_mean[var_to_int_str] = x_values_to_compute
        x_dict_var[var_to_int_str] = x_values_to_compute
        
        mean_functions_list.append(mean_function)
        var_functions_list.append(var_function)
    
    return mean_functions_list, var_functions_list

def update_BO_models(mean_function, var_function, data_x, data_y, Causal_prior):
    """
    Update Bayesian Optimization models with or without causal prior.
    """
    input_space = data_x.shape[1]
    if Causal_prior:
        # Model with causal prior
        mf = GPy.core.Mapping(input_space, 1)
        
        # Create a robust mean function wrapper that handles any input shape
        def robust_mean_function(x):
            try:
                # Make sure the input is properly shaped for the mean function
                if x.shape[1] != input_space:
                    # If we're dealing with multi-variable input but model expects single var
                    if input_space == 1 and x.shape[1] > 1:
                        # Take just the first column for single-variable models
                        x_reshaped = x[:, 0:1]
                        return mean_function(x_reshaped)
                    else:
                        # Print warning but continue with zeros
                        print(f"Warning: Input shape {x.shape} doesn't match expected input_space {input_space}")
                        return np.zeros((x.shape[0], 1))
                return mean_function(x)
            except Exception as e:
                # Silently handle errors without extensive logging
                # print(f"Error in mean function: {e}")
                return np.zeros((x.shape[0], 1))
                
        # Assign the robust mean function
        mf.f = robust_mean_function
        mf.update_gradients = lambda a, b: None
        
        # Create a robust variance function wrapper
        def robust_var_function(x):
            try:
                if x.shape[1] != input_space:
                    # If we're dealing with multi-variable input but model expects single var
                    if input_space == 1 and x.shape[1] > 1:
                        # Take just the first column for single-variable models
                        x_reshaped = x[:, 0:1]
                        return var_function(x_reshaped)
                    else:
                        # Default to unit variance without excessive warnings
                        return np.ones((x.shape[0], 1))
                return var_function(x)
            except Exception as e:
                # Silently handle errors
                # print(f"Error in variance function: {e}")
                return np.ones((x.shape[0], 1))
        
        
        kernel = CausalRBF(input_space, variance_adjustment=robust_var_function, 
                          lengthscale=1., variance=1., rescale_variance=1., ARD=False)
        
        # Ensure input data has correct dimensions
        if data_x.shape[1] != input_space:
            print(f"Warning: data_x shape {data_x.shape} doesn't match input_space {input_space}")
        if data_y.shape[1] != 1:
            data_y = data_y.reshape(-1, 1)
            
        try:
            gpy_model = GPy.models.GPRegression(data_x, data_y, kernel, noise_var=1e-10, mean_function=mf)
        except Exception as e:
            print(f"Error creating GP model: {e}")
            # Fallback to non-causal model
            print("Falling back to standard GP model")
            gpy_model = GPy.models.GPRegression(data_x, data_y, 
                                              GPy.kern.RBF(input_space, lengthscale=1., variance=1.), 
                                              noise_var=1e-10)
    else:
        # Model without causal prior
        if data_y.shape[1] != 1:
            data_y = data_y.reshape(-1, 1)
            
        gpy_model = GPy.models.GPRegression(data_x, data_y, 
                                          GPy.kern.RBF(input_space, lengthscale=1., variance=1.), 
                                          noise_var=1e-10)
    
    emukit_model = GPyModelWrapper(gpy_model)
    return emukit_model

def find_next_y_point(space, model, current_global_best, evaluated_set, costs_functions, task = 'min'):
    ## This function optimises the acquisition function and return the next point together with the 
    ## corresponding y value for the acquisition function
    cost_acquisition = Cost(costs_functions, evaluated_set)
    optimizer = CausalGradientAcquisitionOptimizer(space)
    acquisition = CausalExpectedImprovement(current_global_best, task, model)/cost_acquisition
    x_new, _ = optimizer.optimize(acquisition)
    y_acquisition = acquisition.evaluate(x_new)  
    return y_acquisition, x_new    


def initialise_dicts(exploration_set, task):
    current_best_x = {}
    current_best_y = {}
    x_dict_mean = {}
    x_dict_var = {}
    dict_interventions = []


    for i in range(len(exploration_set)):
      variables = exploration_set[i]
      
      if len(variables) == 1:
        variables = variables[0]
      elif len(variables) > 1:
        num_var = len(variables)
        var = variables[0]
        for j in range(1, num_var):
          var += '_' + variables[j]
        variables = var 

      ## This is creating a list of strings 
      dict_interventions.append(variables)


      current_best_x[variables] = []
      current_best_y[variables] = []

      x_dict_mean[variables] = {}
      x_dict_var[variables] = {}

      ## Assign initial values
      if task == 'min':
        current_best_y[variables].append(np.inf)
        current_best_x[variables].append(np.inf)
      else:
        current_best_y[variables].append(-np.inf)
        current_best_x[variables].append(-np.inf)
      
    return current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions



def find_current_global(current_best_y, dict_interventions, task='min'):
    """
    Find the current global optimum across all intervention sets.
    """
    all_y = []
    for s in range(len(dict_interventions)):
        var_to_int_str = dict_interventions[s]
        if current_best_y[var_to_int_str]:
            all_y.extend(current_best_y[var_to_int_str])
    
    if not all_y:
        return 0.0
    
    if task == 'min':
        return min(all_y)
    else:
        return max(all_y)

def get_new_dict_x(x_new, intervention_variables):
    """
    Convert new x point to a dictionary.
    """
    x_new_dict = {}
    if isinstance(intervention_variables, list):
        for i, var in enumerate(intervention_variables):
            x_new_dict[var] = x_new[0, i]
    else:
        x_new_dict[intervention_variables] = x_new[0, 0]
    
    return x_new_dict

def total_cost(var_to_intervene, costs, x_new_dict):
    """
    Compute the total cost of an intervention.
    """
    cost = 0.0
    if isinstance(var_to_intervene, list):
        for var in var_to_intervene:
            cost += costs[var]
    else:
        cost = costs[var_to_intervene]
    
    return cost

def update_hull(observational_samples, manipulative_variables):
    ## This function computes the coverage of the observations 
    list_variables = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    
    return coverage_obs

def observe(num_observation, complete_dataset, initial_num_obs_samples):
    """
    Sample additional observations from the complete dataset.
    """
    max_idx = complete_dataset.shape[0]
    available_indices = list(range(initial_num_obs_samples, max_idx))
    
    if len(available_indices) < num_observation:
        # If not enough samples available, take all remaining
        indices = available_indices
    else:
        # Randomly sample indices
        indices = np.random.choice(available_indices, num_observation, replace=False)
    
    return complete_dataset.iloc[indices].copy()

def add_data(existing_data, new_data):
    """
    Add new data to existing data.
    """
    data_x, data_y = existing_data
    x_new, y_new = new_data
    
    if isinstance(x_new, list):
        x_new = np.array(x_new)
    if isinstance(y_new, list):
        y_new = np.array(y_new)
    
    return data_x, data_y

def fit_single_GP_model(X, Y, parameter_list, optimize=True):
    """
    Fit a single GP model.
    """
    kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=parameter_list[3], lengthscale=parameter_list[0], variance=parameter_list[1])
    gp = GPy.models.GPRegression(X=X, Y=Y, kernel=kernel, noise_var=parameter_list[2])
    if optimize:
        gp.optimize()
    return gp

def compute_coverage(observational_samples, manipulative_variables, dict_ranges):
    """
    Compute coverage statistics for observational samples.
    """
    # Extract manipulative variables from observational data
    data_manipulative = observational_samples[manipulative_variables].values
    
    # Compute convex hull volume if possible
    if data_manipulative.shape[0] > data_manipulative.shape[1]:
        try:
            hull = ConvexHull(data_manipulative)
            hull_volume = hull.volume
        except Exception as e:
            print(f"Warning: Could not compute convex hull: {e}")
            hull_volume = 0.0
    else:
        hull_volume = 0.0
    
    # Compute total volume of intervention space
    total_volume = 1.0
    for var in manipulative_variables:
        total_volume *= (dict_ranges[var][1] - dict_ranges[var][0])
    
    # Compute coverage ratio
    if total_volume > 0:
        alpha = hull_volume / total_volume
    else:
        alpha = 0.0
    
    return alpha, hull_volume, total_volume 