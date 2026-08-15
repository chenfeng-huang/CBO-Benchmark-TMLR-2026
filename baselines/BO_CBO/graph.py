import numpy as np
import pandas as pd
from collections import OrderedDict
import sys
import os
import inspect
from scipy.optimize import curve_fit
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
import itertools
import json
import math
   

from .utils import fit_single_GP_model, get_do_function_name
from emukit.core.parameter_space import ParameterSpace
from emukit.core.continuous_parameter import ContinuousParameter

class DiscoveredGraph():
    """
    A graph class that converts discovered causal structures to a format compatible with BO/CBO.
    This bridges the gap between causal discovery and optimization algorithms.
    """
    
    def __init__(self, model_results, sem_equations, data, target, config, feature_params, generate_sem=True):
        """
        Initialize with results from causal discovery and the dataset.
        
        Args:
            model_results: Results from causal_discovery's run_icalingam_analysis
            data: DataFrame containing the data used in causal discovery
            target: The target variable (default: 'Y')
            generate_sem: Whether to generate SEM functions (default: True)
        """
        self.model_results = model_results
        self.data = data
        self.node_names = model_results['node_names']
        self.adjacency_matrix = model_results['adjacency_matrix']
        self.causal_order = model_results['causal_order']
        self.target = target
        self.config = config
        self.feature_params = feature_params
        # Store data as numpy arrays
        self.data_arrays = {}
        for col in data.columns:
            self.data_arrays[col] = np.asarray(data[col])[:, np.newaxis]
        
  
        self.sem_functions = self._extract_sem_functions(sem_equations)
        
        # Generate do_functions
        self.do_functions = self._generate_do_functions()
    
    def _extract_sem_functions(self, sem_data):
        """
        Generate simple SEM functions from the discovered causal structure.
        Uses linear relationships based on either the adjacency matrix weights or a JSON file.
        
        Args:
            sem_data (dict): Dictionary containing SEM equations data from JSON.
        """

        sem_functions = {}
        latent_variables = set()

        # Create functions from JSON data
        for node, node_info in sem_data["variables"].items():
            node_type = node_info["type"]
            intercept = node_info["intercept"]
            relationship_type = node_info.get("relationship_type", "linear")
            relationship_params = node_info.get("relationship_params", {})
            latent_variables.update(relationship_params.get("latent_variables", []))

            if node_type == "exogenous":
                if relationship_type == "normal":
                    def create_normal_function(mean_v, std_v):
                        def node_function(epsilon=0, **kwargs):
                            return np.random.normal(mean_v, std_v)
                        return node_function

                    sem_functions[node] = create_normal_function(
                        intercept, relationship_params.get("noise_std", 1.0)
                    )
                else:
                    min_val = relationship_params.get("min_val", intercept - 10)
                    max_val = relationship_params.get("max_val", intercept + 10)
                    def create_uniform_function(min_v, max_v):
                        def node_function(epsilon=0, **kwargs):
                            return np.random.uniform(min_v, max_v) + epsilon
                        return node_function

                    sem_functions[node] = create_uniform_function(min_val, max_val)
            
            else:  # endogenous
                dependencies = node_info["dependencies"]
                coefficients = node_info["coefficients"]
                
                if relationship_type == "linear":
                    def create_linear_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            result = const_intercept
                            for parent in deps:
                                if parent in kwargs:
                                    result += coefs[parent] * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_linear_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "sigmoid":
                    def create_sigmoid_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            linear_result = const_intercept
                            for parent in deps:
                                if parent in kwargs:
                                    linear_result += coefs[parent] * kwargs[parent]
                            
                            sigmoid_result = 1.0 / (1.0 + math.exp(-linear_result))
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return sigmoid_result + noise
                        return node_function
                    
                    sem_functions[node] = create_sigmoid_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "binary_sigmoid":
                    def create_binary_sigmoid_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            linear_result = const_intercept
                            for parent in deps:
                                if parent in kwargs:
                                    linear_result += coefs[parent] * kwargs[parent]
                            
                            sigmoid_result = 1.0 / (1.0 + math.exp(-linear_result))
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            sigmoid_with_noise = sigmoid_result + noise
                            
                            threshold = params.get("threshold", 0.5)
                            return 1.0 if sigmoid_with_noise > threshold else 0.0
                        return node_function
                    
                    sem_functions[node] = create_binary_sigmoid_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "exponential" or relationship_type == "exponential_additive":
                    # exponential: used by multiple datasets
                    # exponential_additive: alias for synthetic_2 (Z = exp(-X) + U_Z)
                    def create_exponential_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 1 and deps[0] in kwargs:
                                result = math.exp(-kwargs[deps[0]])
                            else:
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs[parent] * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_exponential_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "exponential_linear":
                    # For epidemiology L: L = exp(0.5 * T + U) where U ~ N(0,1)
                    def create_exponential_linear_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 1 and deps[0] in kwargs:
                                parent_val = kwargs[deps[0]]
                                coef = coefs.get(deps[0], 0.5)
                                # Add noise U ~ N(0,1) to the linear term before taking exp
                                noise_std = params.get("noise_std", 1.0)
                                if isinstance(epsilon, (int, float)) and epsilon == 0:
                                    u_noise = np.random.normal(0, noise_std)
                                else:
                                    u_noise = epsilon
                                result = math.exp(coef * parent_val + u_noise)
                            else:
                                # Fallback to linear
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs.get(parent, 0) * kwargs[parent]
                                
                                noise_std = params.get("noise_std", 0.01)
                                if isinstance(epsilon, (int, float)) and epsilon == 0:
                                    noise = np.random.normal(0, noise_std)
                                else:
                                    noise = epsilon
                                result += noise
                            
                            return result
                        return node_function
                    
                    sem_functions[node] = create_exponential_linear_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "interaction_linear":
                    # For epidemiology R: R = 4 + L * T (interaction between L and T)
                    def create_interaction_linear_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            # Check if this is an interaction model with specific variables
                            interactions = params.get("interactions", [])
                            
                            if interactions and all(dep in kwargs for dep in deps):
                                result = const_intercept
                                # Linear terms (e.g. chain: Y = -W - 3*Z*X has coefficient W: -1.0)
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs.get(parent, 0) * kwargs[parent]
                                # Process interactions
                                for interaction in interactions:
                                    interaction_vars = interaction.get("variables", [])
                                    interaction_coef = interaction.get("coefficient", 1.0)
                                    
                                    if all(var in kwargs for var in interaction_vars):
                                        # Multiply the variables together
                                        interaction_val = 1.0
                                        for var in interaction_vars:
                                            interaction_val *= kwargs[var]
                                        result += interaction_coef * interaction_val
                            else:
                                # Fallback to linear combination
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs.get(parent, 0) * kwargs[parent]
                            
                            # Note: interaction_linear typically doesn't have additional noise
                            noise_std = params.get("noise_std", 0.0)
                            if noise_std > 0:
                                if isinstance(epsilon, (int, float)) and epsilon == 0:
                                    noise = np.random.normal(0, noise_std)
                                else:
                                    noise = epsilon
                                result += noise
                            
                            return result
                        return node_function
                    
                    sem_functions[node] = create_interaction_linear_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "trigonometric" or relationship_type == "trigonometric_exponential_additive":
                    # trigonometric: used by multiple datasets
                    # trigonometric_exponential_additive: alias for synthetic_2 (Y = cos(Z) - exp(-Z/20) + U_Y)
                    def create_trigonometric_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 1 and deps[0] in kwargs:
                                z_val = kwargs[deps[0]]
                                result = math.cos(z_val) - math.exp(-z_val / 20)
                            else:
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs[parent] * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_trigonometric_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "quadratic_latent":
                    def create_quadratic_latent_function(deps, coefs, const_intercept, params):
                        latent_names = params.get("latent_variables", [])
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 1 and deps[0] in kwargs:
                                f_val = kwargs[deps[0]]
                                if latent_names and latent_names[0] in kwargs:
                                    u1 = kwargs[latent_names[0]]
                                else:
                                    u1 = np.random.normal(0, 1)
                                result = f_val**2 + u1
                            else:
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs[parent] * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_quadratic_latent_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "latent":
                    def create_latent_function(deps, coefs, const_intercept, params):
                        latent_names = params.get("latent_variables", [])
                        def node_function(epsilon=0, **kwargs):
                            if latent_names and latent_names[0] in kwargs:
                                result = kwargs[latent_names[0]]
                            else:
                                result = np.random.normal(0, 1)

                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_latent_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "scaled_exponential":
                    def create_scaled_exponential_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 1 and deps[0] in kwargs:
                                c_val = kwargs[deps[0]]
                                scale_factor = params.get("scale_factor", 0.1)
                                result = math.exp(-c_val) * scale_factor
                            else:
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs[parent] * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_scaled_exponential_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "trigonometric_mixed":
                    def create_trigonometric_mixed_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 2 and all(dep in kwargs for dep in deps):
                                a_val = kwargs[deps[0]]
                                c_val = kwargs[deps[1]]
                                result = math.cos(a_val) + c_val / 10
                            else:
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs[parent] * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_trigonometric_mixed_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "complex_trigonometric_latent":
                    def create_complex_trigonometric_latent_function(deps, coefs, const_intercept, params):
                        latent_names = params.get("latent_variables", [])
                        def node_function(epsilon=0, **kwargs):
                            if len(deps) == 2 and all(dep in kwargs for dep in deps):
                                d_val = kwargs[deps[0]]
                                e_val = kwargs[deps[1]]
                                if len(latent_names) >= 2 and all(u in kwargs for u in latent_names[:2]):
                                    u1 = kwargs[latent_names[0]]
                                    u2 = kwargs[latent_names[1]]
                                else:
                                    u1 = np.random.normal(0, 1)
                                    u2 = np.random.normal(0, 1)
                                # noise_std parameterizes epsilon_y only; there is no
                                # second additive noise term on this node.
                                epsilon_y = np.random.normal(0, params.get("noise_std", 0.1))
                                return math.cos(d_val) + math.sin(e_val) + u1 + u2 * epsilon_y

                            result = const_intercept
                            for parent in deps:
                                if parent in kwargs:
                                    result += coefs[parent] * kwargs[parent]

                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon

                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_complex_trigonometric_latent_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                elif relationship_type == "complex_trigonometric":
                    # For epidemiology: Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε
                    # Dependencies: [T, L, R, B] (in the order they appear in the causal graph)
                    def create_complex_trigonometric_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            # Extract required variables from kwargs
                            # For epidemiology: deps should be ['T', 'L', 'R', 'B'] but may vary by order
                            if all(dep in kwargs for dep in deps):
                                # Try to identify variables by name
                                T_val = kwargs.get('T', 0)
                                L_val = kwargs.get('L', 0)
                                R_val = kwargs.get('R', 0)
                                B_val = kwargs.get('B', 0)
                                
                                # Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε
                                result = const_intercept + math.cos(4 * T_val) + math.sin(-L_val + 2 * R_val) + B_val
                            else:
                                # Fallback to linear combination if variables are missing
                                result = const_intercept
                                for parent in deps:
                                    if parent in kwargs:
                                        result += coefs.get(parent, 0) * kwargs[parent]
                            
                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon
                            
                            return result + noise
                        return node_function
                    
                    sem_functions[node] = create_complex_trigonometric_function(
                        dependencies, coefficients, intercept, relationship_params
                    )
                
                else:
                    def create_endogenous_function(deps, coefs, const_intercept):
                        def node_function(epsilon=0, **kwargs):
                            result = const_intercept
                            for parent in deps:
                                if parent in kwargs:
                                    result += coefs[parent] * kwargs[parent]
                            return result + epsilon
                        return node_function
                    
                    sem_functions[node] = create_endogenous_function(
                        dependencies, coefficients, intercept
                    )

        self.latent_variables = sorted(latent_variables)

        return sem_functions

    def _sample_latents(self):
        # Latent confounders (e.g. U1, U2) must be drawn once per SEM rollout and
        # shared by every node that declares them in relationship_params.latent_variables.
        return {u: np.random.normal(0, 1) for u in self.latent_variables}

    def _generate_do_functions(self):
        """
        Generate do-calculus functions for each manipulative variable.
        """
        do_dict = {}
        
        MIS, POMIS, manipulative_vars = self.get_sets()
        
        for var in MIS:
            if isinstance(var, list):
                var_name = "_".join(var)
            else:
                var_name = var
            func_name = f"compute_do_{var_name}"
            
            def create_do_function(intervention_var):
                def do_function(observational_samples, functions, value):
                    target = self.target
                    
                    try:
                        if isinstance(value, np.ndarray):
                            if isinstance(intervention_var, list) and len(intervention_var) > 1:
                                if value.ndim == 1:
                                    if len(value) < len(intervention_var):
                                        expanded_value = np.zeros(len(intervention_var))
                                        expanded_value[:len(value)] = value
                                        value_arr = expanded_value.reshape(1, -1)
                                    else:
                                        value_arr = value.reshape(1, -1)
                                elif value.ndim == 2:
                                    if value.shape[1] < len(intervention_var):
                                        expanded_value = np.zeros((value.shape[0], len(intervention_var)))
                                        expanded_value[:, :value.shape[1]] = value
                                        value_arr = expanded_value
                                    else:
                                        value_arr = value
                                else:
                                    value_arr = value.reshape(1, -1)
                                    if value_arr.shape[1] < len(intervention_var):
                                        expanded_value = np.zeros((1, len(intervention_var)))
                                        expanded_value[:, :value_arr.shape[1]] = value_arr
                                        value_arr = expanded_value
                            else:
                                if value.ndim == 1:
                                    value_arr = value.reshape(-1, 1)
                                elif value.ndim == 2:
                                    if value.shape[1] != 1 and not isinstance(intervention_var, list):
                                        value_arr = value[:, 0].reshape(-1, 1)
                                    else:
                                        value_arr = value
                                else:
                                    value_arr = value.reshape(1, 1)
                        else:
                            if isinstance(intervention_var, list) and len(intervention_var) > 1:
                                value_arr = np.array([[float(value)] * len(intervention_var)])
                            else:
                                value_arr = np.array([[float(value)]])
                            
                        mean_do = 0.0
                        var_do = 1.0
                        
                        if isinstance(intervention_var, list) and len(intervention_var) > 1:
                            joint_var_key = "_".join(intervention_var)
                            
                            gp_successful = False
                            
                            if joint_var_key in functions and functions[joint_var_key] is not None:
                                try:
                                    gp_target = functions[joint_var_key]
                                    if value_arr.shape[1] == len(intervention_var):
                                        pred_mean, pred_var = gp_target.predict(value_arr)
                                        mean_do = np.mean(pred_mean)
                                        var_do = np.mean(pred_var)
                                        gp_successful = True
                                    else:
                                        pass
                                except Exception as e:
                                    pass
                                    
                            if not gp_successful:
                                model = self.define_SEM()

                                # Monte-Carlo estimate of the interventional mean and
                                # variance under the noisy SCM with shared latents.
                                n_mc = 100
                                y_samples = []
                                for _ in range(n_mc):
                                    result = self._sample_latents()
                                    for variable, func in model.items():
                                        try:
                                            if variable in intervention_var:
                                                idx = intervention_var.index(variable)
                                                if idx < value_arr.shape[1]:
                                                    result[variable] = value_arr[0, idx]
                                                else:
                                                    result[variable] = 0.0
                                            else:
                                                result[variable] = func(**result)
                                        except Exception:
                                            result[variable] = 0.0
                                    if target in result:
                                        y_samples.append(result[target])

                                if y_samples:
                                    mean_do = float(np.mean(y_samples))
                                    var_do = float(np.var(y_samples))
                                else:
                                    mean_do = 0.0
                                    var_do = 0.0
                        else:
                            if isinstance(intervention_var, list):
                                var_key = intervention_var[0]
                            else:
                                var_key = intervention_var
                                
                            if var_key in functions and functions[var_key] is not None:
                                try:
                                    gp_target = functions[var_key]
                                    
                                    if value_arr.ndim == 2 and value_arr.shape[1] != 1:
                                        value_arr = value_arr[:, 0:1]
                                        
                                    pred_mean, pred_var = gp_target.predict(value_arr)
                                    mean_do = np.mean(pred_mean)
                                    var_do = np.mean(pred_var)
                                except Exception as e:
                                    mean_do = 0.0
                                    var_do = 1.0
                            else:
                                mean_do = 0.0
                                var_do = 1.0
                    
                    except Exception as e:
                        mean_do = 0.0
                        var_do = 1.0
                    
                    if np.isscalar(mean_do):
                        mean_do = np.array([mean_do])
                    if np.isscalar(var_do):
                        var_do = np.array([var_do])
                    
                    return mean_do, var_do
                
                return do_function
            
            do_dict[func_name] = create_do_function(var)
        return do_dict
    
    def create_combined_do_function(self, variables):
        """
        Create a do-calculus function for a combined set of intervention variables.
        
        Args:
            variables: List of intervention variables
        
        Returns:
            Function name for the created do-function
        """
        func_name = get_do_function_name(variables)
        
        if func_name in self.do_functions:
            return func_name
        
        def combined_do_function(observational_samples, functions, value):
            target = self.target
            
            try:
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        if len(value) >= len(variables):
                            value_arr = value[:len(variables)]
                        else:
                            value_arr = np.zeros(len(variables))
                            value_arr[:len(value)] = value
                    elif value.ndim == 2:
                        if value.shape[1] >= len(variables):
                            value_arr = value[0, :len(variables)]
                        else:
                            value_arr = np.zeros(len(variables))
                            value_arr[:value.shape[1]] = value[0, :]
                    else:
                        flattened = value.flatten()
                        value_arr = np.zeros(len(variables))
                        value_arr[:min(len(flattened), len(variables))] = flattened[:len(variables)]
                else:
                    try:
                        value_float = float(value)
                        value_arr = np.array([value_float] * len(variables))
                    except (ValueError, TypeError):
                        value_arr = np.zeros(len(variables))
                
                model = self.define_SEM()

                # Monte-Carlo estimate of the interventional mean and variance
                # under the noisy SCM with shared latents.
                n_mc = 100
                y_samples = []
                for _ in range(n_mc):
                    result = self._sample_latents()
                    for variable, func in model.items():
                        try:
                            if variable in variables:
                                idx = variables.index(variable)
                                if idx < len(value_arr):
                                    result[variable] = value_arr[idx]
                                else:
                                    result[variable] = 0.0
                            else:
                                result[variable] = func(**result)
                        except Exception:
                            result[variable] = 0.0
                    if target in result:
                        y_samples.append(result[target])

                if y_samples:
                    mean_do = float(np.mean(y_samples))
                    var_do = float(np.var(y_samples))
                else:
                    mean_do = 0.0
                    var_do = 0.0
            
            except Exception as e:
                mean_do = 0.0
                var_do = 1.0
            
            if np.isscalar(mean_do):
                mean_do = np.array([mean_do])
            if np.isscalar(var_do):
                var_do = np.array([var_do])
            
            return mean_do, var_do
        
        self.do_functions[func_name] = combined_do_function
        
        return func_name
    
    def define_SEM(self):
        """
        Return the SEM as an OrderedDict.
        """
        return OrderedDict([(node, self.sem_functions[node]) for node in self.causal_order])
    

    
    def intervention_function(self, intervention_dict):
        """
        Create a function to evaluate an intervention and its parameter space.
        
        Args:
            intervention_dict: Dictionary mapping variable names to indices
            
        Returns:
            Tuple of (target_function, space)
        """
        model = self.define_SEM()
        target_variable = self.target
        dict_ranges = self.get_interventional_ranges()
        
        min_intervention = []
        max_intervention = []
        for var in intervention_dict:
            min_intervention.append(dict_ranges[var][0])
            max_intervention.append(dict_ranges[var][1])
        
        def target_function(x):
            intervention_values = {}
            for var, idx in intervention_dict.items():
                intervention_values[var] = x[0, idx] if x.ndim > 1 else x[idx]

            # One noisy rollout of the SCM: latent confounders are shared across
            # nodes, and each node draws its declared structural noise.
            result = self._sample_latents()

            for variable, func in model.items():
                if variable in intervention_values:
                    result[variable] = intervention_values[variable]
                else:
                    result[variable] = func(**result)

            return np.array([[result[target_variable]]])
        
        parameters = []
        for var, idx in intervention_dict.items():
            parameters.append(ContinuousParameter(var, min_intervention[idx], max_intervention[idx]))
        space = ParameterSpace(parameters)
        
        return target_function, space
    
    
    def get_interventional_ranges(self):
        """
        Define intervention ranges for each manipulative variable.
        Prefer the config's interventional_domain; fall back to feature_params
        for variables without a configured domain.
        """
        ranges = OrderedDict()
        config_domains = self.config.get('interventional_domain', {}) or {}

        for var in self.config['intervention']:
            if var in config_domains:
                lo, hi = config_domains[var]
                ranges[var] = [float(lo), float(hi)]
            elif var in self.feature_params:
                print(f"Warning: no interventional_domain for {var}; falling back to feature_params range")
                min_val = float(self.feature_params[var]['min'])
                max_val = float(self.feature_params[var]['max'])
                ranges[var] = [min_val, max_val]
            else:
                raise ValueError(f"Variable {var} not found in interventional_domain or feature_params")

        return ranges
    
    def fit_all_models(self):
        """
        Fit Gaussian Process models for all relationships in the graph.
        Optimized to reduce redundant model creation.
        """
        functions = {}
        created_models = set()
        
        for i, node in enumerate(self.causal_order):
            if i == 0:
                functions[node] = []
                continue
                
            parents = []
            node_idx = self.node_names.index(node)
            
            for j, parent in enumerate(self.node_names):
                if self.adjacency_matrix[node_idx, j] != 0:
                    parents.append(parent)
            
            if not parents:
                functions[node] = []
                continue
                
            X = np.hstack([self.data_arrays[p] for p in parents])
            Y = self.data_arrays[node]
            
            parameter_list = [1.0, 1.0, 0.01, False]
            functions[node] = fit_single_GP_model(X, Y, parameter_list)
            
            MIS, _, _ = self.get_sets()
            interventional_vars = set()
            for var_set in MIS:
                if isinstance(var_set, list):
                    interventional_vars.update(var_set)
                else:
                    interventional_vars.add(var_set)
            
            if len(parents) > 1:
                interventional_parents = [p for p in parents if p in interventional_vars]
                
                if len(interventional_parents) > 1:
                    for pair_size in range(2, min(3, len(interventional_parents) + 1)):
                        for parent_combo in itertools.combinations(interventional_parents, pair_size):
                            combo_key = "_".join(sorted(parent_combo))
                            if combo_key not in created_models:
                                print(f"Creating joint GP model for {combo_key}")
                                X_combo = np.hstack([self.data_arrays[p] for p in parent_combo])
                                functions[combo_key] = fit_single_GP_model(X_combo, Y, parameter_list)
                                created_models.add(combo_key)
        
        MIS, _, _ = self.get_sets()
        for var_set in MIS:
            if isinstance(var_set, list) and len(var_set) > 1:
                joint_key = "_".join(sorted(var_set))
                if joint_key not in created_models and joint_key not in functions:
                    print(f"Creating direct intervention GP model for {joint_key}")
                    X_joint = np.hstack([self.data_arrays[v] for v in var_set])
                    Y = self.data_arrays[self.target]
                    functions[joint_key] = fit_single_GP_model(X_joint, Y, [1.0, 1.0, 0.01, False])
                    created_models.add(joint_key)
        
        return functions
    
    def refit_models(self, observational_samples):
        """
        Refit models with new observational data.
        """
        data_arrays = {}
        for col in observational_samples.columns:
            data_arrays[col] = np.asarray(observational_samples[col])[:, np.newaxis]
        
        functions = {}
        
        for i, node in enumerate(self.causal_order):
            if i == 0:
                functions[node] = []
                continue
                
            parents = []
            node_idx = self.node_names.index(node)
            
            for j, parent in enumerate(self.node_names):
                if self.adjacency_matrix[node_idx, j] != 0:
                    parents.append(parent)
            
            if not parents:
                functions[node] = []
                continue
                
            X = np.hstack([data_arrays[p] for p in parents])
            Y = data_arrays[node]
            
            parameter_list = [1.0, 1.0, 0.01, False]
            functions[node] = fit_single_GP_model(X, Y, parameter_list)
        
        return functions
    
    
    def get_cost_structure(self, type_cost):
        """
        Define costs for interventions.
        """
        manipulative_variables = self.config['intervention']
        
        if type_cost == 1:
            costs = {var: 1.0 for var in manipulative_variables}
        elif type_cost == 2:
            costs = {}
            for i, var in enumerate(manipulative_variables):
                costs[var] = 1.0 + i * 0.5
        elif type_cost == 3:
            costs = {}
            for var in manipulative_variables:
                var_idx = self.node_names.index(var)
                outdegree = sum(self.adjacency_matrix[:, var_idx] != 0)
                costs[var] = 1.0 + outdegree * 0.5
        else:
            costs = {var: 1.0 for var in manipulative_variables}
        
        return costs
    
    def get_sets(self):
        """
        Generate sets of manipulative variables for CBO.
        Returns MIS (Minimally Invasive Sets), POMIS (same as MIS for full observability),
        and the list of manipulative variables.
        """
        manipulative_variables = list(self.config['intervention'])
        
        MIS = []
        
        MIS_1 = [[var] for var in manipulative_variables]
        MIS.append(MIS_1)
        
        max_set_size = min(5, len(manipulative_variables))
        for size in range(2, max_set_size + 1):
            MIS_n = [list(combo) for combo in itertools.combinations(manipulative_variables, size)]
            MIS.append(MIS_n)
        
        flat_MIS = [item for sublist in MIS for item in sublist]
        
        POMIS = flat_MIS
        
        return flat_MIS, POMIS, manipulative_variables
    
    def get_all_do(self):
        """
        Return dictionary of do-calculus functions.
        """
        return self.do_functions


def causal_discovery_to_graph(model_results, sem_equations, data, target, config, feature_params):
    """
    Convert the results of causal discovery to a graph structure usable by BO/CBO.
    
    Args:
        model_results: Results from causal_discovery's run_icalingam_analysis
        data: DataFrame containing the data used in causal discovery
        target: The target variable 
        
    Returns:
        A DiscoveredGraph instance compatible with BO/CBO
    """
    return DiscoveredGraph(model_results, sem_equations, data, target, config, feature_params)


def setup_optimization_from_discovery(model_results, sem_equations, data, target, config, feature_params):
    """
    Setup everything needed for BO/CBO from causal discovery results.
    
    Args:
        model_results: Results from causal_discovery's run_icalingam_analysis
        data: DataFrame containing the data used in causal discovery
        target: The target variable 
        
    Returns:
        Tuple of (graph, observational_samples, functions)
    """
     # Randomly select observational samples from the data
    total_rows = data.shape[0]
    random_indices = np.random.choice(total_rows, min(config['num_observations'], total_rows), replace=False)
    observational_samples = data.iloc[random_indices].copy()
    print("Observational samples selected")

    # Create graph from discovery results
    graph = causal_discovery_to_graph(model_results, sem_equations, observational_samples, target, config, feature_params)
    print("Graph created")
   
    # Fit models
    functions = graph.fit_all_models()
    print("Models fitted")
    return graph, observational_samples, functions 