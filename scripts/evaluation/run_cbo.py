
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse
import importlib.util
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.BO_CBO.graph import setup_optimization_from_discovery
from baselines.BO_CBO.utils import compute_coverage, get_do_function_name
from baselines.BO_CBO.BO import NonCausal_BO
from baselines.BO_CBO.CBO import CBO
from metrics.GAP import GAP
from metrics.PA_GAP import PA_GAP
import yaml
import json
import torch
from emukit.core.parameter_space import ParameterSpace
from emukit.core.continuous_parameter import ContinuousParameter


FUNCTION_NETWORK_DATASETS = {
    "ackley": "Ackley",
    "rosenbrock_d3": "Rosenbrock",
    "rosenbrock_d5": "Rosenbrock",
    "rosenbrock_d7": "Rosenbrock",
    "dropwave": "Dropwave",
    "alpine2": "Alpine2",
    "chain_soft": "ChainSoft",
}


def display_algorithm_metrics(outcomes_list, task, algorithm_name, dataset_name, trials):
    # Load dataset configuration for GAP calculation
    with open(f'configs/{dataset_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    with open(f'observational_datasets/{dataset_name}_theoretical_best.json', 'r') as file:
        theoretical_best = json.load(file)

    best_theoretical = theoretical_best['global_best_value']
    
    save_dir = f'results/{algorithm_name}/{dataset_name}'

    gap_calculator = GAP(best_theoretical)
    pa_gap_calculator = PA_GAP(best_theoretical)
    gap_calculator.calculate_GAP(outcomes_list, task)
    pa_gap_calculator.calculate_PA_GAP(outcomes_list, task)
        
    print(f"  Initial Value: {pa_gap_calculator.initial_value:.4f}")
    print(f"  Final Value: {pa_gap_calculator.final_value:.4f}")
    print(f"  Best Theoretical: {pa_gap_calculator.best_theoretical:.4f}")
    print(f"  Optimal Trial: {pa_gap_calculator.optimal_trial}")
    print(f"  Total Trials: {pa_gap_calculator.total_trials}")

    print(f"PA-GAP Metrics:")

    print(f"  PA-GAP Record: {pa_gap_calculator.PA_GAP_record}")
    print(f"  PA-GAP Value: {pa_gap_calculator.PA_GAP_value:.4f}")
    print(f"  PA-GAP std: {pa_gap_calculator.PA_GAP_std:.4f}")
    pa_gap_record = pa_gap_calculator.export_to_dataframe()
    pa_gap_record.to_csv(f'{save_dir}/PA_GAP_record_{trials}.csv', index=False)

    print("GAP Metrics:")
    print(f" GAP Value: {gap_calculator.GAP_value:.4f}")


    print()  # Empty line for separation


def _load_mcbo_function_class(class_name):
    baseline_path = REPO_ROOT / "baselines" / "mcbo"
    functions_path = baseline_path / "scripts" / "functions.py"
    sys.path.insert(0, str(baseline_path))
    spec = importlib.util.spec_from_file_location(
        "mcbo_benchmark_functions", functions_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class FunctionNetworkGraph:
    """Adapter exposing MCBO function networks through the BO graph interface."""

    def __init__(self, env, intervention_variables):
        self.env = env
        self.target = "Y"
        self.intervention_variables = intervention_variables

    def _evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        with torch.no_grad():
            x_tensor = torch.as_tensor(x, dtype=torch.get_default_dtype())
            y_tensor = self.env.evaluate(x_tensor)[..., -1]
        return y_tensor.detach().cpu().numpy().reshape(-1, 1)

    def get_all_do(self):
        do_name = get_do_function_name(self.intervention_variables)

        def compute_do(_observational_samples, _functions, x):
            values = self._evaluate(x)
            return values, np.zeros_like(values)

        return {do_name: compute_do}

    def intervention_function(self, intervention_dict):
        ordered_vars = [
            var for var, _ in sorted(intervention_dict.items(), key=lambda item: item[1])
        ]

        def target_function(x):
            return self._evaluate(x)

        space = ParameterSpace([
            ContinuousParameter(var, 0.0, 1.0) for var in ordered_vars
        ])
        return target_function, space

    def get_cost_structure(self, type_cost=1):
        return {var: 1.0 for var in self.intervention_variables}


def run_function_network_bo(args, config):
    if args.algorithm not in ["BO", "both"]:
        raise ValueError(
            f"Function-network dataset '{args.dataset}' only supports BO in scripts/evaluation/run_cbo.py"
        )
    if args.causal_prior:
        print("Function-network BO does not provide SEM priors; ignoring --causal_prior.")

    torch.set_default_dtype(torch.float64)
    env_class = _load_mcbo_function_class(FUNCTION_NETWORK_DATASETS[args.dataset])
    env_kwargs = {}
    if FUNCTION_NETWORK_DATASETS[args.dataset] == "Rosenbrock":
        # Number of action variables from the config
        env_kwargs["D"] = int(config.get("num_intervention", len(config["intervention"])))
    elif args.dataset == "chain_soft":
        # Wire the config's soft_intervention block into the environment
        soft_cfg = config.get("soft_intervention") or {}
        if soft_cfg:
            coeff_range = soft_cfg.get("coefficient_range", [-0.27, 0.27])
            env_kwargs.update(
                grid_size=soft_cfg.get("grid_size", 10),
                n_alpha=soft_cfg.get("n_alpha", 10),
                n_beta=soft_cfg.get("n_beta", 10),
                coeff_bound=abs(float(coeff_range[1])),
            )
    env = env_class(noise_scales=0.0, **env_kwargs)
    intervention_variables = list(config["intervention"])
    graph = FunctionNetworkGraph(env, intervention_variables)
    input_dim = len(intervention_variables)
    n_init = max(args.init_samples, input_dim + 1)

    data_x = np.random.rand(n_init, input_dim)
    data_y = graph._evaluate(data_x)
    if config["task"] == "min":
        optimal_idx = int(np.argmin(data_y[:, 0]))
    else:
        optimal_idx = int(np.argmax(data_y[:, 0]))
    optimal_intervention_value = data_x[optimal_idx:optimal_idx + 1]
    optimal_y = float(data_y[optimal_idx, 0])

    os.makedirs(f"results/BO/{args.dataset}", exist_ok=True)
    csv_log_file = (
        f"results/BO/{args.dataset}/"
        f"BO_{args.dataset}_{args.num_trials}-trials_progress.csv"
    )
    print(f"BO progress will be saved to: {csv_log_file}")

    bo_result = NonCausal_BO(
        num_trials=args.num_trials,
        graph=graph,
        dict_ranges={var: (0.0, 1.0) for var in intervention_variables},
        interventional_data_x=data_x,
        interventional_data_y=data_y,
        costs=graph.get_cost_structure(),
        observational_samples=pd.DataFrame(),
        functions={},
        initial_intervention_value=optimal_intervention_value,
        initial_y=optimal_y,
        intervention_variables=intervention_variables,
        Causal_prior=False,
        task=config["task"],
        csv_log_file=csv_log_file,
    )

    bo_costs, bo_best_x, _bo_best_y, bo_time, bo_global_opt = bo_result
    print(f"BO completed in {bo_time:.2f} seconds")
    print(f"Final best intervention values: {intervention_variables} = {bo_best_x[-1]}")
    print(f"Final cost: {bo_costs[-1][0]:.4f}")
    display_algorithm_metrics(
        bo_global_opt, config["task"], "BO", args.dataset, args.num_trials
    )


def sample_initial_data(graph, dataname, intervention_sets, num_interventions=5, mode='BO'):
    """
    Generate initial samples for optimization algorithms.
    """
    dict_ranges = graph.get_interventional_ranges()
    dir_path = 'interventional_datasets'
    config_path = 'configs/' + dataname + '.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    
    if mode == 'BO':
        BO_data = np.load(f'{dir_path}/{dataname}_interventional_BO.pkl', allow_pickle=True)
        data_x = BO_data[0]
        data_y = BO_data[1].reshape(-1, 1)
        all_data = np.concatenate((data_x, data_y), axis=1)
        
        state = np.random.get_state()

        np.random.shuffle(all_data)
        
        np.random.set_state(state)

        data_x = all_data[:num_interventions, :len(intervention_sets)]
        data_y = all_data[:num_interventions, len(intervention_sets):]

        # Handle both min and max tasks
        if config['task'] == 'min':
            optimal_y = np.min(data_y)
            optimal_idx = np.where(data_y == optimal_y)[0][0]
        else:  # task == 'max'
            optimal_y = np.max(data_y)
            optimal_idx = np.where(data_y == optimal_y)[0][0]
        
        optimal_intervention_value = np.transpose(all_data[optimal_idx][:len(intervention_sets)][:,np.newaxis])

        return data_x, data_y, optimal_intervention_value, optimal_y
    
    elif mode == 'CBO':
        data_list = []
        data_x_list = []
        data_y_list = []
        opt_list = []
        CBO_data = np.load(f'{dir_path}/{dataname}_interventional_CBO.pkl', allow_pickle=True)
        task = config['task']
        
        # Check if there's a mismatch between intervention_sets and CBO_data
        if len(intervention_sets) > len(CBO_data):
            print(f"Warning: Number of intervention sets ({len(intervention_sets)}) is greater than available data ({len(CBO_data)})")
            print(f"Using only the first {len(CBO_data)} intervention sets")
            intervention_sets = intervention_sets[:len(CBO_data)]
        
        for j in range(len(CBO_data)):
            if j >= len(intervention_sets):
                print(f"Warning: Data item {j} has no corresponding intervention set, skipping")
                continue
                
            # Support both CBO data formats:
            #   1) [data_x, data_y]
            #   2) [n_variables, intervention_set, data_x, data_y]
            data = CBO_data[j]
            if isinstance(data, (list, tuple)) and len(data) == 4:
                _, saved_intervention_set, saved_data_x, saved_data_y = data
                data_x = np.asarray(saved_data_x)
                data_y = np.asarray(saved_data_y)

                if j < len(intervention_sets) and list(saved_intervention_set) != list(intervention_sets[j]):
                    print(
                        f"Warning: Intervention set mismatch at index {j}. "
                        f"Expected {intervention_sets[j]}, got {saved_intervention_set}"
                    )
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                data_x = np.asarray(data[0])
                data_y = np.asarray(data[1])
            else:
                raise ValueError(
                    f"Unsupported CBO data entry format at index {j}: {type(data)} with length "
                    f"{len(data) if hasattr(data, '__len__') else 'unknown'}"
                )
            
            # Number of variables in this intervention set
            num_variables = len(intervention_sets[j]) if j < len(intervention_sets) else (data_x.shape[1] if hasattr(data_x, 'shape') and data_x.ndim == 2 else 1)
            


            if len(data_y.shape) == 1:
                data_y = data_y[:,np.newaxis]

            if len(data_x.shape) == 1:
                # 1D X corresponds to single-variable interventions; make it (N, 1)
                data_x = data_x[:,np.newaxis]
            


            all_data = np.concatenate((data_x, data_y), axis =1)

            ## Need to reset the global seed 
            state = np.random.get_state()

    
            np.random.shuffle(all_data)


            np.random.set_state(state)

            subset_all_data = all_data[:num_interventions]

            data_list.append(subset_all_data)
            data_x_list.append(data_list[j][:,:-1])
            data_y_list.append(data_list[j][:,-1][:,np.newaxis])

            if task == 'min':
                opt_list.append(np.min(subset_all_data[:,-1])) 
                var_min = intervention_sets[np.where(opt_list == np.min(opt_list))[0][0]]
                opt_y = np.min(opt_list)
                opt_intervention_array = data_list[np.where(opt_list == np.min(opt_list))[0][0]]
            else:
                opt_list.append(np.max(subset_all_data[:,-1])) 
                var_min = intervention_sets[np.where(opt_list == np.max(opt_list))[0][0]]
                opt_y = np.max(opt_list)
                opt_intervention_array = data_list[np.where(opt_list == np.max(opt_list))[0][0]]

            if len(var_min) == 4:
                best_variable1 = var_min[0]
                best_variable2 = var_min[1]
                best_variable3 = var_min[2]
                best_variable4 = var_min[3]
                best_variable = "_".join([best_variable1, best_variable2, best_variable3, best_variable4])
                
            if len(var_min) ==  3:
                best_variable1 = var_min[0]
                best_variable2 = var_min[1]
                best_variable3 = var_min[2]
                best_variable = "_".join([best_variable1, best_variable2, best_variable3])
                
            if len(var_min) ==  2:
                best_variable1 = var_min[0]
                best_variable2 = var_min[1]
                best_variable = "_".join([best_variable1, best_variable2])
                
            if len(var_min) ==  1:
                best_variable = var_min[0]


            shape_opt = opt_intervention_array.shape[1] - 1
            if task == 'min':
                best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.min(opt_intervention_array[:,shape_opt]), :shape_opt][0]
            else:
                best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.max(opt_intervention_array[:,shape_opt]), :shape_opt][0]
            
        return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable

    return 


def main(args):
    # Set random seed for reproducibility
    import random
    import os
    from datetime import datetime
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    dataname = args.dataset

    config_path = 'configs/' + dataname + '.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    target = config['target']

    if dataname in FUNCTION_NETWORK_DATASETS:
        run_function_network_bo(args, config)
        return

    df_filtered = pd.read_csv(f'observational_datasets/{dataname}.csv')

    feature_params = json.load(open(f'observational_datasets/{dataname}_feature_params.json'))

    sem_equations = json.load(open(f'sem/{dataname}_sem_equations.json'))

    model_results = np.load(f'sem/{dataname}_sem_model.pkl', allow_pickle=True)
    
    # Create graph, observational samples, and functions
    print("Setting up optimization from discovery results...")
    graph, observational_samples, functions = setup_optimization_from_discovery(
        model_results, sem_equations, df_filtered, target, config, feature_params
    )
    print("Optimization setup complete")
    # Get necessary objects for optimization
    manipulative_variables = list(config['intervention'])
    dict_ranges = {var: (config['interventional_domain'][var][0], config['interventional_domain'][var][1]) for var in manipulative_variables}
    costs = graph.get_cost_structure(type_cost=1)  # Equal costs
     
    print(f"manipulative_variables: {manipulative_variables}")

    
    # Generate initial data
    print("Generating initial data...") 
    if args.algorithm in ['BO', 'both']:
        # For BO, prepare intervention variables based on mode
        bo_vars = manipulative_variables
        print(f"Intervention variables for BO: {bo_vars}")
        
        # Make sure the combined do-function exists
        do_func_name = get_do_function_name(bo_vars)
        if do_func_name not in graph.get_all_do():
            print(f"Creating do-function for combined variables: {do_func_name}")
            graph.create_combined_do_function(bo_vars)
            
        # Generate data for BO
        data_x, data_y, optimal_intervention_value, optimal_y = sample_initial_data(graph, dataname, bo_vars, config['num_intervention'], 'BO')
        
    if args.algorithm in ['CBO', 'both']:
        # For CBO, we need initial data for each intervention set in POMIS
        cbo_vars, POMIS, manipulative_variables = graph.get_sets()
        print(f"Intervention variables for CBO: {cbo_vars}")
        data_x_list, data_y_list, best_intervention_value_cbo, opt_y_cbo, best_variable_cbo = sample_initial_data(graph, dataname, cbo_vars, config['num_intervention'], 'CBO')
        
        # Ensure the cbo_vars match the length of the data lists
        if len(cbo_vars) > len(data_x_list):
            print(f"Warning: Number of intervention sets ({len(cbo_vars)}) is greater than available data ({len(data_x_list)})")
            print(f"Truncating intervention sets to match data")
            cbo_vars = cbo_vars[:len(data_x_list)]

        # Compute coverage for CBO
        alpha_coverage, hull_obs, coverage_total = compute_coverage(
            observational_samples, manipulative_variables, dict_ranges
        )
        print(f"Initial observational coverage: {alpha_coverage:.4f}")
        
    
    # Run optimization algorithms
    if args.algorithm in ['BO', 'both']:
        print("\nRunning Bayesian Optimization...")

        # Create CSV log file path for BO iteration progress
        os.makedirs(f'results/BO/{dataname}', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_log_file = f'results/BO/{dataname}/BO_{dataname}_{args.num_trials}-trials_progress.csv'
        print(f"BO progress will be saved to: {csv_log_file}")
        
        # Run BO
        bo_result = NonCausal_BO(
            num_trials=args.num_trials,
            graph=graph,
            dict_ranges=dict_ranges,
            interventional_data_x=data_x,
            interventional_data_y=data_y,
            costs=costs,
            observational_samples=observational_samples,
            functions=functions,
            initial_intervention_value=optimal_intervention_value,
            initial_y=optimal_y,
            intervention_variables=bo_vars,
            Causal_prior=args.causal_prior,
            task=config['task'],
            csv_log_file=csv_log_file
        )
        
        bo_costs, bo_best_x, bo_best_y, bo_time, bo_global_opt = bo_result
        print(f"BO completed in {bo_time:.2f} seconds")
        print(f"BO iteration progress saved to: {csv_log_file}")
        
        # Display only the final best intervention values
        final_best_x = bo_best_x[-1]
        print(f"Final best intervention values: {bo_vars} = {final_best_x}")
        
        print(f"Final cost: {bo_costs[-1][0]:.4f}")
        
        # Display BO metrics
        display_algorithm_metrics(bo_global_opt, config['task'], 'BO', dataname, args.num_trials)
    
    if args.algorithm in ['CBO', 'both']:
        print("\nRunning Causal Bayesian Optimization...")
        # Make a copy of observational samples for full dataset
        full_observational_samples = observational_samples.copy()
        maxN = config['num_observations'] + 50
        
        # Create CSV log file path for CBO iteration progress
        os.makedirs(f'results/CBO/{dataname}', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cbo_csv_log_file = f'results/CBO/{dataname}/CBO_{dataname}_{args.num_trials}-trials_progress.csv'
        print(f"CBO progress will be saved to: {cbo_csv_log_file}")
        
        # Run CBO
        cbo_result = CBO(
            num_trials=args.num_trials,
            exploration_set=cbo_vars,
            manipulative_variables=manipulative_variables,
            data_x_list=data_x_list,
            data_y_list=data_y_list,
            best_intervention_value=best_intervention_value_cbo,
            opt_y=opt_y_cbo,
            max_N=maxN,
            best_variable=best_variable_cbo,
            dict_ranges=dict_ranges,
            functions=functions,
            initial_num_obs_samples=config['num_observations'],
            observational_samples=observational_samples,
            coverage_total=coverage_total,
            graph=graph,
            num_additional_observations=10,
            costs=costs,
            full_observational_samples=full_observational_samples,
            task=config['task'],
            Causal_prior=args.causal_prior,
            csv_log_file=cbo_csv_log_file
        )
        
        cbo_costs, cbo_best_x, cbo_best_y, cbo_global_opt, cbo_observed, cbo_time = cbo_result
        print(f"CBO completed in {cbo_time:.2f} seconds")
        print(f"CBO iteration progress saved to: {cbo_csv_log_file}")
        
        # Handle the new structured format from CBO
        if isinstance(cbo_best_x, dict) and 'intervention_set' in cbo_best_x:
            # New structured format
            intervention_set = cbo_best_x['intervention_set']
            intervention_values = cbo_best_x['intervention_values']
            print(f"Final best intervention: {intervention_set} = {intervention_values}")
        else:
            # Legacy format - try to handle the old way
            best_set = None
            best_val = None
            for key, value in cbo_best_x.items():
                best_set = key
                best_val = value
                break
            
            if best_set is not None:
                print(f"Final best intervention: {best_set} = {best_val}")
            else:
                print(f"Final best intervention value: {cbo_best_x}")
            
        print(f"Final best value: {cbo_global_opt[-1]:.4f}")
        print(f"Final cost: {cbo_costs[-1]:.4f}")
        print(f"Number of observations: {cbo_observed} / {args.num_trials}")
        
        # Display CBO metrics
        display_algorithm_metrics(cbo_global_opt, config['task'], 'CBO', dataname, args.num_trials)
    

    # Plot results if both algorithms were run
    if args.algorithm == 'both':
        plt.figure(figsize=(12, 6))
        
        # Plot optimization performance
        plt.subplot(1, 2, 1)
        plt.plot(bo_best_y, label='BO')
        plt.plot(cbo_global_opt, label='CBO')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.legend()
        plt.title('Optimization Performance')
        
        # Plot cost
        plt.subplot(1, 2, 2)
        plt.plot(bo_costs, label='BO')
        plt.plot(cbo_costs, label='CBO')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Cost')
        plt.legend()
        plt.title('Cumulative Cost')
        
        plt.tight_layout()
        # Create comparison plots directory and save plot
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig(f'results/plots/bo_cbo_comparison_{args.dataset}_seed{args.seed}_{timestamp}.png')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BO and CBO after causal discovery')
    parser.add_argument('--dataset', type=str, default='toyGraph', help='Dataset name')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--init_samples', type=int, default=5, help='Number of initial samples')
    parser.add_argument('--algorithm', type=str, choices=['BO', 'CBO', 'both'], default='CBO', help='Algorithm to run')
    parser.add_argument('--causal_prior', action='store_true', help='Use causal prior in optimization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    
    args = parser.parse_args()
    main(args) 