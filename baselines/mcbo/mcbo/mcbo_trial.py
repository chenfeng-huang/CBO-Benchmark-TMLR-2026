import numpy as np
import os
import sys
import time
import torch
import csv
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qKnowledgeGradient,
    UpperConfidenceBound,
    qSimpleRegret,
)
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch.sampling.normal import SobolQMCNormalSampler

from torch import Tensor
from typing import Callable, List, Optional

from mcbo.acquisition_function_optimization.optimize_acqf import (
    optimize_acqf_and_get_suggested_point,
)
from mcbo.utils.dag import DAG
from mcbo.utils.initial_design import generate_initial_design
from mcbo.utils.fit_gp_model import fit_gp_model
from mcbo.models.gp_network import GaussianProcessNetwork
from mcbo.utils.posterior_mean import PosteriorMean
from mcbo.models import eta_network
from mcbo.utils.gap_metrics import GapMetricsCalculator
import wandb

def obj_mean(
    X: Tensor, function_network: Callable, network_to_objective_transform: Callable
) -> Tensor:
    '''
    Estimates the mea value of a noisy objective by computing it a lot of times and 
    averaging.
    '''
    X = X[None]  # create new 0th dim
    Ys = function_network(X.repeat(100000, 1, 1))
    return torch.mean(network_to_objective_transform(Ys), dim=0)

def mcbo_trial(
    algo_profile: dict,
    env_profile: dict,
    function_network: Callable,
    network_to_objective_transform: Callable,
    theoretical_optimum: Optional[float] = None,
) -> None:
    r'''
    Interacts with the environment specified by env_profile and function_network for 
    algo_profile['n_bo_iters'] rounds using the algorithm specified in algo_profile. 
    Logs to wandb the average and best score across rounds. 
    '''
    torch.seed()
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # results_folder = script_dir + "/results/" + problem + "/" + algo + "/"

    # Initialize GAP metrics calculator
    gap_calculator = GapMetricsCalculator()
    
    # Setup CSV logging for both causal and function-network environments.
    csv_file = None
    csv_writer = None
    env_name = getattr(wandb.config, 'env', 'unknown') if hasattr(wandb, 'config') else 'unknown'
    seed = getattr(wandb.config, 'seed', algo_profile.get('seed', 0)) if hasattr(wandb, 'config') else algo_profile.get('seed', 0)
    csv_filename = f"trial_results_{algo_profile['algo']}_{env_name}_{seed}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    
    # Get intervention variable names based on environment
    intervention_names = get_intervention_names(env_profile)
    
    # Create CSV header
    header = ['trial_number', 'current_optimal'] + intervention_names
    csv_writer = csv.DictWriter(csv_file, fieldnames=header)
    csv_writer.writeheader()
    print(f"CSV logging enabled: {csv_filename}")

    # Initial evaluations
    X = generate_initial_design(algo_profile, env_profile)
    
    mean_at_X = obj_mean(X, function_network, network_to_objective_transform)
    network_observation_at_X = function_network(X)
    observation_at_X = network_to_objective_transform(network_observation_at_X)
    # Current best objective value.
    best_obs_val = observation_at_X.max().item()

    # Historical best observed objective values and running times.
    hist_best_obs_vals = [best_obs_val]
    runtimes = []

    # Log initial trial (trial 0) to CSV
    if csv_writer is not None:
        initial_row = {'trial_number': 0, 'current_optimal': best_obs_val}
        # Initialize intervention columns as empty for trial 0
        intervention_names = get_intervention_names(env_profile)
        for name in intervention_names:
            initial_row[name] = ''
        csv_writer.writerow(initial_row)
        csv_file.flush()

    init_batch_id = 1

    old_nets = []  # only used by NMCBO to reuse old computation
    for iteration in range(init_batch_id, algo_profile["n_bo_iter"] + 1):
        print("Sampling policy: " + algo_profile["algo"])
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()
        new_x, new_net = get_new_suggested_point(
            X=X,
            network_observation_at_X=network_observation_at_X,
            observation_at_X=observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
            function_network=function_network,
            network_to_objective_transform=network_to_objective_transform,
            old_nets=old_nets,
        )
        if new_net is not None:
            old_nets.append(new_net)
        t1 = time.time()
        runtimes.append(t1 - t0)

        # Evalaute network at new point
        network_observation_at_new_x = function_network(new_x)

        # The mean value of the new action. 
        mean_at_new_x = obj_mean(
            new_x, function_network, network_to_objective_transform
        )
        if mean_at_X is None:
            mean_at_X = mean_at_new_x
        else:
            mean_at_X = torch.cat([mean_at_X, mean_at_new_x], 0)

        # Evaluate objective at new point
        observation_at_new_x = network_to_objective_transform(
            network_observation_at_new_x
        )

        # Update training data
        X = torch.cat([X, new_x], 0)
        network_observation_at_X = torch.cat(
            [network_observation_at_X, network_observation_at_new_x], 0
        )
        observation_at_X = torch.cat([observation_at_X, observation_at_new_x], 0)

        # Update historical best observed objective values
        best_obs_val = observation_at_X.max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))
        
        # Calculate GAP metrics if theoretical optimum is provided
        gap_metrics = {}
        if theoretical_optimum is not None:
            gap_calculator.calculate_gap_metric(
                outcomes=hist_best_obs_vals,
                best_theoretical=theoretical_optimum,
                task='max'  # MCBO maximizes by default
            )
            gap_metrics = gap_calculator.get_metrics_dict()
        
        # average_score includes the random exploration runs at the init.
        log_dict = {
            "score": mean_at_new_x,
            "best_score": torch.max(mean_at_X),
            "average_score": torch.mean(mean_at_X),
            "X": new_x,
        }
        
        # Add GAP metrics to logging
        log_dict.update(gap_metrics)
        
        wandb.log(log_dict)
        
        # Log to CSV if causal environment
        if csv_writer is not None:
            csv_row = {'trial_number': iteration, 'current_optimal': best_obs_val}
            
            # Extract intervention values from new_x
            intervention_values = extract_intervention_values(new_x, env_profile)
            csv_row.update(intervention_values)
            
            csv_writer.writerow(csv_row)
            csv_file.flush()  # Ensure data is written immediately
    
    # Print final GAP metrics summary
    if theoretical_optimum is not None:
        print("\n" + "="*60)
        print("FINAL GAP METRICS SUMMARY")
        print("="*60)
        print(f"Algorithm: {algo_profile['algo']}")
        print(f"Total Trials: {gap_calculator.total_trials}")
        print(f"Initial Value: {gap_calculator.initial_value:.6f}")
        print(f"Final Best Value: {gap_calculator.final_value:.6f}")
        print(f"Theoretical Optimum: {gap_calculator.best_theoretical:.6f}")
        print(f"Improvement Achieved: {gap_calculator.improvement_achieved:.6f}")
        print(f"Improvement Possible: {gap_calculator.improvement_possible:.6f}")
        if gap_calculator.improvement_possible > 1e-10:
            improvement_ratio = gap_calculator.improvement_achieved / gap_calculator.improvement_possible
            print(f"Improvement Ratio: {improvement_ratio:.4f} ({improvement_ratio*100:.2f}%)")
        print(f"Optimal Trial (when best found): {gap_calculator.optimal_trial}")
        print(f"FINAL AVERAGE GAP SCORE: {gap_calculator.gap_value:.6f}")
        print("="*60)
    
    # Close CSV file if it was opened
    if csv_file is not None:
        csv_file.close()
        print(f"CSV results saved to: {csv_filename}")


def get_model(
    X: Tensor,
    network_observation_at_X: Tensor,
    observation_at_X: Tensor,
    algo_profile: dict,
    env_profile: dict,
):
    input_dim = env_profile["input_dim"]
    algo = algo_profile["algo"]
    if algo == "EIFN":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
    elif algo == "EICF":
        model = fit_gp_model(X=X, Y=network_observation_at_X)
    elif algo == "EI":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "KG":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "UCB":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "MCBO":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
        # Make the input dimension bigger to account for the hallucination inputs.
        input_dim = input_dim + env_profile["dag"].get_n_nodes()
    elif algo == "NMCBO":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
    else:
        raise ValueError("No algorithm of this name implemented")
    # Remove the binary target inputs for interventions
    if env_profile["interventional"]:
        input_dim = input_dim - env_profile["dag"].get_n_nodes()
    return model, input_dim


def get_acq_fun(
    model, network_to_objective_transform, observation_at_X, algo_profile, env_profile
):
    algo = algo_profile["algo"]
    if algo == "EIFN":
        # Sampler
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        # Acquisition function
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=observation_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EICF":
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=observation_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EI":
        acquisition_function = ExpectedImprovement(
            model=model, best_f=observation_at_X.max().item()
        )
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo == "KG":
        acquisition_function = qKnowledgeGradient(model=model, num_fantasies=8)
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo == "UCB":
        acquisition_function = UpperConfidenceBound(
            model=model, beta=algo_profile["beta"]
        )
        posterior_mean_function = None
    elif algo == "MCBO":
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        # Acquisition function
        acquisition_function = qSimpleRegret(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )

        posterior_mean_function = None

    return acquisition_function, posterior_mean_function


def get_new_suggested_point_random(env_profile) -> Tensor:
    r"""Returns a new suggested point randomly."""
    if env_profile["interventional"]:
        # For interventional random = random targets, random values
        target_idx = np.random.choice(len(env_profile["valid_targets"]))
        target = env_profile["valid_targets"][target_idx]
        from mcbo.utils.initial_design import random_causal
        return random_causal(target, env_profile), None
    else:
        return torch.rand([1, env_profile["input_dim"]]), None


def get_new_suggested_point(
    X: Tensor,
    network_observation_at_X: Tensor,
    observation_at_X: Tensor,
    algo_profile: dict,
    env_profile: dict,
    function_network: Callable,
    network_to_objective_transform: Callable,
    old_nets: List,
) -> Tensor:

    algo = algo_profile["algo"]

    algos_interventions_not_implemented = ["UCB, KG, EI, EICF"]

    if env_profile["interventional"] and algo in algos_interventions_not_implemented:
        raise ValueError(
            "This algorithm is not implemented for the interventional setting"
        )

    if algo == "Random":
        return get_new_suggested_point_random(env_profile)

    model, acq_fun_input_dim = get_model(
        X, network_observation_at_X, observation_at_X, algo_profile, env_profile
    )

    if algo in algos_interventions_not_implemented:
        acquisition_function, posterior_mean_function = get_acq_fun(
            model,
            network_to_objective_transform,
            observation_at_X,
            algo_profile,
            env_profile,
        )
        new_x, new_score = optimize_acqf_and_get_suggested_point(
            acq_func=acquisition_function,
            bounds=torch.tensor(
                [
                    [0.0 for i in range(acq_fun_input_dim)],
                    [1.0 for i in range(acq_fun_input_dim)],
                ]
            ),
            batch_size=1,
            posterior_mean=posterior_mean_function,
        )
        return new_x, None

    r"""
    The approach is general for both causal BO environments (interventional=True) and 
    function network environments. For both we loop of the possible intervention targets.
    The 'target' for function networks makes no difference to the output and only a 
    single set of targets is contained in valid_targets. 
    """

    best_x = None
    best_score = -torch.inf
    best_target = None

    for target in env_profile["valid_targets"]:
        try:
            model.set_target(target)
        except:
            raise ValueError("Model class isn't able to have interventional targets")
        ## Use a different training procedure if noisy MCBO is used.
        if algo == "NMCBO":
            new_x, new_score, new_net = eta_network.train(
                model,
                network_to_objective_transform,
                env_profile,
                acq_fun_input_dim,
                old_nets,
                batch_size=algo_profile["batch_size"],
            )
        else:
            acquisition_function, posterior_mean_function = get_acq_fun(
                model,
                network_to_objective_transform,
                observation_at_X,
                algo_profile,
                env_profile,
            )

            new_x, new_score = optimize_acqf_and_get_suggested_point(
                acq_func=acquisition_function,
                bounds=torch.tensor(
                    [
                        [0.0 for i in range(acq_fun_input_dim)],
                        [1.0 for i in range(acq_fun_input_dim)],
                    ]
                ),
                batch_size=1,
                posterior_mean=posterior_mean_function,
            )
        if new_score > best_score:
            best_score = new_score
            best_x = new_x
            best_target = torch.tensor(target)

    r"""
    For algorithms that hallucinate inputs, we need to remove these parts of the action
    because they are not used in the real environment. 
    """

    if algo in ["NMCBO", "MCBO"]:
        if env_profile["interventional"]:
            r"""
            For causal BO we also ignore the target dimension which the env_profile
            counts in the input_dim.
            """
            X_dim = env_profile["input_dim"] - env_profile["dag"].get_n_nodes()
            best_x = best_x[:, 0:X_dim]
        else:
            X_dim = env_profile["input_dim"]
            best_x = best_x[:, 0:X_dim]

    # If we're in a causal BO setting to need to prepend the best targets to our ation.
    if env_profile["interventional"]:
        best_x = torch.cat([best_target.unsqueeze(0), best_x], dim=-1)

    # Only NMCBO stores the action and eta networks previously used.
    if algo != "NMCBO":
        new_net = None

    return best_x, new_net


def get_intervention_names(env_profile):
    """
    Get human-readable names for intervention variables based on environment.
    """
    # Map node indices to meaningful names for different environments
    intervention_name_maps = {
        6: {  # PSAGraph has 6 nodes: age, bmi, A, S, cancer, Y
            2: 'Aspirin',
            3: 'Statin'
        },
        3: {  # ToyGraph has 3 nodes
            0: 'X0',
            1: 'X1', 
            2: 'X2'
        },
        6: {  # CompleteGraph has 6 nodes: A, B, C, D, E, Y OR PSA_cdc has 6 nodes: Age, Aspirin, cancer, Statin, BMI, PSA
            0: 'Age',    # PSA_cdc: Age OR CompleteGraph: A
            1: 'Aspirin', # PSA_cdc: Aspirin OR CompleteGraph: B  
            2: 'cancer',  # PSA_cdc: cancer OR CompleteGraph: C
            3: 'Statin',  # PSA_cdc: Statin OR CompleteGraph: D
            4: 'BMI',     # PSA_cdc: BMI OR CompleteGraph: E
            5: 'PSA'      # PSA_cdc: PSA OR CompleteGraph: Y
        },
        9: {  # Diabetes has 9 nodes: DiabetesPedigreeFunction, Age, Pregnancies, BloodPressure, SkinThickness, Insulin, BMI, Glucose, Outcome
            0: 'DiabetesPedigreeFunction',
            1: 'Age',
            2: 'Pregnancies',
            3: 'BloodPressure',
            4: 'SkinThickness',
            5: 'Insulin',
            6: 'BMI',
            7: 'Glucose',
            8: 'Outcome'
        }
    }
    
    n_nodes = env_profile["dag"].get_n_nodes()
    name_map = intervention_name_maps.get(n_nodes, {})
    
    # Get all possible intervention targets
    intervention_names = []
    for target in env_profile["valid_targets"]:
        for i, is_target in enumerate(target):
            if is_target == 1:
                var_name = name_map.get(i, f'X{i}')
                if var_name not in intervention_names:
                    intervention_names.append(var_name)
    
    return intervention_names


def extract_intervention_values(x, env_profile):
    """
    Extract intervention values from action tensor for CSV logging.
    """
    if not env_profile["interventional"]:
        return {}
    
    x_np = x.detach().cpu().numpy().flatten()
    n_nodes = env_profile["dag"].get_n_nodes()
    
    # Extract binary intervention indicators and values
    intervention_indicators = x_np[:n_nodes]  # Binary: whether to intervene
    intervention_values = x_np[n_nodes:]      # Values: what to intervene to
    
    # Map to intervention names
    intervention_names = get_intervention_names(env_profile)
    result = {}
    
    # Initialize all intervention names as empty
    for name in intervention_names:
        result[name] = ''
    
    # Map node indices to names for this environment  
    intervention_name_maps = {
        6: {2: 'Aspirin', 3: 'Statin'},  # PSAGraph
        3: {0: 'X0', 1: 'X1', 2: 'X2'},   # ToyGraph
        6: {1: 'B', 3: 'D', 4: 'E'},   # CompleteGraph
        8: {0: 'DiabetesPedigreeFunction', 1: 'Age', 2: 'Pregnancies', 3: 'BloodPressure', 4: 'SkinThickness'}   # Diabetes
    }
    
    n_nodes = env_profile["dag"].get_n_nodes()
    name_map = intervention_name_maps.get(n_nodes, {})
    
    # Fill in actual intervention values where they occur
    for i, (do_intervene, value) in enumerate(zip(intervention_indicators, intervention_values)):
        if do_intervene > 0.5:  # Binary indicator says to intervene
            var_name = name_map.get(i, f'X{i}')
            if var_name in result:
                result[var_name] = value
    
    return result
