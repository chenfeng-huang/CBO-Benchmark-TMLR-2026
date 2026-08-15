import os
import torch
import botorch
from botorch.acquisition.objective import GenericMCObjective
try:
    from botorch.settings import debug
except ImportError:
    debug = None
from torch import Tensor
import wandb
import argparse

import warnings

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
if debug is not None:
    debug._set_state(True)

from mcbo.mcbo_trial import mcbo_trial
from mcbo.utils import runner_utils

n_bo_iter = 100

def gen_env(env_name, noise_scale):
    from functions import Dropwave, Alpine2, Rosenbrock, ChainSoftPerturb, DropwavePenny, Alpine2Perturb, RosenbrockPerturb, AckleyPerturb, Ackley
    
    if env_name in ["Alpine-Penny"]:
        env = Alpine2(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["Dropwave-Perturb", "Dropwave"]:
        env = Dropwave(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["Dropwave-Penny"]:
        env = DropwavePenny(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["Alpine-Perturb", "Alpine2", "Alpine"]:
        env = Alpine2Perturb(noise_scales=noise_scale, discrete = wandb.config['discrete'])

    elif env_name in ["Rosenbrock-Penny"]:
        env = Rosenbrock(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["Rosenbrock-Perturb", "Rosenbrock"]:
        env = RosenbrockPerturb(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["Ackley-Penny"]:
        env = Ackley(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["Ackley-Perturb", "Ackley"]:
        env = AckleyPerturb(noise_scales=noise_scale, discrete = wandb.config['discrete'])
    elif env_name in ["ChainSoft-Perturb", "ChainSoft", "chain_soft"]:
        env = ChainSoftPerturb(noise_scales=noise_scale, discrete=wandb.config['discrete'])
    else:
        raise ValueError("Invalid environment specified")
    
    return env

def launch_exp(config_dict):
    env = gen_env(config_dict["env"], config_dict["noise_scale"])
    def function_network(X: Tensor, X_a: Tensor):
        return env.evaluate(X=X, X_a=X_a)

    # Function that maps the network output to the objective value
    network_to_objective_transform = lambda Y, X=None: Y[..., -1]
    network_to_objective_transform = GenericMCObjective(network_to_objective_transform)
    if config_dict['noise_scale'] > 0.001:
        batch_size = config_dict["batch_size"]
    else:
        batch_size = 2
    env_profile = env.get_env_profile()
    algo_profile = {
        "algo": config_dict["algo"],
        "output_label": config_dict.get("output_label", config_dict["algo"]),
        "seed": config_dict["seed"],
        "n_init_evals": 2 * (env_profile["input_dim"] + 1),
        "n_bo_iter": config_dict["num_trials"],
        "beta": config_dict["beta"],
        "initial_obs_samples": config_dict["initial_obs_samples"],
        "initial_int_samples": config_dict["initial_int_samples"],
        "batch_size": batch_size,
        "discrete": config_dict["discrete"],
        "eta": config_dict["eta"],
        "adversary_method": config_dict["adversary_method"],
        "adversary_jitter": config_dict["adversary_jitter"],
        "lr": config_dict["lr"],
        "bike_env": None,
    }

    mcbo_trial(
        algo_profile=algo_profile,
        env_profile=env_profile,
        function_network=function_network,
        network_to_objective_transform=network_to_objective_transform,
    )

    # The env standardizes node outputs (target mapped to [0, 1]) for the
    # MW-style algorithms; rescale the exported best-so-far trajectory back to
    # the true objective scale.
    if getattr(env, "output_range_set", False):
        import csv as _csv
        output_label = config_dict.get("output_label") or config_dict["algo"]
        csv_path = f"trial_results_{output_label}_{config_dict['env']}_{config_dict['seed']}.csv"
        if os.path.exists(csv_path):
            lo = float(env.output_range_min[-1])
            hi = float(env.output_range_max[-1])
            with open(csv_path) as f:
                rows = list(_csv.DictReader(f))
            for row in rows:
                row["current_optimal"] = str(float(row["current_optimal"]) * (hi - lo) + lo)
            with open(csv_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=["trial_number", "current_optimal"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Rescaled {csv_path} from standardized [0, 1] to raw objective scale [{lo}, {hi}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-a", "--algo", type=str, default="Random", help="Which algorithm to use"
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float, 
        default=5.0,
        help="Value of beta in UCB algorithms",
    )
    parser.add_argument(
        "-e", "--env", type=str, default="Dropwave", help="Environment to evaluate on"
    )
    parser.add_argument(
        "-n",
        "--noise_scale",
        type=float,
        default=0.0,
        help="Fixed noise scale of every node",
    )
    parser.add_argument(
        "--scratch",
        type=bool,
        default=True,
        help="True if saving wandb logs to a scratch folder",
    )
    parser.add_argument(
        "--initial_obs_samples",
        type=int,
        default=10,
        choices=range(1, 100),
        help="If causal, number of observational samples.",
    )
    parser.add_argument(
        "--initial_int_samples",
        type=int,
        default=2,
        choices=range(1, 100),
        help="If causal, number of initial interventional samples per node.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, choices=range(1, 100), help="Batch size of NMCBO in noisy settings"
    )  

    parser.add_argument(
        "--discrete",
        type=int,
        default=4,
        choices=[-1, 2, 3, 4, 5, 6, 7, 8 ,9],
        help="Number of discrete values for each action node. -1 for continuous. Automatically varies both the environment and the algorithms used"
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Value of eta in MW algorithms",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate of any gradient-based algorithms",
    )

    parser.add_argument(
        "--adversary_method",
        type=str,
        default="Adversary",
        help="Method for adversary to choose actions. Random, Drift or Adversarial"
    )

    parser.add_argument(
        "--adversary_jitter",
        type=float,
        default=0.2,
        help="Probability of adversary choosing a random action"
    )

    parser.add_argument(
        "--group",
        type=str,
        default="env-test",
        help="Group name for wandb"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=n_bo_iter,
        help="Number of BO iterations to run",
    )
    parser.add_argument(
        "--output_label",
        type=str,
        default=None,
        help="Label to use in the raw trial-results CSV filename",
    )

    args = parser.parse_args()

    # if saving to scratch, tell wandb to save there
    WANDB_ENTITY = ""
    WANDB_GROUP = ""
    WANDB_PROJECT = ""
    if args.scratch:
        wandb.init(
            project=WANDB_PROJECT,
            dir=os.environ.get("SCRATCH"),
            entity= WANDB_ENTITY,
            group= WANDB_GROUP,
        )
    else:
        wandb.init(project= WANDB_PROJECT, entity= WANDB_ENTITY, group= WANDB_GROUP)

    wandb.config.update(args)
    config_dict = wandb.config
    launch_exp(config_dict)
