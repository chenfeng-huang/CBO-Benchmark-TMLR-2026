# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run experiment."""
from __future__ import annotations

import os
import random as py_random
from datetime import datetime

import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

from ccbo.experiments import data
from ccbo.methods import cbo
from ccbo.methods import ccbo
from ccbo.methods import random as random_method
from ccbo.utils import constraints_functions
from ccbo.utils import initialisation_utils
from ccbo.utils import sampling_utils
from ccbo.utils import scm_utils
from ccbo.utils import pa_gap_metrics
from ccbo.utils import utilities
from ccbo.experiments.data import build_scm_from_numeric_spec


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config",
                                "ccbo/experiments/config_synthetic1.py")

# Global random seed for reproducibility
flags.DEFINE_integer("seed", 1, "Global random seed.")

### FIXED PARAMETERS ###
# Sampling seed for the ground truth and the sampling of the target function
sampling_seed = 1
# Whether to use noisy observations of the target and the constraints (default).
# This may be overridden inside main() based on the config.
noisy_observations = True
# Produce plot and print statistics
verbose = False
# Number of restarts of GP optimization
n_restart = 5


def main(_):
  flags_config = FLAGS.config
  logging.info("Flags dict is %s", flags_config)

  # Set global seeds for reproducibility
  global sampling_seed
  sampling_seed = FLAGS.seed
  np.random.seed(sampling_seed)
  py_random.seed(sampling_seed)

  ### MISCELLANEOUS PREPARATION ###
  if flags_config.example_name == 'numeric_large':
    # Expect spec JSON to be provided via a file path in env or a default path
    # For simplicity, we read from ./numeric_large_spec.json relative to CWD
    import json
    spec_path = getattr(flags_config, 'spec_path', 'numeric_large_spec.json')
    with open(spec_path, 'r') as f:
      spec = json.load(f)
    scm = build_scm_from_numeric_spec(spec)
    example = None
  else:
    example = data.EXAMPLES_DICT[flags_config.example_name]()
    scm = example.structural_causal_model(**flags_config.constraints)
  graph = scm.graph
  constraints = scm.constraints

  (_, _, intervention_domain, all_ce,
   _, constraints_values, _,
   _, _,
   _) = scm.setup(
       n_grid_points=flags_config.n_grid_points,
       exploration_sets=list(flags_config.exploration_sets) if flags_config.example_name != 'numeric_large' else None,
       n_samples=flags_config.n_samples_ground_truth,
       sampling_seed=sampling_seed)

  # If numeric_large, override exploration sets and intervention domain from spec
  if flags_config.example_name == 'numeric_large':
    # Re-read the spec to fetch interventions and domains
    import json
    spec_path = getattr(flags_config, 'spec_path', 'numeric_large_spec.json')
    with open(spec_path, 'r') as f:
      spec = json.load(f)
    # Build exploration sets: singles and pairs up to max_intervention_set_size
    interventions = [tuple([str(x)]) for x in spec.get('intervention', [])]
    max_set_size = int(spec.get('max_intervention_set_size', 1))
    exploration_sets = set(interventions)
    if max_set_size >= 2:
      import itertools
      for a, b in itertools.combinations([str(x) for x in spec.get('intervention', [])], 2):
        exploration_sets.add(tuple(sorted((a, b))))
    flags_config.exploration_sets = tuple(sorted(exploration_sets))
    # Override domain to use provided interventional_domain for manipulative vars only
    intervention_domain = {str(k): list(map(float, v)) for k, v in spec.get('interventional_domain', {}).items()}
    # Update target if provided (use public attribute)
    if spec.get('target') is not None:
      scm.variables[str(spec['target'])] = ["t"]

  ### GENERATE INITIAL DATA ###

  # Generate observational data by sampling from the true
  # observational distribution
  d_o = sampling_utils.sample_scm(
      scm_funcs=scm.scm_funcs,
      graph=None,
      n_samples=flags_config.n_samples_obs,
      compute_moments=False,
      seed=sampling_seed)

  # Generate interventional data
  d_i = {k: None for k in flags_config.exploration_sets}

  # Optionally seed initial interventional data if provided in config
  cfg_vars = getattr(flags_config, 'intervention_variables', tuple())
  cfg_lvls = getattr(flags_config, 'intervention_levels', tuple())
  if cfg_vars and cfg_lvls:
    for var, level in zip(cfg_vars, cfg_lvls):
      initialisation_utils.assign_interventions(
          variables=var,
          levels=level,
          n_samples_per_intervention=flags_config.n_samples_per_intervention,
          sampling_seed=sampling_seed,
          d_i=d_i,
          graph=graph,
          scm_funcs=scm.scm_funcs)

  # If no initial interventional samples were seeded, pass None to the model
  # so it initializes without pre-existing interventional data.
  if isinstance(d_i, dict) and all(v is None for v in d_i.values()):
    d_i = None

  ### RUN THE ALGORITHM ###
  # Decide noise handling: if only 1 sample per intervention, use noiseless
  noisy_observations_local = getattr(
      flags_config, 'noisy_observations',
      bool(getattr(flags_config, 'n_samples_per_intervention', 1) and
           flags_config.n_samples_per_intervention > 1)
  )
  model_name = flags_config.model_name
  use_causal_prior = model_name in [
      "ccbo_single_task_causal_prior", "ccbo_dag_multi_task"
  ]
  is_multi_task = model_name in [
      "ccbo_multi_task", "ccbo_multi_task_causal_prior", "ccbo_dag_multi_task"
  ]
  use_prior_mean = model_name in ["ccbo_single_task_causal_prior",
                                  "ccbo_multi_task_causal_prior",
                                  "ccbo_dag_multi_task"]
  add_rbf_kernel = (
      flags_config.add_rbf_kernel and model_name in ["ccbo_dag_multi_task"])
  update_scm = flags_config.update_scm and model_name in ["ccbo_dag_multi_task"]

  # Setup input params
  input_params = {
      "graph": graph,
      "scm": scm,
      "make_scm_estimator": scm_utils.build_fitted_scm,
      "exploration_sets": list(flags_config.exploration_sets),
      "observation_samples": d_o,
      "intervention_samples": d_i,
      "intervention_domain": intervention_domain,
      "number_of_trials": flags_config.n_trials,
      # Honor task from config: 'min' or 'max'
      "task": utilities.Task.MAX if getattr(flags_config, 'task', 'min') == 'max' else utilities.Task.MIN,
      "sample_anchor_points": flags_config.sample_anchor_points,
      "seed_anchor_points": flags_config.seed_anchor_points,
      "num_anchor_points": flags_config.n_grid_points,
      "ground_truth": all_ce,
      "sampling_seed": sampling_seed,
      "n_restart": n_restart,
      "verbose": verbose,
      "causal_prior": use_causal_prior,
      "hp_prior": flags_config.use_hp_prior,
      # Noisy observations
      "noisy_observations": noisy_observations_local,
      "noisy_acquisition": flags_config.noisy_acquisition,
      "n_samples_per_intervention": flags_config.n_samples_per_intervention,
      "fix_likelihood_noise_var": flags_config.fix_likelihood_noise_var
  }

  if model_name == "cbo":
    model = cbo.CBO(**input_params)
  elif model_name == "random":
    model = random_method.Random(**input_params)
  else:
    # Add constraints
    input_params["ground_truth_constraints"] = constraints_values
    input_params["constraints"] = constraints
    input_params["multi_task_model"] = is_multi_task
    input_params["use_prior_mean"] = use_prior_mean
    input_params["update_scm"] = update_scm
    input_params["add_rbf_kernel"] = add_rbf_kernel

    if model_name == "ccbo_dag_multi_task":
      # Monte Carlo construction of the kernel
      input_params["n_kernel_samples"] = flags_config.n_kernel_samples

    model = ccbo.CCBO(**input_params)

  # Run method
  model.run()

  # If model is not constrained compute feasibility after running it
  if model_name in ["random", "cbo"]:
    (constraints_dict, _, _, _,
     _) = constraints_functions.get_constraints_dicts(
         flags_config.exploration_sets, constraints, graph,
         model.target_variable, d_o)
    for i, v in enumerate(model.best_es_over_trials_list):
      if len(v) > 1:
        value = model.best_level_over_trials_list[i].tolist()[0]
      else:
        value = model.best_level_over_trials_list[i]
      is_feasible, _, _, _ = constraints_functions.verify_feasibility(
          optimal_unconstrained_set=v,
          optimal_level=value,
          exploration_sets=flags_config.exploration_sets,
          all_ce=all_ce,
          constraints=constraints,
          constraints_dict=constraints_dict,
          scm_funcs=scm.scm_funcs,
          graph=graph,
          dict_variables=scm.variables,
          interventional_grids=model.interventional_grids,
          n_samples=flags_config.n_samples_ground_truth,
          sampling_seed=sampling_seed,
          task=model.task,
      )
      model.best_es_over_trials_list[i] = [v, int(is_feasible)]

  logging.info(
      "The optimal intervention set found by the algorithm is %s",
      model.best_es_over_trials_list[-1][0],
  )

  logging.info(
      "The optimal target effect value found by the algorithm is %s",
      model.optimal_outcome_values_during_trials[-1],
  )

  ### PA_GAP METRICS CALCULATION ###
  # Calculate theoretical best value from ground truth
  task_type = 'min' if getattr(model.task, 'value', None) == 'min' else 'max'
  theoretical_best = pa_gap_metrics.calculate_theoretical_best(all_ce, task_type)
  
  # Initialize PA_GAP calculator
  pa_gap_calculator = pa_gap_metrics.PA_GAP(best_theoretical=theoretical_best)
  
  # Calculate PA_GAP metrics
  pa_gap_results = pa_gap_calculator.calculate_PA_GAP(
      outcomes=model.optimal_outcome_values_during_trials,
      task=task_type
  )
  
  # Log PA_GAP results
  logging.info("PA_GAP Metrics:")
  logging.info("Average PA_GAP: %s", pa_gap_results['PA_GAP_mean'])
  logging.info("Final PA_GAP: %s", pa_gap_results['PA_GAP_final'])
  logging.info("Improvement Ratio: %s", pa_gap_results['improvement_ratio'])
  logging.info("Optimal found at trial: %s", pa_gap_results['optimal_trial'])
  
  # Print detailed summary
  pa_gap_calculator.print_summary()
  
  # Export PA_GAP data to DataFrame for further analysis
  pa_gap_df = pa_gap_calculator.export_to_dataframe()
  logging.info("PA_GAP trial-by-trial data exported to DataFrame with %d rows", len(pa_gap_df))
  
  ### SAVE RESULTS TO CSV ###
  # Create results directory if it doesn't exist
  results_dir = "experiment_results"
  os.makedirs(results_dir, exist_ok=True)
  
  # Create timestamp for unique file naming
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  
  # Calculate best-so-far values (cumulative best) using RAW outcomes
  # and collate current optimal values tracked by the algorithm
  task_type = 'min' if getattr(model.task, 'value', None) == 'min' else 'max'
  n_trials = len(model.optimal_outcome_values_during_trials)
  best_so_far_values = []
  current_best = model.outcome_values[0]
  
  for i in range(n_trials):
    value = model.outcome_values[i]
    if i == 0:
      best_so_far_values.append(value)
      current_best = value
    else:
      if task_type == 'min':
        current_best = min(current_best, value)
      else:
        current_best = max(current_best, value)
      best_so_far_values.append(current_best)
  
  # Create DataFrame with trial, best_so_far, y (raw per trial) and current_optimal
  simple_results_data = {
      'trial': list(range(n_trials)),
      'best_so_far': best_so_far_values,
      'y': model.outcome_values[:n_trials],
      'current_optimal': model.optimal_outcome_values_during_trials
  }
  
  simple_results_df = pd.DataFrame(simple_results_data)
  
  # Save single CSV file, allowing a custom results label if provided
  results_label = getattr(flags_config, 'results_label', flags_config.example_name)
  results_file = os.path.join(
      results_dir,
      f"{results_label}_{flags_config.model_name}_{timestamp}.csv",
  )
  simple_results_df.to_csv(results_file, index=False)
  
  logging.info("=== CSV EXPORT COMPLETE ===")
  logging.info("Results saved to: %s", results_file)
  logging.info("Columns: trial, best_so_far, y, current_optimal")
  logging.info("Total trials: %d", n_trials)

if __name__ == "__main__":
  app.run(main)
