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

"""Config for Chain example (X, W, Z, Y) per user specification.

Interventions:
- W, Z in [-1, 1]
Observations:
- n_samples_obs = 200
Target:
- Y (task: min)
"""

import ml_collections


def get_config():
  """Return the configuration for the chain example."""
  config = ml_collections.ConfigDict()

  # Example key to select the SCM
  config.example_name = 'chain'
  # Label used for naming result files; can differ from example_name
  config.results_label = 'chain_ccbo_single_task'

  # Trials and dataset sizes
  config.n_trials = 100  # Number of trials to run
  config.n_samples_obs = 200  # Number of initial observational data points
  # Number of samples per interventional distribution
  config.n_samples_per_intervention = 1

  # Number of samples to estimate ground truth for plotting/metrics
  config.n_samples_ground_truth = 100

  # Anchor points config for acquisition optimization
  config.seed_anchor_points = 1
  config.sample_anchor_points = False
  config.n_grid_points = 100

  # Noise/likelihood settings
  config.fix_likelihood_noise_var = True
  config.noisy_acquisition = False

  # No explicit constraints specified
  config.constraints = ml_collections.ConfigDict()
  config.constraints.variables = tuple()
  config.constraints.lambdas = tuple()

  # User-specified intervention setup for W and Z
  config.intervention_variables = (("W",), ("Z",))
  # Seed initial intervention levels (inside domains)
  config.intervention_levels = ((0.0,), (0.0,))

  # Exploration sets match the intervention variables
  config.exploration_sets = (("W",), ("Z",))

  # Kernel/hyperparameters options
  config.add_rbf_kernel = False
  config.update_scm = False
  config.use_hp_prior = True
  config.n_kernel_samples = 10

  # Choose model to run
  # Options: "cbo", "ccbo_single_task", "ccbo_single_task_causal_prior",
  #          "ccbo_multi_task", "ccbo_multi_task_causal_prior", "ccbo_dag_multi_task"
  config.model_name = 'ccbo_single_task'

  # Explicit task
  config.task = 'min'

  return config


