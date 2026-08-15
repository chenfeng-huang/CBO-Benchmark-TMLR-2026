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

"""Config for Diabetes experiment."""

import ml_collections


def get_config():
  """Return the default configuration for Diabetes experiment."""
  config = ml_collections.ConfigDict()

  # Name associated with this SCM
  config.example_name = 'diabetes'

  config.n_trials = 100  # Number of trials to run.
  config.n_samples_obs = 250  # Number of initial observational data points (as specified).
  # Number of samples per interventional distribution.
  config.n_samples_per_intervention = 50

  # Number to sample to use to get the ground truth function
  config.n_samples_ground_truth = 100

  # Seed to use to sample the anchor points.
  config.seed_anchor_points = 1
  # Use a regular grid of points to evaluate the acquisition function
  # or sample points uniformly.
  config.sample_anchor_points = False

  # Number of points on a regular grid to evaluate the acquisition function.
  config.n_grid_points = 100

  # Learn or fix the likelihoood noise in the GP model.
  config.fix_likelihood_noise_var = True

  # Learn or fix the likelihoood noise in the GP model.
  config.noisy_acquisition = False

  # Intervention variables and levels based on Diabetes specification
  # Single variable interventions: Insulin and BloodPressure
  # The algorithm will explore different levels within the variable domains
  config.intervention_variables = (
      ('Insulin',), ('BloodPressure',)
  )  # Intervention variables.
  
  config.intervention_levels = (
      (0.,), (0.,)   # Initial intervention levels (algorithm will optimize these)
  )  # Intervention values.

  # No explicit constraints mentioned in the specification
  config.constraints = ml_collections.ConfigDict()
  config.constraints.variables = ()
  config.constraints.lambdas = ()  # No constraint values.

  # Exploration sets - unique intervention sets
  config.exploration_sets = (
      ('Insulin',), ('BloodPressure',)
  )

  # Sum the RBF kernel to the Monte Carlo one
  config.add_rbf_kernel = False

  # Whether to update the SCM at every iteration for G-MTGP
  config.update_scm = False

  # Use hp_prior in kernel
  config.use_hp_prior = True

  # Number of samples for the kernel computation
  config.n_kernel_samples = 10

  # Specify which model to run with possible values:
  # "cbo", "ccbo_single_task", "ccbo_single_task_causal_prior",
  # "ccbo_multi_task", "ccbo_multi_task_causal_prior", "ccbo_dag_multi_task"
  config.model_name = 'ccbo_single_task'

  return config


