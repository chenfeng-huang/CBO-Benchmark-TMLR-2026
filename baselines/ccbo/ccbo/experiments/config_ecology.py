# Config for ecology experiment using a JSON numeric spec

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  # Use the numeric_large pathway to load from spec
  config.example_name = 'numeric_large'
  config.results_label = 'ecology'

  # Trials and data sizes
  config.n_trials = 100
  config.n_samples_obs = 46
  config.n_samples_per_intervention = 1
  config.n_samples_ground_truth = 100

  # Acquisition grid / anchor points
  config.seed_anchor_points = 1
  config.sample_anchor_points = False
  config.n_grid_points = 100

  # GP settings
  config.fix_likelihood_noise_var = True
  config.noisy_acquisition = False
  config.add_rbf_kernel = False
  config.update_scm = False
  config.use_hp_prior = True
  config.n_kernel_samples = 10

  # Model and task
  config.model_name = 'ccbo_single_task'
  config.task = 'min'

  # The runner will compute exploration sets from the spec interventions
  config.exploration_sets = tuple()

  # Constraints empty
  config.constraints = ml_collections.ConfigDict()
  config.constraints.variables = tuple()
  config.constraints.lambdas = tuple()

  # Path to the ecology JSON spec (relative to project root or absolute)
  config.spec_path = 'ecology_spec.json'

  return config
