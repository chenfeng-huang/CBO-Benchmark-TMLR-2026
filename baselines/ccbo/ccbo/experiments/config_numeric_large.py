# Copyright 2025

"""Config for numeric_large experiment built from user-provided JSON spec."""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  # Use special example name the runner will treat as numeric_large
  config.example_name = 'numeric_large'
  config.results_label = 'numeric_large'

  # Trials and data sizes
  config.n_trials = 100
  config.n_samples_obs = 300
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

  # Model
  config.model_name = 'ccbo_single_task'

  # Task
  config.task = 'max'

  # Placeholders; the runner will compute exploration_sets from interventions
  config.exploration_sets = tuple()

  # Constraints empty
  config.constraints = ml_collections.ConfigDict()
  config.constraints.variables = tuple()
  config.constraints.lambdas = tuple()

  return config


