# Config for protein experiment using a JSON numeric spec.
# Mirrors config_ecology.py but loads the protein spec and uses the
# CBO_Benchmark protein settings (target=Erk, task=min, n_obs=250).

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  # Use the numeric_large pathway to load from spec
  config.example_name = 'numeric_large'
  config.results_label = 'protein'

  # Trials and data sizes (match CBO_Benchmark/configs/protein.yaml).
  # Note: ccbo's run_optimization writes n_trials rows total, where the first
  # row is a sentinel (`inf`) for the "before any intervention" state. We
  # request 101 here so that after the exporter drops the sentinel we land at
  # 100 real-iteration rows -- matching the BO/CBO 100-trial progress format.
  config.n_trials = 101
  config.n_samples_obs = 250
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

  # The runner derives exploration sets from spec.intervention +
  # spec.max_intervention_set_size when example_name == 'numeric_large'.
  config.exploration_sets = tuple()

  # No constraints for the protein benchmark
  config.constraints = ml_collections.ConfigDict()
  config.constraints.variables = tuple()
  config.constraints.lambdas = tuple()

  # Path to the protein JSON spec (relative to project root or absolute)
  config.spec_path = 'protein_spec.json'

  return config
