# Causal Bayesian Optimization: <br>Foundations, Methods, and Applications (TMLR 2026)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

Official benchmark implementation accompanying **Causal Bayesian Optimization: Foundations, Methods, and Applications**, accepted to **Transactions on Machine Learning Research (TMLR) 2026**. The repository provides a benchmark covering hard- and soft-intervention settings, the GAP and Path-Aware GAP (PA-GAP) metrics, and seven CBO methods evaluated alongside a non-causal BO baseline under a common scoring protocol.



## Authors

**Chenfeng Huang**, Thuy T. Le, Zixuan Ma, Hien Tran

> Causal Bayesian Optimization (CBO) integrates causal inference with Bayesian optimization to enable sample-efficient intervention selection in systems governed by causal structure. This survey provides a comprehensive and systematic review of the CBO landscape, organizing the growing literature through a unified BO-loop perspective that reveals how causal assumptions shape four core components: intervention search spaces, surrogate construction, acquisition design, and decision policies. We organize methods along recurring design axes, including graph and system-knowledge assumptions, environmental assumptions, intervention representation, surrogate architecture, and decision rules, and we clarify conceptual and notational connections between CBO and adjacent fields, including causal bandits, Bayesian experimental design, safe optimization, policy search, and causal abstraction. To address the lack of standardized evaluation in the field, we introduce a reproducibility-oriented benchmark that covers hard- and soft-intervention settings, implements both the standard GAP metric and a new trajectory-aware Path-Aware GAP (PA-GAP) metric, and evaluates seven CBO methods alongside a non-causal BO baseline under a common scoring protocol.
>
> Within this benchmark-specified regime, where the benchmark SCM, intervention domains, and reference optima are fixed and known-graph methods receive the benchmark graph unless explicitly specified otherwise, our empirical study across thirteen datasets, three budget levels, and two metrics reveals that no single method dominates uniformly: rankings depend critically on the dataset, budget, metric, and method-specific use of causal information, and strong non-causal baselines remain competitive in several settings. We further add controlled graph-misspecification and omitted-variable stress tests, which show that rankings can change substantially when the learner-side causal information is perturbed. We therefore identify robustness to causal-assumption violations, scalable unknown-graph optimization, mixed intervention types, realistic cost models, tighter theoretical guarantees, and integration with modern representation learning and causal abstractions as open challenges for moving CBO from controlled benchmarks toward reliable deployment.

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional; CPU is supported)

### Setup

1. **Clone the repository**:

```bash
git clone https://github.com/chenfeng-huang/CBO-Benchmark-TMLR-2026
cd CBO-Benchmark-TMLR-2026
```

2. **Create a conda environment** (recommended):

```bash
conda create -n CBO-Benchmark python=3.10 -y
conda activate CBO-Benchmark
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Each baseline is designed to run in its own isolated environment to avoid dependency conflicts; check the corresponding folder under `baselines/` before running that method. The function-network wrappers additionally require PyTorch, since `scripts/evaluation/run_cbo.py` and the MCBO-style environments import `torch`.

## Methods

Seven CBO methods are evaluated against a non-causal BO baseline under the same trial budget, seed, and scoring protocol.

| Method | Key | Description |
|--------|-----|-------------|
| **BO** | `BO` | Non-causal Bayesian optimization baseline (no graph information) |
| **CBO** | `CBO` | Causal Bayesian optimization with observational priors over intervention sets |
| **CEO** | `CEO` | Causal entropy search over a posterior of candidate graphs |
| **CoCaBO** | `CoCaBO` | Contextual causal BO with context-dependent intervention policies |
| **DCBO** | `DCBO` | Dynamic CBO over temporally indexed SCMs |
| **cCBO** | `ccbo` | Constrained CBO with feasibility constraints on interventions |
| **MCBO** | `MCBO` | Mechanism-level CBO propagating uncertainty through the graph |
| **HCBO** | `HCBO` | High-dimensional CBO exploiting coverage structure (protein-only wrapper) |
| **ACBO** | `ACBO` | Adversarial / non-stationary CBO for soft-intervention networks |

## Benchmark Datasets

Thirteen datasets span hard-intervention SCMs and soft-intervention function networks. Configuration files under `configs/` define the target, manipulable variables, intervention domains, and task direction (`min` or `max`).

### Hard-intervention SCMs

| Dataset | Nodes | Manipulable | Task |
|---------|-------|-------------|------|
| **`toyGraph`** | 3 | `X`, `Z` | Min `Y` |
| **`synthetic`** | 7 (+2 latent) | `B`, `D`, `E` | Min `Y` |
| **`synthetic_2`** | 3 | `X`, `Z` | Min `Y` |
| **`chain`** | 4 | `W`, `Z` | Min `Y` |
| **`ecology`** | 11 | `N`, `O`, `C`, `T`, `D` | Max `Y` |
| **`protein`** | 8 | `PKC`, `PKA`, `Mek`, `Akt` | Min `Erk` |
| **`healthcare`** | 6 | `Aspirin`, `Statin` | Min `PSA` |
| **`epidemiology`** | 5 | `L`, `B` | Min `Y` |

### Soft-intervention function networks

| Dataset | Actions | Task |
|---------|---------|------|
| **`ackley`** | 6 | Max `Y` |
| **`rosenbrock_d5`** | 5 (`rosenbrock_d3` / `rosenbrock_d7` variants available) | Max `Y` |
| **`dropwave`** | 2 | Max `Y` |
| **`alpine2`** | 6 | Max `Y` |
| **`chain_soft`** | policy on `Z` + hard `W` | Min `Y` |

Reference optima used as the GAP / PA-GAP denominators are fixed in `observational_datasets/<DATASET>_theoretical_best.json`. 

## Repository Layout

| Path | Description |
|------|-------------|
| **`configs/`** | Dataset targets, intervention variables, domains, and task type |
| **`dataset_generators/`** | Observational and interventional data generation scripts (`ecology` / `protein` are interventional-only; their observational data is bundled) |
| **`observational_datasets/`** | Observational CSVs, feature parameters, and reference optima |
| **`interventional_datasets/`** | Interventional seed data consumed by the BO / CBO runner |
| **`sem/`** | SEM equation JSONs and SEM model pickle files (the benchmark's structural ground truth) |
| **`metrics/`** | Shared GAP and PA-GAP implementations |
| **`scripts/`** | Experiment and evaluation entry points (preferred interface) |
| **`baselines/`** | Vendored baseline implementations plus BO/CBO, CEO, and CoCaBO support code |

The shared helper `scripts/common.sh` moves commands to the benchmark root and writes method logs under `results/logs/<METHOD>/<DATASET>/`, so scripts can be run from anywhere inside the repository.

## Reproducing Paper Results

The main benchmark uses a budget of 100 interventional trials, with metrics also reported at 50 and 20 effective trial lengths. All methods run with released default hyperparameters; wrapper-level overrides are limited to trial budget, random seed, dataset mapping, objective direction, and trajectory export format.

| Stage | Key settings |
|-------|--------------|
| **Data** | Generators are seeded (`seed=42`); reference optima fixed per dataset |
| **Optimization** | `100` trials, seed `123` (hard) / `42` (soft), 20 random seeds for stress tests |
| **Evaluation** | GAP and PA-GAP at `100`, `50`, and `20` effective trial lengths |

```bash
# Full hard-intervention batch on the default dataset list
sh scripts/hard_internvention/run_all_baselines.sh 100 123

# Full soft-intervention batch on the default dataset list
sh scripts/soft_intervention/run_all_baselines.sh 100 42
```

## Usage

### 1) Generate hard-intervention data

```bash
sh scripts/hard_internvention/generate_regular_hard.sh toyGraph
```

The observation generators write `sem/<DATASET>_sem_model.pkl`; the intervention generators and the BO/CBO runner load that pickle later.

`ecology` and `protein` are built on bundled real data, so they have no observation generator: their observational CSVs, feature parameters, and fitted SEM equations are fixed assets shipped with the repository. Running the script on either dataset skips the observation step and regenerates only the interventional data.

### 2) Run hard-intervention experiments

```bash
# BO / CBO
sh scripts/hard_internvention/run_bo_cbo.sh CBO 100 123 toyGraph chain
sh scripts/hard_internvention/run_bo_cbo.sh both 100 123 protein

# CBO-family baselines
sh scripts/hard_internvention/run_ceo.sh 100 123 toyGraph
sh scripts/hard_internvention/run_cocabo.sh 100 123 toyGraph
sh scripts/hard_internvention/run_dcbo.sh 100 123 toyGraph
sh scripts/hard_internvention/run_mcbo.sh 100 42 toyGraph
sh scripts/hard_internvention/run_ccbo.sh 123 100 toyGraph
sh scripts/hard_internvention/run_hcbo.sh test protein   # HCBO wrapper is protein-only

# Full batch
sh scripts/hard_internvention/run_all_baselines.sh 100 123 toyGraph
```

If the dataset list is omitted, hard-intervention scripts use the default list in `scripts/common.sh`:

`chain ecology epidemiology healthcare protein synthetic_2 synthetic toyGraph`

### 3) Run soft-intervention / function-network experiments

```bash
# BO + ACBO + MCBO
sh scripts/soft_intervention/run_all_baselines.sh 100 42 ackley alpine2 chain_soft

# Individual soft-intervention baselines
sh scripts/soft_intervention/run_acbo.sh 100 42 ackley
sh scripts/soft_intervention/run_mcbo_fn.sh 100 42 ackley
sh scripts/hard_internvention/run_bo_cbo.sh BO 100 42 ackley
```

The soft-intervention default dataset list is:

`ackley rosenbrock_d5 dropwave alpine2 chain_soft`

### 4) Evaluate GAP / PA-GAP

```bash
sh scripts/evaluation/eval.sh CBO toyGraph min
sh scripts/evaluation/eval.sh BO alpine2 max
```

`scripts/evaluation/calculate_gap_metrics.py` uses the shared metric code in `metrics/`. It expects a 100-trial progress file and reports metrics for 100, 50, and 20 effective trial lengths.

### Script reference

| Script | Arguments | Description |
|--------|-----------|-------------|
| `generate_regular_hard.sh` | `[dataset]` | Regenerate observational + interventional artifacts (one dataset per call) |
| `run_bo_cbo.sh` | `[BO\|CBO\|both] [trials] [seed] [dataset ...]` | Run the BO / CBO runner |
| `run_ceo.sh`, `run_cocabo.sh`, `run_dcbo.sh`, `run_mcbo.sh` | `[trials] [seed] [dataset ...]` | Run a CBO-family baseline |
| `run_ccbo.sh` | `[seed] [trials] [dataset ...]` | Run cCBO (note the argument order) |
| `run_hcbo.sh` | `[mode] [dataset ...]` | Run HCBO (protein-only) |
| `run_acbo.sh`, `run_mcbo_fn.sh` | `[trials] [seed] [dataset ...]` | Run a soft-intervention baseline |
| `eval.sh` | `[method] [dataset] [task]` | Compute GAP / PA-GAP from a progress CSV |

## Results

Every method exports a common trajectory format, so scoring differences reflect algorithmic behavior rather than implementation artifacts. A budget of `B` interventions produces `B+1` rows: an initial best-so-far value followed by one entry per interventional trial.

```
results/
├── <METHOD>/<DATASET>/<METHOD>_<DATASET>_<trials>-trials_progress.csv
└── logs/<METHOD>/<DATASET>/          # Per-run stdout logs

eval_results/
├── <METHOD>/<DATASET>/gap_metrics_<METHOD>_<DATASET>.csv
└── <METHOD>/<DATASET>/detailed_pa_gap_<METHOD>_<DATASET>.csv
```

Metrics reported per method:

- **GAP**: normalized improvement from the initial value toward the reference optimum, measured at the final trial
- **PA-GAP**: trajectory-aware Path-Aware GAP, rewarding progress across the full optimization path rather than only the final discovery
- **Optimal trial**: the trial index at which the best value was first attained


## Citation

```bibtex
@article{huang2026cbo,
  title   = {Causal Bayesian Optimization: Foundations, Methods, and Applications},
  author  = {Huang, Chenfeng and Le, Thuy T. and Ma, Zixuan and Tran, Hien},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```
