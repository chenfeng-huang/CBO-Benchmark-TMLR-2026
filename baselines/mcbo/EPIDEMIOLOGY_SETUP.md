# Epidemiology Experiment Setup

## Overview

This document describes the setup for the Epidemiology causal environment in the MCBO (Multi-objective Causal Bayesian Optimization) framework.

## Causal Model Specification

### Variables and Relationships

The epidemiology model includes 5 variables with the following causal structure:

**Exogenous Variables:**
- `B` ~ Uniform(-1, 1) - Baseline factor
- `T` ~ Uniform(4, 8) - Treatment factor

**Endogenous Variables:**
- `L` = exp(0.5 * T + U) where U ~ N(0,1) - Latent factor
- `R` = 4 + L * T (deterministic) - Response factor  
- `Y` = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε where ε ~ N(0,1) - **Target variable**

### Causal Order
B → T → L → R → Y

### Interventional Setup

**Target:** Y (minimize)
**Task:** min
**Number of interventions:** 1
**Number of observations:** 200

**Interventional Variables:**
- L: Can be intervened upon
- B: Can be intervened upon

**Interventional Domains:**
- L: [0.479, 801.787]
- B: [-0.994, 0.999]

## Implementation Details

### File Structure

```
mcbo/
├── scripts/
│   ├── functions.py          # Contains Epidemiology class
│   └── runner.py            # Updated to include epidemiology
├── run_epidemiology_example.py  # Python example script
├── run_epidemiology.sh         # Shell script for quick execution
└── EPIDEMIOLOGY_SETUP.md       # This documentation
```

### Epidemiology Class

The `Epidemiology` class in `scripts/functions.py` implements:

- **Node ordering:** B(0), T(1), L(2), R(3), Y(4)
- **DAG structure:** Proper parent-child relationships
- **Intervention mapping:** Maps [0,1] inputs to intervention domains
- **Evaluation function:** Computes all variables following causal order
- **Noise handling:** Proper noise distribution for stochastic variables

### Key Features

1. **Proper Causal Structure:** Follows the specified DAG with correct dependencies
2. **Intervention Support:** Only allows interventions on L and B as specified
3. **Domain Mapping:** Correctly maps optimization inputs to intervention ranges
4. **Noise Modeling:** Includes appropriate noise for L and Y variables
5. **Target Optimization:** Negates Y for maximization (since task is minimize)

## Usage

### Quick Start

**Option 1: Shell Script**
```bash
./run_epidemiology.sh
```

**Option 2: Python Script**
```bash
python run_epidemiology_example.py
```

**Option 3: Direct Runner**
```bash
python scripts/runner.py -e epidemiology -a MCBO -s 42
```

### Command Line Arguments

- `-e epidemiology`: Use the epidemiology environment
- `-a MCBO`: Algorithm (MCBO, Random, UCB, EI, etc.)
- `-s 42`: Random seed
- `-b 10.0`: Beta parameter for exploration
- `-n 1.0`: Noise scale
- `--initial_obs_samples 20`: Initial observational samples
- `--initial_int_samples 5`: Initial interventional samples per node

### Example Experiments

**Single Experiment:**
```bash
python scripts/runner.py -e epidemiology -a MCBO -s 42 -b 10.0 -n 1.0
```

**Comparison Study:**
```bash
# MCBO
python scripts/runner.py -e epidemiology -a MCBO -s 42
python scripts/runner.py -e epidemiology -a MCBO -s 68
python scripts/runner.py -e epidemiology -a MCBO -s 123

# Random baseline
python scripts/runner.py -e epidemiology -a Random -s 42
python scripts/runner.py -e epidemiology -a Random -s 68
python scripts/runner.py -e epidemiology -a Random -s 123
```

## Expected Output

### Results Files
- `trial_results_MCBO_epidemiology_42.csv`: Detailed results for MCBO with seed 42
- Similar files for other algorithm/seed combinations

### Performance Metrics
- **Regret:** Distance from theoretical optimum
- **Best Value:** Best Y value found (remember: minimization task)
- **Convergence:** How quickly the algorithm finds good solutions

## Validation

The setup has been validated with:

✅ **Environment Creation:** Successfully creates Epidemiology instance  
✅ **Dimension Check:** Input dimension = 10 (5 nodes × 2 for interventions)  
✅ **Evaluation Test:** Properly evaluates random inputs  
✅ **Target Output:** Returns appropriate Y values (negated for maximization)  
✅ **Integration:** Works with existing MCBO framework  

## Technical Notes

### Intervention Encoding
- Each variable has 2 dimensions: [intervention_flag, intervention_value]
- Total input dimension: 5 variables × 2 = 10 dimensions
- Only L and B can have intervention_flag = 1

### Optimization Direction
- Original task: minimize Y
- Framework expects: maximize objective
- Solution: Return -Y as the objective value

### Noise Handling
- L has noise from N(0,1) in the exponential
- Y has additive noise from N(0,1)
- B, T are exogenous (sampled fresh each evaluation)
- R is deterministic

## Troubleshooting

**Common Issues:**

1. **Import Error:** Ensure you're in the mcbo directory
2. **Module Not Found:** Check that scripts/ is in Python path
3. **Dimension Mismatch:** Verify input has 10 dimensions
4. **Convergence Issues:** Try different beta values or initial samples

**Debug Commands:**
```python
# Test environment
from scripts.functions import Epidemiology
env = Epidemiology()
print(f"Input dim: {env.input_dim}")
print(f"Valid targets: {env.valid_targets}")
```

## Next Steps

1. **Run Experiments:** Execute the provided scripts
2. **Analyze Results:** Compare MCBO vs Random performance
3. **Parameter Tuning:** Experiment with different beta values
4. **Extended Studies:** Try different seeds and noise levels

For questions or issues, refer to the main MCBO documentation or examine the existing environment implementations in `scripts/functions.py`.





