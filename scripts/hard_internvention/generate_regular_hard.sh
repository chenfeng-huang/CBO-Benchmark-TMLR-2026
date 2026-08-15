#!/bin/sh
# Usage: sh scripts/hard_internvention/generate_regular_hard.sh [dataset]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

# The generators import baselines.BO_CBO.graph, which must resolve from the
# benchmark root regardless of where the script was invoked.
PYTHONPATH="$BENCHMARK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

DATASET=${1:-toyGraph}

# Datasets built on bundled real data (ecology, protein) have no observation
# generator: their observational CSVs, feature parameters and fitted SEM
# artifacts are fixed assets. Only their interventional data is regenerated.
OBS_SCRIPT="dataset_generators/generate_${DATASET}_observation.py"
if [ -f "$OBS_SCRIPT" ]; then
    python "$OBS_SCRIPT"
else
    echo "No observation generator for '${DATASET}': its observational data and SEM artifacts are pre-provided."
fi

python "dataset_generators/generate_${DATASET}_intervention.py"
