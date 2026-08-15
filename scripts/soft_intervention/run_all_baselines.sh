#!/bin/sh
# Usage: sh scripts/soft_intervention/run_bo_acbo_mcbo_fn.sh [num_trials] [seed] [dataset1 dataset2 ...]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

FN_DATASETS="ackley rosenbrock_d5 dropwave alpine2 chain_soft"
NUM_TRIALS=${1:-100}
SEED=${2:-42}
if [ "$#" -gt 2 ]; then
    shift 2
    DATASETS="$*"
else
    DATASETS="$FN_DATASETS"
fi

failed=""

run_group() {
    name=$1
    shift
    echo "================================================================"
    echo "Running $name"
    echo "================================================================"
    if ! "$@"; then
        failed="$failed $name"
    fi
}

run_group BO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_bo_cbo.sh" BO "$NUM_TRIALS" "$SEED" $DATASETS
run_group ACBO sh "$BENCHMARK_ROOT/scripts/soft_intervention/run_acbo.sh" "$NUM_TRIALS" "$SEED" $DATASETS
run_group MCBO sh "$BENCHMARK_ROOT/scripts/soft_intervention/run_mcbo_fn.sh" "$NUM_TRIALS" "$SEED" $DATASETS

if [ -n "$failed" ]; then
    echo "Some function-network experiment groups failed:$failed"
    exit 1
fi
echo "All BO/ACBO/MCBO function-network experiments completed successfully."
