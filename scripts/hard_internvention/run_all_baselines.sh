#!/bin/sh
# Usage: sh scripts/hard_internvention/run_all_baselines.sh [num_trials] [seed] [dataset1 dataset2 ...]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

NUM_TRIALS=${1:-100}
SEED=${2:-123}
if [ "$#" -gt 2 ]; then
    shift 2
    DATASETS="$*"
else
    DATASETS="$DEFAULT_DATASETS"
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
run_group CBO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_bo_cbo.sh" CBO "$NUM_TRIALS" "$SEED" $DATASETS
run_group CEO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_ceo.sh" "$NUM_TRIALS" "$SEED" $DATASETS
run_group CoCaBO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_cocabo.sh" "$NUM_TRIALS" "$SEED" $DATASETS
run_group DCBO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_dcbo.sh" "$NUM_TRIALS" "$SEED" $DATASETS
run_group MCBO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_mcbo.sh" "$NUM_TRIALS" "$SEED" $DATASETS
run_group HCBO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_hcbo.sh" test $DATASETS
run_group CCBO sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_ccbo.sh" "$SEED" "$NUM_TRIALS" $DATASETS
run_group ACBO sh "$BENCHMARK_ROOT/scripts/soft_intervention/run_acbo.sh" "$NUM_TRIALS" "$SEED" $DATASETS

if [ -n "$failed" ]; then
    echo "Some baseline groups failed or are unsupported:$failed"
    exit 1
fi
echo "All baseline groups completed successfully."
