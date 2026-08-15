#!/bin/sh
# Usage: sh scripts/hard_internvention/run_cocabo.sh [num_trials] [seed] [dataset1 dataset2 ...]

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
for ds in $DATASETS; do
    if ! run_logged CoCaBO "$ds" "$NUM_TRIALS" \
        python -u "$BASELINES_ROOT/CoCaBO/run_benchmark.py" \
            --dataset "$ds" --num_trials "$NUM_TRIALS" --seed "$SEED"; then
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "CoCaBO finished with failures:$failed"
    exit 1
fi
echo "All CoCaBO runs completed successfully."
