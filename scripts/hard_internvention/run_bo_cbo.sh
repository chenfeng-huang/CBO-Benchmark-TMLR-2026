#!/bin/sh
# Usage: sh scripts/hard_internvention/run_bo_cbo.sh [BO|CBO|both] [num_trials] [seed] [dataset1 dataset2 ...]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

ALGORITHM=${1:-CBO}
NUM_TRIALS=${2:-100}
SEED=${3:-123}
if [ "$#" -gt 3 ]; then
    shift 3
    DATASETS="$*"
else
    DATASETS="$DEFAULT_DATASETS"
fi

case "$ALGORITHM" in
    BO|CBO|both) ;;
    *)
        echo "Invalid algorithm: $ALGORITHM (expected BO, CBO, or both)" >&2
        exit 2
        ;;
esac

failed=""
for ds in $DATASETS; do
    if ! run_logged "$ALGORITHM" "$ds" "$NUM_TRIALS" \
        python -u scripts/evaluation/run_cbo.py --algorithm "$ALGORITHM" --dataset "$ds" \
            --num_trials "$NUM_TRIALS" --causal_prior --seed "$SEED"; then
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "$ALGORITHM finished with failures:$failed"
    exit 1
fi
echo "All $ALGORITHM runs completed successfully."
