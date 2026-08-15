#!/bin/sh
# Usage: sh scripts/hard_internvention/run_dcbo.sh [num_trials] [seed] [dataset1 dataset2 ...]

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

dcbo_setup() {
    case "$1" in
        chain) echo chain ;;
        ecology) echo ecology ;;
        epidemiology) echo epidemiology ;;
        healthcare) echo psa ;;
        protein) echo protein ;;
        synthetic_2) echo synthetic_2 ;;
        synthetic) echo complete ;;
        toyGraph) echo toygraph ;;
        *) return 1 ;;
    esac
}

failed=""
for ds in $DATASETS; do
    setup=$(dcbo_setup "$ds") || {
        unsupported_dataset DCBO "$ds" "no setup mapping"
        failed="$failed $ds"
        continue
    }
    task=$(dataset_task "$ds")
    src="$BASELINES_ROOT/DCBO/results/dcbo_best_so_far_seed${SEED}_T1_trials${NUM_TRIALS}_reps1.csv"
    if run_logged DCBO "$ds" "$NUM_TRIALS" \
        python -u "$BASELINES_ROOT/DCBO/run_dcbo.py" \
            --setup "$setup" --T 1 --trials "$NUM_TRIALS" --reps 1 \
            --seed "$SEED" --config "$BENCHMARK_ROOT/configs/$ds.yaml"; then
        python "$BENCHMARK_ROOT/scripts/evaluation/export_progress.py" dcbo \
            --src "$src" --dataset "$ds" --task "$task" --num_trials "$NUM_TRIALS"
    else
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "DCBO finished with failures:$failed"
    exit 1
fi
echo "All DCBO runs completed successfully."
