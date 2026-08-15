#!/bin/sh
# Usage: sh scripts/hard_internvention/run_mcbo.sh [num_trials] [seed] [dataset1 dataset2 ...]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

NUM_TRIALS=${1:-100}
SEED=${2:-42}
if [ "$#" -gt 2 ]; then
    shift 2
    DATASETS="$*"
else
    DATASETS="$DEFAULT_DATASETS"
fi

mcbo_env() {
    case "$1" in
        chain) echo chain ;;
        ecology) echo ecology ;;
        epidemiology) echo epidemiology ;;
        healthcare) echo PSAGraph ;;
        protein) echo protein ;;
        synthetic_2) echo synthetic_2 ;;
        synthetic) echo CompleteGraph ;;
        toyGraph) echo ToyGraph ;;
        *) return 1 ;;
    esac
}

failed=""
for ds in $DATASETS; do
    env_name=$(mcbo_env "$ds") || {
        unsupported_dataset MCBO "$ds" "no environment mapping"
        failed="$failed $ds"
        continue
    }
    task=$(dataset_task "$ds")
    src="$BASELINES_ROOT/mcbo/trial_results_MCBO_${env_name}_${SEED}.csv"
    if run_logged MCBO "$ds" "$NUM_TRIALS" \
        sh -c 'cd "$1" && shift && "$@"' sh "$BASELINES_ROOT/mcbo" \
        env WANDB_MODE=offline PYTHONPATH=. python -u scripts/runner.py \
            -a MCBO -s "$SEED" -e "$env_name" -n 0.0 \
            --batch_size 32 -b 10.0 --num_trials "$NUM_TRIALS"; then
        python "$BENCHMARK_ROOT/scripts/evaluation/export_progress.py" mcbo \
            --src "$src" --dataset "$ds" --task "$task" --num_trials "$NUM_TRIALS"
    else
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "MCBO finished with failures:$failed"
    exit 1
fi
echo "All MCBO runs completed successfully."
