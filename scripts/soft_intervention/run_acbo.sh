#!/bin/sh
# Usage: sh scripts/soft_intervention/run_acbo.sh [num_trials] [seed] [dataset1 dataset2 ...]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

FN_DATASETS="ackley rosenbrock_d5 dropwave alpine2 chain_soft"
NUM_TRIALS=${1:-100}
SEED=${2:-123}
ACBO_DISCRETE=${ACBO_DISCRETE:-4}
if [ "$#" -gt 2 ]; then
    shift 2
    DATASETS="$*"
else
    DATASETS="$FN_DATASETS"
fi

acbo_env() {
    case "$1" in
        ackley) echo Ackley-Perturb ;;
        rosenbrock_d5) echo Rosenbrock-Perturb ;;
        dropwave) echo Dropwave-Perturb ;;
        alpine2) echo Alpine-Perturb ;;
        chain_soft) echo ChainSoft-Perturb ;;
        *) return 1 ;;
    esac
}

failed=""
for ds in $DATASETS; do
    env_name=$(acbo_env "$ds") || {
        unsupported_dataset ACBO "$ds" "no ACBO function-network mapping"
        failed="$failed $ds"
        continue
    }
    task=$(dataset_task "$ds")
    src="$BASELINES_ROOT/acbo/trial_results_ACBO_${env_name}_${SEED}.csv"
    if run_logged ACBO "$ds" "$NUM_TRIALS" \
        sh -c 'cd "$1" && shift && "$@"' sh "$BASELINES_ROOT/acbo" \
        env WANDB_MODE=offline PYTHONPATH=. python -u scripts/runner.py \
            -a CBO-MW -s "$SEED" -e "$env_name" -n 0.0 \
            --batch_size 32 -b 10.0 --num_trials "$NUM_TRIALS" \
            --discrete "$ACBO_DISCRETE" --output_label ACBO; then
        python "$BENCHMARK_ROOT/scripts/evaluation/export_progress.py" acbo \
            --src "$src" --dataset "$ds" --task "$task" --num_trials "$NUM_TRIALS"
    else
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "ACBO finished with failures:$failed"
    exit 1
fi
echo "All ACBO runs completed successfully."
