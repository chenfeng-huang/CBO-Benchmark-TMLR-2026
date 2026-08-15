#!/bin/sh
# Usage: sh scripts/hard_internvention/run_ccbo.sh [seed] [num_trials] [dataset1 dataset2 ...]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

SEED=${1:-1}
NUM_TRIALS=${2:-100}
if [ "$#" -gt 2 ]; then
    shift 2
    DATASETS="$*"
else
    DATASETS="$DEFAULT_DATASETS"
fi

ccbo_config() {
    case "$1" in
        chain) echo "experiments/config_chain.py chain_ccbo_single_task" ;;
        ecology) echo "experiments/config_ecology.py ecology" ;;
        epidemiology) echo "experiments/config_epidemiology.py epidemiology" ;;
        healthcare) echo "experiments/config_health.py health" ;;
        protein) echo "experiments/config_protein.py protein" ;;
        synthetic_2) echo "experiments/config_synthetic2.py synthetic2" ;;
        synthetic) echo "experiments/config_completegraph.py completegraph" ;;
        toyGraph) echo "experiments/config_synthetic1.py synthetic1" ;;
        *) return 1 ;;
    esac
}

failed=""
for ds in $DATASETS; do
    set -- $(ccbo_config "$ds") || {
        unsupported_dataset CCBO "$ds" "no config mapping"
        failed="$failed $ds"
        continue
    }
    config=$1
    source_label=$2
    task=$(dataset_task "$ds")
    out="$BENCHMARK_ROOT/results/CCBO/$ds/CCBO_${ds}_${NUM_TRIALS}-trials_progress.csv"

    if run_logged CCBO "$ds" "$NUM_TRIALS" \
        sh -c 'cd "$1" && shift && "$@"' sh "$BASELINES_ROOT/ccbo/ccbo" \
        python -m ccbo.experiments.run_optimization --config="$config" --seed="$SEED"; then
        python "$BASELINES_ROOT/ccbo/scripts/export_progress.py" \
            --label "$source_label" --method CCBO --task "$task" \
            --num_trials "$NUM_TRIALS" --out "$out"
    else
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "CCBO finished with failures:$failed"
    exit 1
fi
echo "All CCBO runs completed successfully."
