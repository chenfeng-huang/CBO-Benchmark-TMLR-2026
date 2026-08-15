#!/bin/sh
# Usage: sh scripts/hard_internvention/run_hcbo.sh [mode] [dataset1 dataset2 ...]
# Modes: test | try | full. HCBO is currently wired for protein only.

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

MODE=${1:-test}
if [ "$#" -gt 1 ]; then
    shift 1
    DATASETS="$*"
else
    DATASETS="$DEFAULT_DATASETS"
fi

case "$MODE" in
    test) MODE_FLAG="--is_test_mode"; EXP_NAME="test-ProteinGraph" ;;
    try) MODE_FLAG="--is_try_mode"; EXP_NAME="try-ProteinGraph" ;;
    full) MODE_FLAG=""; EXP_NAME="ProteinGraph" ;;
    *)
        echo "Unknown HCBO mode: $MODE (expected test, try, or full)" >&2
        exit 2
        ;;
esac

failed=""
for ds in $DATASETS; do
    if [ "$ds" != "protein" ]; then
        unsupported_dataset HCBO "$ds" "vendored HCBO wrapper currently exposes ProteinGraph only"
        failed="$failed $ds"
        continue
    fi

    if run_logged HCBO "$ds" 100 \
        sh -c 'cd "$1" && shift && "$@"' sh "$BASELINES_ROOT/HCBO" \
        sh -c '
            set -eu
            mkdir -p data/real_data/ProteinGraph
            python - "$1" <<PY
import pandas as pd
import sys
benchmark_root = sys.argv[1]
src = f"{benchmark_root}/observational_datasets/protein.csv"
dst = "data/real_data/ProteinGraph/true_observations.pkl"
mapping = {"PKC": "C", "PKA": "A", "Raf": "R", "Mek": "M", "Erk": "Y"}
df = pd.read_csv(src)[list(mapping.keys())].rename(columns=mapping)
df.to_pickle(dst)
print(f"wrote {dst}: shape={df.shape}")
PY
            rm -rf data/experiments/ProteinGraph
            python initialize_real.py ProteinGraph
            python run.py ProteinGraph --run_performance --HCBO_only --process_number 1 '"$MODE_FLAG"'
        ' sh "$BENCHMARK_ROOT"; then
        python "$BENCHMARK_ROOT/scripts/evaluation/export_progress.py" hcbo \
            --src "$BASELINES_ROOT/HCBO/result/$EXP_NAME/Baseline/Causal/HCBO" \
            --dataset protein --task min --num_trials 100 --replicate 0
    else
        failed="$failed $ds"
    fi
done

if [ -n "$failed" ]; then
    echo "HCBO finished with unsupported/failed datasets:$failed"
    exit 1
fi
echo "All HCBO runs completed successfully."
