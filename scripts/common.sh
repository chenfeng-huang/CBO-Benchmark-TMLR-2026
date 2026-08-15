#!/bin/sh
# Shared helpers for CBO_Benchmark shell entry points.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$(basename "$SCRIPT_DIR")" in
    hard_internvention|soft_intervention|evaluation)
        SCRIPTS_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
        ;;
    *)
        SCRIPTS_ROOT="$SCRIPT_DIR"
        ;;
esac
BENCHMARK_ROOT=$(CDPATH= cd -- "$SCRIPTS_ROOT/.." && pwd)
BASELINES_ROOT="$BENCHMARK_ROOT/baselines"
DEFAULT_DATASETS="chain ecology epidemiology healthcare protein synthetic_2 synthetic toyGraph"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

cd_benchmark_root() {
    cd "$BENCHMARK_ROOT" || exit 1
}

dataset_task() {
    python - "$BENCHMARK_ROOT/configs/$1.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r") as f:
    print(yaml.safe_load(f).get("task", "min"))
PY
}

run_logged() {
    method=$1
    dataset=$2
    trials=$3
    shift 3

    log_dir="$BENCHMARK_ROOT/results/logs/$method/$dataset"
    mkdir -p "$log_dir"
    log_file="$log_dir/${method}_${dataset}_trials${trials}_${RUN_TIMESTAMP}.log"
    status_file=$(mktemp "${TMPDIR:-/tmp}/cbo_${method}_${dataset}.XXXXXX") || exit 1

    echo "========================================"
    echo "$method on dataset=$dataset  trials=$trials"
    echo "Time: $(date)"
    echo "Log file: $log_file"
    echo "Command: $*"
    echo "========================================"

    (
        "$@" 2>&1
        echo $? > "$status_file"
    ) | tee "$log_file"

    rc=$(cat "$status_file" 2>/dev/null || echo 1)
    rm -f "$status_file"
    if [ "$rc" -ne 0 ]; then
        echo "[$method] dataset=$dataset FAILED (exit $rc)" | tee -a "$log_file"
        return "$rc"
    fi
    echo "[$method] dataset=$dataset OK" | tee -a "$log_file"
    return 0
}

unsupported_dataset() {
    method=$1
    dataset=$2
    reason=$3
    echo "[$method] dataset=$dataset unsupported: $reason"
}

