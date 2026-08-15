#!/bin/sh
# Usage: sh scripts/evaluation/eval.sh [method] [dataset] [task]

. "$(dirname "$0")/../common.sh"
cd_benchmark_root

METHOD=${1:-BO}
DATASET=${2:-alpine2}
TASK=${3:-max}

python scripts/evaluation/calculate_gap_metrics.py --dataset "$DATASET" --method "$METHOD" --task "$TASK"
