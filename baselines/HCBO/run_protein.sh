#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
exec sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_hcbo.sh" "${1:-test}" protein
