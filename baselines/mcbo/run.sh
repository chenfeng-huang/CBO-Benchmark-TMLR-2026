#!/bin/sh
# Legacy wrapper. Prefer: sh scripts/hard_internvention/run_mcbo.sh ... from CBO_Benchmark.

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
exec sh "$BENCHMARK_ROOT/scripts/hard_internvention/run_mcbo.sh" "$@"