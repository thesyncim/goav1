#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 REPO_ROOT OUTDIR" >&2
  exit 2
fi

REPO_ROOT=$(cd "$1" && pwd -P)
mkdir -p "$2"
OUTDIR=$(cd "$2" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
FIXTURE_DIR="$OUTDIR/fixtures"
OVERLAY="$OUTDIR/go-overlay.json"
VIRTUAL_TEST="$REPO_ROOT/internal/av1/motion/hbd_motion_bench_fixture_export_overlay_test.go"
mkdir -p "$FIXTURE_DIR"

python3 - "$VIRTUAL_TEST" "$SCRIPT_DIR/export_fixtures_test.go.in" "$OVERLAY" <<'PY'
import json
import sys
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump({"Replace": {sys.argv[1]: sys.argv[2]}}, f)
PY

PROJECT_ENV=${PROJECT_ENV:-}
GOTOOLCHAIN=${GOTOOLCHAIN:-auto}
GOEXPERIMENT=${GOEXPERIMENT:-simd}
export GOTOOLCHAIN GOEXPERIMENT

cd "$REPO_ROOT"
if [ -n "$PROJECT_ENV" ]; then
  GOAV1_HBD_FIXTURE_DIR="$FIXTURE_DIR" "$PROJECT_ENV" go test \
    -overlay="$OVERLAY" \
    ./internal/av1/motion \
    -run '^TestHBDMotionBenchExportFixtures$' -count=1
else
  GOAV1_HBD_FIXTURE_DIR="$FIXTURE_DIR" go test \
    -overlay="$OVERLAY" \
    ./internal/av1/motion \
    -run '^TestHBDMotionBenchExportFixtures$' -count=1
fi
