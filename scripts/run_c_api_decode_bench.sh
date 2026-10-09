#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${GOAV1_C_API_BENCH_BUILD_DIR:-"${TMPDIR:-/tmp}/goav1-c-api-decode-bench-${CODEX_AGENT_ID:-local}"}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 aom|dav1d [--corpus DIR] [--clip NAME] [--validate-only] [--profile-repeat COUNT]" >&2
  exit 2
fi
backend=$1
shift
case "$backend" in
  aom|dav1d) ;;
  *)
    echo "backend must be aom or dav1d, got: $backend" >&2
    exit 2
    ;;
esac

binary="$build_dir/c-api-decode-bench-$backend"
if [[ ! -x "$binary" ||
      "$root/tools/c_api_decode_bench/bench.c" -nt "$binary" ||
      "$root/tools/c_api_decode_bench/build.sh" -nt "$binary" ]]; then
  bash "$root/tools/c_api_decode_bench/build.sh" >&2
fi
exec "$binary" "$@"
