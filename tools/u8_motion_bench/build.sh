#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: DAV1D_STATIC_LIB=/path/to/libdav1d.a [DAV1D_SOURCE=/path/to/dav1d] $0 REPO_ROOT OUTDIR" >&2
  exit 2
fi
if [ -z "${DAV1D_STATIC_LIB:-}" ]; then
  echo "DAV1D_STATIC_LIB must point to the optimized static dav1d archive" >&2
  exit 2
fi

REPO_ROOT=$(cd "$1" && pwd -P)
mkdir -p "$2"
OUTDIR=$(cd "$2" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
DAV1D_STATIC_LIB=$(cd "$(dirname "$DAV1D_STATIC_LIB")" && pwd -P)/$(basename "$DAV1D_STATIC_LIB")
if [ ! -f "$DAV1D_STATIC_LIB" ]; then
  echo "static library not found: $DAV1D_STATIC_LIB" >&2
  exit 2
fi

EXPECTED_DAV1D_COMMIT=b546257f770768b2c88258c533da38b91a06f737
DAV1D_SOURCE=${DAV1D_SOURCE:-$REPO_ROOT/third_party/upstream/dav1d}
if [ ! -d "$DAV1D_SOURCE" ]; then
  echo "pinned dav1d source checkout not found; set DAV1D_SOURCE" >&2
  exit 2
fi
DAV1D_SOURCE=$(cd "$DAV1D_SOURCE" && pwd -P)
ACTUAL_DAV1D_COMMIT=$(git -C "$DAV1D_SOURCE" rev-parse HEAD)
if [ "$ACTUAL_DAV1D_COMMIT" != "$EXPECTED_DAV1D_COMMIT" ]; then
  echo "dav1d source commit mismatch: got $ACTUAL_DAV1D_COMMIT, expected $EXPECTED_DAV1D_COMMIT" >&2
  exit 2
fi

PROJECT_ENV=${PROJECT_ENV:-}
GOTOOLCHAIN=${GOTOOLCHAIN:-local}
GOEXPERIMENT=${GOEXPERIMENT:-simd}
export PROJECT_ENV GOTOOLCHAIN GOEXPERIMENT

"$SCRIPT_DIR/prepare_fixtures.sh" "$REPO_ROOT" "$OUTDIR"
clang -O3 -DNDEBUG -Wall -Wextra -Werror -std=c11 \
  "$SCRIPT_DIR/u8_motion_bench.c" "$DAV1D_STATIC_LIB" \
  -o "$OUTDIR/u8_motion_bench"
if [ -n "$PROJECT_ENV" ]; then
  GO_VERSION=$(cd "$REPO_ROOT" && "$PROJECT_ENV" go version)
  GO_CACHE=$(cd "$REPO_ROOT" && "$PROJECT_ENV" go env GOCACHE)
else
  GO_VERSION=$(cd "$REPO_ROOT" && go version)
  GO_CACHE=$(cd "$REPO_ROOT" && go env GOCACHE)
fi
{
  echo "Host: $(uname -a)"
  echo "Compiler: $(clang --version | head -n 2 | tr '\n' ' ')"
  echo "C flags: -O3 -DNDEBUG -Wall -Wextra -Werror -std=c11"
  echo "GOTOOLCHAIN: $GOTOOLCHAIN"
  echo "GOEXPERIMENT: $GOEXPERIMENT"
  echo "Go: $GO_VERSION"
  echo "Go build cache: $GO_CACHE"
  echo "Go repository root: $REPO_ROOT"
  echo "Go revision: $(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "Go motion package status:"
  git -C "$REPO_ROOT" status --short -- internal/av1/motion
  echo "Go tracked motion package diff SHA-256: $(git -C "$REPO_ROOT" diff --binary HEAD -- internal/av1/motion | shasum -a 256 | awk '{print $1}')"
  echo "Go I8MM source: $REPO_ROOT/internal/av1/motion/convolve_i8mm_arm64.go"
  echo "Go I8MM source SHA-256: $(shasum -a 256 "$REPO_ROOT/internal/av1/motion/convolve_i8mm_arm64.go" | awk '{print $1}')"
  echo "Go I8MM assembly: $REPO_ROOT/internal/av1/motion/convolve_i8mm_arm64.s"
  echo "Go I8MM assembly SHA-256: $(shasum -a 256 "$REPO_ROOT/internal/av1/motion/convolve_i8mm_arm64.s" | awk '{print $1}')"
  echo "fixture test template SHA-256: $(shasum -a 256 "$SCRIPT_DIR/export_fixtures_test.go.in" | awk '{print $1}')"
  echo "dav1d source checkout: $DAV1D_SOURCE"
  echo "dav1d source revision: $ACTUAL_DAV1D_COMMIT"
  echo "dav1d static archive: $DAV1D_STATIC_LIB"
  echo "direct C symbols: dav1d_put_8tap_regular_smooth_8bpc_neon_i8mm; dav1d_prep_8tap_regular_smooth_8bpc_neon_i8mm"
  echo "static library SHA-256: $(shasum -a 256 "$DAV1D_STATIC_LIB" | awk '{print $1}')"
  echo "harness SHA-256: $(shasum -a 256 "$OUTDIR/u8_motion_bench" | awk '{print $1}')"
} > "$OUTDIR/build_metadata.txt"

"$OUTDIR/u8_motion_bench" --verify --fixtures "$OUTDIR/fixtures"
