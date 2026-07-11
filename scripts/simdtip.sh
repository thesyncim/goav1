#!/usr/bin/env bash
#
# simdtip.sh — run the Go toolchain from tip (Go 1.27+) with the arm64 SIMD
# experiment enabled, for the Go-native-SIMD kernel port (see SIMD_PORT.md).
#
# The simd/archsimd package (arm64 NEON intrinsics) only exists at Go tip behind
# GOEXPERIMENT=simd. Install/refresh tip with:
#     go install golang.org/dl/gotip@latest && gotip download
#
# Usage:
#     scripts/simdtip.sh build ./...
#     scripts/simdtip.sh test ./internal/av1/dsp/ -run TestAddResidualSIMD
#     scripts/simdtip.sh test -bench BenchmarkAddResidual ./internal/av1/dsp/
#     scripts/simdtip.sh conformance          # strict-MD5 dryrun under gotip+simd
#
set -euo pipefail

GOTIP_ROOT="${GOTIP_ROOT:-$HOME/sdk/tsgo}"
if [[ ! -x "$GOTIP_ROOT/bin/go" ]]; then
	echo "gotip not found at $GOTIP_ROOT. Install: go install golang.org/dl/gotip@latest && gotip download" >&2
	exit 1
fi

# GOTOOLCHAIN=local pins this toolchain (no re-exec to the module's go line);
# a dedicated GOCACHE avoids version-skew errors against the 1.26 build cache.
export GOROOT="$GOTIP_ROOT"
export GOTOOLCHAIN=local
export GOCACHE="${GOAV1_SIMD_GOCACHE:-/tmp/goav1-simdcache}"
export GOEXPERIMENT=simd
GO="$GOTIP_ROOT/bin/go"

cmd="${1:-}"; shift || true
case "$cmd" in
	conformance)
		GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 \
			"$GO" test -tags goav1_oracle ./internal/av1/testvector \
			-run 'TestLibaomFastFrameWorkDryRun' -count=1 -timeout 600s "$@"
		;;
	conformance-extended)
		GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 \
			"$GO" test -tags goav1_oracle ./internal/av1/testvector \
			-run 'TestLibaomExtendedFrameWorkDryRun' -count=1 -timeout 1800s "$@"
		;;
	go)
		exec "$GO" "$@"
		;;
	*)
		exec "$GO" "$cmd" "$@"
		;;
esac
