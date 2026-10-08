#!/usr/bin/env bash
#
# simdtip.sh — run the selected official Go toolchain with arm64 SIMD enabled.
# The historical filename remains for convenience. Go 1.27.0 or newer is
# required; select the executable with GO=/path/to/go or use PATH's `go`.
# See SIMD_PORT.md for the current toolchain and validation workflow.
#
# Usage:
#     scripts/simdtip.sh build ./...
#     scripts/simdtip.sh test ./internal/av1/dsp/ -run TestAddResidualSIMD
#     scripts/simdtip.sh test -bench BenchmarkAddResidual ./internal/av1/dsp/
#     scripts/simdtip.sh conformance          # strict-MD5 dryrun with SIMD
#
set -euo pipefail

GO_BIN="${GO:-go}"
if ! command -v "$GO_BIN" >/dev/null 2>&1; then
	echo "Go executable '$GO_BIN' was not found; install official Go 1.27+ on PATH or set GO=/path/to/go" >&2
	exit 1
fi

# Preserve other experiment selections, but force simd on even if the caller
# supplied none, nosimd, or a contradictory simd/nosimd sequence. GOEXPERIMENT
# processes entries from left to right, so retain only entries after the last
# none reset and append simd last.
simd_experiments=""
simd_reset=0
simd_remaining="${GOEXPERIMENT:-}"
while [[ -n "$simd_remaining" ]]; do
	if [[ "$simd_remaining" == *,* ]]; then
		simd_token="${simd_remaining%%,*}"
		simd_remaining="${simd_remaining#*,}"
	else
		simd_token="$simd_remaining"
		simd_remaining=""
	fi
	case "$simd_token" in
		"") ;;
		none)
			simd_reset=1
			simd_experiments=""
			;;
		simd|nosimd) ;;
		*)
			if [[ -n "$simd_experiments" ]]; then
				simd_experiments="${simd_experiments},${simd_token}"
			else
				simd_experiments="$simd_token"
			fi
			;;
	esac
done

if (( simd_reset )); then
	GOEXPERIMENT=none
	if [[ -n "$simd_experiments" ]]; then
		GOEXPERIMENT+=",${simd_experiments}"
	fi
else
	GOEXPERIMENT="$simd_experiments"
fi
if [[ -n "$GOEXPERIMENT" ]]; then
	GOEXPERIMENT+=",simd"
else
	GOEXPERIMENT=simd
fi
export GOEXPERIMENT

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
	echo "usage: scripts/simdtip.sh {build|test|go|conformance|conformance-extended} [go arguments...]" >&2
	exit 2
fi
shift
case "$cmd" in
	conformance)
		GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 \
			"$GO_BIN" test -tags goav1_oracle ./internal/av1/testvector \
			-run 'TestLibaomFastFrameWorkDryRun' -count=1 -timeout 600s "$@"
		;;
	conformance-extended)
		GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 \
			"$GO_BIN" test -tags goav1_oracle ./internal/av1/testvector \
			-run 'TestLibaomExtendedFrameWorkDryRun' -count=1 -timeout 1800s "$@"
		;;
	go)
		exec "$GO_BIN" "$@"
		;;
	*)
		exec "$GO_BIN" "$cmd" "$@"
		;;
esac
