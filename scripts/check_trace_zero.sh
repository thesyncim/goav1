#!/usr/bin/env sh
set -eu

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

release_bin="$tmp/entropy-release.test"
trace_bin="$tmp/entropy-trace.test"
tile_release_bin="$tmp/tile-release.test"
tile_rng_trace_bin="$tmp/tile-rng-trace.test"
tile_coeff_trace_bin="$tmp/tile-coeff-trace.test"
reader_symbols='github.com/thesyncim/goav1/internal/av1/entropy\.\(\*Reader\)\.(ReadBit|ReadBoolQ15|ReadBits|ReadSymbol|ReadCDF|ReadCDFTrusted|ReadCDF3Trusted|ReadCDF4Trusted|ReadBinaryCDFTrusted|readSymbolTrusted)$'
tile_symbols='github.com/thesyncim/goav1/internal/av1/tile\.\(\*DecodeState\)\.ReadCoefficientsTXB$|github.com/thesyncim/goav1/internal/av1/tile\.decodeBlockLoopVisitWithCoeffController|github.com/thesyncim/goav1/internal/av1/tile\.\(\*BlockModeContext\)\.BuildReferenceMVStack$'

go test -c -o "$release_bin" ./internal/av1/entropy

if go tool nm "$release_bin" | grep -E 'trace(CDF|Bool)Read|Trace(Label|SetFrame|ResetSeq)' >/dev/null; then
	echo "release binary retained entropy trace symbols" >&2
	exit 1
fi

release_dump="$tmp/release.objdump"
go tool objdump -s "$reader_symbols" "$release_bin" >"$release_dump"
if grep -E 'trace(CDF|Bool)Read|BitsRead' "$release_dump" >/dev/null; then
	echo "release entropy reader still prepares trace calls or tell state" >&2
	grep -E 'trace(CDF|Bool)Read|BitsRead' "$release_dump" >&2
	exit 1
fi

go test -c -tags goav1_trace_rng -o "$trace_bin" ./internal/av1/entropy
trace_dump="$tmp/trace.objdump"
go tool objdump -s "$reader_symbols" "$trace_bin" >"$trace_dump"
if ! grep -E 'trace(CDF|Bool)Read' "$trace_dump" >/dev/null; then
	echo "trace build did not retain entropy trace calls" >&2
	exit 1
fi

go test -c -o "$tile_release_bin" ./internal/av1/tile
if go tool nm "$tile_release_bin" | grep -E 'coeffTrace(Block|Coeff|CulLevel)|TraceLabel|debugReferenceMVStack|debugRefMV' >/dev/null; then
	echo "release tile binary retained trace symbols" >&2
	exit 1
fi

tile_release_dump="$tmp/tile-release.objdump"
go tool objdump -s "$tile_symbols" "$tile_release_bin" >"$tile_release_dump"
if grep -E 'coeffTrace(Block|Coeff|CulLevel)|TraceLabel|GOAV1_MARK|debugReferenceMVStack|debugRefMV' "$tile_release_dump" >/dev/null; then
	echo "release tile hot paths still prepare trace calls" >&2
	grep -E 'coeffTrace(Block|Coeff|CulLevel)|TraceLabel|GOAV1_MARK|debugReferenceMVStack|debugRefMV' "$tile_release_dump" >&2
	exit 1
fi

go test -c -tags goav1_trace_rng -o "$tile_rng_trace_bin" ./internal/av1/tile
tile_rng_trace_dump="$tmp/tile-rng-trace.objdump"
go tool objdump -s "$tile_symbols" "$tile_rng_trace_bin" >"$tile_rng_trace_dump"
if ! grep -E 'TraceLabel' "$tile_rng_trace_dump" >/dev/null; then
	echo "tile rng trace build did not retain block trace labels" >&2
	exit 1
fi

go test -c -tags goav1_coeff_trace -o "$tile_coeff_trace_bin" ./internal/av1/tile
tile_coeff_trace_dump="$tmp/tile-coeff-trace.objdump"
go tool objdump -s "$tile_symbols" "$tile_coeff_trace_bin" >"$tile_coeff_trace_dump"
if ! grep -E 'coeffTrace(Block|Coeff|CulLevel)' "$tile_coeff_trace_dump" >/dev/null; then
	echo "tile coeff trace build did not retain coefficient trace calls" >&2
	exit 1
fi

echo "trace hooks compile out of release reader and tile hot paths"
