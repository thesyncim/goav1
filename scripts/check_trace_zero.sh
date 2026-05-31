#!/usr/bin/env sh
set -eu

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

release_bin="$tmp/entropy-release.test"
trace_bin="$tmp/entropy-trace.test"
reader_symbols='github.com/thesyncim/goav1/internal/av1/entropy\.\(\*Reader\)\.(ReadBit|ReadBoolQ15|ReadBits|ReadSymbol|ReadCDF|ReadCDFTrusted|ReadBinaryCDFTrusted|readSymbolTrusted)$'

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

echo "entropy trace hooks compile out of release reader hot paths"
