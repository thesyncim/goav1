#!/usr/bin/env sh
set -eu

LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
allowlist="$repo_root/scripts/compiler_hotpath_escape_allowlist.txt"
tmp=${TMPDIR:-/tmp}/goav1-compiler-reports.$$
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT INT HUP TERM

raw="$tmp/compiler.txt"
escapes="$tmp/hot-escapes.txt"
bce="$tmp/hot-bce.txt"

packages="./internal/av1/bitstream ./internal/av1/entropy ./internal/av1/tile ./internal/av1/threading ./internal/av1/decoder ./internal/av1/encoder ./internal/av1/motion ./internal/av1/transform ./internal/av1/cdef ./internal/av1/prediction ./internal/av1/reconstruct ./internal/av1/loopfilter ./internal/av1/restoration ."

go build -gcflags='all=-m=2 -d=ssa/check_bce/debug=1' $packages >"$raw" 2>&1

awk '
/^# / {
	own = ($0 ~ /^# github.com\/thesyncim\/goav1(\/| |\[)/)
	next
}
own &&
$0 ~ /^(decode|decoder|encoder|rtp|internal\/av1\/(bitstream|entropy|tile|threading|decoder|encoder|motion|transform|cdef|prediction|reconstruct|loopfilter|restoration)\/)/ &&
$0 ~ /(make\(|new\(|func literal|moved to heap)/ &&
$0 ~ /(escapes to heap|moved to heap)/ {
	print
}
' "$raw" | sed -E 's/^([^:]+):[0-9]+:[0-9]+:/\1:*:*:/' | sort -u >"$escapes"

awk '
/^# / {
	own = ($0 ~ /^# github.com\/thesyncim\/goav1(\/| |\[)/)
	next
}
own &&
$0 ~ /^(decode|decoder|encoder|rtp|internal\/av1\/(bitstream|entropy|tile|threading|decoder|encoder|motion|transform|cdef|prediction|reconstruct|loopfilter|restoration)\/)/ &&
$0 ~ /Found (IsInBounds|IsSliceInBounds|IsSliceInBounds64)/ {
	print
}
' "$raw" | sort -u >"$bce"

if [ ! -f "$allowlist" ]; then
	echo "missing hot escape allowlist: $allowlist" >&2
	exit 1
fi

if ! diff -u "$allowlist" "$escapes"; then
	echo "new or changed hot-package heap escapes detected; update code or allowlist intentionally" >&2
	exit 1
fi

escape_count=$(wc -l <"$escapes" | tr -d ' ')
bce_count=$(wc -l <"$bce" | tr -d ' ')
printf 'compiler reports: %s hot heap escapes allowlisted; %s hot BCE sites reported\n' "$escape_count" "$bce_count"

if [ -n "${GOAV1_COMPILER_REPORT_DIR:-}" ]; then
	mkdir -p "$GOAV1_COMPILER_REPORT_DIR"
	cp "$raw" "$GOAV1_COMPILER_REPORT_DIR/compiler.txt"
	cp "$escapes" "$GOAV1_COMPILER_REPORT_DIR/hot-escapes.txt"
	cp "$bce" "$GOAV1_COMPILER_REPORT_DIR/hot-bce.txt"
fi
