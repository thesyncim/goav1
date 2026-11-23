#!/usr/bin/env sh
# Merge fresh CPU profiles into the PGO profile the CLIs and benchmarks build
# with (cmd/qualitybench/default.pgo).
#
# usage: scripts/regen_pgo.sh [-o out.pgo] [-e clip.yuv:WIDTHxHEIGHT] stream.ivf...
#
# Every IVF is decoded once on one worker by cmd/aom-go-dec. With -e, the raw
# 8-bit I420 clip is encoded by cmd/encbench on one thread at 0.5, 1 and 2
# Mbps. The new profiles are merged into the existing one, so stages these
# runs do not exercise keep their samples; a profile rebuilt from scratch
# measured worse. Without -o the committed profile is rewritten in place.
#
# A/B the result before committing it: build cmd/aom-go-dec and cmd/encbench
# with -pgo=<old> and -pgo=<new> and compare instructions and cycles.
set -eu

usage() {
	echo "usage: scripts/regen_pgo.sh [-o out.pgo] [-e clip.yuv:WIDTHxHEIGHT] stream.ivf..." >&2
	exit 2
}

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
base="$root/cmd/qualitybench/default.pgo"
out=$base
enc=
while getopts o:e: opt; do
	case $opt in
	o) out=$OPTARG ;;
	e) enc=$OPTARG ;;
	*) usage ;;
	esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || [ -n "$enc" ] || usage

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GOEXPERIMENT=simd

if [ $# -gt 0 ]; then
	(cd "$root" && go build -pgo="$base" -o "$tmp/dec" ./cmd/aom-go-dec)
	i=0
	for ivf in "$@"; do
		i=$((i + 1))
		"$tmp/dec" -quiet -workers 1 -cpuprofile "$tmp/dec$i.pprof" -o /dev/null "$ivf"
	done
fi

if [ -n "$enc" ]; then
	clip=${enc%:*}
	size=${enc##*:}
	width=${size%x*}
	height=${size#*x}
	[ "$clip" != "$enc" ] && [ "$width" != "$size" ] || usage
	bytes=$(wc -c <"$clip")
	frames=$((bytes / (width * height * 3 / 2)))
	[ "$frames" -gt 0 ] || { echo "regen_pgo: $clip holds no ${width}x${height} frame" >&2; exit 1; }
	(cd "$root" && go build -pgo="$base" -o "$tmp/enc" ./cmd/encbench)
	for rate in 500000 1000000 2000000; do
		GOMAXPROCS=1 "$tmp/enc" -input "$clip" -frames "$frames" -width "$width" -height "$height" \
			-fps 30 -bitrate "$rate" -nopsnr -cpuprofile "$tmp/enc$rate.pprof" -o /dev/null >/dev/null
	done
fi

(cd "$root" && go tool pprof -proto "$base" "$tmp"/*.pprof >"$tmp/merged.pgo")
mv "$tmp/merged.pgo" "$out"
echo "regen_pgo: wrote $out ($(wc -c <"$out" | tr -d ' ') bytes)"
