#!/usr/bin/env sh
set -eu

go test . ./internal/av1/bitstream ./internal/av1/ivf ./internal/av1/obu ./internal/av1/rtp ./internal/av1/entropy ./internal/av1/parser ./internal/av1/decoder ./internal/av1/frame ./internal/av1/memory ./internal/av1/dsp ./internal/av1/cdef ./internal/av1/prediction ./internal/av1/motion ./internal/av1/quantize ./internal/av1/transform ./internal/av1/reconstruct ./internal/av1/loopfilter ./internal/av1/restoration ./internal/av1/tile ./internal/av1/threading

# Keep the benchmark count high enough that one-time runtime first touches do
# not show up as fractional B/op noise; package tests above enforce exact
# steady-state allocation counts with testing.AllocsPerRun.
bench_time=${GOAV1_ALLOC_BENCHTIME:-200x}
bench_re='^BenchmarkDecode(PostFilteredProfileClip|SuperResInterProfileClip|SuperResInterHighBDProfileClip|SuperResRestorationProfileClip|FilmGrainProfileClip|FullVectorAllocs)$'
bench_out=$(GOMAXPROCS=1 GOGC=off go test . -run '^$' -bench="$bench_re" -benchmem -benchtime="$bench_time" -count=1)
printf '%s\n' "$bench_out"
printf '%s\n' "$bench_out" | awk '
BEGIN { count = 0; failed = 0 }
$1 ~ /^BenchmarkDecode/ {
	count++
	b = ""
	a = ""
	for (i = 1; i <= NF; i++) {
		if ($i == "B/op") {
			b = $(i - 1)
		}
		if ($i == "allocs/op") {
			a = $(i - 1)
		}
	}
	if (b == "" || a == "") {
		printf("missing allocation metrics for %s\n", $1) > "/dev/stderr"
		failed = 1
		next
	}
	if (b != 0 || a != 0) {
		printf("%s allocated: %s B/op %s allocs/op\n", $1, b, a) > "/dev/stderr"
		failed = 1
	}
}
END {
	if (count != 6) {
		printf("ran %d allocation benchmarks, want 6\n", count) > "/dev/stderr"
		failed = 1
	}
	exit failed
}'
