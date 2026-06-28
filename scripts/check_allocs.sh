#!/usr/bin/env sh
set -eu

go test . ./internal/av1/bitstream ./internal/av1/ivf ./internal/av1/obu ./internal/av1/rtp ./internal/av1/entropy ./internal/av1/parser ./internal/av1/decoder ./internal/av1/encoder ./internal/av1/frame ./internal/av1/memory ./internal/av1/dsp ./internal/av1/cdef ./internal/av1/prediction ./internal/av1/motion ./internal/av1/quantize ./internal/av1/transform ./internal/av1/reconstruct ./internal/av1/loopfilter ./internal/av1/restoration ./internal/av1/tile ./internal/av1/threading

check_zero_alloc_benchmarks() {
	expected_count=$1
	name_re=$2
	bench_output=$3
	printf '%s\n' "$bench_output" | awk -v expected="$expected_count" -v name_re="$name_re" '
	BEGIN { count = 0; failed = 0 }
	$1 ~ name_re {
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
		if (count != expected) {
			printf("ran %d allocation benchmarks, want %d\n", count, expected) > "/dev/stderr"
			failed = 1
		}
		exit failed
		}'
}

check_zero_alloc_benchmarks_at_least() {
	min_count=$1
	name_re=$2
	bench_output=$3
	printf '%s\n' "$bench_output" | awk -v min="$min_count" -v name_re="$name_re" '
		BEGIN { count = 0; failed = 0 }
		$1 ~ name_re {
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
			if (count < min) {
				printf("ran %d allocation benchmarks, want at least %d\n", count, min) > "/dev/stderr"
				failed = 1
			}
			exit failed
		}'
}

# Keep the benchmark count high enough that one-time runtime first touches do
# not show up as fractional B/op noise; package tests above enforce exact
# steady-state allocation counts with testing.AllocsPerRun.
bench_time=${GOAV1_ALLOC_BENCHTIME:-200x}
bench_re='^BenchmarkDecode(PostFilteredProfileClip|SuperResInterProfileClip|SuperResInterHighBDProfileClip|SuperResRestorationProfileClip|FilmGrainProfileClip|FullVectorAllocs)$'
bench_out=$(GOMAXPROCS=1 GOGC=off go test . -run '^$' -bench="$bench_re" -benchmem -benchtime="$bench_time" -count=1)
printf '%s\n' "$bench_out"
check_zero_alloc_benchmarks 6 '^BenchmarkDecode' "$bench_out"

# 1080p allocation canaries run in a stable allocation-only mode. The default
# parallel throughput benchmarks can observe process-level runtime first touches
# at -benchtime=1x; these rows prove the reusable encoder/decoder hot paths
# themselves stay at zero heap traffic.
bench1080_time=${GOAV1_ALLOC_1080P_BENCHTIME:-1x}
encoder1080_re='^Benchmark.*1080p$'
encoder1080_out=$(GOMAXPROCS=1 GOGC=off go test ./internal/av1/encoder -run '^$' -bench="$encoder1080_re" -benchmem -benchtime="$bench1080_time" -count=1)
printf '%s\n' "$encoder1080_out"
check_zero_alloc_benchmarks_at_least 6 '^Benchmark.*1080p' "$encoder1080_out"

# Keep the public 1080p benchmark canaries isolated from each other so runtime
# first touches in one benchmark do not get charged to a later hot-path row.
public_rtc1080_out=$(GOMAXPROCS=1 GOGC=off go test . -run '^$' -bench='^BenchmarkPublicRTCEncoderEncodePicture1080p$' -benchmem -benchtime="$bench1080_time" -count=1)
printf '%s\n' "$public_rtc1080_out"
check_zero_alloc_benchmarks 2 '^BenchmarkPublicRTCEncoderEncodePicture1080p' "$public_rtc1080_out"

public_decode1080_out=$(GOMAXPROCS=1 GOGC=off go test . -run '^$' -bench='^BenchmarkPublicDecodeRTPPayload1080p$' -benchmem -benchtime="$bench1080_time" -count=1)
printf '%s\n' "$public_decode1080_out"
check_zero_alloc_benchmarks 1 '^BenchmarkPublicDecodeRTPPayload1080p' "$public_decode1080_out"

public_layered_decode1080_out=$(GOMAXPROCS=1 GOGC=off go test . -run '^$' -bench='^BenchmarkPublicLayeredDecodeRTPPayload1080p$' -benchmem -benchtime="$bench1080_time" -count=1)
printf '%s\n' "$public_layered_decode1080_out"
check_zero_alloc_benchmarks 1 '^BenchmarkPublicLayeredDecodeRTPPayload1080p' "$public_layered_decode1080_out"
