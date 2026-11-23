//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import "testing"

// SIMD (archsimd) benchmark variants, paired with the NEON/scalar benchmarks in
// metric_neon_bench_test.go for a three-way comparison.

func BenchmarkPixelStats8x8SIMD(b *testing.B)   { benchPixelStats(b, pixelStats8x8SIMD) }
func BenchmarkPixelStats16x16SIMD(b *testing.B) { benchPixelStats(b, pixelStats16x16SIMD) }
func BenchmarkPixelStats32x32SIMD(b *testing.B) { benchPixelStats(b, pixelStats32x32SIMD) }
