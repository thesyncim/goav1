//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import "testing"

// SIMD (archsimd) benchmark variants, paired with the NEON/scalar benchmarks in
// metric_neon_bench_test.go for a three-way comparison.

func BenchmarkSATDCoeffs16SIMD(b *testing.B)   { benchSATDCoeffs(b, satdCoeffsSIMD, 16) }
func BenchmarkSATDCoeffs64SIMD(b *testing.B)   { benchSATDCoeffs(b, satdCoeffsSIMD, 64) }
func BenchmarkSATDCoeffs256SIMD(b *testing.B)  { benchSATDCoeffs(b, satdCoeffsSIMD, 256) }
func BenchmarkSATDCoeffs1024SIMD(b *testing.B) { benchSATDCoeffs(b, satdCoeffsSIMD, 1024) }

func BenchmarkHadamard4x4SIMD(b *testing.B)   { benchHadamard4x4(b, hadamard4x4SIMD) }
func BenchmarkHadamard8x8SIMD(b *testing.B)   { benchHadamard8x8(b, hadamard8x8SIMD) }
func BenchmarkHadamard16x16SIMD(b *testing.B) { benchHadamard16x16(b, hadamard16x16SIMD) }
func BenchmarkHadamard32x32SIMD(b *testing.B) { benchHadamard32x32(b, hadamard32x32SIMD) }

func BenchmarkPixelStats8x8SIMD(b *testing.B)   { benchPixelStats(b, pixelStats8x8SIMD) }
func BenchmarkPixelStats16x16SIMD(b *testing.B) { benchPixelStats(b, pixelStats16x16SIMD) }
func BenchmarkPixelStats32x32SIMD(b *testing.B) { benchPixelStats(b, pixelStats32x32SIMD) }
func BenchmarkPixelStats8x4SIMD(b *testing.B)   { benchPixelStats(b, pixelStats8x4SIMD) }
