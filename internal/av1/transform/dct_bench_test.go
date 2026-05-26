package transform

import "testing"

// Extended InverseDCTBlock benchmarks for the 16x16, 32x32, and 64x64
// transform sizes. The 4x4 and 8x8 variants live alongside the unit
// tests in dct_test.go; this file fills in the rest of the size ladder
// so callers profiling the inverse transform stage see how cost scales
// with block area (the dominant cost in residual reconstruction for
// high-quality vectors).
//
// All variants use caller-owned scratch and report b.SetBytes so the
// standard go test bench output column surfaces a byte throughput
// figure proportional to coefficient and destination plane volume.

func benchmarkInverseDCTBlockSquare(b *testing.B, side int) {
	b.Helper()
	coeff := make([]int32, side*side)
	dst := make([]int16, side*side)
	scratch := make([]int32, side*side)
	for i := range coeff {
		coeff[i] = int32(i%17) - 8
	}
	size := Size{Width: side, Height: side}
	// 2 bytes per dst sample; 4 bytes per coefficient. Reporting the
	// combined memory traffic keeps the MB/s column comparable across
	// block sizes.
	b.SetBytes(int64(side*side) * 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := InverseDCTBlock(dst, side, coeff, side, scratch, size); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInverseDCTBlock16x16(b *testing.B) { benchmarkInverseDCTBlockSquare(b, 16) }
func BenchmarkInverseDCTBlock32x32(b *testing.B) { benchmarkInverseDCTBlockSquare(b, 32) }
func BenchmarkInverseDCTBlock64x64(b *testing.B) { benchmarkInverseDCTBlockSquare(b, 64) }

// BenchmarkInverseBlockHybrid covers the dispatcher path that the
// reconstruction pipeline actually invokes during decode. The DCT-only
// micros measure the transform kernel; this one measures the kernel
// plus the type-dispatched horizontal/vertical orchestration.
func BenchmarkInverseBlockHybrid16x16(b *testing.B) {
	const side = 16
	coeff := make([]int32, side*side)
	dst := make([]int16, side*side)
	scratch := make([]int32, side*side)
	for i := range coeff {
		coeff[i] = int32(i%13) - 6
	}
	size := Size{Width: side, Height: side}
	b.SetBytes(int64(side*side) * 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := InverseBlock(dst, side, coeff, side, scratch, size, TypeADSTADST); err != nil {
			b.Fatal(err)
		}
	}
}
