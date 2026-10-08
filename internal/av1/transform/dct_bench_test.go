package transform

import (
	"fmt"
	"testing"
)

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
	size := Size{Width: uint8(side), Height: uint8(side)}
	// 2 bytes per dst sample; 4 bytes per coefficient. Reporting the
	// combined memory traffic keeps the MB/s column comparable across
	// block sizes.
	b.SetBytes(int64(side*side) * 6)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
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
	size := Size{Width: uint8(side), Height: uint8(side)}
	b.SetBytes(int64(side*side) * 6)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := InverseBlock(dst, side, coeff, side, scratch, size, TypeADSTADST); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkInverseBlockBitDepth8 exercises the public 8-bit API that selects
// the int16 column pipeline. The bounded fixture reaches that pipeline for
// every size; the high-range fixture forces the certified DCT16/32/64 guard
// to fall back to the exact int32 column kernel. DCT8 is full-range exact.
func BenchmarkInverseBlockBitDepth8(b *testing.B) {
	for _, side := range []int{8, 16, 32, 64} {
		for _, pattern := range []string{"bounded", "high-range"} {
			b.Run(inverseBenchSizeName(side)+"/"+pattern, func(b *testing.B) {
				coeff := inverseColumnBenchmarkCoefficients(side, pattern)
				dst := make([]int16, side*side)
				scratch := make([]int32, side*side)
				size := Size{Width: uint8(side), Height: uint8(side)}
				b.SetBytes(int64(side * side * 6))
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if err := InverseBlockBitDepth(dst, side, coeff, side, scratch, size, TypeDCTDCT, 8); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func inverseColumnBenchmarkCoefficients(side int, pattern string) []int32 {
	coeff := make([]int32, side*side)
	for i := range coeff {
		if pattern == "bounded" {
			coeff[i] = int32((i*7)%3 - 1)
		} else if i&1 == 0 {
			coeff[i] = 30000
		} else {
			coeff[i] = -30000
		}
	}
	return coeff
}

func inverseBenchSizeName(side int) string {
	return fmt.Sprintf("DCT%dx%d", side, side)
}
