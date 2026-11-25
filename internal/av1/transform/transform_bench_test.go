package transform

import (
	"fmt"
	"testing"
)

// benchmarkCol2 drives a batched column kernel over two adjacent columns of a
// row-major scratch buffer repeatedly. It is used to compare the live (NEON)
// dispatch slot against its pure-Go reference. The buffer uses a realistic
// stride (the two target columns plus padding) so the strided loads match the
// decode-time layout.
func benchmarkCol2(b *testing.B, length int, fn func(buf []int32, rowStride int, min, max int32)) {
	b.Helper()
	const rowStride = 4
	buf := make([]int32, (length-1)*rowStride+2)
	for i := range buf {
		buf[i] = int32((i*1103 - 4000) & 0x7fff)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(buf, rowStride, minInt16, maxInt16)
	}
}

func BenchmarkDCT16Col2Dispatch(b *testing.B) { benchmarkCol2(b, dct16Size, inverseDCT16Col2Impl) }
func BenchmarkDCT16Col2PureGo(b *testing.B)   { benchmarkCol2(b, dct16Size, inverseDCT16Col2PureGo) }
func BenchmarkDCT32Col2Dispatch(b *testing.B) { benchmarkCol2(b, dct32Size, inverseDCT32Col2Impl) }
func BenchmarkDCT32Col2PureGo(b *testing.B)   { benchmarkCol2(b, dct32Size, inverseDCT32Col2PureGo) }
func BenchmarkDCT64Col2Dispatch(b *testing.B) { benchmarkCol2(b, dct64Size, inverseDCT64Col2Impl) }
func BenchmarkDCT64Col2PureGo(b *testing.B)   { benchmarkCol2(b, dct64Size, inverseDCT64Col2PureGo) }

// benchmarkCol4 is benchmarkCol2 for the batched four-column kernels; per
// call it transforms four columns, so compare ns/col against two Col2 calls.
func benchmarkCol4(b *testing.B, length int, fn func(buf []int32, rowStride int, min, max int32)) {
	b.Helper()
	const rowStride = 6
	buf := make([]int32, (length-1)*rowStride+4)
	for i := range buf {
		buf[i] = int32((i*1103 - 4000) & 0x7fff)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(buf, rowStride, minInt16, maxInt16)
	}
}

// DCT16 four-column vs the older two-column pair: the four-lane int32 kernel
// multiplies four columns per instruction and halves the loads/stores/srshr
// relative to two inverseDCT16Col2 calls.
func BenchmarkDCT16Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct16Size, inverseDCT16Col4Impl) }
func BenchmarkDCT16Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct16Size, inverseDCT16Col4PureGo) }
func BenchmarkDCT16Col4Col2(b *testing.B) {
	benchmarkCol4(b, dct16Size, func(buf []int32, rowStride int, min, max int32) {
		inverseDCT16Col2Impl(buf, rowStride, min, max)
		inverseDCT16Col2Impl(buf[2:], rowStride, min, max)
	})
}

// DCT4 four-column vs the scalar column path it replaces (DCT4 had no batched
// column kernel; the column pass ran four scalar inverseDCT4 calls).
func BenchmarkDCT4Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct4Size, inverseDCT4Col4Impl) }
func BenchmarkDCT4Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct4Size, inverseDCT4Col4PureGo) }

// DCT8 four-column vs the older two-column pair: the four-lane int32 kernel
// transforms four columns per instruction, replacing two int64-lane
// inverseDCT8Col2 calls for each group of four columns.
func BenchmarkDCT8Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct8Size, inverseDCT8Col4Impl) }
func BenchmarkDCT8Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct8Size, inverseDCT8Col4PureGo) }
func BenchmarkDCT8Col4Col2(b *testing.B) {
	benchmarkCol4(b, dct8Size, func(buf []int32, rowStride int, min, max int32) {
		inverseDCT8Col2Impl(buf, rowStride, min, max)
		inverseDCT8Col2Impl(buf[2:], rowStride, min, max)
	})
}

func BenchmarkDCT32Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct32Size, inverseDCT32Col4Impl) }
func BenchmarkDCT32Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct32Size, inverseDCT32Col4PureGo) }
func BenchmarkDCT64Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct64Size, inverseDCT64Col4Impl) }
func BenchmarkDCT64Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct64Size, inverseDCT64Col4PureGo) }

func BenchmarkADST16Col4Dispatch(b *testing.B) { benchmarkCol4(b, adst16Size, inverseADST16Col4Impl) }
func BenchmarkADST16Col4PureGo(b *testing.B)   { benchmarkCol4(b, adst16Size, inverseADST16Col4PureGo) }

// Extended DCT_DCT InverseBlock benchmarks for the 16x16, 32x32, and 64x64
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
		if err := InverseBlock(dst, side, coeff, side, scratch, size, TypeDCTDCT); err != nil {
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

func BenchmarkSizeFromDimensions(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := SizeFromDimensions(32, 32); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkScanSize(b *testing.B) {
	size := Size{Width: 64, Height: 64}
	for i := 0; i < b.N; i++ {
		if _, err := ScanSize(size); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSizeShift(b *testing.B) {
	size := Size{Width: 32, Height: 32}
	for i := 0; i < b.N; i++ {
		if _, err := size.Shift(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDefaultScanMode(b *testing.B) {
	size := Size{Width: 16, Height: 32}
	for i := 0; i < b.N; i++ {
		if _, err := DefaultScanMode(size, Class2D); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFillDefaultScan32x32(b *testing.B) {
	var scan [32 * 32]int16
	var inverse [32 * 32]int16
	size := Size{Width: 32, Height: 32}
	for i := 0; i < b.N; i++ {
		if err := FillDefaultScan(scan[:], inverse[:], size, Class2D); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFillDefaultScan4x4(b *testing.B) {
	var scan [16]int16
	var inverse [16]int16
	size := Size{Width: 4, Height: 4}
	for i := 0; i < b.N; i++ {
		if err := FillDefaultScan(scan[:], inverse[:], size, Class2D); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMaxEOB(b *testing.B) {
	size := Size{Width: 64, Height: 64}
	for i := 0; i < b.N; i++ {
		if _, err := MaxEOB(size); err != nil {
			b.Fatal(err)
		}
	}
}

// benchmarkRow2 drives a batched row kernel over two rows repeatedly. It is
// used to compare the live (NEON) dispatch slot against its pure-Go reference.
func benchmarkRow2(b *testing.B, length int, fn func(r0, r1 []int32, min, max int32)) {
	b.Helper()
	r0 := make([]int32, length)
	r1 := make([]int32, length)
	for i := 0; i < length; i++ {
		r0[i] = int32((i*1103 - 4000) & 0x7fff)
		r1[i] = int32((i*-877 + 2500) & 0x7fff)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(r0, r1, minInt16, maxInt16)
	}
}

func BenchmarkDCT4Row2Dispatch(b *testing.B)  { benchmarkRow2(b, dct4Size, inverseDCT4Row2Impl) }
func BenchmarkDCT4Row2PureGo(b *testing.B)    { benchmarkRow2(b, dct4Size, inverseDCT4Row2PureGo) }
func BenchmarkDCT8Row2Dispatch(b *testing.B)  { benchmarkRow2(b, dct8Size, inverseDCT8Row2Impl) }
func BenchmarkDCT8Row2PureGo(b *testing.B)    { benchmarkRow2(b, dct8Size, inverseDCT8Row2PureGo) }
func BenchmarkDCT16Row2Dispatch(b *testing.B) { benchmarkRow2(b, dct16Size, inverseDCT16Row2Impl) }
func BenchmarkDCT16Row2PureGo(b *testing.B)   { benchmarkRow2(b, dct16Size, inverseDCT16Row2PureGo) }
func BenchmarkDCT32Row2Dispatch(b *testing.B) { benchmarkRow2(b, dct32Size, inverseDCT32Row2Impl) }
func BenchmarkDCT32Row2PureGo(b *testing.B)   { benchmarkRow2(b, dct32Size, inverseDCT32Row2PureGo) }
func BenchmarkDCT64Row2Dispatch(b *testing.B) { benchmarkRow2(b, dct64Size, inverseDCT64Row2Impl) }
func BenchmarkDCT64Row2PureGo(b *testing.B)   { benchmarkRow2(b, dct64Size, inverseDCT64Row2PureGo) }
func BenchmarkADST4Row2Dispatch(b *testing.B) { benchmarkRow2(b, adst4Size, inverseADST4Row2Impl) }
func BenchmarkADST4Row2PureGo(b *testing.B)   { benchmarkRow2(b, adst4Size, inverseADST4Row2PureGo) }
func BenchmarkADST8Row2Dispatch(b *testing.B) { benchmarkRow2(b, adst8Size, inverseADST8Row2Impl) }
func BenchmarkADST8Row2PureGo(b *testing.B)   { benchmarkRow2(b, adst8Size, inverseADST8Row2PureGo) }

// benchmarkRow4 drives a batched four-row kernel; per call it transforms four
// rows, so compare ns/row against two Row2 calls.
func benchmarkRow4(b *testing.B, length int, fn func(r0, r1, r2, r3 []int32, min, max int32)) {
	b.Helper()
	rows := make([][]int32, 4)
	for r := range rows {
		rows[r] = make([]int32, length)
		for i := range rows[r] {
			rows[r][i] = int32((i*1103 - 4000 + r*997) & 0x7fff)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(rows[0], rows[1], rows[2], rows[3], minInt16, maxInt16)
	}
}

func BenchmarkDCT32Row4Dispatch(b *testing.B) { benchmarkRow4(b, dct32Size, inverseDCT32Row4Impl) }
func BenchmarkDCT32Row4PureGo(b *testing.B)   { benchmarkRow4(b, dct32Size, inverseDCT32Row4PureGo) }
func BenchmarkDCT64Row4Dispatch(b *testing.B) { benchmarkRow4(b, dct64Size, inverseDCT64Row4Impl) }
func BenchmarkDCT64Row4PureGo(b *testing.B)   { benchmarkRow4(b, dct64Size, inverseDCT64Row4PureGo) }

func BenchmarkADST16Row4Dispatch(b *testing.B) { benchmarkRow4(b, adst16Size, inverseADST16Row4Impl) }
func BenchmarkADST16Row4PureGo(b *testing.B)   { benchmarkRow4(b, adst16Size, inverseADST16Row4PureGo) }
