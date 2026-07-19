// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import "testing"

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
