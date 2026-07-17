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

func BenchmarkDCT32Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct32Size, inverseDCT32Col4Impl) }
func BenchmarkDCT32Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct32Size, inverseDCT32Col4PureGo) }
func BenchmarkDCT64Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct64Size, inverseDCT64Col4Impl) }
func BenchmarkDCT64Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct64Size, inverseDCT64Col4PureGo) }

func BenchmarkDCT8Col4Dispatch(b *testing.B) { benchmarkCol4(b, dct8Size, inverseDCT8Col4Impl) }
func BenchmarkDCT8Col4ViaCol2(b *testing.B)  { benchmarkCol4(b, dct8Size, inverseDCT8Col4ViaCol2) }
func BenchmarkDCT8Col4PureGo(b *testing.B)   { benchmarkCol4(b, dct8Size, inverseDCT8Col4PureGo) }
func BenchmarkDCT16Col4Dispatch(b *testing.B) {
	benchmarkCol4(b, dct16Size, inverseDCT16Col4Impl)
}
func BenchmarkDCT16Col4ViaCol2(b *testing.B) {
	benchmarkCol4(b, dct16Size, inverseDCT16Col4ViaCol2)
}
func BenchmarkDCT16Col4PureGo(b *testing.B) {
	benchmarkCol4(b, dct16Size, inverseDCT16Col4PureGo)
}

func BenchmarkADST8Col4Dispatch(b *testing.B)  { benchmarkCol4(b, adst8Size, inverseADST8Col4Impl) }
func BenchmarkADST8Col4PureGo(b *testing.B)    { benchmarkCol4(b, adst8Size, inverseADST8Col4PureGo) }
func BenchmarkADST16Col4Dispatch(b *testing.B) { benchmarkCol4(b, adst16Size, inverseADST16Col4Impl) }
func BenchmarkADST16Col4PureGo(b *testing.B)   { benchmarkCol4(b, adst16Size, inverseADST16Col4PureGo) }

func BenchmarkIdentity8Col4Dispatch(b *testing.B) { benchmarkCol4(b, 8, inverseIdentity8Col4Impl) }
func BenchmarkIdentity8Col4PureGo(b *testing.B)   { benchmarkCol4(b, 8, inverseIdentity8Col4PureGo) }
func BenchmarkIdentity16Col4Dispatch(b *testing.B) {
	benchmarkCol4(b, 16, inverseIdentity16Col4Impl)
}
func BenchmarkIdentity16Col4PureGo(b *testing.B) {
	benchmarkCol4(b, 16, inverseIdentity16Col4PureGo)
}
