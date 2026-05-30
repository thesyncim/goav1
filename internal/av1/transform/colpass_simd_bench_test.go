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
