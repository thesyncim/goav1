// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import "testing"

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
func BenchmarkADST4Row2Dispatch(b *testing.B) { benchmarkRow2(b, adst4Size, inverseADST4Row2Impl) }
func BenchmarkADST4Row2PureGo(b *testing.B)   { benchmarkRow2(b, adst4Size, inverseADST4Row2PureGo) }
func BenchmarkADST8Row2Dispatch(b *testing.B) { benchmarkRow2(b, adst8Size, inverseADST8Row2Impl) }
func BenchmarkADST8Row2PureGo(b *testing.B)   { benchmarkRow2(b, adst8Size, inverseADST8Row2PureGo) }
