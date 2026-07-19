// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "testing"

// BenchmarkBoxsumNEON measures the current NEON brute-force box sum for the
// same 70x70 extended block used by BenchmarkBoxsumScalar / ...Separable, so the
// three can be compared directly for the A/B decision.
func BenchmarkBoxsumNEON(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsumNEON(src, so, w, h, ss, 2, true, dst, ds)
		boxsumNEON(src, so, w, h, ss, 2, false, dst, ds)
	}
}

func BenchmarkBoxsumNEONR1(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsumNEON(src, so, w, h, ss, 1, true, dst, ds)
		boxsumNEON(src, so, w, h, ss, 1, false, dst, ds)
	}
}
