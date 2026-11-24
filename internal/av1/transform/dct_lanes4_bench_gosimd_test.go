// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package transform

import "testing"

// BenchmarkLanes4 times the four-lane DCT32/DCT64 adapters on the shapes the
// row and column passes hand them: a 64-wide column buffer and four rows.
func BenchmarkLanes4(b *testing.B) {
	const stride = 64
	fill := func(s []int32) {
		for i := range s {
			s[i] = int32((i*1103-4000)&0x3fff) - 0x2000
		}
	}
	col := make([]int32, 63*stride+4)
	rows := make([]int32, 4*dct64Size)
	lo, hi := int32(-1<<17), int32(1<<17-1)
	run := func(name string, fn func()) {
		b.Run(name, func(b *testing.B) {
			fill(col)
			fill(rows)
			for b.Loop() {
				fn()
			}
		})
	}
	r := func(n, i int) []int32 { return rows[i*n : (i+1)*n] }
	run("dct64col", func() { inverseDCT64Col4SIMD(col, stride, lo, hi) })
	run("dct64lowcol", func() { inverseDCT64LowCol4SIMD(col, stride, 32, lo, hi) })
	run("dct64row", func() { inverseDCT64Row4SIMD(r(64, 0), r(64, 1), r(64, 2), r(64, 3), lo, hi) })
	run("dct64rowlow", func() { inverseDCT64Row4LowSIMD(r(64, 0), r(64, 1), r(64, 2), r(64, 3), 32, lo, hi) })
	run("dct32col", func() { inverseDCT32Lanes4ColSIMD(col, stride, lo, hi) })
	run("dct32row", func() { inverseDCT32Lanes4RowSIMD(r(32, 0), r(32, 1), r(32, 2), r(32, 3), lo, hi) })
}
