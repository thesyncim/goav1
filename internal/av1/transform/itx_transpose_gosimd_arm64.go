// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// transpose4x4 transposes a 4x4 block of int32 lanes held as four row vectors.
// It is its own inverse, so it both gathers rows into per-element lane vectors
// (rows become lanes) and scatters them back.
func transpose4x4(a, b, c, d archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	t0 := a.InterleaveLo(b) // a0 b0 a1 b1
	t1 := a.InterleaveHi(b) // a2 b2 a3 b3
	t2 := c.InterleaveLo(d) // c0 d0 c1 d1
	t3 := c.InterleaveHi(d) // c2 d2 c3 d3
	e0 := fdctInt32AsInt64(t0)
	e2 := fdctInt32AsInt64(t2)
	f1 := fdctInt32AsInt64(t1)
	f3 := fdctInt32AsInt64(t3)
	return fdctInt64AsInt32(e0.InterleaveLo(e2)), fdctInt64AsInt32(e0.InterleaveHi(e2)),
		fdctInt64AsInt32(f1.InterleaveLo(f3)), fdctInt64AsInt32(f1.InterleaveHi(f3))
}
