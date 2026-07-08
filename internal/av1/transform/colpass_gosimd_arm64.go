// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD inverse column-pass transforms (Go 1.27+ GOEXPERIMENT=simd).
// The two columns of a *Col2 transform map onto the two lanes of an Int64x2, so
// the whole int64-precision butterfly runs in one vector. Loaded coefficients
// stay Int32x4 (low 2 lanes = the 2 columns) and feed MulWidenLo (int32*const ->
// int64), matching the scalar's int64(c)*const products exactly. roundShift is
// Add(1<<(bits-1)).ShiftAllRight(bits) in the int64 domain. clipRange to
// [min,max] is SaturateToInt32 then Int32x4 Max(min).Min(max): saturating to the
// int32 range first never changes a clamp into [min,max] (both fit int32), so it
// is byte-identical to the scalar's int64 clip.
//
// Column-pass callers always pass stride == transform width (>= 4), so the
// 4-wide array-pointer loads never see a short slice. See SIMD_PORT.md.

package transform

import "simd/archsimd"

// inverseDCT8Col2SIMD reproduces inverseDCT8 on the two adjacent columns at
// buf[k*stride] and buf[k*stride+1], byte-for-byte with inverseDCT8Col2PureGo.
func inverseDCT8Col2SIMD(buf []int32, stride int, min int32, max int32) {
	// Broadcast every constant once into a local. These stay register-resident
	// for the whole function; they must NOT be package globals, which would
	// force a memory load per use (SIMD vectors belong in registers, not the
	// data segment / heap).
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	dctC181 := archsimd.BroadcastInt32x4(181)
	dctC799 := archsimd.BroadcastInt32x4(799)
	dctC1567 := archsimd.BroadcastInt32x4(1567)
	dctC1138 := archsimd.BroadcastInt32x4(1138)
	dctC1703 := archsimd.BroadcastInt32x4(1703)
	dctCn312 := archsimd.BroadcastInt32x4(3784 - 4096)
	dctCn79 := archsimd.BroadcastInt32x4(4017 - 4096)
	dctR8 := archsimd.BroadcastInt64x2(1 << 7)
	dctR11 := archsimd.BroadcastInt64x2(1 << 10)
	dctR12 := archsimd.BroadcastInt64x2(1 << 11)
	ld := func(k int) archsimd.Int32x4 { return archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int32x4) { v.StorePart(buf[k*stride : k*stride+2]) }
	clip := func(v archsimd.Int64x2) archsimd.Int32x4 { return v.SaturateToInt32().Max(minV).Min(maxV) }
	w := func(v archsimd.Int32x4) archsimd.Int64x2 { return v.ExtendLo2ToInt64() }

	// inverseDCT4 on even rows 0,2,4,6.
	e0, e1, e2, e3 := ld(0), ld(2), ld(4), ld(6)
	t0 := e0.Add(e2).MulWidenLo(dctC181).Add(dctR8).ShiftAllRight(8)
	t1 := e0.Sub(e2).MulWidenLo(dctC181).Add(dctR8).ShiftAllRight(8)
	t2 := e1.MulWidenLo(dctC1567).Sub(e3.MulWidenLo(dctCn312)).Add(dctR12).ShiftAllRight(12).Sub(w(e3))
	t3 := e1.MulWidenLo(dctCn312).Add(e3.MulWidenLo(dctC1567)).Add(dctR12).ShiftAllRight(12).Add(w(e1))
	d0 := clip(t0.Add(t3)) // c[0]
	d2 := clip(t1.Add(t2)) // c[2]
	d4 := clip(t1.Sub(t2)) // c[4]
	d6 := clip(t0.Sub(t3)) // c[6]

	// inverseDCT8 odd part on rows 1,3,5,7.
	o1, o3, o5, o7 := ld(1), ld(3), ld(5), ld(7)
	t4a := o1.MulWidenLo(dctC799).Sub(o7.MulWidenLo(dctCn79)).Add(dctR12).ShiftAllRight(12).Sub(w(o7))
	t5a := o5.MulWidenLo(dctC1703).Sub(o3.MulWidenLo(dctC1138)).Add(dctR11).ShiftAllRight(11)
	t6a := o5.MulWidenLo(dctC1138).Add(o3.MulWidenLo(dctC1703)).Add(dctR11).ShiftAllRight(11)
	t7a := o1.MulWidenLo(dctCn79).Add(o7.MulWidenLo(dctC799)).Add(dctR12).ShiftAllRight(12).Add(w(o1))
	t4 := clip(t4a.Add(t5a))
	t5c := clip(t4a.Sub(t5a))
	t7 := clip(t7a.Add(t6a))
	t6c := clip(t7a.Sub(t6a))
	t5 := t6c.Sub(t5c).MulWidenLo(dctC181).Add(dctR8).ShiftAllRight(8)
	t6 := t6c.Add(t5c).MulWidenLo(dctC181).Add(dctR8).ShiftAllRight(8)

	st(0, clip(w(d0).Add(w(t7))))
	st(1, clip(w(d2).Add(t6)))
	st(2, clip(w(d4).Add(t5)))
	st(3, clip(w(d6).Add(w(t4))))
	st(4, clip(w(d6).Sub(w(t4))))
	st(5, clip(w(d4).Sub(t5)))
	st(6, clip(w(d2).Sub(t6)))
	st(7, clip(w(d0).Sub(w(t7))))
}
