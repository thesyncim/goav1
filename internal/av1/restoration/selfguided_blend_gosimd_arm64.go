// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD (simd/archsimd, Go 1.27+ GOEXPERIMENT=simd) port of the SGR
// self-guided final projection ("blend") for the high-bit-depth uint16 path.
// It is compute-bound per-pixel int32 arithmetic with no data dependency and no
// transpose — the ideal SIMD target: eight pixels are projected per iteration.
//
// Per pixel the scalar reference computes
//
//	u  = int32(s) << SGRProjRstBits          (s widened from uint16)
//	v  = u << SGRProjPrjBits + xq0*(f0-u) + xq1*(f1-u)
//	r  = roundPowerOfTwo(v, SGRProjPrjBits+SGRProjRstBits)   // (v + 1<<10) >> 11
//	w  = int32(int16(r))                     // libaom's wrapping int16 cast
//	d  = clampInt32(w, 0, max)               // -> uint16
//
// The SIMD kernel processes 16 columns per iteration as four Int32x4 halves.
// Every op maps to the exact scalar integer op, so byte-identity follows
// directly (int32 wraps mod 2^32 exactly like Go int32, and the three addends
// commute, so the projection is regrouped to fold the u<<7 and the rounding
// bias into the multiply-accumulate chain — see sgrProjectHalf):
//   - u = s<<RstBits: u16 keeps a register VSSHL after the widen.
//   - roundPowerOfTwo is + bias then arithmetic >>11 (VSSHL by a negative
//     count); the bias seeds the accumulator so no separate add is emitted.
//   - the wrapping int16 cast is TruncToInt16 (VXTN keeps the low 16 bits =
//     exactly Go's int16(r)); the [0,maxI] clip runs as Max/Min in the int16
//     domain (maxI < 65535) before the uint16 reinterpret.
//
// The int32 flt loads and the uint16 dst store are natural array-pointer
// accesses (f0/f1/dst always hold >= 8 elements from column i in a full
// 8-group). Columns beyond width&^7 fall back to the scalar reference.

package restoration

import (
	"unsafe"

	"simd/archsimd"
)

// init binds the winning Go-native SIMD high-bit-depth SGR blend under the
// goexperiment.simd build. Measured-losing Wiener and 8-bit SGR candidates
// remain on their NEON dispatchers.
func init() {
	sgrWeightedRowImpl = sgrWeightedRowSIMD
}

// sgrConsts holds the loop-invariant broadcast vectors for sgrProjectHalf.
//
// The shift amounts are pre-broadcast to vectors so each per-pixel shift issues
// a single register VSSHL: archsimd's ShiftAll{Left,Right} always lower to
// variable VSSHL and would otherwise re-materialise the constant with VMOV+VDUP
// on every use inside the loop (a negative Shift count is an arithmetic right
// shift, byte-identical to ShiftAllRight). shRst (=RstBits) is used by the u16
// widen.
//
// bias (the rounding 1<<10) seeds the multiply-accumulate chain and k128
// (=1<<PrjBits) turns u<<PrjBits into a fused u*128 MulAdd, so the projection is
// one Sub + one MulAdd per filter term plus a single MulAdd for the u<<7 term,
// then one arithmetic >>11 — no separate shift for <<7 and no separate add for
// the bias, matching the NEON asm's MLA + rounding-shift structure.
type sgrConsts struct {
	xq0V, xq1V, bias  archsimd.Int32x4
	cuV, shRst, shRnd archsimd.Int32x4
}

func newSGRConsts(xq0, xq1 int32) sgrConsts {
	return sgrConsts{
		xq0V:  archsimd.BroadcastInt32x4(xq0),
		xq1V:  archsimd.BroadcastInt32x4(xq1),
		bias:  archsimd.BroadcastInt32x4(1 << (SGRProjPrjBits + SGRProjRstBits - 1)),
		cuV:   archsimd.BroadcastInt32x4((1 << SGRProjPrjBits) - xq0 - xq1),
		shRst: archsimd.BroadcastInt32x4(SGRProjRstBits),
		shRnd: archsimd.BroadcastInt32x4(-(SGRProjPrjBits + SGRProjRstBits)),
	}
}

// sgrProjectHalf computes roundPowerOfTwo(v, 11) for four columns, where
// v = u<<7 + xq0*(f0-u) + xq1*(f1-u) and u = s<<4 (the caller supplies u
// pre-shifted). The result is the raw int32 projection BEFORE the wrapping
// int16 cast and the [0,max] clip.
//
// All adds/muls are int32 and commute mod 2^32, so seeding the accumulator with
// the rounding bias and adding u<<7 as the last term is bit-identical to the
// scalar order: (f0-u)*xq0 + bias, then +(f1-u)*xq1, then +u*128, equals
// u<<7 + xq0*(f0-u) + xq1*(f1-u) + bias == v + bias, and u*128 == u<<7 in the
// low 32 bits. Each MulAdd is one VMLA, the same fused op the asm uses.
func sgrProjectHalf(u, f0, f1 archsimd.Int32x4, c sgrConsts) archsimd.Int32x4 {
	// Algebraic factor: u<<7 + xq0*(f0-u) + xq1*(f1-u) == u*(128-xq0-xq1) +
	// xq0*f0 + xq1*f1 -- drops both f-u subtracts (6 ops/group -> 4). The round
	// bias seeds the first MulAdd; c.shRnd then arithmetic-shifts right by 11.
	v := c.xq0V.MulAdd(f0, c.bias) // xq0*f0 + bias
	v = c.xq1V.MulAdd(f1, v)       // xq1*f1 + v
	v = c.cuV.MulAdd(u, v)         // (128-xq0-xq1)*u + v
	return v.Shift(c.shRnd)
}

// loadI32x4 is a bounds-check-free array-pointer load of four int32 at p[i];
// callers guarantee i+4 <= len(p).
func loadI32x4(p []int32, i int) archsimd.Int32x4 {
	return archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&p[i])))
}

// sgrWeightedRowSIMD is the Go-native-SIMD high-bit-depth (uint16) final
// projection row. The bulk runs 16 pixels per iteration (two 16-byte loads,
// four Int32x4 projections, two 16-byte stores); a trailing 8-wide group and
// the <8 tail keep the scalar path. Byte-identical to sgrWeightedRow.
//
// The widen uses ExtendLo4ToUint32 and HiToLo+ExtendLo4ToUint32. The int32->int16
// pack uses TruncToInt16 and InterleaveLo. Unlike
// the [0,maxI] clamp cannot fold into the narrow (maxI < 65535, so
// SQXTUN's full-range saturation is too wide), so it stays as Max/Min in the
// int16 domain before the uint16 reinterpret.
func sgrWeightedRowSIMD(dst []uint16, src []uint16, f0 []int32, f1 []int32, xq0 int32, xq1 int32, maxI int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := newSGRConsts(xq0, xq1)
	zero16 := archsimd.BroadcastInt16x8(0)
	max16 := archsimd.BroadcastInt16x8(int16(maxI))
	narrow := func(a, b archsimd.Int32x4) archsimd.Uint16x8 {
		return restorationTruncateInt32PairToInt16(a, b).Max(zero16).Min(max16).ConvertToUint16()
	}
	// u16 has no widen-with-shift op, so u = s<<RstBits is a register VSSHL after
	// the UXTL/UXTL2 widen (uSh applies c.shRst).
	uSh := func(v archsimd.Uint32x4) archsimd.Int32x4 { return v.ConvertToInt32().Shift(c.shRst) }
	i := 0
	for ; i+16 <= width; i += 16 {
		sv0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[i])))
		sv1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[i+8])))
		r0 := sgrProjectHalf(uSh(sv0.ExtendLo4ToUint32()), loadI32x4(f0, i), loadI32x4(f1, i), c)
		r1 := sgrProjectHalf(uSh(sv0.HiToLo().ExtendLo4ToUint32()), loadI32x4(f0, i+4), loadI32x4(f1, i+4), c)
		r2 := sgrProjectHalf(uSh(sv1.ExtendLo4ToUint32()), loadI32x4(f0, i+8), loadI32x4(f1, i+8), c)
		r3 := sgrProjectHalf(uSh(sv1.HiToLo().ExtendLo4ToUint32()), loadI32x4(f0, i+12), loadI32x4(f1, i+12), c)
		narrow(r0, r1).StoreArray((*[8]uint16)(unsafe.Pointer(&dst[i])))
		narrow(r2, r3).StoreArray((*[8]uint16)(unsafe.Pointer(&dst[i+8])))
	}
	if i+8 <= width {
		sv := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[i])))
		rLo := sgrProjectHalf(uSh(sv.ExtendLo4ToUint32()), loadI32x4(f0, i), loadI32x4(f1, i), c)
		rHi := sgrProjectHalf(uSh(sv.HiToLo().ExtendLo4ToUint32()), loadI32x4(f0, i+4), loadI32x4(f1, i+4), c)
		narrow(rLo, rHi).StoreArray((*[8]uint16)(unsafe.Pointer(&dst[i])))
		i += 8
	}
	if i < width {
		sgrWeightedRow(dst[i:], src[i:], f0[i:], f1[i:], xq0, xq1, maxI)
	}
}
