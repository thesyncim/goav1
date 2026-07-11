// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD (simd/archsimd, Go 1.27+ GOEXPERIMENT=simd) port of the SGR
// self-guided final projection ("blend"), for both the 8-bit-pixel path
// (sgrWeightedRowU8) and the high-bit-depth uint16 path (sgrWeightedRow). This
// is compute-bound per-pixel int32 arithmetic with no data dependency and no
// transpose — the ideal SIMD target: eight pixels are projected per iteration.
//
// Per pixel the scalar reference computes
//
//	u  = int32(s) << SGRProjRstBits          (s widened from uint8/uint16)
//	v  = u << SGRProjPrjBits + xq0*(f0-u) + xq1*(f1-u)
//	r  = roundPowerOfTwo(v, SGRProjPrjBits+SGRProjRstBits)   // (v + 1<<10) >> 11
//	w  = int32(int16(r))                     // libaom's wrapping int16 cast
//	d  = clampInt32(w, 0, max)               // -> uint8 / uint16
//
// The SIMD kernel processes 16 columns per iteration as four Int32x4 halves.
// Every op maps to the exact scalar integer op, so byte-identity follows
// directly (int32 wraps mod 2^32 exactly like Go int32, and the three addends
// commute, so the projection is regrouped to fold the u<<7 and the rounding
// bias into the multiply-accumulate chain — see sgrProjectHalf):
//   - u = s<<RstBits: the u8 path folds it into the USHLL widen; u16 keeps a
//     register VSSHL after the widen.
//   - roundPowerOfTwo is + bias then arithmetic >>11 (VSSHL by a negative
//     count); the bias seeds the accumulator so no separate add is emitted.
//   - the wrapping int16 cast is TruncToInt16 (VXTN keeps the low 16 bits =
//     exactly Go's int16(r)); for u8 SaturateToUint8 (SQXTUN) folds the
//     [0,255] clip into the narrow, for u16 the [0,maxI] clip runs as Max/Min
//     in the int16 domain (maxI < 65535) before the uint16 reinterpret.
//
// The int32 flt loads and the uint16 dst store are natural array-pointer
// accesses (f0/f1/dst always hold >= 8 elements from column i in a full
// 8-group). The uint8 source and destination lack an 8-byte archsimd
// load/store (arm64 exposes only 128-bit vectors), so 8 bytes are staged
// through a function-local [16]uint8 exactly as minMaxAbsDiff8x8SIMD does; this
// also matches the NEON asm, which never over-reads past the 8-pixel group.
// Columns beyond width&^7 fall back to the scalar reference.

package restoration

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"

	"simd/archsimd"
)

// init binds the Go-native SIMD SGR blend kernels under the goexperiment.simd
// build (the NEON binding in u8_dispatch_arm64.go is excluded there via
// !goexperiment.simd). Both Wiener u8 passes are Go-SIMD here: the box sums
// and the A/B stencil keep their NEON bindings from
// selfguided_dispatch_arm64.go, which is not gated out of the simd build.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		wienerHorizontalU8Impl = wienerHorizontalU8SIMD // symmetric-pair FIR: Go-SIMD beats asm
		wienerVerticalU8Impl = wienerVerticalU8SIMD     // column MAC: Go-SIMD beats asm
	}
	sgrWeightedRowU8Impl = sgrWeightedRowU8SIMD
	sgrWeightedRowImpl = sgrWeightedRowSIMD
}

// sgrConsts holds the loop-invariant broadcast vectors for sgrProjectHalf.
//
// The shift amounts are pre-broadcast to vectors so each per-pixel shift issues
// a single register VSSHL: archsimd's ShiftAll{Left,Right} always lower to
// variable VSSHL and would otherwise re-materialise the constant with VMOV+VDUP
// on every use inside the loop (a negative Shift count is an arithmetic right
// shift, byte-identical to ShiftAllRight). shRst (=RstBits) is only used by the
// u16 widen; the u8 path folds <<RstBits into its USHLL widen instead.
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

// sgrWeightedRowU8SIMD is the Go-native-SIMD 8-bit final projection row. The
// bulk runs 16 pixels per iteration through one 16-byte load, four Int32x4
// projections, and one 16-byte store (no per-pixel staging); a trailing 8-wide
// group and the <8 tail keep the scalar path. Byte-identical to
// sgrWeightedRowU8.
//
// The widen folds the u = s<<RstBits shift into a single USHLL/USHLL2
// (ExtendLo8ShlToUint16/ExtendHi8ShlToUint16 with amount RstBits): u comes
// straight out of the widen with no separate shift (s<<4 <= 4080 fits uint16).
// The two int32->int16 packs use TruncToInt16 + TruncToInt16Hi (XTN/XTN2), and
// the int16->uint8 narrow uses SaturateToUint8 + SaturateToUint8Hi
// (SQXTUN/SQXTUN2). TruncToInt16 is libaom's wrapping int16 cast, and SQXTUN's
// signed->unsigned saturation is exactly clampInt32(int16(rounded), 0, 255), so
// the explicit [0,255] clamp is folded into the narrow — the same instruction
// sequence as the NEON asm.
func sgrWeightedRowU8SIMD(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	// Reslice to width so the compiler can prove the i+16<=width loads in bounds
	// and drop the per-load bounds checks (the loop only reads indices < width).
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := newSGRConsts(xq0, xq1)
	i := 0
	for ; i+16 <= width; i += 16 {
		sv := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&src[i])))
		lo16 := sv.ExtendLo8ShlToUint16(SGRProjRstBits) // (s0..s7)<<4   (USHLL #4)
		hi16 := sv.ExtendHi8ShlToUint16(SGRProjRstBits) // (s8..s15)<<4  (USHLL2 #4)
		r0 := sgrProjectHalf(lo16.ExtendLo4ToUint32().ConvertToInt32(), loadI32x4(f0, i), loadI32x4(f1, i), c)
		r1 := sgrProjectHalf(lo16.ExtendHi4ToUint32().ConvertToInt32(), loadI32x4(f0, i+4), loadI32x4(f1, i+4), c)
		r2 := sgrProjectHalf(hi16.ExtendLo4ToUint32().ConvertToInt32(), loadI32x4(f0, i+8), loadI32x4(f1, i+8), c)
		r3 := sgrProjectHalf(hi16.ExtendHi4ToUint32().ConvertToInt32(), loadI32x4(f0, i+12), loadI32x4(f1, i+12), c)
		lo := r0.TruncToInt16().TruncToInt16Hi(r1) // w0..w7
		hi := r2.TruncToInt16().TruncToInt16Hi(r3) // w8..w15
		lo.SaturateToUint8().SaturateToUint8Hi(hi).StoreArray((*[16]uint8)(unsafe.Pointer(&dst[i])))
	}
	if i+8 <= width {
		// uint8 lacks an 8-byte archsimd load/store, so stage 8 bytes through a
		// function-local array (never over-reading past the group), matching the
		// NEON asm's 8-pixel load width.
		var sbuf, obuf [16]uint8
		sbuf[0], sbuf[1], sbuf[2], sbuf[3] = src[i], src[i+1], src[i+2], src[i+3]
		sbuf[4], sbuf[5], sbuf[6], sbuf[7] = src[i+4], src[i+5], src[i+6], src[i+7]
		lo16 := archsimd.LoadUint8x16Array(&sbuf).ExtendLo8ShlToUint16(SGRProjRstBits)
		rLo := sgrProjectHalf(lo16.ExtendLo4ToUint32().ConvertToInt32(), loadI32x4(f0, i), loadI32x4(f1, i), c)
		rHi := sgrProjectHalf(lo16.ExtendHi4ToUint32().ConvertToInt32(), loadI32x4(f0, i+4), loadI32x4(f1, i+4), c)
		rLo.TruncToInt16().TruncToInt16Hi(rHi).SaturateToUint8().StoreArray(&obuf)
		dst[i], dst[i+1], dst[i+2], dst[i+3] = obuf[0], obuf[1], obuf[2], obuf[3]
		dst[i+4], dst[i+5], dst[i+6], dst[i+7] = obuf[4], obuf[5], obuf[6], obuf[7]
		i += 8
	}
	if i < width {
		sgrWeightedRowU8(dst[i:], src[i:], f0[i:], f1[i:], xq0, xq1)
	}
}

// sgrWeightedRowSIMD is the Go-native-SIMD high-bit-depth (uint16) final
// projection row. The bulk runs 16 pixels per iteration (two 16-byte loads,
// four Int32x4 projections, two 16-byte stores); a trailing 8-wide group and
// the <8 tail keep the scalar path. Byte-identical to sgrWeightedRow.
//
// The widen uses ExtendLo4/ExtendHi4ToUint32 (UXTL/UXTL2) and the int32->int16
// pack uses TruncToInt16 + TruncToInt16Hi (XTN/XTN2) — no lane shuffles. Unlike
// the u8 path the [0,maxI] clamp cannot fold into the narrow (maxI < 65535, so
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
		return a.TruncToInt16().TruncToInt16Hi(b).Max(zero16).Min(max16).ConvertToUint16()
	}
	// u16 has no widen-with-shift op, so u = s<<RstBits is a register VSSHL after
	// the UXTL/UXTL2 widen (uSh applies c.shRst).
	uSh := func(v archsimd.Uint32x4) archsimd.Int32x4 { return v.ConvertToInt32().Shift(c.shRst) }
	i := 0
	for ; i+16 <= width; i += 16 {
		sv0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[i])))
		sv1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[i+8])))
		r0 := sgrProjectHalf(uSh(sv0.ExtendLo4ToUint32()), loadI32x4(f0, i), loadI32x4(f1, i), c)
		r1 := sgrProjectHalf(uSh(sv0.ExtendHi4ToUint32()), loadI32x4(f0, i+4), loadI32x4(f1, i+4), c)
		r2 := sgrProjectHalf(uSh(sv1.ExtendLo4ToUint32()), loadI32x4(f0, i+8), loadI32x4(f1, i+8), c)
		r3 := sgrProjectHalf(uSh(sv1.ExtendHi4ToUint32()), loadI32x4(f0, i+12), loadI32x4(f1, i+12), c)
		narrow(r0, r1).StoreArray((*[8]uint16)(unsafe.Pointer(&dst[i])))
		narrow(r2, r3).StoreArray((*[8]uint16)(unsafe.Pointer(&dst[i+8])))
	}
	if i+8 <= width {
		sv := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[i])))
		rLo := sgrProjectHalf(uSh(sv.ExtendLo4ToUint32()), loadI32x4(f0, i), loadI32x4(f1, i), c)
		rHi := sgrProjectHalf(uSh(sv.ExtendHi4ToUint32()), loadI32x4(f0, i+4), loadI32x4(f1, i+4), c)
		narrow(rLo, rHi).StoreArray((*[8]uint16)(unsafe.Pointer(&dst[i])))
		i += 8
	}
	if i < width {
		sgrWeightedRow(dst[i:], src[i:], f0[i:], f1[i:], xq0, xq1, maxI)
	}
}
