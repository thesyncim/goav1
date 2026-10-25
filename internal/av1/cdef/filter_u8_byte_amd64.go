// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package cdef

// Byte-lane AVX2 CDEF kernels for real-interior 8-bit units, ported from
// dav1d's 8bpc edges == 0xf path (src/x86/cdef_avx2.asm CDEF_FILTER,
// LOAD_BLOCK / ACCUMULATE_TAP_BYTE / ADJUST_PIXEL). Every tap is an 8-bit
// sample, so 32 pixels share one ymm: four 8-pixel rows ([r0 r1 | r2 r3]) for
// 8-wide blocks, and four 4-pixel rows ([r0 r1 . . | r2 r3 . .]) for 4-wide
// blocks, exactly dav1d's LOAD_BLOCK lane layouts.
//
//   - constrain() runs on bytes as in ACCUMULATE_TAP_BYTE: |t-x| from the two
//     saturating differences, the per-block shift emulated with a 16-bit
//     shift plus a byte mask (dav1d's tap_table shift masks), psubusb for the
//     limit. The sign is folded back as min(t-x, lim) - min(x-t, lim) (one of
//     the two saturating differences is zero), which needs no psignb mask.
//   - dav1d widens with pmaddubsw after pairing taps with punpcklbw; archsimd
//     has no byte unpack on amd64, so the pairing is by pixel parity instead:
//     pmaddubsw against [w,0,w,0..] and [0,w,0,w..] yields the even and odd
//     pixels' weighted sums in separate int16 accumulators. Taps sharing a
//     weight are pre-summed in int8 first (a primary pair is at most 30, the
//     secondary 2*S0+S1 at most 48, and the primary (4,2) pairing is folded
//     to 2*(2*P0+P1) <= 180 via a per-block doubling mask), so the pri+sec
//     split costs four pmaddubsw per 32 pixels.
//   - (8 + sum - (sum < 0)) >> 4 is taken in int16 (no pmulhrsw on amd64);
//     the int8 deltas (|d| <= 15) are recombined from the two parities and
//     added to px in bytes. Primary-only and secondary-only results stay in
//     [0, 255] (tap weights sum to 12 < 16), so a wrapping add is exact; with
//     both strengths the add saturates in the 0x80-biased domain and the
//     result is clamped to the tap neighbourhood's [min, max].
//
// Output is bit-identical to filterBlockU8PureGo on the widened pixels; see
// filter_u8_byte_simd_test.go.

import (
	"simd/archsimd"
	"unsafe"
)

// cdefHasAVX2 gates the amd64 byte route at run time.
var cdefHasAVX2 = archsimd.X86.AVX2()

func u8ByteInteriorAvailable() bool { return cdefHasAVX2 }

// cdefByteAVX2Ctx is the per-block state of the byte kernels. Offsets are
// byte offsets into the BStride-spaced byte tap buffer.
type cdefByteAVX2Ctx struct {
	dst    unsafe.Pointer
	src    unsafe.Pointer
	dstStr int
	height int

	pri0, pri1             int
	sec0, sec1, sec2, sec3 int

	priStr, secStr uint8
	priSh, secSh   uint64
	// priW is the primary weight applied after the int8 pre-sum (tap1), and
	// priDbl is 0xff when tap0 == 2*tap1 (strength even: taps 4, 2).
	priW   uint8
	priDbl int8
}

func (c *cdefByteAVX2Ctx) setPrimary(s, damping int) {
	taps := cdefPrimaryTaps[s&1]
	c.priW = taps[1]
	c.priDbl = 0
	if taps[0] == 2*taps[1] {
		c.priDbl = -1
	}
	c.priStr = uint8(s)
	c.priSh = uint64(constrainShift(s, damping))
}

// filterUnitBlocksU8Byte is filterUnitBlocksU8PureGo on a byte tap buffer:
// per-block luma strength adjust, blocks with both strengths zero skipped
// (dav1d cdef_apply_tmpl.c), and the pri / sec / pri+sec kernel picked per
// block as dav1d's cdef_filter_%1x%2_8bpc does.
func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	ctx := cdefByteAVX2Ctx{dstStr: dstStride, height: u.blockHeight}
	sec := u.secondaryStrength
	ctx.secStr = uint8(sec)
	ctx.secSh = uint64(constrainShift(sec, u.damping))
	strength := u.primaryStrength
	if !u.lumaAdjust {
		ctx.setPrimary(strength, u.damping)
	}
	base := unsafe.Pointer(unsafe.SliceData(input))
	dstBase := unsafe.Pointer(unsafe.SliceData(dst))
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			ctx.setPrimary(strength, u.damping)
		}
		if strength == 0 && sec == 0 {
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.pri0 = int(cdefDirections[dir+2][0])
		ctx.pri1 = int(cdefDirections[dir+2][1])
		ctx.sec0 = int(cdefDirections[dir+4][0])
		ctx.sec1 = int(cdefDirections[dir][0])
		ctx.sec2 = int(cdefDirections[dir+4][1])
		ctx.sec3 = int(cdefDirections[dir][1])
		ctx.src = unsafe.Add(base, inputOrigin+((by*BStride)<<u.bhLog2)+(bx<<u.bwLog2))
		ctx.dst = unsafe.Add(dstBase, (by<<u.bhLog2)*dstStride+(bx<<u.bwLog2))
		w8 := u.blockWidth == 8
		switch {
		case strength != 0 && sec != 0:
			if w8 {
				cdefByteAVX2Filter8PriSec(&ctx)
			} else {
				cdefByteAVX2Filter4PriSec(&ctx)
			}
		case strength != 0:
			if w8 {
				cdefByteAVX2Filter8Pri(&ctx)
			} else {
				cdefByteAVX2Filter4Pri(&ctx)
			}
		default:
			if w8 {
				cdefByteAVX2Filter8Sec(&ctx)
			} else {
				cdefByteAVX2Filter4Sec(&ctx)
			}
		}
	}
}

func cdefByteLd16(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// cdefByteAVX2Load8 packs four 8-pixel rows at p as [r0 r1 | r2 r3]
// (LOAD_BLOCK w=8: movq + vinserti128 + punpcklqdq).
func cdefByteAVX2Load8(p unsafe.Pointer) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	ac := z.SetLo(cdefByteLd16(p)).SetHi(cdefByteLd16(unsafe.Add(p, 2*BStride))).AsUint64x4()
	bd := z.SetLo(cdefByteLd16(unsafe.Add(p, BStride))).SetHi(cdefByteLd16(unsafe.Add(p, 3*BStride))).AsUint64x4()
	return ac.InterleaveLoGrouped(bd).AsUint8x32()
}

// cdefByteAVX2Load4 packs four 4-pixel rows at p as [r0 r1 . . | r2 r3 . .]
// (LOAD_BLOCK w=4 h=4: movd + vinserti128 + punpckldq). Bytes 8..15 of each
// lane are don't-care.
func cdefByteAVX2Load4(p unsafe.Pointer) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	ac := z.SetLo(cdefByteLd16(p)).SetHi(cdefByteLd16(unsafe.Add(p, 2*BStride))).AsUint32x8()
	bd := z.SetLo(cdefByteLd16(unsafe.Add(p, BStride))).SetHi(cdefByteLd16(unsafe.Add(p, 3*BStride))).AsUint32x8()
	return ac.InterleaveLoGrouped(bd).AsUint8x32()
}

func cdefByteAVX2Store8(dst unsafe.Pointer, dstStr int, v archsimd.Uint8x32) {
	lo := v.GetLo().AsUint64x2()
	hi := v.GetHi().AsUint64x2()
	*(*uint64)(dst) = lo.GetElem(0)
	*(*uint64)(unsafe.Add(dst, dstStr)) = lo.GetElem(1)
	*(*uint64)(unsafe.Add(dst, 2*dstStr)) = hi.GetElem(0)
	*(*uint64)(unsafe.Add(dst, 3*dstStr)) = hi.GetElem(1)
}

func cdefByteAVX2Store4(dst unsafe.Pointer, dstStr int, v archsimd.Uint8x32) {
	lo := v.GetLo().AsUint32x4()
	hi := v.GetHi().AsUint32x4()
	*(*uint32)(dst) = lo.GetElem(0)
	*(*uint32)(unsafe.Add(dst, dstStr)) = lo.GetElem(1)
	*(*uint32)(unsafe.Add(dst, 2*dstStr)) = hi.GetElem(0)
	*(*uint32)(unsafe.Add(dst, 3*dstStr)) = hi.GetElem(1)
}

// cdefByteAVX2Constrain is ACCUMULATE_TAP_BYTE's constrain(): the signed
// byte min(|t-x|, max(0, str - (|t-x| >> sh))) carrying the sign of t-x.
// The byte shift is a 16-bit shift masked to the surviving bits (mask is
// 0xff >> sh broadcast).
func cdefByteAVX2Constrain(t, x, str archsimd.Uint8x32, sh uint64, mask archsimd.Uint8x32) archsimd.Int8x32 {
	pos := t.SubSaturated(x)
	neg := x.SubSaturated(t)
	ad := pos.Or(neg)
	lim := str.SubSaturated(ad.AsUint16x16().ShiftAllRight(sh).AsUint8x32().And(mask))
	return pos.Min(lim).AsInt8x32().Sub(neg.Min(lim).AsInt8x32())
}

// cdefByteAVX2Delta is ((8 + sum - (sum < 0)) >> 4) on one parity's int16
// sums.
func cdefByteAVX2Delta(sum, eight archsimd.Int16x16) archsimd.Int16x16 {
	return sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4)
}

// cdefByteAVX2Join recombines the even- and odd-pixel int16 deltas into
// one int8 delta per byte lane.
func cdefByteAVX2Join(dE, dO archsimd.Int16x16, lowMask archsimd.Uint16x16) archsimd.Uint8x32 {
	return dE.AsUint16x16().And(lowMask).Or(dO.AsUint16x16().ShiftAllLeft(8)).AsUint8x32()
}

// cdefByteAVX2Weights returns the even- and odd-pixel pmaddubsw weight
// patterns [w,0,w,0..] and [0,w,0,w..].
func cdefByteAVX2Weights(w uint8) (archsimd.Uint8x32, archsimd.Uint8x32) {
	return archsimd.BroadcastUint16x16(uint16(w)).AsUint8x32(), archsimd.BroadcastUint16x16(uint16(w) << 8).AsUint8x32()
}

func cdefByteAVX2Filter8PriSec(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	priSh, secSh := c.priSh, c.secSh
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> priSh))
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> secSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	bias := archsimd.BroadcastUint8x32(0x80)
	pri0, pri1 := c.pri0, c.pri1
	sec0, sec1, sec2, sec3 := c.sec0, c.sec1, c.sec2, c.sec3
	src, dst, dstStr := c.src, c.dst, c.dstStr
	for h := c.height; h > 0; h -= 4 {
		x := cdefByteAVX2Load8(src)
		p, q := cdefByteAVX2Load8(unsafe.Add(src, pri0)), cdefByteAVX2Load8(unsafe.Add(src, -pri0))
		mn, mx := x.Min(p).Min(q), x.Max(p).Max(q)
		p0 := cdefByteAVX2Constrain(p, x, priStr, priSh, priMask).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load8(unsafe.Add(src, pri1)), cdefByteAVX2Load8(unsafe.Add(src, -pri1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		w := p0.Add(p0.And(dbl)).Add(cdefByteAVX2Constrain(p, x, priStr, priSh, priMask)).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load8(unsafe.Add(src, sec0)), cdefByteAVX2Load8(unsafe.Add(src, -sec0))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 := cdefByteAVX2Constrain(p, x, secStr, secSh, secMask).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(unsafe.Add(src, sec1)), cdefByteAVX2Load8(unsafe.Add(src, -sec1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 = s0.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(unsafe.Add(src, sec2)), cdefByteAVX2Load8(unsafe.Add(src, -sec2))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s := s0.Add(s0).Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(unsafe.Add(src, sec3)), cdefByteAVX2Load8(unsafe.Add(src, -sec3))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s = s.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		sumE := pE.DotProductPairsSaturated(w).Add(sE.DotProductPairsSaturated(s))
		sumO := pO.DotProductPairsSaturated(w).Add(sO.DotProductPairsSaturated(s))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sumE, eight), cdefByteAVX2Delta(sumO, eight), lowMask)
		y := x.Xor(bias).AsInt8x32().AddSaturated(d.AsInt8x32()).AsUint8x32().Xor(bias).Min(mx).Max(mn)
		cdefByteAVX2Store8(dst, dstStr, y)
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteAVX2Filter8Pri(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	priSh := c.priSh
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> priSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	pri0, pri1 := c.pri0, c.pri1
	src, dst, dstStr := c.src, c.dst, c.dstStr
	for h := c.height; h > 0; h -= 4 {
		x := cdefByteAVX2Load8(src)
		p0 := cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, pri0)), x, priStr, priSh, priMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, -pri0)), x, priStr, priSh, priMask))
		w := p0.Add(p0.And(dbl)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, pri1)), x, priStr, priSh, priMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, -pri1)), x, priStr, priSh, priMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(pE.DotProductPairsSaturated(w), eight), cdefByteAVX2Delta(pO.DotProductPairsSaturated(w), eight), lowMask)
		cdefByteAVX2Store8(dst, dstStr, x.Add(d))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteAVX2Filter8Sec(c *cdefByteAVX2Ctx) {
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	secSh := c.secSh
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> secSh))
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	sec0, sec1, sec2, sec3 := c.sec0, c.sec1, c.sec2, c.sec3
	src, dst, dstStr := c.src, c.dst, c.dstStr
	for h := c.height; h > 0; h -= 4 {
		x := cdefByteAVX2Load8(src)
		s0 := cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, sec0)), x, secStr, secSh, secMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, -sec0)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, sec1)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, -sec1)), x, secStr, secSh, secMask))
		s := s0.Add(s0).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, sec2)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, -sec2)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, sec3)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(unsafe.Add(src, -sec3)), x, secStr, secSh, secMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sE.DotProductPairsSaturated(s), eight), cdefByteAVX2Delta(sO.DotProductPairsSaturated(s), eight), lowMask)
		cdefByteAVX2Store8(dst, dstStr, x.Add(d))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteAVX2Filter4PriSec(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	priSh, secSh := c.priSh, c.secSh
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> priSh))
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> secSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	bias := archsimd.BroadcastUint8x32(0x80)
	pri0, pri1 := c.pri0, c.pri1
	sec0, sec1, sec2, sec3 := c.sec0, c.sec1, c.sec2, c.sec3
	src, dst, dstStr := c.src, c.dst, c.dstStr
	for h := c.height; h > 0; h -= 4 {
		x := cdefByteAVX2Load4(src)
		p, q := cdefByteAVX2Load4(unsafe.Add(src, pri0)), cdefByteAVX2Load4(unsafe.Add(src, -pri0))
		mn, mx := x.Min(p).Min(q), x.Max(p).Max(q)
		p0 := cdefByteAVX2Constrain(p, x, priStr, priSh, priMask).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load4(unsafe.Add(src, pri1)), cdefByteAVX2Load4(unsafe.Add(src, -pri1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		w := p0.Add(p0.And(dbl)).Add(cdefByteAVX2Constrain(p, x, priStr, priSh, priMask)).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load4(unsafe.Add(src, sec0)), cdefByteAVX2Load4(unsafe.Add(src, -sec0))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 := cdefByteAVX2Constrain(p, x, secStr, secSh, secMask).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(unsafe.Add(src, sec1)), cdefByteAVX2Load4(unsafe.Add(src, -sec1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 = s0.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(unsafe.Add(src, sec2)), cdefByteAVX2Load4(unsafe.Add(src, -sec2))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s := s0.Add(s0).Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(unsafe.Add(src, sec3)), cdefByteAVX2Load4(unsafe.Add(src, -sec3))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s = s.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		sumE := pE.DotProductPairsSaturated(w).Add(sE.DotProductPairsSaturated(s))
		sumO := pO.DotProductPairsSaturated(w).Add(sO.DotProductPairsSaturated(s))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sumE, eight), cdefByteAVX2Delta(sumO, eight), lowMask)
		y := x.Xor(bias).AsInt8x32().AddSaturated(d.AsInt8x32()).AsUint8x32().Xor(bias).Min(mx).Max(mn)
		cdefByteAVX2Store4(dst, dstStr, y)
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteAVX2Filter4Pri(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	priSh := c.priSh
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> priSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	pri0, pri1 := c.pri0, c.pri1
	src, dst, dstStr := c.src, c.dst, c.dstStr
	for h := c.height; h > 0; h -= 4 {
		x := cdefByteAVX2Load4(src)
		p0 := cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, pri0)), x, priStr, priSh, priMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, -pri0)), x, priStr, priSh, priMask))
		w := p0.Add(p0.And(dbl)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, pri1)), x, priStr, priSh, priMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, -pri1)), x, priStr, priSh, priMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(pE.DotProductPairsSaturated(w), eight), cdefByteAVX2Delta(pO.DotProductPairsSaturated(w), eight), lowMask)
		cdefByteAVX2Store4(dst, dstStr, x.Add(d))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteAVX2Filter4Sec(c *cdefByteAVX2Ctx) {
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	secSh := c.secSh
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> secSh))
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	sec0, sec1, sec2, sec3 := c.sec0, c.sec1, c.sec2, c.sec3
	src, dst, dstStr := c.src, c.dst, c.dstStr
	for h := c.height; h > 0; h -= 4 {
		x := cdefByteAVX2Load4(src)
		s0 := cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, sec0)), x, secStr, secSh, secMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, -sec0)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, sec1)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, -sec1)), x, secStr, secSh, secMask))
		s := s0.Add(s0).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, sec2)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, -sec2)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, sec3)), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(unsafe.Add(src, -sec3)), x, secStr, secSh, secMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sE.DotProductPairsSaturated(s), eight), cdefByteAVX2Delta(sO.DotProductPairsSaturated(s), eight), lowMask)
		cdefByteAVX2Store4(dst, dstStr, x.Add(d))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}
