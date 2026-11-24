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

import "simd/archsimd"

// cdefHasAVX2 gates the amd64 byte route at run time.
var cdefHasAVX2 = archsimd.X86.AVX2()

func u8ByteInteriorAvailable() bool { return cdefHasAVX2 }

// cdefByteAVX2Ctx is the per-block state of the byte kernels. Offsets are
// byte offsets into the BStride-spaced byte tap buffer.
type cdefByteAVX2Ctx struct {
	input  []byte
	dst    []byte
	srcOff int // the block's first tap-buffer byte
	dstOff int // the block's first output byte
	dstStr int
	height int

	// Tap window indices cdefByteCenter +- the direction's tap offsets
	// (cdefTapIdx's layout: p0+, p0-, p1+, p1-, s0+, s0-, ..., s3-).
	taps *[12]uint

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
	ctx.input, ctx.dst = input, dst
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
		ctx.taps = &cdefByteTapIdx[dir&7]
		ctx.srcOff = inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		ctx.dstOff = (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
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

// The kernels take a four-row group's taps from one window of the byte tap
// buffer starting cdefByteCenter bytes (two rows and two columns, the
// farthest tap) before the group's first pixel; each row's taps are a
// constant-offset sub-array of it, and a tap at offset t sits at
// cdefByteCenter + t there. One bounds check per group.
const (
	cdefByteCenter = 2*BStride + 2
	cdefByteMaxOff = 2 * cdefByteCenter
	cdefByteWinRow = cdefByteMaxOff + 16
	cdefByteWin4   = cdefByteWinRow + 3*BStride
)

// cdefByteTapIdx is cdefTapIdx for the byte tap buffer.
var cdefByteTapIdx = func() (t [8][12]uint) {
	for d := range t {
		offs := [6]int{
			int(cdefDirections[d+2][0]), int(cdefDirections[d+2][1]),
			int(cdefDirections[d+4][0]), int(cdefDirections[d][0]),
			int(cdefDirections[d+4][1]), int(cdefDirections[d][1]),
		}
		for k, o := range offs {
			t[d][2*k], t[d][2*k+1] = uint(cdefByteCenter+o), uint(cdefByteCenter-o)
		}
	}
	return
}()

const errCDEFByteWindow = "cdef: byte block outside the tap buffer or destination"

// cdefByteRows4 returns the four rows' tap windows of the group whose window
// starts at o.
func cdefByteRows4(in []byte, o int) (*[cdefByteWinRow]byte, *[cdefByteWinRow]byte, *[cdefByteWinRow]byte, *[cdefByteWinRow]byte) {
	g := (*[cdefByteWin4]byte)(in[o : o+cdefByteWin4])
	return (*[cdefByteWinRow]byte)(g[0:cdefByteWinRow]), (*[cdefByteWinRow]byte)(g[BStride : BStride+cdefByteWinRow]),
		(*[cdefByteWinRow]byte)(g[2*BStride : 2*BStride+cdefByteWinRow]), (*[cdefByteWinRow]byte)(g[3*BStride : 3*BStride+cdefByteWinRow])
}

// rows checks the block origin, height and output layout, so each group
// window and output row below is one compare, and returns the first group's
// window start.
func (c *cdefByteAVX2Ctx) rows() int {
	if c.srcOff < cdefByteCenter || c.srcOff > 1<<24 || uint(c.height) > 128 ||
		uint(c.dstOff) > 1<<30 || uint(c.dstStr) > 1<<20 {
		panic(errCDEFByteWindow)
	}
	return c.srcOff - cdefByteCenter
}

// priTaps / secTaps0 / secTaps1 return tap window indices checked against
// the row window, so the tap loads need no bounds checks.
func (c *cdefByteAVX2Ctx) priTaps() (uint, uint, uint, uint) {
	t := c.taps
	a, b, d, e := t[0], t[1], t[2], t[3]
	if a > cdefByteMaxOff || b > cdefByteMaxOff || d > cdefByteMaxOff || e > cdefByteMaxOff {
		panic(errCDEFByteWindow)
	}
	return a, b, d, e
}

func (c *cdefByteAVX2Ctx) secTaps0() (uint, uint, uint, uint) {
	t := c.taps
	a, b, d, e := t[4], t[5], t[6], t[7]
	if a > cdefByteMaxOff || b > cdefByteMaxOff || d > cdefByteMaxOff || e > cdefByteMaxOff {
		panic(errCDEFByteWindow)
	}
	return a, b, d, e
}

func (c *cdefByteAVX2Ctx) secTaps1() (uint, uint, uint, uint) {
	t := c.taps
	a, b, d, e := t[8], t[9], t[10], t[11]
	if a > cdefByteMaxOff || b > cdefByteMaxOff || d > cdefByteMaxOff || e > cdefByteMaxOff {
		panic(errCDEFByteWindow)
	}
	return a, b, d, e
}

func cdefByteLd16(r *[cdefByteWinRow]byte, off uint) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(r[off : off+16]))
}

// cdefByteAVX2Load8 packs the four rows' 8 pixels at off as [r0 r1 | r2 r3]
// (LOAD_BLOCK w=8: movq + vinserti128 + punpcklqdq).
func cdefByteAVX2Load8(r0, r1, r2, r3 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	ac := z.SetLo(cdefByteLd16(r0, off)).SetHi(cdefByteLd16(r2, off)).AsUint64x4()
	bd := z.SetLo(cdefByteLd16(r1, off)).SetHi(cdefByteLd16(r3, off)).AsUint64x4()
	return ac.InterleaveLoGrouped(bd).AsUint8x32()
}

// cdefByteAVX2Load4 packs the four rows' 4 pixels at off as
// [r0 r1 . . | r2 r3 . .] (LOAD_BLOCK w=4 h=4: movd + vinserti128 +
// punpckldq). Bytes 8..15 of each lane are don't-care.
func cdefByteAVX2Load4(r0, r1, r2, r3 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	ac := z.SetLo(cdefByteLd16(r0, off)).SetHi(cdefByteLd16(r2, off)).AsUint32x8()
	bd := z.SetLo(cdefByteLd16(r1, off)).SetHi(cdefByteLd16(r3, off)).AsUint32x8()
	return ac.InterleaveLoGrouped(bd).AsUint8x32()
}

func cdefBytePut8(b []byte, o int, v uint64) {
	w := (*[8]byte)(b[o : o+8])
	w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

func cdefBytePut4(b []byte, o int, v uint32) {
	w := (*[4]byte)(b[o : o+4])
	w[0], w[1], w[2], w[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func cdefByteAVX2Store8(dst []byte, o, dstStr int, v archsimd.Uint8x32) {
	lo := v.GetLo().AsUint64x2()
	hi := v.GetHi().AsUint64x2()
	cdefBytePut8(dst, o, lo.GetElem(0))
	cdefBytePut8(dst, o+dstStr, lo.GetElem(1))
	cdefBytePut8(dst, o+2*dstStr, hi.GetElem(0))
	cdefBytePut8(dst, o+3*dstStr, hi.GetElem(1))
}

func cdefByteAVX2Store4(dst []byte, o, dstStr int, v archsimd.Uint8x32) {
	lo := v.GetLo().AsUint32x4()
	hi := v.GetHi().AsUint32x4()
	cdefBytePut4(dst, o, lo.GetElem(0))
	cdefBytePut4(dst, o+dstStr, lo.GetElem(1))
	cdefBytePut4(dst, o+2*dstStr, hi.GetElem(0))
	cdefBytePut4(dst, o+3*dstStr, hi.GetElem(1))
}

// cdefByteAVX2Constrain is ACCUMULATE_TAP_BYTE's constrain(): the signed
// byte min(|t-x|, max(0, str - (|t-x| >> sh))) carrying the sign of t-x.
// The byte shift is a dword shift masked to the surviving bits (mask is
// 0xff >> sh broadcast). It is VPSRLVD by a hoisted count vector rather than
// VPSRLW by an xmm count: the compiler rematerializes that count with a
// legacy-SSE MOVQ inside the row loop, an SSE/AVX transition while the ymm
// upper halves are live.
func cdefByteAVX2Constrain(t, x, str archsimd.Uint8x32, sh archsimd.Uint32x8, mask archsimd.Uint8x32) archsimd.Int8x32 {
	pos := t.SubSaturated(x)
	neg := x.SubSaturated(t)
	ad := pos.Or(neg)
	lim := str.SubSaturated(ad.AsUint32x8().ShiftRight(sh).AsUint8x32().And(mask))
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
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(c.priSh)), archsimd.BroadcastUint32x8(uint32(c.secSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	bias := archsimd.BroadcastUint8x32(0x80)
	p0p, p0m, p1p, p1m := c.priTaps()
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load8(r0, r1, r2, r3, cdefByteCenter)
		p, q := cdefByteAVX2Load8(r0, r1, r2, r3, p0p), cdefByteAVX2Load8(r0, r1, r2, r3, p0m)
		mn, mx := x.Min(p).Min(q), x.Max(p).Max(q)
		p0 := cdefByteAVX2Constrain(p, x, priStr, priSh, priMask).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, p1p), cdefByteAVX2Load8(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		w := p0.Add(p0.And(dbl)).Add(cdefByteAVX2Constrain(p, x, priStr, priSh, priMask)).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s0p), cdefByteAVX2Load8(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 := cdefByteAVX2Constrain(p, x, secStr, secSh, secMask).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s1p), cdefByteAVX2Load8(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 = s0.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s2p), cdefByteAVX2Load8(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s := s0.Add(s0).Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s3p), cdefByteAVX2Load8(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s = s.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		sumE := pE.DotProductPairsSaturated(w).Add(sE.DotProductPairsSaturated(s))
		sumO := pO.DotProductPairsSaturated(w).Add(sO.DotProductPairsSaturated(s))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sumE, eight), cdefByteAVX2Delta(sumO, eight), lowMask)
		y := x.Xor(bias).AsInt8x32().AddSaturated(d.AsInt8x32()).AsUint8x32().Xor(bias).Min(mx).Max(mn)
		cdefByteAVX2Store8(dst, do, dstStr, y)
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter8Pri(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	priSh := archsimd.BroadcastUint32x8(uint32(c.priSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	p0p, p0m, p1p, p1m := c.priTaps()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load8(r0, r1, r2, r3, cdefByteCenter)
		p0 := cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p0p), x, priStr, priSh, priMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p0m), x, priStr, priSh, priMask))
		w := p0.Add(p0.And(dbl)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p1p), x, priStr, priSh, priMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p1m), x, priStr, priSh, priMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(pE.DotProductPairsSaturated(w), eight), cdefByteAVX2Delta(pO.DotProductPairsSaturated(w), eight), lowMask)
		cdefByteAVX2Store8(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter8Sec(c *cdefByteAVX2Ctx) {
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	secSh := archsimd.BroadcastUint32x8(uint32(c.secSh))
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load8(r0, r1, r2, r3, cdefByteCenter)
		s0 := cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s0p), x, secStr, secSh, secMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s0m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s1p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s1m), x, secStr, secSh, secMask))
		s := s0.Add(s0).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s2p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s2m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s3p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s3m), x, secStr, secSh, secMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sE.DotProductPairsSaturated(s), eight), cdefByteAVX2Delta(sO.DotProductPairsSaturated(s), eight), lowMask)
		cdefByteAVX2Store8(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter4PriSec(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(c.priSh)), archsimd.BroadcastUint32x8(uint32(c.secSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	bias := archsimd.BroadcastUint8x32(0x80)
	p0p, p0m, p1p, p1m := c.priTaps()
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load4(r0, r1, r2, r3, cdefByteCenter)
		p, q := cdefByteAVX2Load4(r0, r1, r2, r3, p0p), cdefByteAVX2Load4(r0, r1, r2, r3, p0m)
		mn, mx := x.Min(p).Min(q), x.Max(p).Max(q)
		p0 := cdefByteAVX2Constrain(p, x, priStr, priSh, priMask).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, p1p), cdefByteAVX2Load4(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		w := p0.Add(p0.And(dbl)).Add(cdefByteAVX2Constrain(p, x, priStr, priSh, priMask)).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s0p), cdefByteAVX2Load4(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 := cdefByteAVX2Constrain(p, x, secStr, secSh, secMask).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s1p), cdefByteAVX2Load4(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 = s0.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s2p), cdefByteAVX2Load4(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s := s0.Add(s0).Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s3p), cdefByteAVX2Load4(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s = s.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		sumE := pE.DotProductPairsSaturated(w).Add(sE.DotProductPairsSaturated(s))
		sumO := pO.DotProductPairsSaturated(w).Add(sO.DotProductPairsSaturated(s))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sumE, eight), cdefByteAVX2Delta(sumO, eight), lowMask)
		y := x.Xor(bias).AsInt8x32().AddSaturated(d.AsInt8x32()).AsUint8x32().Xor(bias).Min(mx).Max(mn)
		cdefByteAVX2Store4(dst, do, dstStr, y)
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter4Pri(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	priSh := archsimd.BroadcastUint32x8(uint32(c.priSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	p0p, p0m, p1p, p1m := c.priTaps()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load4(r0, r1, r2, r3, cdefByteCenter)
		p0 := cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p0p), x, priStr, priSh, priMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p0m), x, priStr, priSh, priMask))
		w := p0.Add(p0.And(dbl)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p1p), x, priStr, priSh, priMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p1m), x, priStr, priSh, priMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(pE.DotProductPairsSaturated(w), eight), cdefByteAVX2Delta(pO.DotProductPairsSaturated(w), eight), lowMask)
		cdefByteAVX2Store4(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter4Sec(c *cdefByteAVX2Ctx) {
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	secSh := archsimd.BroadcastUint32x8(uint32(c.secSh))
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load4(r0, r1, r2, r3, cdefByteCenter)
		s0 := cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s0p), x, secStr, secSh, secMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s0m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s1p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s1m), x, secStr, secSh, secMask))
		s := s0.Add(s0).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s2p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s2m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s3p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s3m), x, secStr, secSh, secMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sE.DotProductPairsSaturated(s), eight), cdefByteAVX2Delta(sO.DotProductPairsSaturated(s), eight), lowMask)
		cdefByteAVX2Store4(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}
