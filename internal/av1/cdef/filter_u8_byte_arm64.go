// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package cdef

// Safe-Go byte-lane CDEF kernels for real-interior 8-bit units, ported from
// dav1d's 8bpc edged NEON filters (src/arm/64/cdef.S filter_func_8 /
// handle_pixel_8). Every tap is an 8-bit sample, so the whole filter runs in
// 16 byte lanes: two 8-pixel rows per vector for 8-wide blocks, four 4-pixel
// rows for 4-wide blocks.
//
// Each strip slices its local tap window into fixed-size array pointers
// (*[cdefByteWinRow]byte per row) once per strip with bounded positive tap
// offsets, so Go's bounds-check elimination drops all inner load checks
// without using unsafe.
//
// Output is bit-identical to filterBlockU8PureGo on the same pixels; see
// filter_u8_byte_simd_test.go.

import (
	"encoding/binary"
	"simd/archsimd"
)

const (
	cdefByteCenter = 2*BStride + 2
	cdefByteMaxOff = 4*BStride + 4
	cdefByteWinRow = 4*BStride + 20 // max tap (4*BStride+4) + 16
	cdefByteWin8   = 5*BStride + 20 // max tap + BStride + 16
	cdefByteWin4   = 7*BStride + 20 // max tap + 3*BStride + 16
)

func u8ByteInteriorAvailable() bool { return true }

type cdefByteCtx struct {
	input  []byte
	dst    []byte
	srcOff int
	dstOff int
	dstStr int
	height int

	p0p, p0m uint
	p1p, p1m uint
	s0p, s0m uint
	s1p, s1m uint
	s2p, s2m uint
	s3p, s3m uint

	priStr, secStr archsimd.Uint8x16
	priSh, secSh   archsimd.Int8x16
	tap0, tap1     archsimd.Int8x16
}

// filterUnitBlocksU8Byte is filterUnitBlocksU8PureGo on a byte tap buffer:
// per-block luma strength adjust, blocks with both strengths zero skipped
// (dav1d cdef_apply_tmpl.c), and the pri / sec / pri+sec kernel picked per
// block as dav1d's cdef_filter_block_*_neon does.
func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	sec := u.secondaryStrength
	ctx := cdefByteCtx{
		input:  input,
		dst:    dst,
		dstStr: dstStride,
		height: u.blockHeight,
		secStr: archsimd.BroadcastUint8x16(uint8(sec)),
		secSh:  archsimd.BroadcastInt8x16(-int8(constrainShift(sec, u.damping))),
	}

	strength := u.primaryStrength
	prevStrength := -1
	if !u.lumaAdjust {
		prevStrength = strength
		if strength != 0 {
			taps := cdefPrimaryTaps[strength&1]
			ctx.priStr = archsimd.BroadcastUint8x16(uint8(strength))
			ctx.priSh = archsimd.BroadcastInt8x16(-int8(constrainShift(strength, u.damping)))
			ctx.tap0 = archsimd.BroadcastInt8x16(int8(taps[0]))
			ctx.tap1 = archsimd.BroadcastInt8x16(int8(taps[1]))
		}
	}

	bwLog2 := u.bwLog2
	bhLog2 := u.bhLog2
	isW8 := u.blockWidth == 8

	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			if strength != prevStrength {
				prevStrength = strength
				if strength != 0 {
					taps := cdefPrimaryTaps[strength&1]
					ctx.priStr = archsimd.BroadcastUint8x16(uint8(strength))
					ctx.priSh = archsimd.BroadcastInt8x16(-int8(constrainShift(strength, u.damping)))
					ctx.tap0 = archsimd.BroadcastInt8x16(int8(taps[0]))
					ctx.tap1 = archsimd.BroadcastInt8x16(int8(taps[1]))
				}
			}
		}
		if strength == 0 && sec == 0 {
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.srcOff = inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		ctx.dstOff = (by<<bhLog2)*dstStride + (bx << bwLog2)
		if strength != 0 {
			pri0 := int(cdefDirections[dir+2][0])
			pri1 := int(cdefDirections[dir+2][1])
			ctx.p0p, ctx.p0m = uint(cdefByteCenter+pri0), uint(cdefByteCenter-pri0)
			ctx.p1p, ctx.p1m = uint(cdefByteCenter+pri1), uint(cdefByteCenter-pri1)
		}
		if sec != 0 {
			sec0, sec1 := int(cdefDirections[dir+4][0]), int(cdefDirections[dir][0])
			sec2, sec3 := int(cdefDirections[dir+4][1]), int(cdefDirections[dir][1])
			ctx.s0p, ctx.s0m = uint(cdefByteCenter+sec0), uint(cdefByteCenter-sec0)
			ctx.s1p, ctx.s1m = uint(cdefByteCenter+sec1), uint(cdefByteCenter-sec1)
			ctx.s2p, ctx.s2m = uint(cdefByteCenter+sec2), uint(cdefByteCenter-sec2)
			ctx.s3p, ctx.s3m = uint(cdefByteCenter+sec3), uint(cdefByteCenter-sec3)
		}
		if isW8 {
			switch {
			case strength != 0 && sec != 0:
				cdefByteFilter8PriSec(&ctx)
			case strength != 0:
				cdefByteFilter8Pri(&ctx)
			default:
				cdefByteFilter8Sec(&ctx)
			}
			continue
		}
		switch {
		case strength != 0 && sec != 0:
			cdefByteFilter4PriSec(&ctx)
		case strength != 0:
			cdefByteFilter4Pri(&ctx)
		default:
			cdefByteFilter4Sec(&ctx)
		}
	}
}

// cdefByteLoad8x2Rows packs the 8 bytes at r0[off:] (low half) and r1[off:]
// (high half). Callers bound off <= cdefByteMaxOff so BCE drops all checks.
func cdefByteLoad8x2Rows(r0, r1 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16(r0[off : off+16]).ReshapeToUint64s()
	b := archsimd.LoadUint8x16(r1[off : off+16]).ReshapeToUint64s()
	return a.InterleaveLo(b).ReshapeToUint8s()
}

// cdefByteLoad4x4Rows packs the 4 bytes at r0..r3[off:] in row order.
// Callers bound off <= cdefByteMaxOff so BCE drops all checks.
func cdefByteLoad4x4Rows(r0, r1, r2, r3 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16(r0[off : off+16]).ReshapeToUint32s()
	b := archsimd.LoadUint8x16(r1[off : off+16]).ReshapeToUint32s()
	c := archsimd.LoadUint8x16(r2[off : off+16]).ReshapeToUint32s()
	d := archsimd.LoadUint8x16(r3[off : off+16]).ReshapeToUint32s()
	ab := a.InterleaveLo(b).ReshapeToUint64s()
	cd := c.InterleaveLo(d).ReshapeToUint64s()
	return ab.InterleaveLo(cd).ReshapeToUint8s()
}

// cdefByteConstrain is handle_pixel_8's constrain(): the signed byte
// min(|t-x|, max(0, str - (|t-x| >> shift))) carrying the sign of t-x.
func cdefByteConstrain(t, x, str archsimd.Uint8x16, sh archsimd.Int8x16) archsimd.Int8x16 {
	pos := t.SubSaturated(x)
	neg := x.SubSaturated(t)
	lim := str.SubSaturated(pos.Or(neg).Shift(sh))
	return pos.Min(lim).BitsToInt8().Sub(neg.Min(lim).BitsToInt8())
}

// cdefByteFinish turns the two accumulators (a seeded with -1) into the
// filtered pixels: x + ((8 + sum - (sum < 0)) >> 4) saturated to [0, 255].
func cdefByteFinish(x archsimd.Uint8x16, a, b archsimd.Int8x16, four archsimd.Int8x16, bias archsimd.Uint8x16) archsimd.Uint8x16 {
	r := a.Average(b)
	odd := a.Xor(b).ShiftAllLeft(7).ShiftAllRight(7)
	h := r.Add(odd.And(r.ShiftAllRight(7)))
	d := h.Add(four).ShiftAllRight(3)
	return x.Xor(bias).BitsToInt8().AddSaturated(d).ToBits().Xor(bias)
}

func cdefByteFilter8PriSec(ctx *cdefByteCtx) {
	p0p, p0m := ctx.p0p, ctx.p0m
	p1p, p1m := ctx.p1p, ctx.p1m
	s0p, s0m := ctx.s0p, ctx.s0m
	s1p, s1m := ctx.s1p, ctx.s1m
	s2p, s2m := ctx.s2p, ctx.s2m
	s3p, s3m := ctx.s3p, ctx.s3m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff ||
		s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	priStr, secStr := ctx.priStr, ctx.secStr
	priSh, secSh := ctx.priSh, ctx.secSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		w := (*[cdefByteWin8]byte)(input[base : base+cdefByteWin8])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
		mn, mx := x, x
		p, q := cdefByteLoad8x2Rows(r0, r1, p0p), cdefByteLoad8x2Rows(r0, r1, p0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad8x2Rows(r0, r1, p1p), cdefByteLoad8x2Rows(r0, r1, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad8x2Rows(r0, r1, s0p), cdefByteLoad8x2Rows(r0, r1, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad8x2Rows(r0, r1, s1p), cdefByteLoad8x2Rows(r0, r1, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad8x2Rows(r0, r1, s2p), cdefByteLoad8x2Rows(r0, r1, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad8x2Rows(r0, r1, s3p), cdefByteLoad8x2Rows(r0, r1, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		y := cdefByteFinish(x, a, b, four, bias).Min(mx).Max(mn).ReshapeToUint64s()
		binary.LittleEndian.PutUint64(dst[dstOff:dstOff+8], y.GetElem(0))
		binary.LittleEndian.PutUint64(dst[dstOff+dstStr:dstOff+dstStr+8], y.GetElem(1))
		base += 2 * BStride
		dstOff += 2 * dstStr
	}
}

func cdefByteFilter8Pri(ctx *cdefByteCtx) {
	p0p, p0m := ctx.p0p, ctx.p0m
	p1p, p1m := ctx.p1p, ctx.p1m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff {
		return
	}
	priStr, priSh := ctx.priStr, ctx.priSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		w := (*[cdefByteWin8]byte)(input[base : base+cdefByteWin8])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
		a := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p0p), x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p0m), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p1p), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p1m), x, priStr, priSh).MulAdd(tap1, b)
		y := cdefByteFinish(x, a, b, four, bias).ReshapeToUint64s()
		binary.LittleEndian.PutUint64(dst[dstOff:dstOff+8], y.GetElem(0))
		binary.LittleEndian.PutUint64(dst[dstOff+dstStr:dstOff+dstStr+8], y.GetElem(1))
		base += 2 * BStride
		dstOff += 2 * dstStr
	}
}

func cdefByteFilter8Sec(ctx *cdefByteCtx) {
	s0p, s0m := ctx.s0p, ctx.s0m
	s1p, s1m := ctx.s1p, ctx.s1m
	s2p, s2m := ctx.s2p, ctx.s2m
	s3p, s3m := ctx.s3p, ctx.s3m
	if s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	secStr, secSh := ctx.secStr, ctx.secSh
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		w := (*[cdefByteWin8]byte)(input[base : base+cdefByteWin8])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
		ka := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s0p), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s1p), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s0m), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s1m), x, secStr, secSh))
		a := ka.Add(ka).Add(neg1).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s2p), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s3p), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s2m), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s3m), x, secStr, secSh))
		y := cdefByteFinish(x, a, b, four, bias).ReshapeToUint64s()
		binary.LittleEndian.PutUint64(dst[dstOff:dstOff+8], y.GetElem(0))
		binary.LittleEndian.PutUint64(dst[dstOff+dstStr:dstOff+dstStr+8], y.GetElem(1))
		base += 2 * BStride
		dstOff += 2 * dstStr
	}
}

func cdefByteStore4x4(dst []byte, dstOff, dstStr int, v archsimd.Uint8x16) {
	y := v.ReshapeToUint32s()
	binary.LittleEndian.PutUint32(dst[dstOff:dstOff+4], y.GetElem(0))
	binary.LittleEndian.PutUint32(dst[dstOff+dstStr:dstOff+dstStr+4], y.GetElem(1))
	binary.LittleEndian.PutUint32(dst[dstOff+2*dstStr:dstOff+2*dstStr+4], y.GetElem(2))
	binary.LittleEndian.PutUint32(dst[dstOff+3*dstStr:dstOff+3*dstStr+4], y.GetElem(3))
}

func cdefByteFilter4PriSec(ctx *cdefByteCtx) {
	p0p, p0m := ctx.p0p, ctx.p0m
	p1p, p1m := ctx.p1p, ctx.p1m
	s0p, s0m := ctx.s0p, ctx.s0m
	s1p, s1m := ctx.s1p, ctx.s1m
	s2p, s2m := ctx.s2p, ctx.s2m
	s3p, s3m := ctx.s3p, ctx.s3m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff ||
		s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	priStr, secStr := ctx.priStr, ctx.secStr
	priSh, secSh := ctx.priSh, ctx.secSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		mn, mx := x, x
		p, q := cdefByteLoad4x4Rows(r0, r1, r2, r3, p0p), cdefByteLoad4x4Rows(r0, r1, r2, r3, p0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, p1p), cdefByteLoad4x4Rows(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s0p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s1p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s2p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s3p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias).Min(mx).Max(mn))
		base += 4 * BStride
		dstOff += 4 * dstStr
	}
}

func cdefByteFilter4Pri(ctx *cdefByteCtx) {
	p0p, p0m := ctx.p0p, ctx.p0m
	p1p, p1m := ctx.p1p, ctx.p1m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff {
		return
	}
	priStr, priSh := ctx.priStr, ctx.priSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		a := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p0p), x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p0m), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p1p), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p1m), x, priStr, priSh).MulAdd(tap1, b)
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias))
		base += 4 * BStride
		dstOff += 4 * dstStr
	}
}

func cdefByteFilter4Sec(ctx *cdefByteCtx) {
	s0p, s0m := ctx.s0p, ctx.s0m
	s1p, s1m := ctx.s1p, ctx.s1m
	s2p, s2m := ctx.s2p, ctx.s2m
	s3p, s3m := ctx.s3p, ctx.s3m
	if s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	secStr, secSh := ctx.secStr, ctx.secSh
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		ka := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s0p), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s1p), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s0m), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s1m), x, secStr, secSh))
		a := ka.Add(ka).Add(neg1).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s2p), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s3p), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s2m), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s3m), x, secStr, secSh))
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias))
		base += 4 * BStride
		dstOff += 4 * dstStr
	}
}
