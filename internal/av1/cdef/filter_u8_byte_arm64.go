// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package cdef

// Byte-lane CDEF kernels for real-interior 8-bit units, ported from dav1d's
// 8bpc edged NEON filters (src/arm/64/cdef.S filter_func_8 /
// handle_pixel_8). Every tap is an 8-bit sample, so the whole filter runs in
// 16 byte lanes: two 8-pixel rows per vector for 8-wide blocks, four 4-pixel
// rows for 4-wide blocks.
//
//   - constrain() is uabd / ushl / uqsub / umin with the sign applied by
//     cmhi + neg + bsl, exactly handle_pixel_8.
//   - |sum| can reach 15*2*6 + 4*4*3 = 228, past int8, so the taps on the
//     +offset side and the -offset side go to two int8 accumulators (each at
//     most 114 in magnitude), the first seeded with -1 as dav1d does.
//   - (8 + sum - (sum < 0)) >> 4 is taken as dav1d's halving adds followed
//     by a rounding shift by 3; shadd is srhadd minus the parity bit.
//   - px + delta saturates to [0, 255] (dav1d's usqadd), done as a signed
//     saturating add in the 0x80-biased domain. With both strengths the
//     result is clamped to the tap neighbourhood's [min, max].
//
// Output is bit-identical to filterBlockU8PureGo on the same pixels; see
// filter_u8_byte_arm64_test.go.

import (
	"simd/archsimd"
	"unsafe"
)

func u8ByteInteriorAvailable() bool { return true }

// cdefByteCtx is the per-block state of the byte kernels. Offsets are byte
// offsets into the BStride-spaced byte tap buffer; shifts are negative
// counts for USHL.
type cdefByteCtx struct {
	dst    unsafe.Pointer
	src    unsafe.Pointer
	dstStr int
	height int

	pri0, pri1             int
	sec0, sec1, sec2, sec3 int

	priStr, secStr uint8
	priSh, secSh   int8
	priTap0        int8
	priTap1        int8
}

// filterUnitBlocksU8Byte is filterUnitBlocksU8PureGo on a byte tap buffer:
// per-block luma strength adjust, blocks with both strengths zero skipped
// (dav1d cdef_apply_tmpl.c), and the pri / sec / pri+sec kernel picked per
// block as dav1d's cdef_filter_block_*_neon does.
func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	ctx := cdefByteCtx{dstStr: dstStride, height: u.blockHeight}
	sec := u.secondaryStrength
	ctx.secStr = uint8(sec)
	ctx.secSh = -int8(constrainShift(sec, u.damping))
	strength := u.primaryStrength
	setPrimary := func(s int) {
		taps := cdefPrimaryTaps[s&1]
		ctx.priTap0 = int8(taps[0])
		ctx.priTap1 = int8(taps[1])
		ctx.priStr = uint8(s)
		ctx.priSh = -int8(constrainShift(s, u.damping))
	}
	if !u.lumaAdjust {
		setPrimary(strength)
	}
	base := unsafe.Pointer(unsafe.SliceData(input))
	dstBase := unsafe.Pointer(unsafe.SliceData(dst))
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			setPrimary(strength)
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
		if u.blockWidth == 8 {
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

// cdefByteLoad8x2 packs the 8 bytes at p (low half) and at p+BStride (high
// half): two full loads and a ZIP1.2D, dav1d's ld1 {v.d}[0] / {v.d}[1].
func cdefByteLoad8x2(p unsafe.Pointer) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p)).ReshapeToUint64s()
	b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, BStride))).ReshapeToUint64s()
	return a.InterleaveLo(b).ReshapeToUint8s()
}

// cdefByteLoad4x4 packs the 4 bytes at p and the next three BStride rows in
// row order (dav1d's ld1 {v.s}[0..3]).
func cdefByteLoad4x4(p unsafe.Pointer) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p)).ReshapeToUint32s()
	b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, BStride))).ReshapeToUint32s()
	c := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 2*BStride))).ReshapeToUint32s()
	d := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 3*BStride))).ReshapeToUint32s()
	ab := a.InterleaveLo(b).ReshapeToUint64s()
	cd := c.InterleaveLo(d).ReshapeToUint64s()
	return ab.InterleaveLo(cd).ReshapeToUint8s()
}

// cdefByteConstrain is handle_pixel_8's constrain(): the signed byte
// min(|t-x|, max(0, str - (|t-x| >> shift))) carrying the sign of t-x. The
// sign is applied without a select: of the two saturating differences
// t-x and x-t one is zero and the other is |t-x|, so clamping each to the
// limit and subtracting gives the signed value (dav1d uses cmhi + neg + bsl).
func cdefByteConstrain(t, x, str archsimd.Uint8x16, sh archsimd.Int8x16) archsimd.Int8x16 {
	pos := t.SubSaturated(x)
	neg := x.SubSaturated(t)
	lim := str.SubSaturated(pos.Or(neg).Shift(sh))
	return pos.Min(lim).BitsToInt8().Sub(neg.Min(lim).BitsToInt8())
}

// cdefByteFinish turns the two accumulators (a seeded with -1) into the
// filtered pixels: x + ((8 + sum - (sum < 0)) >> 4) saturated to [0, 255].
func cdefByteFinish(x archsimd.Uint8x16, a, b archsimd.Int8x16) archsimd.Uint8x16 {
	r := a.Average(b) // srhadd: sum >> 1
	// shadd = srhadd - ((a ^ b) & 1); dav1d picks it when sum < 0.
	odd := a.Xor(b).ShiftAllLeft(7).ShiftAllRight(7)
	h := r.Add(odd.And(r.ShiftAllRight(7)))
	d := h.Add(archsimd.BroadcastInt8x16(4)).ShiftAllRight(3)
	bias := archsimd.BroadcastUint8x16(0x80)
	return x.Xor(bias).BitsToInt8().AddSaturated(d).ToBits().Xor(bias)
}

func cdefByteFilter8PriSec(ctx *cdefByteCtx) {
	priStr := archsimd.BroadcastUint8x16(ctx.priStr)
	secStr := archsimd.BroadcastUint8x16(ctx.secStr)
	priSh := archsimd.BroadcastInt8x16(ctx.priSh)
	secSh := archsimd.BroadcastInt8x16(ctx.secSh)
	tap0 := archsimd.BroadcastInt8x16(ctx.priTap0)
	tap1 := archsimd.BroadcastInt8x16(ctx.priTap1)
	pri0, pri1 := ctx.pri0, ctx.pri1
	sec0, sec1, sec2, sec3 := ctx.sec0, ctx.sec1, ctx.sec2, ctx.sec3
	src, dst, dstStr := ctx.src, ctx.dst, ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		// One tap pair at a time, as handle_pixel_8: load, fold min/max,
		// constrain, accumulate, so few vectors stay live.
		x := cdefByteLoad8x2(src)
		mn, mx := x, x
		p, q := cdefByteLoad8x2(unsafe.Add(src, pri0)), cdefByteLoad8x2(unsafe.Add(src, -pri0))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, archsimd.BroadcastInt8x16(-1))
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad8x2(unsafe.Add(src, pri1)), cdefByteLoad8x2(unsafe.Add(src, -pri1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad8x2(unsafe.Add(src, sec0)), cdefByteLoad8x2(unsafe.Add(src, -sec0))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad8x2(unsafe.Add(src, sec1)), cdefByteLoad8x2(unsafe.Add(src, -sec1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad8x2(unsafe.Add(src, sec2)), cdefByteLoad8x2(unsafe.Add(src, -sec2))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad8x2(unsafe.Add(src, sec3)), cdefByteLoad8x2(unsafe.Add(src, -sec3))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		y := cdefByteFinish(x, a, b).Min(mx).Max(mn).ReshapeToUint64s()
		*(*uint64)(dst) = y.GetElem(0)
		*(*uint64)(unsafe.Add(dst, dstStr)) = y.GetElem(1)
		if h > 2 {
			src = unsafe.Add(src, 2*BStride)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

func cdefByteFilter8Pri(ctx *cdefByteCtx) {
	priStr := archsimd.BroadcastUint8x16(ctx.priStr)
	priSh := archsimd.BroadcastInt8x16(ctx.priSh)
	tap0 := archsimd.BroadcastInt8x16(ctx.priTap0)
	tap1 := archsimd.BroadcastInt8x16(ctx.priTap1)
	pri0, pri1 := ctx.pri0, ctx.pri1
	src, dst, dstStr := ctx.src, ctx.dst, ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		x := cdefByteLoad8x2(src)
		a := cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, pri0)), x, priStr, priSh).MulAdd(tap0, archsimd.BroadcastInt8x16(-1))
		b := cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, -pri0)), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, pri1)), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, -pri1)), x, priStr, priSh).MulAdd(tap1, b)
		y := cdefByteFinish(x, a, b).ReshapeToUint64s()
		*(*uint64)(dst) = y.GetElem(0)
		*(*uint64)(unsafe.Add(dst, dstStr)) = y.GetElem(1)
		if h > 2 {
			src = unsafe.Add(src, 2*BStride)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

func cdefByteFilter8Sec(ctx *cdefByteCtx) {
	secStr := archsimd.BroadcastUint8x16(ctx.secStr)
	secSh := archsimd.BroadcastInt8x16(ctx.secSh)
	sec0, sec1, sec2, sec3 := ctx.sec0, ctx.sec1, ctx.sec2, ctx.sec3
	src, dst, dstStr := ctx.src, ctx.dst, ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		x := cdefByteLoad8x2(src)
		ka := cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, sec0)), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, sec1)), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, -sec0)), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, -sec1)), x, secStr, secSh))
		a := ka.Add(ka).Add(archsimd.BroadcastInt8x16(-1)).
			Add(cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, sec2)), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, sec3)), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, -sec2)), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad8x2(unsafe.Add(src, -sec3)), x, secStr, secSh))
		y := cdefByteFinish(x, a, b).ReshapeToUint64s()
		*(*uint64)(dst) = y.GetElem(0)
		*(*uint64)(unsafe.Add(dst, dstStr)) = y.GetElem(1)
		if h > 2 {
			src = unsafe.Add(src, 2*BStride)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

func cdefByteStore4x4(dst unsafe.Pointer, dstStr int, v archsimd.Uint8x16) {
	y := v.ReshapeToUint32s()
	*(*uint32)(dst) = y.GetElem(0)
	*(*uint32)(unsafe.Add(dst, dstStr)) = y.GetElem(1)
	*(*uint32)(unsafe.Add(dst, 2*dstStr)) = y.GetElem(2)
	*(*uint32)(unsafe.Add(dst, 3*dstStr)) = y.GetElem(3)
}

func cdefByteFilter4PriSec(ctx *cdefByteCtx) {
	priStr := archsimd.BroadcastUint8x16(ctx.priStr)
	secStr := archsimd.BroadcastUint8x16(ctx.secStr)
	priSh := archsimd.BroadcastInt8x16(ctx.priSh)
	secSh := archsimd.BroadcastInt8x16(ctx.secSh)
	tap0 := archsimd.BroadcastInt8x16(ctx.priTap0)
	tap1 := archsimd.BroadcastInt8x16(ctx.priTap1)
	pri0, pri1 := ctx.pri0, ctx.pri1
	sec0, sec1, sec2, sec3 := ctx.sec0, ctx.sec1, ctx.sec2, ctx.sec3
	src, dst, dstStr := ctx.src, ctx.dst, ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		x := cdefByteLoad4x4(src)
		mn, mx := x, x
		p, q := cdefByteLoad4x4(unsafe.Add(src, pri0)), cdefByteLoad4x4(unsafe.Add(src, -pri0))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, archsimd.BroadcastInt8x16(-1))
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad4x4(unsafe.Add(src, pri1)), cdefByteLoad4x4(unsafe.Add(src, -pri1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad4x4(unsafe.Add(src, sec0)), cdefByteLoad4x4(unsafe.Add(src, -sec0))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad4x4(unsafe.Add(src, sec1)), cdefByteLoad4x4(unsafe.Add(src, -sec1))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad4x4(unsafe.Add(src, sec2)), cdefByteLoad4x4(unsafe.Add(src, -sec2))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad4x4(unsafe.Add(src, sec3)), cdefByteLoad4x4(unsafe.Add(src, -sec3))
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		cdefByteStore4x4(dst, dstStr, cdefByteFinish(x, a, b).Min(mx).Max(mn))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteFilter4Pri(ctx *cdefByteCtx) {
	priStr := archsimd.BroadcastUint8x16(ctx.priStr)
	priSh := archsimd.BroadcastInt8x16(ctx.priSh)
	tap0 := archsimd.BroadcastInt8x16(ctx.priTap0)
	tap1 := archsimd.BroadcastInt8x16(ctx.priTap1)
	pri0, pri1 := ctx.pri0, ctx.pri1
	src, dst, dstStr := ctx.src, ctx.dst, ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		x := cdefByteLoad4x4(src)
		a := cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, pri0)), x, priStr, priSh).MulAdd(tap0, archsimd.BroadcastInt8x16(-1))
		b := cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, -pri0)), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, pri1)), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, -pri1)), x, priStr, priSh).MulAdd(tap1, b)
		cdefByteStore4x4(dst, dstStr, cdefByteFinish(x, a, b))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}

func cdefByteFilter4Sec(ctx *cdefByteCtx) {
	secStr := archsimd.BroadcastUint8x16(ctx.secStr)
	secSh := archsimd.BroadcastInt8x16(ctx.secSh)
	sec0, sec1, sec2, sec3 := ctx.sec0, ctx.sec1, ctx.sec2, ctx.sec3
	src, dst, dstStr := ctx.src, ctx.dst, ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		x := cdefByteLoad4x4(src)
		ka := cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, sec0)), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, sec1)), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, -sec0)), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, -sec1)), x, secStr, secSh))
		a := ka.Add(ka).Add(archsimd.BroadcastInt8x16(-1)).
			Add(cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, sec2)), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, sec3)), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, -sec2)), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4(unsafe.Add(src, -sec3)), x, secStr, secSh))
		cdefByteStore4x4(dst, dstStr, cdefByteFinish(x, a, b))
		if h > 4 {
			src = unsafe.Add(src, 4*BStride)
			dst = unsafe.Add(dst, 4*dstStr)
		}
	}
}
