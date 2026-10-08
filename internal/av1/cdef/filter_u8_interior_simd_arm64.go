// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

// CDEF interior 8-bit-dst fused kernel, the Go SIMD counterpart of dav1d's
// edged 8bpc .16b path (src/arm/64/cdef.S cdef_filter8_*_edged_8bpc_neon). When
// every CDEF tap is a real sample (no VeryLarge sentinel in the unit's tap
// footprint, see cdefUnitInteriorU8), the taps are 8-bit values and the whole
// constrain runs in 16 byte-lanes per op. Output is bit-identical to
// filterBlockU8PureGo; see u8_interior_differential_test.go.
//
// Constrain in byte lanes: with a = |t-x|, lim = max(str - (a>>shift), 0) is
// the saturating subtract, and clamp(t-x, -lim, +lim) + x equals
// min(max(t, x-lim), x+lim) with saturating bounds, since t is already in
// [0, 255]. The sum is then taken in 16-bit lanes, one half of the 16 bytes at
// a time (row r in the low half, row r+1 in the high half).

import (
	"simd/archsimd"
	"unsafe"
)

// cdefTapReach is the maximum |offset|, in samples, that any CDEF tap reads
// beyond the current pixel along either axis (cdefDirections secondary taps
// reach two rows and two columns). A block whose whole footprint expanded by
// this reach holds only real samples has no VeryLarge sentinel in any tap.
const cdefTapReach = 2

// cdefUnitInteriorU8 reports whether every CDEF tap the unit's blocks read
// lands on a real 0..255 sample, i.e. the tap footprint of the block bounding
// box (expanded by cdefTapReach on every side) is free of the VeryLarge border
// sentinel. That is exactly dav1d's edges == 0xf predicate for goav1's tap
// buffer: when it holds, the interior .16b kernels may narrow the uint16 taps
// to uint8 losslessly. The scan is one strided pass over the footprint, run
// once per unit and amortized across all of the unit's blocks.
func cdefUnitInteriorU8(input []uint16, inputOrigin int, blocks []BlockPosition, bwLog2 int, bhLog2 int) bool {
	if len(input) == 0 || len(blocks) == 0 {
		return false
	}
	blockWidth := 1 << bwLog2
	blockHeight := 1 << bhLog2
	minBX := int(blocks[0].BX)
	maxBX := minBX
	minBY := int(blocks[0].BY)
	maxBY := minBY
	for i := 1; i < len(blocks); i++ {
		bx := int(blocks[i].BX)
		by := int(blocks[i].BY)
		if bx < minBX {
			minBX = bx
		}
		if bx > maxBX {
			maxBX = bx
		}
		if by < minBY {
			minBY = by
		}
		if by > maxBY {
			maxBY = by
		}
	}
	// inputOrigin addresses the unit's first interior sample (buffer row
	// VerticalBorder, column HorizontalBorder). Walk out cdefTapReach on each
	// side of the block bounding box.
	topLeft := inputOrigin + ((minBY * BStride) << bhLog2) + (minBX << bwLog2)
	scanStart := topLeft - cdefTapReach*BStride - cdefTapReach
	nRows := (maxBY-minBY+1)*blockHeight + 2*cdefTapReach
	nCols := (maxBX-minBX+1)*blockWidth + 2*cdefTapReach
	if scanStart < 0 {
		return false
	}
	end := scanStart + (nRows-1)*BStride + nCols
	if end > len(input) {
		return false
	}
	return cdefScanU16RowsAtMost255(input, scanStart, nRows, nCols)
}

// cdefPack2Rows packs two rows of eight samples into one 16-byte vector: row
// a in bytes 0..7, row b in bytes 8..15. The samples are 0..255 by the
// interior contract, so the narrowing truncation is exact.
func cdefPack2Rows(a, b archsimd.Int16x8) archsimd.Uint8x16 {
	a8 := a.ToBits().TruncToUint8().ReshapeToUint64s()
	b8 := b.ToBits().TruncToUint8().ReshapeToUint64s()
	return a8.InterleaveLo(b8).ReshapeToUint8s()
}

// cdefHalvesU8 widens the two packed rows back to 16-bit lanes: lo is row r
// (bytes 0..7), hi is row r+1 (bytes 8..15).
func cdefHalvesU8(v archsimd.Uint8x16) (archsimd.Int16x8, archsimd.Int16x8) {
	return v.ExtendLo8ToUint16().BitsToInt16(), v.HiToLo().ExtendLo8ToUint16().BitsToInt16()
}

// cdefConstrainU8 returns the tap value after constrain(), in byte lanes:
// x + clamp(t-x, -lim, +lim) with lim = max(str - (|t-x| >> shift), 0),
// written as min(max(t, x-lim), x+lim) with saturating bounds.
func cdefConstrainU8(t, x, str archsimd.Uint8x16, sh archsimd.Int8x16) archsimd.Uint8x16 {
	ab := t.Max(x).Sub(t.Min(x))
	lim := str.SubSaturated(ab.Shift(sh))
	return t.Max(x.SubSaturated(lim)).Min(x.AddSaturated(lim))
}

// cdefFilterBlock8InteriorSIMD filters a fused 8-wide block whose taps are all
// real 8-bit samples, two rows per vector. Height is even by routing contract.
// Primary and secondary are both enabled (the unit loop routes single-strength
// blocks to cdefFilterBlockSIMD), so the clamp is always active.
func cdefFilterBlock8InteriorSIMD(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastUint8x16(uint8(ctx.priStrength))
	secStr := archsimd.BroadcastUint8x16(uint8(ctx.secStrength))
	priSh := archsimd.BroadcastInt8x16(-int8(ctx.priShift))
	secSh := archsimd.BroadcastInt8x16(-int8(ctx.secShift))
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x8(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x8(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x8(8)
	wtot := archsimd.BroadcastInt16x8(int16(2*ctx.priTap0 + 2*ctx.priTap1 + 4*ctx.secTap0 + 4*ctx.secTap1))
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefPack2Rows(cdefLoadU16P(src), cdefLoadU16P(src2))
		p0 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, pri0)), cdefLoadU16P(unsafe.Add(src2, pri0)))
		p1 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, -pri0)), cdefLoadU16P(unsafe.Add(src2, -pri0)))
		p2 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, pri1)), cdefLoadU16P(unsafe.Add(src2, pri1)))
		p3 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, -pri1)), cdefLoadU16P(unsafe.Add(src2, -pri1)))
		s0 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, sec0)), cdefLoadU16P(unsafe.Add(src2, sec0)))
		s1 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, -sec0)), cdefLoadU16P(unsafe.Add(src2, -sec0)))
		s2 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, sec1)), cdefLoadU16P(unsafe.Add(src2, sec1)))
		s3 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, -sec1)), cdefLoadU16P(unsafe.Add(src2, -sec1)))
		s4 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, sec2)), cdefLoadU16P(unsafe.Add(src2, sec2)))
		s5 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, -sec2)), cdefLoadU16P(unsafe.Add(src2, -sec2)))
		s6 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, sec3)), cdefLoadU16P(unsafe.Add(src2, sec3)))
		s7 := cdefPack2Rows(cdefLoadU16P(unsafe.Add(src, -sec3)), cdefLoadU16P(unsafe.Add(src2, -sec3)))

		// Clamp bounds fold over the raw taps and the centre, as maxClip/min4 do.
		mx := x.Max(p0).Max(p1).Max(p2).Max(p3).Max(s0).Max(s1).Max(s2).Max(s3)
		mx = mx.Max(s4).Max(s5).Max(s6).Max(s7)
		mn := x.Min(p0).Min(p1).Min(p2).Min(p3).Min(s0).Min(s1).Min(s2).Min(s3)
		mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)

		c0 := cdefConstrainU8(p0, x, priStr, priSh)
		c1 := cdefConstrainU8(p1, x, priStr, priSh)
		c2 := cdefConstrainU8(p2, x, priStr, priSh)
		c3 := cdefConstrainU8(p3, x, priStr, priSh)
		d0 := cdefConstrainU8(s0, x, secStr, secSh)
		d1 := cdefConstrainU8(s1, x, secStr, secSh)
		d2 := cdefConstrainU8(s2, x, secStr, secSh)
		d3 := cdefConstrainU8(s3, x, secStr, secSh)
		d4 := cdefConstrainU8(s4, x, secStr, secSh)
		d5 := cdefConstrainU8(s5, x, secStr, secSh)
		d6 := cdefConstrainU8(s6, x, secStr, secSh)
		d7 := cdefConstrainU8(s7, x, secStr, secSh)

		// Widen to 16-bit halves and take the signed differences from x.
		xl, xh := cdefHalvesU8(x)
		ml, mh := cdefHalvesU8(mx)
		nl, nh := cdefHalvesU8(mn)
		c0l, c0h := cdefHalvesU8(c0)
		c1l, c1h := cdefHalvesU8(c1)
		c2l, c2h := cdefHalvesU8(c2)
		c3l, c3h := cdefHalvesU8(c3)
		d0l, d0h := cdefHalvesU8(d0)
		d1l, d1h := cdefHalvesU8(d1)
		d2l, d2h := cdefHalvesU8(d2)
		d3l, d3h := cdefHalvesU8(d3)
		d4l, d4h := cdefHalvesU8(d4)
		d5l, d5h := cdefHalvesU8(d5)
		d6l, d6h := cdefHalvesU8(d6)
		d7l, d7h := cdefHalvesU8(d7)

		// sum = primary taps (weights {p0,p1}) + secondary taps (weights {2,1}).
		// sum = sum_i w_i*(t_i - x) = sum_i w_i*t_i - wtot*x, with wtot the sum of
		// the tap weights (two taps per primary weight, four per secondary weight).
		sumL := c0l.Add(c1l).Mul(priTap0).Add(c2l.Add(c3l).Mul(priTap1))
		sumL = sumL.Add(d0l.Add(d1l).Add(d2l).Add(d3l).Mul(secTap0)).Add(d4l.Add(d5l).Add(d6l).Add(d7l).Mul(secTap1))
		sumL = sumL.Sub(xl.Mul(wtot))
		sumH := c0h.Add(c1h).Mul(priTap0).Add(c2h.Add(c3h).Mul(priTap1))
		sumH = sumH.Add(d0h.Add(d1h).Add(d2h).Add(d3h).Mul(secTap0)).Add(d4h.Add(d5h).Add(d6h).Add(d7h).Mul(secTap1))
		sumH = sumH.Sub(xh.Mul(wtot))

		// y = x + ((8 + sum - (sum<0)) >> 4), clamped to [min, max].
		yL := xl.Add(sumL.Add(sumL.ShiftAllRight(15)).Add(eight).ShiftAllRight(4)).Max(nl).Min(ml)
		yH := xh.Add(sumH.Add(sumH.ShiftAllRight(15)).Add(eight).ShiftAllRight(4)).Max(nh).Min(mh)
		*(*uint64)(dst) = yL.ToBits().TruncToUint8().ReshapeToUint64s().GetElem(0)
		*(*uint64)(unsafe.Add(dst, dstStr)) = yH.ToBits().TruncToUint8().ReshapeToUint64s().GetElem(0)
		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

// filterBlockU8InteriorSIMD runs the interior kernel on one fused 8-wide block.
// It is the test entry: the unit loop reaches the kernel only after
// cdefUnitInteriorU8 proves the footprint is sentinel-free.
func filterBlockU8InteriorSIMD(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	var ctx cdefSIMDCtx
	ctx.setBlock(unsafe.Pointer(&dst[dstOrigin]), dstStride, unsafe.Pointer(&input[inputOrigin]), params, true)
	cdefFilterBlock8InteriorSIMD(&ctx)
}
