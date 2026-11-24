// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// The six-tap entries and kernels over footprint-checked unsafeutil.Grid
// views, as on arm64 (mc8_gosimd_arm64.go, mc8grid_gosimd_arm64.go): Put8
// checks the common case (destination and source window inside their planes)
// in line and runs the kernel with no call before it; everything else is
// mc8PutEdge, which materializes the clamped window (mc8SourceGrid).
//
// Each kernel takes fixed-width column strips of its views (Fits, then Sub:
// one test per strip) and names every row by the row loop variable, bounded
// by the strip's height, or by a constant, so the rows pay no check. A
// kernel restates what it relies on with Fits and returns early when it does
// not hold, which the entries make unreachable. The 2D intermediate (int16)
// and the prep CONV_BUF keep their fixed-array and slice windows; the
// eight-tap (sharp) kernels keep their slice windows (mc8Source).

// mc8MaxRows bounds the row counts the kernels take (h+5 rows of a block at
// most maxBlockSize high). The kernels test it first, which tells the compiler
// their row loops cannot overflow, so the row checks drop.
const mc8MaxRows = maxBlockSize + 8

// mc8G4..mc8G24 are row r of g from column 0, as a fixed-size window.
func mc8G4(g unsafeutil.Grid[uint8], r int) *[4]byte   { return g.Row4(r, 0) }
func mc8G8(g unsafeutil.Grid[uint8], r int) *[8]byte   { return g.Row8(r, 0) }
func mc8G16(g unsafeutil.Grid[uint8], r int) *[16]byte { return g.Row16(r, 0) }
func mc8G24(g unsafeutil.Grid[uint8], r int) *[24]byte { return g.Row24(r, 0) }

// mc8Halo is the source window halo of an axis: 2 when its phase is nonzero.
func mc8Halo(sub int) int {
	if sub != 0 {
		return 2
	}
	return 0
}

// mc8SourceGrid resolves the six-tap source window of a w x h block whose
// first output sample reads ref at (refX, refY) as a Grid view from its
// top-left tap sample: bh rows (h, plus 5 on a filtered vertical axis) of
// max(w+8, 16) bytes, the bytes every kernel may load. When any tap falls
// outside the plane the clamped window is first materialized in edge (dav1d
// src/recon_tmpl.c mc() emu_edge). ok is false only when the reference plane
// itself is unusable.
func mc8SourceGrid(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src unsafeutil.Grid[uint8], ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	if hx {
		x0 -= 2
		bw += 5
	}
	if hy {
		y0 -= 2
		bh += 5
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < ref.Width {
		return src, false
	}
	sw := max(w+8, 16)
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + x0
		if unsafeutil.GridFit(len(ref.Pix), start, ref.Stride, sw, bh) {
			return unsafeutil.NewGrid(ref.Pix, start, ref.Stride, sw, bh), true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return src, false
	}
	// The edge window holds 135 rows of 136 bytes; bh <= 133 and the kernels
	// read at most max(w+8, 16) <= 136 bytes per row.
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	if !unsafeutil.GridFit(len(edge), 0, emuEdgeStride, sw, bh) {
		return src, false
	}
	return unsafeutil.NewGrid(edge[:], 0, emuEdgeStride, sw, bh), true
}

// Put8 is dav1d's mc() put for an unscaled 8-bit single-reference block:
// the copy (integer vector) or the put kernels for the block's filters, from
// the reference origin (refX, refY) and Q4 subpel offsets (subX, subY). It
// returns false, with nothing written, for shapes it does not cover; the
// caller then runs the general path.
func Put8(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 {
		return mc8CopyFast(dst, ref, dstX, dstY, refX, refY, w, h)
	}
	if !mc8AVX2 || (w&7 != 0 && w != 4) || w <= 0 || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	eight, ok := mc8Taps(w, kx, ky, subX, subY)
	if !ok {
		return false
	}
	if eight {
		return mc8Put8Tap(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The six-tap source window, in place when it lies inside the plane:
	// halo 2 before and 3 after each sample on a filtered axis. The common
	// case, destination and source window both in their planes, is checked
	// in line and runs the kernel with no call before it.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	dOff, sOff, sw := dstY*dst.Stride+dstX, y0*ref.Stride+x0, max(w+8, 16)
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < ref.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		return mc8PutEdge(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	dg, sg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	// mc8PutGrid written out: one call level less for the common kernels.
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX != 0 && subY != 0:
		im := &scratch.im
		if w == 4 {
			mc8AVX2MidH4W4(sg, h+5, phx, im)
			mc8AVX2PutV6MidW4(im, phy, dg)
		} else {
			mc8AVX2H6Mid(sg, w, h+5, phx, im)
			mc8AVX2PutV6Mid(im, phy, dg)
		}
	case subX != 0:
		if w == 4 {
			mc8AVX2PutH4W4(sg, phx, dg)
		} else {
			mc8AVX2PutH6(sg, phx, dg)
		}
	default:
		if w == 4 {
			mc8AVX2PutV6W4(sg, phy, &scratch.im, dg)
		} else {
			mc8AVX2PutV6(sg, phy, dg)
		}
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// mc8PutEdge is Put8's six-tap path for a block whose destination or source
// window is not inside its plane (or whose views do not fit): the checked
// destination view and the source window from mc8SourceGrid, which clamps it.
func mc8PutEdge(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	dOff := dstY*dst.Stride + dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) {
		return false
	}
	sg, ok := mc8SourceGrid(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	mc8PutGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), sg, w, h, subX, subY, &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15], scratch)
	return true
}

// mc8PutGrid runs the six-tap put kernel of a validated block (w 4 or a
// multiple of 8, a nonzero phase on some axis): dg is the w x h destination,
// sg the source window. Like dav1d's RET (VZEROUPPER) it clears the upper
// YMM halves before returning.
func mc8PutGrid(dg, sg unsafeutil.Grid[uint8], w, h, subX, subY int, phx, phy *mc8Phase, scratch *ConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		im := &scratch.im
		if w == 4 {
			mc8AVX2MidH4W4(sg, h+5, phx, im)
			mc8AVX2PutV6MidW4(im, phy, dg)
		} else {
			mc8AVX2H6Mid(sg, w, h+5, phx, im)
			mc8AVX2PutV6Mid(im, phy, dg)
		}
	case subX != 0:
		if w == 4 {
			mc8AVX2PutH4W4(sg, phx, dg)
		} else {
			mc8AVX2PutH6(sg, phx, dg)
		}
	default:
		if w == 4 {
			mc8AVX2PutV6W4(sg, phy, &scratch.im, dg)
		} else {
			mc8AVX2PutV6(sg, phy, dg)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc8PrepFast fills a w x h CONV_BUF (row stride w) with the AVX2 prep
// kernels. It returns false for shapes it does not cover.
func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if !mc8AVX2 || (w&7 != 0 && w != 4) || w <= 0 || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil || len(out) < w*h {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	eight, ok := mc8Taps(w, kx, ky, subX, subY)
	if !ok {
		return false
	}
	o := out[:w*h]
	if eight {
		return mc8Prep8Tap(o, ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	sOff, sw := y0*ref.Stride+x0, max(w+8, 16)
	var sg unsafeutil.Grid[uint8]
	if x0 >= 0 && y0 >= 0 && x0+w+hx+hx+hx>>1 <= ref.Width && y0+bh <= ref.Height && ref.Stride >= ref.Width &&
		unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		sg = unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	} else if sg, ok = mc8SourceGrid(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge); !ok {
		return false
	}
	im := (*mc8IM)(&scratch.im8)
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX|subY == 0:
		mc8AVX2PrepCopy(sg, w, h, o)
	case subX != 0 && subY != 0:
		if w == 4 {
			mc8AVX2MidH4W4(sg, h+5, phx, im)
			mc8AVX2PrepV6MidW4(im, h, phy, o)
		} else {
			mc8AVX2H6Mid(sg, w, h+5, phx, im)
			mc8AVX2PrepV6Mid(im, w, h, phy, o)
		}
	case subX != 0:
		if w == 4 {
			mc8AVX2PrepH4W4(sg, h, phx, o)
		} else {
			mc8AVX2PrepH6(sg, w, h, phx, o)
		}
	default:
		if w == 4 {
			mc8AVX2PrepV6W4(sg, h, phy, im, o)
		} else {
			mc8AVX2PrepV6(sg, w, h, phy, o)
		}
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// mc8AVX2PutH6 is the put horizontal-only pass, clip((s + 34) >> 6): sg
// holds the h = dg.Height() source rows (max(w+8, 16) bytes each), dg the
// destination block.
func mc8AVX2PutH6(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h, w := dg.Height(), dg.Width()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	if w == 8 {
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		r := 0
		for ; r < h-1; r += 2 {
			a := mc8H6(mc8Ld2(mc8G16(s, r), mc8G16(s, r+1)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint).ReshapeToUint64s()
			binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], v.GetLo().GetElem(0))
			binary.LittleEndian.PutUint64(mc8G8(dg, r+1)[:], v.GetHi().GetElem(0))
		}
		if r < h {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], mc8Pack(a, a, zero, c255, deint).ReshapeToUint64s().GetLo().GetElem(0))
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, h) || !dg.Fits(0, c, 16, h) {
			return
		}
		s, d := sg.Sub(0, c, 24, h), dg.Sub(0, c, 16, h)
		r := 0
		for ; r < h-1; r += 2 {
			x0, x1 := mc8G24(s, r), mc8G24(s, r+1)
			a := mc8H6(mc8Ld2((*[16]byte)(x0[0:16]), (*[16]byte)(x1[0:16])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			b := mc8H6(mc8Ld2((*[16]byte)(x0[8:24]), (*[16]byte)(x1[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, b, zero, c255, deint)
			v.GetLo().StoreArray(mc8G16(d, r))
			v.GetHi().StoreArray(mc8G16(d, r+1))
		}
		if r < h {
			x := mc8G24(s, r)
			a := mc8H6(mc8Ld2((*[16]byte)(x[0:16]), (*[16]byte)(x[0:16])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			b := mc8H6(mc8Ld2((*[16]byte)(x[8:24]), (*[16]byte)(x[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			mc8Pack(a, b, zero, c255, deint).GetLo().StoreArray(mc8G16(d, r))
		}
	}
	if c < w && sg.Fits(0, c, 16, h) && dg.Fits(0, c, 8, h) {
		// The last eight columns of a 16k+8-wide block (no AV1 block is).
		mc8AVX2PutH6(sg.Sub(0, c, 16, h), ph, dg.Sub(0, c, 8, h))
	}
}

// mc8AVX2PrepH6 is the prep horizontal-only pass into int16 rows of stride w:
// ((s + 2 + 6144 << 2) >> 2) as unsigned.
func mc8AVX2PrepH6(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		r := 0
		for ; r < h-1; r += 2 {
			a := mc8H6(mc8Ld2(mc8G16(s, r), mc8G16(s, r+1)), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
		}
		if r < h {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, h) {
			return
		}
		s := sg.Sub(0, c, 24, h)
		for r := 0; r < h; r++ {
			x := mc8G24(s, r)
			a := mc8H6(mc8Ld2((*[16]byte)(x[0:16]), (*[16]byte)(x[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*w+c, a.ToBits().ShiftAllRight(2))
		}
	}
	if c < w {
		if !sg.Fits(0, c, 16, h) {
			return
		}
		s := sg.Sub(0, c, 16, h)
		for r := 0; r < h; r++ {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*w+c, a.ToBits().ShiftAllRight(2).GetLo())
		}
	}
}

// mc8AVX2H6Mid writes rows x w rounded horizontal sums m = (s + 2) >> 2 as
// int16 with row stride w (the 2D intermediate) from the rows of sg.
func mc8AVX2H6Mid(sg unsafeutil.Grid[uint8], w, rows int, ph *mc8Phase, im *mc8IM) {
	if uint(rows-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2)
	if w == 8 {
		if !sg.Fits(0, 0, 16, rows) {
			return
		}
		s := sg.Sub(0, 0, 16, rows)
		r := 0
		for ; r < rows-1; r += 2 {
			a := mc8H6(mc8Ld2(mc8G16(s, r), mc8G16(s, r+1)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, r*8, a)
		}
		if r < rows {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, r*8, a.GetLo())
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, rows) {
			return
		}
		s := sg.Sub(0, c, 24, rows)
		for r := 0; r < rows; r++ {
			x := mc8G24(s, r)
			a := mc8H6(mc8Ld2((*[16]byte)(x[0:16]), (*[16]byte)(x[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, r*w+c, a)
		}
	}
	if c < w {
		if !sg.Fits(0, c, 16, rows) {
			return
		}
		s := sg.Sub(0, c, 16, rows)
		for r := 0; r < rows; r++ {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, r*w+c, a.GetLo())
		}
	}
}

// mc8AVX2PutV6 is the put vertical-only pass: clip((s + 32) >> 6). sg holds
// the h+5 source rows, dg the h destination rows.
//
// The kernels wider than 8 run 16-column strips; when w%16 == 8 the last
// strip starts at w-16 and rewrites eight columns of the previous one with
// the same values, so every strip stores 16 columns per row.
func mc8AVX2PutV6(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h, w := dg.Height(), dg.Width()
	if w == 8 {
		mc8AVX2PutV6W8(sg, ph, dg)
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		if !sg.Fits(0, c, 16, h+5) || !dg.Fits(0, c, 16, h) {
			return
		}
		mc8AVX2PutV6Strip(sg.Sub(0, c, 16, h+5), ph, dg.Sub(0, c, 16, h))
	}
}

// mc8AVX2PutV6Strip is one 16-column strip of mc8AVX2PutV6: s holds the h+5
// source rows of d's h rows. The five leading rows are rows 0..4 of s, the
// rest the rows of t (s from row 5), indexed by the output row.
func mc8AVX2PutV6Strip(s unsafeutil.Grid[uint8], ph *mc8Phase, d unsafeutil.Grid[uint8]) {
	h := d.Height()
	if uint(h-1) >= mc8MaxRows || !s.Fits(0, 0, 16, h+5) {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	z0 := archsimd.LoadUint8x16Array(mc8G16(s, 0)).ExtendToUint16()
	z1 := archsimd.LoadUint8x16Array(mc8G16(s, 1)).ExtendToUint16()
	z2 := archsimd.LoadUint8x16Array(mc8G16(s, 2)).ExtendToUint16()
	z3 := archsimd.LoadUint8x16Array(mc8G16(s, 3)).ExtendToUint16()
	z4 := archsimd.LoadUint8x16Array(mc8G16(s, 4)).ExtendToUint16()
	t := s.Sub(5, 0, 16, h)
	p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
	r := 0
	for ; r < h-1; r += 2 {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
		p4 := mc8Pair(z4, z5)
		p5 := mc8Pair(z5, z6)
		a := mc8V6(p0, p2, p4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		b := mc8V6(p1, p3, p5, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, b, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
		v.GetLo().StoreArray(mc8G16(d, r))
		v.GetHi().StoreArray(mc8G16(d, r+1))
		p0, p1, p2, p3, z4 = p2, p3, p4, p5, z6
	}
	if r < h {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		a := mc8V6(p0, p2, mc8Pair(z4, z5), t0, t1, t2).Add(rnd).ShiftAllRight(6)
		mc8Pack(a, a, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s().GetLo().StoreArray(mc8G16(d, r))
	}
}

// mc8AVX2PutV6W8 runs the eight-column vertical pass two rows per vector (row
// r in the low lane, r+1 in the high lane).
func mc8AVX2PutV6W8(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 16, h+5) || !dg.Fits(0, 0, 8, h) {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	z0 := archsimd.LoadUint8x16Array(mc8G16(sg, 0)).ExtendToUint16()
	z1 := archsimd.LoadUint8x16Array(mc8G16(sg, 1)).ExtendToUint16()
	z2 := archsimd.LoadUint8x16Array(mc8G16(sg, 2)).ExtendToUint16()
	z3 := archsimd.LoadUint8x16Array(mc8G16(sg, 3)).ExtendToUint16()
	z4 := archsimd.LoadUint8x16Array(mc8G16(sg, 4)).ExtendToUint16()
	t := sg.Sub(5, 0, 16, h)
	q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
	q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
	r := 0
	for ; r < h-1; r += 2 {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
		q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
		a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], v.GetLo().ReshapeToUint64s().GetElem(0))
		binary.LittleEndian.PutUint64(mc8G8(dg, r+1)[:], v.GetHi().ReshapeToUint64s().GetElem(0))
		q0, q2, z4 = q2, q4, z6
	}
	if r < h {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z5))
		a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], mc8Pack(a, a, zero, c255, deint).GetLo().ReshapeToUint64s().GetElem(0))
	}
}

// mc8AVX2PrepV6 is the prep vertical-only pass, ((s + 2) >> 2) + 6144, from
// the h+5 rows of sg into the w x h CONV_BUF out.
func mc8AVX2PrepV6(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		if !sg.Fits(0, 0, 16, h+5) {
			return
		}
		z0 := archsimd.LoadUint8x16Array(mc8G16(sg, 0)).ExtendToUint16()
		z1 := archsimd.LoadUint8x16Array(mc8G16(sg, 1)).ExtendToUint16()
		z2 := archsimd.LoadUint8x16Array(mc8G16(sg, 2)).ExtendToUint16()
		z3 := archsimd.LoadUint8x16Array(mc8G16(sg, 3)).ExtendToUint16()
		z4 := archsimd.LoadUint8x16Array(mc8G16(sg, 4)).ExtendToUint16()
		t := sg.Sub(5, 0, 16, h)
		q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
		q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
		r := 0
		for ; r < h-1; r += 2 {
			z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
			z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
			q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
			a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
			q0, q2, z4 = q2, q4, z6
		}
		if r < h {
			z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
			q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z5))
			a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		if !sg.Fits(0, c, 16, h+5) {
			return
		}
		mc8AVX2PrepV6Strip(sg.Sub(0, c, 16, h+5), w, h, ph, out[c:])
	}
}

// mc8AVX2PrepV6Strip is one 16-column strip of mc8AVX2PrepV6 (out row
// stride w), two rows per step.
func mc8AVX2PrepV6Strip(s unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, oc []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !s.Fits(0, 0, 16, h+5) {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	z0 := archsimd.LoadUint8x16Array(mc8G16(s, 0)).ExtendToUint16()
	z1 := archsimd.LoadUint8x16Array(mc8G16(s, 1)).ExtendToUint16()
	z2 := archsimd.LoadUint8x16Array(mc8G16(s, 2)).ExtendToUint16()
	z3 := archsimd.LoadUint8x16Array(mc8G16(s, 3)).ExtendToUint16()
	z4 := archsimd.LoadUint8x16Array(mc8G16(s, 4)).ExtendToUint16()
	t := s.Sub(5, 0, 16, h)
	p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
	r := 0
	for ; r < h-1; r += 2 {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
		p4, p5 := mc8Pair(z4, z5), mc8Pair(z5, z6)
		a := mc8V6(p0, p2, p4, t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		b := mc8V6(p1, p3, p5, t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		win := oc[uint(r*w) : uint(r*w)+uint(w)+16]
		a.StoreArray((*[16]uint16)(win))
		b.StoreArray((*[16]uint16)(win[w:]))
		p0, p1, p2, p3, z4 = p2, p3, p4, p5, z6
	}
	if r < h {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		a := mc8V6(p0, p2, mc8Pair(z4, z5), t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		mc8StU16(oc, r*w, a)
	}
}

// mc8AVX2PutV6Mid runs the put 2D vertical pass over the int16 intermediate
// (row stride w = dg.Width(), h+5 rows): clip((sum(half tap * m) + 512) >> 10)
// into dg.
//
// For w == 8 a 32-byte load at row k holds rows k and k+1, so the per-lane
// word interleave of rows (k, k+1) gives row pairs (k, k+1) in the low lane and
// (k+1, k+2) in the high lane: output rows r and r+1 come out one per lane.
// Wider blocks run 16-column strips one row at a time, where the per-lane
// interleave and VPACKUSDW undo each other's lane split.
func mc8AVX2PutV6Mid(im *mc8IM, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h, w := dg.Height(), dg.Width()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2 := mc8TapsD(ph.d[0]), mc8TapsD(ph.d[1]), mc8TapsD(ph.d[2])
	rnd := archsimd.BroadcastInt32x8(512)
	c255 := archsimd.BroadcastUint16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	if w == 8 {
		l0 := mc8LdI16(im, 0)
		l1 := mc8LdI16(im, 8)
		l2 := mc8LdI16(im, 16)
		l3 := mc8LdI16(im, 24)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		r := 0
		for ; r < h; r += 2 {
			q := 32 + r*8
			l4 := mc8LdI16(im, q)
			l5 := mc8LdI16(im, q+8)
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			lo := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(10)
			hi := mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(10)
			v := lo.SaturateToUint16ConcatGrouped(hi).Min(c255).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], v.GetLo().ReshapeToUint64s().GetElem(0))
			if r+1 < h {
				binary.LittleEndian.PutUint64(mc8G8(dg, r+1)[:], v.GetHi().ReshapeToUint64s().GetElem(0))
			}
			a0, b0, a2, b2 = a2, b2, a4, b4
		}
		return
	}
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		if !dg.Fits(0, c, 16, h) {
			return
		}
		dc := dg.Sub(0, c, 16, h)
		l0 := mc8LdI16(im, c)
		l1 := mc8LdI16(im, c+w)
		l2 := mc8LdI16(im, c+2*w)
		l3 := mc8LdI16(im, c+3*w)
		l4 := mc8LdI16(im, c+4*w)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a1, b1 := l1.InterleaveLoGrouped(l2), l1.InterleaveHiGrouped(l2)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a3, b3 := l3.InterleaveLoGrouped(l4), l3.InterleaveHiGrouped(l4)
		for r := 0; r < h; r += 2 {
			q := c + (r+5)*w
			l5 := mc8LdI16(im, q)
			l6 := l5
			if r+1 < h {
				l6 = mc8LdI16(im, q+w)
			}
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			a5, b5 := l5.InterleaveLoGrouped(l6), l5.InterleaveHiGrouped(l6)
			x := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(10).SaturateToUint16ConcatGrouped(mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(10)).Min(c255)
			y := mc8VMid(a1, a3, a5, d0, d1, d2, rnd).ShiftAllRight(10).SaturateToUint16ConcatGrouped(mc8VMid(b1, b3, b5, d0, d1, d2, rnd).ShiftAllRight(10)).Min(c255)
			// x and y are rows r and r+1 in natural order, each word < 256.
			v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			v.GetLo().StoreArray(mc8G16(dc, r))
			if r+1 < h {
				v.GetHi().StoreArray(mc8G16(dc, r+1))
			}
			a0, b0, a1, b1, a2, b2, a3, b3, l4 = a2, b2, a3, b3, a4, b4, a5, b5, l6
		}
	}
}

// Width-4 kernels (dav1d put_6tap/prep_6tap .h_w4, .v_w4, .hv_w4). Each
// 256-bit vector holds four rows of four samples: rows r, r+1 in the low lane
// and r+2, r+3 in the high lane. The source view's rows are sixteen bytes
// (max(w+8, 16)); the four-tap horizontal pass reads bytes 1..8 of each.

// mc8H4Row is the eight bytes at window tap 1 of row r of s.
func mc8H4Row(s unsafeutil.Grid[uint8], r int) uint64 {
	return binary.LittleEndian.Uint64(mc8G16(s, r)[1:9])
}

// mc8StPutW4G stores the rows r..r+3 (r+k < h only) of v to dg: dwords 0, 1
// of the low lane and 0, 1 of the high lane.
func mc8StPutW4G(dg unsafeutil.Grid[uint8], r, h int, v archsimd.Uint8x32) {
	u := v.ReshapeToUint32s()
	lo, hi := u.GetLo(), u.GetHi()
	le := binary.LittleEndian
	if r+4 <= h && dg.Fits(r, 0, 4, 4) { // a full group
		d := dg.Sub(r, 0, 4, 4)
		le.PutUint32(mc8G4(d, 0)[:], lo.GetElem(0))
		le.PutUint32(mc8G4(d, 1)[:], lo.GetElem(1))
		le.PutUint32(mc8G4(d, 2)[:], hi.GetElem(0))
		le.PutUint32(mc8G4(d, 3)[:], hi.GetElem(1))
		return
	}
	le.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
	if r+1 < h {
		le.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
	}
	if r+2 < h {
		le.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
	}
	if r+3 < h {
		le.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
	}
}

// mc8AVX2PutH4W4 is the w=4 put horizontal-only pass.
func mc8AVX2PutH4W4(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 16, h) {
		return
	}
	s := sg.Sub(0, 0, 16, h)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	r := 0
	for ; r < h-3; r += 4 {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, r+1), mc8H4Row(s, r+2), mc8H4Row(s, r+3), sA, sB, t0, t1).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		u4 := v.ReshapeToUint32s()
		lo, hi := u4.GetLo(), u4.GetHi()
		binary.LittleEndian.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
	}
	if r < h {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, min(r+1, h-1)), mc8H4Row(s, min(r+2, h-1)), mc8H4Row(s, h-1), sA, sB, t0, t1).Add(rnd).ShiftAllRight(6)
		mc8StPutW4G(dg, r, h, mc8Pack(a, a, zero, c255, deint))
	}
}

// mc8AVX2PrepH4W4 is the w=4 prep horizontal-only pass.
func mc8AVX2PrepH4W4(sg unsafeutil.Grid[uint8], h int, ph *mc8Phase, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 16, h) {
		return
	}
	s := sg.Sub(0, 0, 16, h)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	r := 0
	for ; r < h-3; r += 4 {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, r+1), mc8H4Row(s, r+2), mc8H4Row(s, r+3), sA, sB, t0, t1).Add(rnd)
		mc8StU16(out, r*4, a.ToBits().ShiftAllRight(2))
	}
	if r < h {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, min(r+1, h-1)), mc8H4Row(s, min(r+2, h-1)), mc8H4Row(s, h-1), sA, sB, t0, t1).Add(rnd)
		mc8StPrepW4(out, r, h, a.ToBits().ShiftAllRight(2))
	}
}

// mc8AVX2MidH4W4 writes the w=4 2D intermediate (row stride 4) from the rows
// rows of sg. The last group may write up to three rows past rows inside the
// scratch block.
func mc8AVX2MidH4W4(sg unsafeutil.Grid[uint8], rows int, ph *mc8Phase, im *mc8IM) {
	if uint(rows-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	if !sg.Fits(0, 0, 16, rows) {
		return
	}
	s := sg.Sub(0, 0, 16, rows)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(2)
	r := 0
	for ; r < rows-3; r += 4 {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, r+1), mc8H4Row(s, r+2), mc8H4Row(s, r+3), sA, sB, t0, t1).Add(rnd).ShiftAllRight(2)
		mc8StI16(im, r*4, a)
	}
	if r < rows {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, min(r+1, rows-1)), mc8H4Row(s, min(r+2, rows-1)), mc8H4Row(s, rows-1), sA, sB, t0, t1).Add(rnd).ShiftAllRight(2)
		mc8StI16(im, r*4, a)
	}
}

// mc8PackW4 packs the h+5 four-byte source rows of sg into im at stride 4
// bytes (two words per row), so a 16-byte load holds four consecutive rows.
func mc8PackW4(sg unsafeutil.Grid[uint8], h int, im *mc8IM) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 4, h+5) {
		return
	}
	s := sg.Sub(0, 0, 4, h+5)
	for k := 0; k < h+5; k++ {
		v := binary.LittleEndian.Uint32(mc8G4(s, k)[:])
		p := (*[2]int16)(im[2*k : 2*k+2])
		p[0], p[1] = int16(v), int16(v>>16) // one 32-bit store
	}
}

// mc8AVX2PutV6W4 is the w=4 put vertical-only pass. The h+5 source rows are
// first packed into im (mc8PackW4); zero-extended, row k+1 shifted into the
// high bytes forms the vertical byte pairs for four output rows at once.
func mc8AVX2PutV6W4(sg unsafeutil.Grid[uint8], ph *mc8Phase, im *mc8IM, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	mc8PackW4(sg, h, im)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	rnd := archsimd.BroadcastInt16x16(32)
	pa := mc8Pair(mc8ZxW4(im, 0), mc8ZxW4(im, 2))
	pb := mc8Pair(mc8ZxW4(im, 4), mc8ZxW4(im, 6))
	for r := 0; r < h; r += 4 {
		q := 2 * (r + 4)
		pc := mc8Pair(mc8ZxW4(im, q), mc8ZxW4(im, q+2))
		a := mc8V6(pa, pb, pc, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		if r+4 > h {
			mc8StPutW4G(dg, r, h, v)
			return
		}
		u4 := v.ReshapeToUint32s()
		lo, hi := u4.GetLo(), u4.GetHi()
		binary.LittleEndian.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
		pa = pc
		pb = mc8Pair(mc8ZxW4(im, q+4), mc8ZxW4(im, q+6))
	}
}

// mc8AVX2PrepV6W4 is the w=4 prep vertical-only pass, laid out as
// mc8AVX2PutV6W4.
func mc8AVX2PrepV6W4(sg unsafeutil.Grid[uint8], h int, ph *mc8Phase, im *mc8IM, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	mc8PackW4(sg, h, im)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	pa := mc8Pair(mc8ZxW4(im, 0), mc8ZxW4(im, 2))
	pb := mc8Pair(mc8ZxW4(im, 4), mc8ZxW4(im, 6))
	for r := 0; r < h; r += 4 {
		q := 2 * (r + 4)
		pc := mc8Pair(mc8ZxW4(im, q), mc8ZxW4(im, q+2))
		s := mc8V6(pa, pb, pc, t0, t1, t2)
		mc8StPrepW4(out, r, h, s.Add(rnd).ToBits().ShiftAllRight(2))
		pa = pc
		pb = mc8Pair(mc8ZxW4(im, q+4), mc8ZxW4(im, q+6))
	}
}

// mc8AVX2PutV6MidW4 is the w=4 put 2D vertical pass. A 32-byte load at row k
// of the stride-4 intermediate holds rows k..k+3; the per-lane word interleave
// with the load at row k+1 gives the row pairs (k, k+1) | (k+2, k+3) (low
// halves) and (k+1, k+2) | (k+3, k+4) (high halves), so VPACKUSDW of the two
// sums is rows r..r+3 in order.
func mc8AVX2PutV6MidW4(im *mc8IM, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2 := mc8TapsD(ph.d[0]), mc8TapsD(ph.d[1]), mc8TapsD(ph.d[2])
	// clip((S + 512) >> 10) is ((S + 512) >> 6) >> 4 after the
	// unsigned-saturating pack (nested floor division; the >> 6 values fit).
	// Both shifts stay immediates: a variable count makes the compiler emit a
	// legacy-SSE MOVQ into an XMM register.
	rnd := archsimd.BroadcastInt32x8(512)
	c255 := archsimd.BroadcastUint16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	l0, l1 := mc8LdI16(im, 0), mc8LdI16(im, 4)
	l2, l3 := mc8LdI16(im, 8), mc8LdI16(im, 12)
	pa, qa := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
	pb, qb := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
	for r := 0; r < h; r += 4 {
		q := 4 * (r + 4)
		l4, l5 := mc8LdI16(im, q), mc8LdI16(im, q+4)
		pc, qc := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		x := mc8VMid(pa, pb, pc, d0, d1, d2, rnd).ShiftAllRight(6)
		y := mc8VMid(qa, qb, qc, d0, d1, d2, rnd).ShiftAllRight(6)
		u := x.SaturateToUint16ConcatGrouped(y)
		v := u.ShiftAllRight(4).Min(c255).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
		if r+4 > h {
			mc8StPutW4G(dg, r, h, v)
			return
		}
		u4 := v.ReshapeToUint32s()
		lo, hi := u4.GetLo(), u4.GetHi()
		binary.LittleEndian.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
		pa, qa = pc, qc
		l6, l7 := mc8LdI16(im, q+8), mc8LdI16(im, q+12)
		pb, qb = l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
	}
}

// mc8AVX2PrepCopy is the prep copy (dav1d prep_8bpc), 16 * px + 6144, from
// the h rows of sg into the w x h CONV_BUF out.
func mc8AVX2PrepCopy(sg unsafeutil.Grid[uint8], w, h int, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	off := archsimd.BroadcastUint16x16(6144)
	le := binary.LittleEndian
	switch w {
	case 4:
		if !sg.Fits(0, 0, 4, h) {
			return
		}
		s := sg.Sub(0, 0, 4, h)
		for r := 0; r < h; r += 4 {
			var z archsimd.Uint32x4
			v := z.SetElem(0, le.Uint32(mc8G4(s, r)[:])).SetElem(1, le.Uint32(mc8G4(s, min(r+1, h-1))[:])).
				SetElem(2, le.Uint32(mc8G4(s, min(r+2, h-1))[:])).SetElem(3, le.Uint32(mc8G4(s, min(r+3, h-1))[:]))
			mc8StPrepW4(out, r, h, v.ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off))
		}
	case 8:
		if !sg.Fits(0, 0, 8, h) {
			return
		}
		s := sg.Sub(0, 0, 8, h)
		r := 0
		for ; r < h-1; r += 2 {
			var z archsimd.Uint64x2
			u := z.SetElem(0, le.Uint64(mc8G8(s, r)[:])).SetElem(1, le.Uint64(mc8G8(s, r+1)[:])).ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off)
			mc8StU16(out, r*8, u)
		}
		if r < h {
			var z archsimd.Uint64x2
			u := z.SetElem(0, le.Uint64(mc8G8(s, r)[:])).ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off)
			mc8StU16x8(out, r*8, u.GetLo())
		}
	default:
		c := 0
		for ; c+16 <= w; c += 16 {
			if !sg.Fits(0, c, 16, h) {
				return
			}
			s := sg.Sub(0, c, 16, h)
			for r := 0; r < h; r++ {
				mc8StU16(out, r*w+c, archsimd.LoadUint8x16Array(mc8G16(s, r)).ExtendToUint16().ShiftAllLeft(4).Add(off))
			}
		}
		if c < w {
			if !sg.Fits(0, c, 16, h) {
				return
			}
			s := sg.Sub(0, c, 16, h)
			for r := 0; r < h; r++ {
				mc8StU16x8(out, r*w+c, archsimd.LoadUint8x16Array(mc8G16(s, r)).ExtendToUint16().ShiftAllLeft(4).Add(off).GetLo())
			}
		}
	}
}
