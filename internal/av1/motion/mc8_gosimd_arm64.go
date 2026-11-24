// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"errors"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// dav1d-style 8-bit translational motion compensation (src/mc_tmpl.c put_8tap
// and prep_8tap, src/arm/64/mc.S filter_fn). The decoder spends most of its
// inter-prediction time in the regular and smooth filters, whose outer taps are
// zero, so like dav1d these kernels run them as six taps (window taps 1..6).
// The <=4 filters and bilinear also fit that window. Sharp (eight taps), width
// 4 and the high-bit-depth paths keep the general kernels.
//
// Arithmetic follows the libaom-exact single-prediction and CONV_BUF formulas
// of convolve{X,Y,2D}8PureGo and predictInterCompoundRef8ToConvBuf*PureGo. All
// AV1 subpel taps are even, so the horizontal and 1D passes use halved taps in
// int16 lanes (the sums are exact modulo 2^16 and the final sums fit). The 2D
// intermediate is the rounded horizontal sum widened to int32 so the vertical
// pass is one multiply-accumulate per tap and lane half:
//
//	put H:   clip((s + 34) >> 6)               s = sum(half tap * px)
//	put V:   clip((s + 32) >> 6)
//	put 2D:  m = (s + 2) >> 2;  clip((sum(32 * tap * m) + 2^15) >> 16)
//	prep H:  ((s + 2) >> 2) + 6144              (as (s + 24578) >> 2, unsigned)
//	prep V:  ((s + 2) >> 2) + 6144
//	prep 2D: (sum(tap * m) + 64 + 6144 << 7) >> 7
//
// put H folds libaom's round3-then-round4 into one shift (nested floor
// division), put 2D reads the >>16 straight from the high half of each int32
// lane, and prep H/V fold the CONV_BUF offset into the rounding constant.
//
// Every access is a bounds-checked fixed-size window of a row slice: the
// source window from mc8Source, the destination rows Put8 checked, and the
// intermediate and CONV_BUF rows of their scratch arrays.

const (
	mc8KindRegular = iota
	mc8KindSmooth
	mc8KindSharp
	mc8KindBilinear
	mc8KindRegular4
	mc8KindSmooth4
	mc8KindCount
)

// mc8Phase is one filter phase pre-broadcast for the six-tap window.
type mc8Phase struct {
	h [6][8]int16 // halved taps
	v [6][4]int32 // put 2D vertical: 32 * tap
	p [6][4]int32 // prep 2D vertical: 512 * tap
	w [6][4]int32 // 16bpc one-axis passes: halved taps on int32 lanes
}

var mc8Phases [mc8KindCount][16]mc8Phase

// mc8Vec holds the kernels' constant rounding and shift vectors. A kernel
// loads them (one table address, one vector load each) rather than
// broadcasting: a broadcast builds the value in a general register, inserts
// it into a lane and duplicates it.
var mc8Vec = struct {
	rnd32, rnd34, prepRnd, rnd2, sh6, sh2 [8]int16
	w32, wsh6, w15                        [4]int32
	mul16                                 [16]uint8
	off6144                               [8]uint16
}{
	rnd2:    [8]int16{2, 2, 2, 2, 2, 2, 2, 2},
	w15:     [4]int32{1 << 15, 1 << 15, 1 << 15, 1 << 15},
	mul16:   [16]uint8{16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},
	off6144: [8]uint16{6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144},
	rnd32:   [8]int16{32, 32, 32, 32, 32, 32, 32, 32},
	rnd34:   [8]int16{34, 34, 34, 34, 34, 34, 34, 34},
	prepRnd: [8]int16{2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2},
	sh6:     [8]int16{-6, -6, -6, -6, -6, -6, -6, -6},
	sh2:     [8]int16{-2, -2, -2, -2, -2, -2, -2, -2},
	w32:     [4]int32{32, 32, 32, 32},
	wsh6:    [4]int32{-6, -6, -6, -6},
}

// mc8SixTap reports whether every phase of kind lies inside window taps 1..6.
var mc8SixTap [mc8KindCount]bool

func init() {
	tables := [mc8KindCount]*[16][filterTaps]int16{
		mc8KindRegular:  &subpelFilters8,
		mc8KindSmooth:   &subpelFilters8Smooth,
		mc8KindSharp:    &subpelFilters8Sharp,
		mc8KindBilinear: &bilinearFilters,
		mc8KindRegular4: &subpelFilters4,
		mc8KindSmooth4:  &subpelFilters4Smooth,
	}
	for kind, tab := range tables {
		six := true
		for s := range 16 {
			k := &tab[s]
			if k[0] != 0 || k[7] != 0 {
				six = false
			}
			for _, c := range k {
				if c&1 != 0 {
					six = false
				}
			}
			ph := &mc8Phases[kind][s]
			for i := range 6 {
				c := k[1+i]
				for l := range 8 {
					ph.h[i][l] = c / 2
				}
				for l := range 4 {
					ph.v[i][l] = int32(c) * 32
					ph.p[i][l] = int32(c) * 512
					ph.w[i][l] = int32(c / 2)
				}
			}
		}
		mc8SixTap[kind] = six
	}
}

// mc8Kind mirrors interpKernel's table selection: one table load indexed by
// the filter (values past the last filter select bilinear, as interpKernel's
// default arm does) and whether the block is at most four samples wide.
func mc8Kind(f InterpFilter, blockSize int) int {
	small := 0
	if blockSize <= 4 {
		small = 1
	}
	return int(mc8KindTab[small][min(f, InterpBilinear)])
}

var mc8KindTab = [2][interpFilterCount]uint8{
	{mc8KindRegular, mc8KindSmooth, mc8KindSharp, mc8KindBilinear},
	{mc8KindRegular4, mc8KindSmooth4, mc8KindRegular4, mc8KindBilinear},
}

// mc8Source resolves the six-tap source window of a w x h block whose
// first output sample reads ref at (refX, refY). halo selects the filtered
// axes. The window is returned as a Grid view from its top-left tap sample:
// bh rows (h, plus 5 on a filtered vertical axis) of max(w+8, 16) bytes, the
// bytes every kernel may load (16 per eight output columns). When any tap
// falls outside the plane the clamped window is first materialized in edge
// (dav1d src/recon_tmpl.c mc() emu_edge). ok is false only when the reference
// plane itself is unusable.
func mc8Source(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src unsafeutil.Grid[uint8], ok bool) {
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
	// The edge window holds 135 rows of 136 bytes; bh <= 133 and the
	// kernels read at most max(w+8, 16) <= 136 bytes per row.
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
		// mc8CopyFast written out for the power-of-two widths (every AV1
		// block width): one call, CopyRect's.
		if w&(w-1) != 0 || w&3 != 0 || w > maxBlockSize || h <= 0 ||
			dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
			refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < ref.Width {
			return mc8CopyFast(dst, ref, dstX, dstY, refX, refY, w, h)
		}
		return unsafeutil.CopyRect(dst.Pix, dstY*dst.Stride+dstX, dst.Stride, ref.Pix, refY*ref.Stride+refX, ref.Stride, w, h)
	}
	if w == 2 {
		return mc8PutW2(dst, ref, dstX, dstY, refX, refY, h, filterW, filterH, subX, subY, filters, scratch)
	}
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && mc8Put8TapChecked(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The six-tap source window (mc8Source), in place when it lies inside
	// the plane: halo 2 before and 3 after each sample on a filtered axis.
	// The common case, destination and source window both in their planes,
	// is checked in line with one footprint test for the pair and runs the
	// kernel with no call before it; everything else is mc8PutEdge.
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
		im := scratch.imHBD[:]
		if w == 4 {
			mc8H6ToI32W4(sg, h+5, phx, im)
			mc8PutV6FromI32W4(im, h, phy, dg)
		} else {
			mc8H6ToI32(sg, w, h+5, phx, im, mc8PutH6ToI32Rnd)
			mc8PutV6FromI32(im, w, h, phy, dg)
		}
	case subX != 0:
		switch {
		case w == 4:
			mc8PutH6S4(sg, phx, dg)
		case w == 8:
			mc8PutH6S8(sg, phx, dg)
		case w&15 == 0:
			mc8PutH6W16(sg, phx, dg)
		default:
			mc8PutH6Split(sg, phx, dg)
		}
	default:
		switch {
		case w == 4:
			mc8PutV6S4(sg, phy, dg)
		case w == 8:
			mc8PutV6S8(sg, phy, dg)
		case w&15 == 0:
			mc8PutV6W16(sg, phy, dg)
		default:
			mc8PutV6Split(sg, phy, dg)
		}
	}
	return true
}

// mc8PutEdge is Put8's six-tap path for a block whose destination or source
// window is not inside its plane (or whose views do not fit): the checked
// destination view and the source window from mc8Source, which clamps it.
func mc8PutEdge(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	dg, ok := mc8DstGrid(dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	sg, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	mc8PutGrid(dg, sg, w, h, subX, subY, &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15], scratch)
	return true
}

// mc8SameGeometry reports whether two planes have the same dimensions,
// stride and buffer length, so one footprint check covers both.
func mc8SameGeometry(a, b *frame.Plane) bool {
	return a.Width == b.Width && a.Height == b.Height && a.Stride == b.Stride && len(a.Pix) == len(b.Pix)
}

// Put8UV is Put8 for the U and V blocks of one chroma prediction: both planes
// share the block, the reference origin, the phase and the filters (dav1d
// recon_b_inter's two chroma mc() calls). When the two destination planes and
// the two reference planes have the same geometry, the shape, kernel and phase
// selection and every footprint check run once for the pair; the per-plane
// kernels then run back to back. It returns false when some plane was not
// written; the caller then predicts both planes plane by plane (rewriting a
// plane already written here yields the same samples).
func Put8UV(dstU, dstV, refU, refV *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if w == 2 || !mc8SameGeometry(dstU, dstV) || !mc8SameGeometry(refU, refV) {
		return Put8(dstU, refU, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY, filters, scratch) &&
			Put8(dstV, refV, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY, filters, scratch)
	}
	if subX|subY == 0 {
		if w&3 != 0 || w > maxBlockSize || h <= 0 {
			return false
		}
		if dstX < 0 || dstY < 0 || dstX+w > dstU.Width || dstY+h > dstU.Height || dstU.Stride < dstU.Width ||
			refX < 0 || refY < 0 || refX+w > refU.Width || refY+h > refU.Height || refU.Stride < refU.Width {
			return false
		}
		// The planes of each pair share their geometry, so the V views fit
		// exactly when the U views do.
		dOff, sOff := dstY*dstU.Stride+dstX, refY*refU.Stride+refX
		if w&(w-1) == 0 { // every AV1 block width
			return unsafeutil.CopyRect(dstU.Pix, dOff, dstU.Stride, refU.Pix, sOff, refU.Stride, w, h) &&
				unsafeutil.CopyRect(dstV.Pix, dOff, dstV.Stride, refV.Pix, sOff, refV.Stride, w, h)
		}
		if !unsafeutil.GridFit(len(dstU.Pix), dOff, dstU.Stride, w, h) || !unsafeutil.GridFit(len(refU.Pix), sOff, refU.Stride, w, h) {
			return false
		}
		du, su := unsafeutil.NewGrid(dstU.Pix, dOff, dstU.Stride, w, h), unsafeutil.NewGrid(refU.Pix, sOff, refU.Stride, w, h)
		dv, sv := unsafeutil.NewGrid(dstV.Pix, dOff, dstV.Stride, w, h), unsafeutil.NewGrid(refV.Pix, sOff, refV.Stride, w, h)
		return mc8CopyStrips(du, su) && mc8CopyStrips(dv, sv)
	}
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && mc8Put8TapChecked(dstU, refU, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch) &&
			mc8Put8TapChecked(dstV, refV, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The source window, in-plane for both planes at once: that case checks
	// the U views in line (the V planes share their geometry, so the V views
	// fit exactly when they do) and runs the two planes' kernels back to back
	// with no call before them; mc8PutUVEdge takes the rest.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	dOff, sOff, sw := dstY*dstU.Stride+dstX, y0*refU.Stride+x0, max(w+8, 16)
	if dstX < 0 || dstY < 0 || dstX+w > dstU.Width || dstY+h > dstU.Height || dstU.Stride < dstU.Width ||
		x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > refU.Width || y0+bh > refU.Height || refU.Stride < refU.Width ||
		!unsafeutil.GridFit(len(dstU.Pix), dOff, dstU.Stride, w, h) || !unsafeutil.GridFit(len(refU.Pix), sOff, refU.Stride, sw, bh) {
		return mc8PutUVEdge(dstU, dstV, refU, refV, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	du, su := unsafeutil.NewGrid(dstU.Pix, dOff, dstU.Stride, w, h), unsafeutil.NewGrid(refU.Pix, sOff, refU.Stride, sw, bh)
	dv, sv := unsafeutil.NewGrid(dstV.Pix, dOff, dstV.Stride, w, h), unsafeutil.NewGrid(refV.Pix, sOff, refV.Stride, sw, bh)
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX != 0 && subY != 0:
		im := scratch.imHBD[:]
		if w == 4 {
			mc8H6ToI32W4(su, h+5, phx, im)
			mc8PutV6FromI32W4(im, h, phy, du)
			mc8H6ToI32W4(sv, h+5, phx, im)
			mc8PutV6FromI32W4(im, h, phy, dv)
		} else {
			mc8H6ToI32(su, w, h+5, phx, im, mc8PutH6ToI32Rnd)
			mc8PutV6FromI32(im, w, h, phy, du)
			mc8H6ToI32(sv, w, h+5, phx, im, mc8PutH6ToI32Rnd)
			mc8PutV6FromI32(im, w, h, phy, dv)
		}
	case subX != 0:
		switch {
		case w == 4:
			mc8PutH6S4(su, phx, du)
			mc8PutH6S4(sv, phx, dv)
		case w == 8:
			mc8PutH6S8(su, phx, du)
			mc8PutH6S8(sv, phx, dv)
		case w&15 == 0:
			mc8PutH6W16(su, phx, du)
			mc8PutH6W16(sv, phx, dv)
		default:
			mc8PutH6Split(su, phx, du)
			mc8PutH6Split(sv, phx, dv)
		}
	default:
		switch {
		case w == 4:
			mc8PutV6S4(su, phy, du)
			mc8PutV6S4(sv, phy, dv)
		case w == 8:
			mc8PutV6S8(su, phy, du)
			mc8PutV6S8(sv, phy, dv)
		case w&15 == 0:
			mc8PutV6W16(su, phy, du)
			mc8PutV6W16(sv, phy, dv)
		default:
			mc8PutV6Split(su, phy, du)
			mc8PutV6Split(sv, phy, dv)
		}
	}
	return true
}

// mc8PutUVEdge is Put8UV's six-tap path when the destination or the shared
// source window is not inside its planes (or the views do not fit): the
// checked destination views, and each plane's clamped window materialized in
// turn in the shared edge buffer.
func mc8PutUVEdge(dstU, dstV, refU, refV *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	dOff := dstY*dstU.Stride + dstX
	if dstX < 0 || dstY < 0 || dstX+w > dstU.Width || dstY+h > dstU.Height || dstU.Stride < dstU.Width ||
		!unsafeutil.GridFit(len(dstU.Pix), dOff, dstU.Stride, w, h) {
		return false
	}
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	for p := 0; p < 2; p++ {
		dst, ref := dstU, refU
		if p == 1 {
			dst, ref = dstV, refV
		}
		sg, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
		if !ok {
			return false
		}
		mc8PutGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), sg, w, h, subX, subY, phx, phy, scratch)
	}
	return true
}

// mc8DstGrid is Put8's destination check: the w x h block at (x, y) of dst as
// a Grid view.
func mc8DstGrid(dst *frame.Plane, x, y, w, h int) (unsafeutil.Grid[uint8], bool) {
	off := y*dst.Stride + x
	if x < 0 || y < 0 || x+w > dst.Width || y+h > dst.Height || dst.Stride < dst.Width ||
		!unsafeutil.GridFit(len(dst.Pix), off, dst.Stride, w, h) {
		return unsafeutil.Grid[uint8]{}, false
	}
	return unsafeutil.NewGrid(dst.Pix, off, dst.Stride, w, h), true
}

// mc8Halo is the source window halo of an axis: 2 when its phase is nonzero.
func mc8Halo(sub int) int {
	if sub != 0 {
		return 2
	}
	return 0
}

// mc8PutGrid runs the six-tap put kernel of a validated block (w 4 or a
// multiple of 8, a nonzero phase on some axis): dg is the w x h destination,
// sg the source window (mc8Source).
func mc8PutGrid(dg, sg unsafeutil.Grid[uint8], w, h, subX, subY int, phx, phy *mc8Phase, scratch *ConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		im := scratch.imHBD[:]
		if w == 4 {
			mc8H6ToI32W4(sg, h+5, phx, im)
			mc8PutV6FromI32W4(im, h, phy, dg)
		} else {
			mc8H6ToI32(sg, w, h+5, phx, im, mc8PutH6ToI32Rnd)
			mc8PutV6FromI32(im, w, h, phy, dg)
		}
	case subX != 0:
		switch {
		case w == 4:
			mc8PutH6S4(sg, phx, dg)
		case w == 8:
			mc8PutH6S8(sg, phx, dg)
		case w&15 == 0:
			mc8PutH6W16(sg, phx, dg)
		default:
			mc8PutH6Split(sg, phx, dg)
		}
	default:
		switch {
		case w == 4:
			mc8PutV6S4(sg, phy, dg)
		case w == 8:
			mc8PutV6S8(sg, phy, dg)
		case w&15 == 0:
			mc8PutV6W16(sg, phy, dg)
		default:
			mc8PutV6Split(sg, phy, dg)
		}
	}
}

// mc8PutH6Split is the put horizontal pass of a block 16k+8 wide (no AV1
// block is; the kernels allow it): the sixteen-column strips, then the last
// eight columns.
func mc8PutH6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	w, h := dg.Width(), dg.Height()
	w16 := w &^ 15
	mc8PutH6W16(sg, ph, dg.Sub(0, 0, w16, h))
	mc8PutH6S8(sg.Sub(0, w16, 16, h), ph, dg.Sub(0, w16, 8, h))
}

// mc8PutV6Split is mc8PutH6Split for the vertical pass.
func mc8PutV6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	w, h := dg.Width(), dg.Height()
	w16 := w &^ 15
	mc8PutV6W16(sg, ph, dg.Sub(0, 0, w16, h))
	mc8PutV6S8(sg.Sub(0, w16, 16, h+5), ph, dg.Sub(0, w16, 8, h))
}

// mc8PrepH6Split is mc8PutH6Split for the prep horizontal pass.
func mc8PrepH6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	w, h := og.Width(), og.Height()
	w16 := w &^ 15
	mc8PrepH6W16(sg, ph, og.Sub(0, 0, w16, h))
	mc8PrepH6S8(sg.Sub(0, w16, 16, h), ph, og.Sub(0, w16, 8, h))
}

// mc8PrepV6Split is mc8PutH6Split for the prep vertical pass.
func mc8PrepV6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	w, h := og.Width(), og.Height()
	w16 := w &^ 15
	mc8PrepV6W16(sg, ph, og.Sub(0, 0, w16, h))
	mc8PrepV6S8(sg.Sub(0, w16, 16, h+5), ph, og.Sub(0, w16, 8, h))
}

// mc8PutW2 is Put8 at width 2, dav1d's put_8tap w2 for the sub-8x8 chroma
// blocks of recon_b_inter: the width-4 kernels filter into a local block of
// 4-byte rows whose first two columns are copied out, so nothing past the
// block is written. Output columns 0 and 1 read only the taps a 2-wide filter
// reads; mc8Source sizes the window for the 4-wide kernels, and any taps it
// edge-extends past the plane feed only the discarded columns.
func mc8PutW2(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if h <= 0 || h > mc8W2MaxH || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+2 > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	if dOff+(h-1)*dst.Stride+2 > len(dst.Pix) {
		return false
	}
	sg, ok := mc8Source(ref, refX, refY, 4, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	var tmp [4 * mc8W2MaxH]byte
	tg := unsafeutil.NewGrid(tmp[:], 0, 4, 4, h)
	switch {
	case subX != 0 && subY != 0:
		im := scratch.imHBD[:]
		mc8H6ToI32W4(sg, h+5, &mc8Phases[kx][subX], im)
		mc8PutV6FromI32W4(im, h, &mc8Phases[ky][subY], tg)
	case subX != 0:
		mc8PutH6S4(sg, &mc8Phases[kx][subX], tg)
	default:
		mc8PutV6S4(sg, &mc8Phases[ky][subY], tg)
	}
	rows := dst.Pix[dOff : dOff+(h-1)*dst.Stride+2]
	for r := 0; r < h; r++ {
		row := rows[r*dst.Stride : r*dst.Stride+2]
		row[0], row[1] = tmp[4*r], tmp[4*r+1]
	}
	return true
}

// mc8W2MaxH bounds mc8PutW2's block height: a 2-wide chroma block comes from
// a 4-wide luma block, at most 4x16 (2x16 under 4:2:2).
const mc8W2MaxH = 16

// mc8CopyFast is the put copy (dav1d put_neon) for an in-plane w x h 8-bit
// block with w a multiple of 4: vector rows instead of one memmove per row.
// It returns false when the source leaves the plane (the clamped copy runs).
func mc8CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	if refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+dstX, refY*ref.Stride+refX
	if w&(w-1) == 0 { // every AV1 block width
		return unsafeutil.CopyRect(dst.Pix, dOff, dst.Stride, ref.Pix, sOff, ref.Stride, w, h)
	}
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, w, h) {
		return false
	}
	return mc8CopyStrips(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, w, h))
}

// Window helpers. Offsets are uint32 so o+N cannot overflow and each window
// costs one compare; the entries keep every window slice below 1<<32
// elements (mc8WindowMax), so the offsets never wrap.
func mc8W4(s []byte, o uint32) *[4]byte      { return (*[4]byte)(s[uint(o) : uint(o)+4]) }
func mc8W8(s []byte, o uint32) *[8]byte      { return (*[8]byte)(s[uint(o) : uint(o)+8]) }
func mc8W16(s []byte, o uint32) *[16]byte    { return (*[16]byte)(s[uint(o) : uint(o)+16]) }
func mc8W32(s []byte, o uint32) *[32]byte    { return (*[32]byte)(s[uint(o) : uint(o)+32]) }
func mc8WU4(s []uint16, o uint32) *[4]uint16 { return (*[4]uint16)(s[uint(o) : uint(o)+4]) }
func mc8WU8(s []uint16, o uint32) *[8]uint16 { return (*[8]uint16)(s[uint(o) : uint(o)+8]) }

// mc8WindowMax bounds the window slices handed to the kernels (see mc8W16).
const mc8WindowMax = 1<<32 - 1

// Limit-checked windows. A kernel computes lim = len(s)-N once and walks a
// plain int offset; each window is then a sign test and one compare against
// lim, which together prove the slice expression (0 <= o, o+N <= len(s)), so
// no zero-extension or end-offset arithmetic is left per row.
func mc8R8(s []byte, o, lim int) *[8]byte {
	if o < 0 || o > lim {
		panic(errMC8Window)
	}
	return (*[8]byte)(s[o : o+8])
}

func mc8R16(s []byte, o, lim int) *[16]byte {
	if o < 0 || o > lim {
		panic(errMC8Window)
	}
	return (*[16]byte)(s[o : o+16])
}

func mc8R32(s []byte, o, lim int) *[32]byte {
	if o < 0 || o > lim {
		panic(errMC8Window)
	}
	return (*[32]byte)(s[o : o+32])
}

// errMC8Window is the panic value of a window past its limit (a caller bug:
// the entries validate every block footprint).
var errMC8Window = errors.New("motion: kernel window out of range")

// mc8PrepFast fills a w x h CONV_BUF (row stride w) with the dav1d-style prep
// kernels. It returns false for shapes it does not cover.
func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && len(out) >= w*h && mc8Prep8Tap(out[:w*h], ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The six-tap source window (mc8Source), in place when it lies inside
	// the plane, as in Put8: that case is checked in line and runs its
	// kernel with no call before it; mc8PrepEdge takes the rest.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	sOff, sw := y0*ref.Stride+x0, max(w+8, 16)
	if x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < ref.Width ||
		!unsafeutil.GridFit(len(out), 0, w, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		return mc8PrepEdge(out, ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	og, sg := unsafeutil.NewGrid(out, 0, w, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	// mc8PrepGrid written out, as in Put8.
	switch {
	case subX != 0 && subY != 0:
		im := scratch.im[:]
		if w == 4 {
			mc8H6ToI32W4(sg, h+5, &mc8Phases[kx][subX], im)
			mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], out[:4*h], mc8PrepV6Rnd)
		} else {
			mc8H6ToI32(sg, w, h+5, &mc8Phases[kx][subX], im, 2)
			mc8PrepV6FromI32S(im, w, h, &mc8Phases[ky][subY], og, mc8PrepV6Rnd)
		}
	case subX != 0:
		ph := &mc8Phases[kx][subX]
		switch {
		case w == 4:
			mc8PrepH6S4(sg, ph, og)
		case w == 8:
			mc8PrepH6S8(sg, ph, og)
		case w&15 == 0:
			mc8PrepH6W16(sg, ph, og)
		default:
			mc8PrepH6Split(sg, ph, og)
		}
	case subY != 0:
		ph := &mc8Phases[ky][subY]
		switch {
		case w == 4:
			mc8PrepV6S4(sg, ph, og)
		case w == 8:
			mc8PrepV6S8(sg, ph, og)
		case w&15 == 0:
			mc8PrepV6W16(sg, ph, og)
		default:
			mc8PrepV6Split(sg, ph, og)
		}
	default:
		if w == 8 {
			mc8PrepCopyW8(sg, out, h)
		} else {
			mc8PrepCopy(sg, og)
		}
	}
	return true
}

// mc8PrepEdge is mc8PrepFast's six-tap path for a source window that is not
// inside the plane (or views that do not fit): the window from mc8Source,
// which clamps it.
func mc8PrepEdge(out []uint16, ref *frame.Plane, refX, refY, w, h, subX, subY, kx, ky int, scratch *CompoundConvolveScratch) bool {
	if !unsafeutil.GridFit(len(out), 0, w, w, h) {
		return false
	}
	sg, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	mc8PrepGrid(out, unsafeutil.NewGrid(out, 0, w, w, h), sg, w, h, subX, subY, &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15], scratch)
	return true
}

// mc8PrepGrid runs the six-tap prep kernel of a validated block (w 4 or a
// multiple of 8): og is the w x h CONV_BUF view of out, sg the source window
// (mc8Source).
func mc8PrepGrid(out []uint16, og unsafeutil.Grid[uint16], sg unsafeutil.Grid[uint8], w, h, subX, subY int, phx, phy *mc8Phase, scratch *CompoundConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		im := scratch.im[:]
		if w == 4 {
			mc8H6ToI32W4(sg, h+5, phx, im)
			mc8PrepV6FromI32W4(im, h, phy, out[:4*h], mc8PrepV6Rnd)
		} else {
			mc8H6ToI32(sg, w, h+5, phx, im, 2)
			mc8PrepV6FromI32S(im, w, h, phy, og, mc8PrepV6Rnd)
		}
	case subX != 0:
		switch {
		case w == 4:
			mc8PrepH6S4(sg, phx, og)
		case w == 8:
			mc8PrepH6S8(sg, phx, og)
		case w&15 == 0:
			mc8PrepH6W16(sg, phx, og)
		default:
			mc8PrepH6Split(sg, phx, og)
		}
	case subY != 0:
		switch {
		case w == 4:
			mc8PrepV6S4(sg, phy, og)
		case w == 8:
			mc8PrepV6S8(sg, phy, og)
		case w&15 == 0:
			mc8PrepV6W16(sg, phy, og)
		default:
			mc8PrepV6Split(sg, phy, og)
		}
	default:
		if w == 8 {
			mc8PrepCopyW8(sg, out, h)
		} else {
			mc8PrepCopy(sg, og)
		}
	}
}

// mc8LoadWiden widens the 16 bytes of p to two uint16 vectors, returned as
// bytes for the tap shifts; the six-tap horizontal sums read lanes 0..12.
func mc8LoadWiden(p *[16]uint8) (archsimd.Uint8x16, archsimd.Uint8x16) {
	a := archsimd.LoadUint8x16Array(p)
	return a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
}

// mc8LoadWiden3 is mc8LoadWiden passing o through, so a call written as
// mc8LoadWiden3(mc8R16(s, o+st, lim), o+st) carries an instruction of its own
// line (the add) and leaves no inline-mark NOP.
func mc8LoadWiden3(p *[16]uint8, o int) (archsimd.Uint8x16, archsimd.Uint8x16, int) {
	lo, hi := mc8LoadWiden(p)
	return lo, hi, o
}

func mc8I16(x archsimd.Uint8x16) archsimd.Int16x8 {
	return x.ReshapeToUint16s().BitsToInt16()
}

// mc8LoadRow16 loads the 16 bytes at p widened to int16 lanes 0..7.
func mc8LoadRow16(p *[16]uint8) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array(p).ExtendLo8ToUint16().BitsToInt16()
}

// mc8Lo64 is the low 8 bytes of v (the eight saturated output pixels).
func mc8Lo64(v archsimd.Uint8x16) uint64 {
	return v.ReshapeToUint64s().GetElem(0)
}

// mc8LoadI32x8 loads the eight int32 intermediate samples at p as two halves.
func mc8LoadI32x8(p *[8]int32) (archsimd.Int32x4, archsimd.Int32x4) {
	return archsimd.LoadInt32x4Array((*[4]int32)(p[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(p[4:8]))
}

// mc8PutH6ToI32Rnd is the put 2D horizontal rounding constant: (s + 2) >> 2
// plus a bias of 8 on every intermediate sample, ((s + 34) >> 2) exactly
// (floor division by 4 commutes with adding a multiple of 4). The six
// vertical taps of every six-tap phase sum to 128, scaled by 32 to 4096, so
// the bias adds 8 * 4096 = 2^15 to every vertical sum: the vertical pass's
// rounding, which mc8V6PutPair therefore no longer adds. The biased samples
// stay below 2^15 (|s| < 20000) and the vertical sums inside int32.
const mc8PutH6ToI32Rnd = 2 + 4*8

// mc8V6PutPair is the put 2D vertical chain of output rows r and r+1 from
// the seven intermediate rows (x, y halves) r..r+6, saturated to bytes. The
// intermediate carries the vertical rounding (mc8PutH6ToI32Rnd).
func mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5 archsimd.Int32x4) (uint64, uint64) {
	a0 := x0.Mul(c0)
	b0 := y0.Mul(c0)
	a1 := x1.Mul(c0)
	b1 := y1.Mul(c0)
	a0 = x1.MulAdd(c1, a0)
	b0 = y1.MulAdd(c1, b0)
	a1 = x2.MulAdd(c1, a1)
	b1 = y2.MulAdd(c1, b1)
	a0 = x2.MulAdd(c2, a0)
	b0 = y2.MulAdd(c2, b0)
	a1 = x3.MulAdd(c2, a1)
	b1 = y3.MulAdd(c2, b1)
	a0 = x3.MulAdd(c3, a0)
	b0 = y3.MulAdd(c3, b0)
	a1 = x4.MulAdd(c3, a1)
	b1 = y4.MulAdd(c3, b1)
	a0 = x4.MulAdd(c4, a0)
	b0 = y4.MulAdd(c4, b0)
	a1 = x5.MulAdd(c4, a1)
	b1 = y5.MulAdd(c4, b1)
	a0 = x5.MulAdd(c5, a0)
	b0 = y5.MulAdd(c5, b0)
	a1 = x6.MulAdd(c5, a1)
	b1 = y6.MulAdd(c5, b1)
	v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16()
	v1 := a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s()).BitsToInt16()
	return mc8Lo64(v0.SaturateToUint8()), mc8Lo64(v1.SaturateToUint8())
}

// mc8PrepV6Rnd is the 8-bit prep 2D vertical rounding constant, pre-scaled
// by the 512x vertical taps: (64 + 6144 << 7) << 9.
const mc8PrepV6Rnd = (64 + 6144<<7) << 9

// mc8V6PrepPair is the prep 2D vertical chain of CONV_BUF rows r and r+1
// from the seven intermediate rows r..r+6.
func mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (archsimd.Uint16x8, archsimd.Uint16x8) {
	// Multiply first and add rnd last, as in mc8V6PutPair.
	a0 := x0.Mul(c0)
	b0 := y0.Mul(c0)
	a1 := x1.Mul(c0)
	b1 := y1.Mul(c0)
	a0 = x1.MulAdd(c1, a0)
	b0 = y1.MulAdd(c1, b0)
	a1 = x2.MulAdd(c1, a1)
	b1 = y2.MulAdd(c1, b1)
	a0 = x2.MulAdd(c2, a0)
	b0 = y2.MulAdd(c2, b0)
	a1 = x3.MulAdd(c2, a1)
	b1 = y3.MulAdd(c2, b1)
	a0 = x3.MulAdd(c3, a0)
	b0 = y3.MulAdd(c3, b0)
	a1 = x4.MulAdd(c3, a1)
	b1 = y4.MulAdd(c3, b1)
	a0 = x4.MulAdd(c4, a0)
	b0 = y4.MulAdd(c4, b0)
	a1 = x5.MulAdd(c4, a1)
	b1 = y5.MulAdd(c4, b1)
	a0 = x5.MulAdd(c5, a0)
	b0 = y5.MulAdd(c5, b0)
	a1 = x6.MulAdd(c5, a1)
	b1 = y6.MulAdd(c5, b1)
	a0, b0, a1, b1 = a0.Add(rnd), b0.Add(rnd), a1.Add(rnd), b1.Add(rnd)
	return a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()), a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s())
}

// mc8H6Sum is the six-tap halved horizontal sum (plus rnd) of the eight
// output columns whose window starts at p.
func mc8H6Sum(p *[16]uint8, c0, c1, c2, c3, c4, c5, rnd archsimd.Int16x8) archsimd.Int16x8 {
	lo, hi := mc8LoadWiden(p)
	sum := mc8I16(lo).MulAdd(c0, rnd)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
	return mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
}

// mc8BlendAvgFast is the 8-bit plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w: libaom's
// clip(round4(((a*8 + b*8) >> 4) - 6144)) is (a + b - 12272) >> 5 by nested
// floor division, evaluated in int16 lanes (8-bit CONV_BUF values stay below
// 2^14, so a + b cannot wrap) and saturated to bytes.
func mc8BlendAvgFast(dst *frame.Plane, a, b []uint16, dstX, dstY, w, h int) bool {
	if w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	off := dstY*dst.Stride + dstX
	ext := (h-1)*dst.Stride + w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+ext > len(dst.Pix) || ext > mc8WindowMax {
		return false
	}
	mc8BlendAvg(dst.Pix[off:off+ext], dst.Stride, a[:w*h], b[:w*h], w, h)
	return true
}

func mc8BlendAvg(dst []byte, dstStride int, a, b []uint16, w, h int) {
	// clamp((a + b - 12272) >> 5, 0, 255) in the shape of dav1d's add +
	// sqrshrun pair: a saturating subtract clamps below at 0, a saturating
	// << 3 clamps floor(u / 32) at 255 in each high byte, and UZP2 gathers
	// those bytes.
	bias := archsimd.BroadcastUint16x8(12272)
	sh := archsimd.BroadcastInt16x8(3)
	avg := func(pa, pb *[8]uint16) archsimd.Uint8x16 {
		return archsimd.LoadUint16x8Array(pa).Add(archsimd.LoadUint16x8Array(pb)).SubSaturated(bias).ShiftSaturated(sh).ReshapeToUint8s()
	}
	// The predictors are contiguous: they are walked as one stream whose
	// loop bound proves each window; only the destination rows are checked.
	n := w * h
	a, b = a[:n:n], b[:n:n]
	ds := uint32(dstStride)
	var d uint32
	switch {
	case w == 4:
		for i := 0; i <= len(a)-8; i += 8 {
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			x := v.ConcatOdd(v).ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint32(mc8W4(dst, d)[:], uint32(x))
			binary.LittleEndian.PutUint32(mc8W4(dst, d+ds)[:], uint32(x>>32))
			d += 2 * ds
		}
	case w == 8:
		for i := 0; i <= len(a)-8; i += 8 {
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.ConcatOdd(v).ReshapeToUint64s().GetElem(0))
			d += ds
		}
	case w&15 == 0:
		ws := uint32(w)
		var c uint32
		for i := 0; i <= len(a)-16; i += 16 {
			pa, pb := (*[16]uint16)(a[i:i+16]), (*[16]uint16)(b[i:i+16])
			v0 := avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8]))
			v1 := avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16]))
			v0.ConcatOdd(v1).StoreArray(mc8W16(dst, d+c))
			if c += 16; c == ws {
				c, d = 0, d+ds
			}
		}
	default: // w = 16k + 8
		ws := uint32(w)
		var c uint32
		for i := 0; i <= len(a)-8; {
			if ws-c == 8 {
				pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
				v := avg(pa, pb)
				binary.LittleEndian.PutUint64(mc8W8(dst, d+c)[:], v.ConcatOdd(v).ReshapeToUint64s().GetElem(0))
				i += 8
				c, d = 0, d+ds
				continue
			}
			pa, pb := (*[16]uint16)(a[i:i+16]), (*[16]uint16)(b[i:i+16])
			v0 := avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8]))
			v1 := avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16]))
			v0.ConcatOdd(v1).StoreArray(mc8W16(dst, d+c))
			i += 16
			c += 16
		}
	}
}

// mc8L16U loads eight uint16 lanes from p (len(p) >= 8).
func mc8L16U(p []uint16) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p))
}
