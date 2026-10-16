// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import "math/bits"

// This file ports dav1d's superblock deblock entry points
// (src/loopfilter_tmpl.c loop_filter_{h,v}_sb128{y,uv}_c and the NEON
// lpf_{h,v}_sb_{y,uv} drivers in src/arm/64/loopfilter.S) for 8-bit planes.
//
// One call filters every edge of one 4-pixel column (vertical edges) or one
// 4-pixel row (horizontal edges) of 4x4 units straight from the dav1d edge
// bitmasks: bit n of vm[k] marks unit n as an edge whose filter width class is
// k (luma 4/8/16, chroma 4/6). Levels come from the decode-time level cache with
// dav1d's "level from the previous unit" fallback. The SIMD build processes four
// units (sixteen samples) per kernel call with every filter width fused, as the
// NEON code does; the portable build filters unit by unit with the C reference.
//
// The caller guarantees the geometry: every unit selected by vm lies inside the
// plane's deblock extent and every tap the selected width touches is inside
// pix. Units are trusted, so no per-edge validation happens here.

// SBLUT holds the per-level interior and block limits for one sharpness value
// (dav1d Av1FilterLUT e/i). H, the high-edge-variance threshold, is level>>4.
type SBLUT struct {
	E [MaxLevel + 1]uint8
	I [MaxLevel + 1]uint8
}

// NewSBLUT derives the limit tables for a frame's loop-filter sharpness exactly
// as ThresholdsForLevel does per level.
func NewSBLUT(sharpness uint8) SBLUT {
	var lut SBLUT
	for level := 0; level <= MaxLevel; level++ {
		th, err := ThresholdsForLevel(uint8(level), sharpness)
		if err != nil {
			continue
		}
		lut.E[level] = th.BlockLimit
		lut.I[level] = th.Limit
	}
	return lut
}

// SBStats reports the edges a superblock call filtered: Applied counts every
// selected unit whose resolved level is non-zero and MaxLevel is the largest
// such level.
type SBStats struct {
	Applied  uint32
	MaxLevel uint8
}

// Add accumulates another call's counts.
func (s *SBStats) Add(o SBStats) {
	s.Applied += o.Applied
	if o.MaxLevel > s.MaxLevel {
		s.MaxLevel = o.MaxLevel
	}
}

// FilterSBVertical8 filters the vertical edges of one 4-pixel column of 4x4
// units in an 8-bit plane. pix[off] is the sample just right of the edge (q0)
// on the first unit's first row; unit n starts 4*n rows below. lvl[lvlOff +
// n*lvlStride][comp] is unit n's level; a zero level falls back to the unit on
// the left (lvl[... - 1]). vm[2] must be zero for chroma. rowBytes is how many
// bytes from off to the end of the row the call may read and rewrite.
func FilterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	if vm[0]|vm[1]|vm[2] == 0 {
		return SBStats{}
	}
	return filterSBVertical8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

// FilterSBHorizontal8 filters the horizontal edges of one 4-pixel row of 4x4
// units in an 8-bit plane. pix[off] is the sample just below the edge (q0) at
// the first unit's first column; unit n starts 4*n samples to the right.
// lvl[lvlOff + n][comp] is unit n's level; a zero level falls back to the unit
// above (lvl[... - lvlStride]). rowBytes is how many bytes from off to the end
// of the row the call may read and rewrite; whole sixteen-sample groups that
// would cross it are filtered unit by unit instead.
func FilterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	if vm[0]|vm[1]|vm[2] == 0 {
		return SBStats{}
	}
	return filterSBHorizontal8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

// sbUnitWidth returns unit n's filter width (4, 6, 8 or 16) from the masks.
func sbUnitWidth(vm *[3]uint32, luma bool, bit uint32) int {
	switch {
	case vm[2]&bit != 0:
		return 16
	case vm[1]&bit != 0:
		if luma {
			return 8
		}
		return 6
	default:
		return 4
	}
}

// filterSBVertical8Scalar is the unit-by-unit reference: dav1d
// loop_filter_h_sb128{y,uv}_c over loop_filter() with stridea = row stride.
func filterSBVertical8Scalar(pix []byte, off, stride int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	var st SBStats
	all := vm[0] | vm[1] | vm[2]
	for all != 0 {
		n := bits.TrailingZeros32(all)
		bit := uint32(1) << uint(n)
		all &^= bit
		li := lvlOff + n*lvlStride
		L := lvl[li][comp]
		if L == 0 {
			L = lvl[li-1][comp]
		}
		if L == 0 {
			continue
		}
		st.Applied++
		if L > st.MaxLevel {
			st.MaxLevel = L
		}
		lpfUnit8(pix, off+4*n*stride, stride, 1, int(lut.E[L]), int(lut.I[L]), int(L>>4), sbUnitWidth(&vm, luma, bit))
	}
	return st
}

// filterSBHorizontal8Scalar is the unit-by-unit reference for horizontal edges:
// dav1d loop_filter_v_sb128{y,uv}_c with stridea = 1, strideb = row stride.
func filterSBHorizontal8Scalar(pix []byte, off, stride int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	var st SBStats
	all := vm[0] | vm[1] | vm[2]
	for all != 0 {
		n := bits.TrailingZeros32(all)
		bit := uint32(1) << uint(n)
		all &^= bit
		li := lvlOff + n
		L := lvl[li][comp]
		if L == 0 {
			L = lvl[li-lvlStride][comp]
		}
		if L == 0 {
			continue
		}
		st.Applied++
		if L > st.MaxLevel {
			st.MaxLevel = L
		}
		lpfUnit8(pix, off+4*n, 1, stride, int(lut.E[L]), int(lut.I[L]), int(L>>4), sbUnitWidth(&vm, luma, bit))
	}
	return st
}

// lpfUnit8 is dav1d's loop_filter() for one 4-sample edge unit of an 8-bit
// plane: pix[off] is q0 of the first sample, stridea steps along the edge and
// strideb steps across it.
func lpfUnit8(pix []byte, off, stridea, strideb, E, I, H, wd int) {
	for i := 0; i < 4; i, off = i+1, off+stridea {
		p1 := int(pix[off-2*strideb])
		p0 := int(pix[off-strideb])
		q0 := int(pix[off])
		q1 := int(pix[off+strideb])
		fm := absInt(p1-p0) <= I && absInt(q1-q0) <= I && absInt(p0-q0)*2+(absInt(p1-q1)>>1) <= E
		var p2, q2, p3, q3 int
		if wd > 4 {
			p2 = int(pix[off-3*strideb])
			q2 = int(pix[off+2*strideb])
			fm = fm && absInt(p2-p1) <= I && absInt(q2-q1) <= I
			if wd > 6 {
				p3 = int(pix[off-4*strideb])
				q3 = int(pix[off+3*strideb])
				fm = fm && absInt(p3-p2) <= I && absInt(q3-q2) <= I
			}
		}
		if !fm {
			continue
		}
		flat8out := false
		var p6, p5, p4, q4, q5, q6 int
		if wd >= 16 {
			p6 = int(pix[off-7*strideb])
			p5 = int(pix[off-6*strideb])
			p4 = int(pix[off-5*strideb])
			q4 = int(pix[off+4*strideb])
			q5 = int(pix[off+5*strideb])
			q6 = int(pix[off+6*strideb])
			flat8out = absInt(p6-p0) <= 1 && absInt(p5-p0) <= 1 && absInt(p4-p0) <= 1 &&
				absInt(q4-q0) <= 1 && absInt(q5-q0) <= 1 && absInt(q6-q0) <= 1
		}
		flat8in := false
		if wd >= 6 {
			flat8in = absInt(p2-p0) <= 1 && absInt(p1-p0) <= 1 && absInt(q1-q0) <= 1 && absInt(q2-q0) <= 1
		}
		if wd >= 8 {
			flat8in = flat8in && absInt(p3-p0) <= 1 && absInt(q3-q0) <= 1
		}
		switch {
		case wd >= 16 && flat8out && flat8in:
			pix[off-6*strideb] = uint8((p6*7 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4)
			pix[off-5*strideb] = uint8((p6*5 + p5*2 + p4*2 + p3*2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4)
			pix[off-4*strideb] = uint8((p6*4 + p5 + p4*2 + p3*2 + p2*2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4)
			pix[off-3*strideb] = uint8((p6*3 + p5 + p4 + p3*2 + p2*2 + p1*2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4)
			pix[off-2*strideb] = uint8((p6*2 + p5 + p4 + p3 + p2*2 + p1*2 + p0*2 + q0 + q1 + q2 + q3 + q4 + 8) >> 4)
			pix[off-strideb] = uint8((p6 + p5 + p4 + p3 + p2 + p1*2 + p0*2 + q0*2 + q1 + q2 + q3 + q4 + q5 + 8) >> 4)
			pix[off] = uint8((p5 + p4 + p3 + p2 + p1 + p0*2 + q0*2 + q1*2 + q2 + q3 + q4 + q5 + q6 + 8) >> 4)
			pix[off+strideb] = uint8((p4 + p3 + p2 + p1 + p0 + q0*2 + q1*2 + q2*2 + q3 + q4 + q5 + q6*2 + 8) >> 4)
			pix[off+2*strideb] = uint8((p3 + p2 + p1 + p0 + q0 + q1*2 + q2*2 + q3*2 + q4 + q5 + q6*3 + 8) >> 4)
			pix[off+3*strideb] = uint8((p2 + p1 + p0 + q0 + q1 + q2*2 + q3*2 + q4*2 + q5 + q6*4 + 8) >> 4)
			pix[off+4*strideb] = uint8((p1 + p0 + q0 + q1 + q2 + q3*2 + q4*2 + q5*2 + q6*5 + 8) >> 4)
			pix[off+5*strideb] = uint8((p0 + q0 + q1 + q2 + q3 + q4*2 + q5*2 + q6*7 + 8) >> 4)
		case wd >= 8 && flat8in:
			pix[off-3*strideb] = uint8((p3*3 + p2*2 + p1 + p0 + q0 + 4) >> 3)
			pix[off-2*strideb] = uint8((p3*2 + p2 + p1*2 + p0 + q0 + q1 + 4) >> 3)
			pix[off-strideb] = uint8((p3 + p2 + p1 + p0*2 + q0 + q1 + q2 + 4) >> 3)
			pix[off] = uint8((p2 + p1 + p0 + q0*2 + q1 + q2 + q3 + 4) >> 3)
			pix[off+strideb] = uint8((p1 + p0 + q0 + q1*2 + q2 + q3*2 + 4) >> 3)
			pix[off+2*strideb] = uint8((p0 + q0 + q1 + q2*2 + q3*3 + 4) >> 3)
		case wd == 6 && flat8in:
			pix[off-2*strideb] = uint8((p2*3 + p1*2 + p0*2 + q0 + 4) >> 3)
			pix[off-strideb] = uint8((p2 + p1*2 + p0*2 + q0*2 + q1 + 4) >> 3)
			pix[off] = uint8((p1 + p0*2 + q0*2 + q1*2 + q2 + 4) >> 3)
			pix[off+strideb] = uint8((p0 + q0*2 + q1*2 + q2*3 + 4) >> 3)
		default:
			hev := absInt(p1-p0) > H || absInt(q1-q0) > H
			var f int
			if hev {
				f = clampDiff8(clampDiff8(p1-q1) + 3*(q0-p0))
			} else {
				f = clampDiff8(3 * (q0 - p0))
			}
			f1 := min(f+4, 127) >> 3
			f2 := min(f+3, 127) >> 3
			pix[off-strideb] = clampPixel8(p0 + f2)
			pix[off] = clampPixel8(q0 - f1)
			if !hev {
				f = (f1 + 1) >> 1
				pix[off-2*strideb] = clampPixel8(p1 + f)
				pix[off+strideb] = clampPixel8(q1 - f)
			}
		}
	}
}

func clampDiff8(v int) int {
	if v < -128 {
		return -128
	}
	if v > 127 {
		return 127
	}
	return v
}

func clampPixel8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// sbLUTs holds the limit tables for every legal sharpness value.
var sbLUTs = func() (t [MaxSharpness + 1]SBLUT) {
	for s := range t {
		t[s] = NewSBLUT(uint8(s))
	}
	return t
}()

// SBLUTFor returns the shared, read-only limit table for a sharpness value
// (clamped to MaxSharpness).
func SBLUTFor(sharpness uint8) *SBLUT {
	if sharpness > MaxSharpness {
		sharpness = MaxSharpness
	}
	return &sbLUTs[sharpness]
}
