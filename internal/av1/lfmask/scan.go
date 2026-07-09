package lfmask

import "math/bits"

// This file ports the pure bit-scan at the heart of dav1d's deblock apply pass.
// dav1d consumes the edge bitmasks with zero neighbour or tree lookups: for one
// 4x4 column (vertical edges) or row (horizontal edges) position it combines the
// three strength lanes into a 32-bit run mask and walks the set bits in
// ascending 4x4 order, recovering each edge's filter-width class from which
// strength lane holds the bit.
//
// The scan here reproduces exactly:
//   - the lo/hi 16-bit lane assembly in filter_plane_cols_y / filter_plane_rows_y
//     (src/lf_apply_tmpl.c:176): hmask[s] = mask[s][0] | (mask[s][1] << 16),
//     with the hi lane included only when the SB-row reaches past row 16, and the
//     hi lane used alone for the bottom SB-row of a 64px superblock (starty4>0);
//   - the run walk and width selection in loop_filter_{h,v}_sb128y_c
//     (src/loopfilter_tmpl.c:169): for (y=1; vm & ~(y-1); y<<=1) with
//     idx = (mask[2] & y) ? 2 : !!(mask[1] & y).
//
// Level resolution (the l[0][0] ? l[0][0] : l[-1][0] "level-from-previous"
// fallback, the E/I/H lookup, and the actual filter dispatch) depends on the
// concrete decode-time level cache pointer layout and is the caller's job; this
// function yields only the geometry: which 4x4 edges are filtered, and at what
// width class.

// EdgeVisitor receives one deblock edge: offset4 is the edge's 4x4 offset from
// the SB-row window start (starty4); widthClass is 0/1/2 selecting luma filter
// widths 4/8/16 (4 << widthClass).
type EdgeVisitor func(offset4 int, widthClass int)

// ScanLuma walks the set edges of one luma 4x4 position (one entry of
// filter_y[dir][pos]) over the SB-row window [starty4, endy4) in ascending 4x4
// order, invoking visit for each. It performs the same lane assembly and run
// walk dav1d uses at deblock time. widthClass recovers the strength lane that
// owns each bit (the mask builder writes each edge into exactly one lane, so the
// highest set lane is that edge's class).
func ScanLuma(mask *[3][2]uint16, starty4, endy4 int, visit EdgeVisitor) {
	var m0, m1, m2 uint32
	if starty4 == 0 {
		m0 = uint32(mask[0][0])
		m1 = uint32(mask[1][0])
		m2 = uint32(mask[2][0])
		if endy4 > 16 {
			m0 |= uint32(mask[0][1]) << 16
			m1 |= uint32(mask[1][1]) << 16
			m2 |= uint32(mask[2][1]) << 16
		}
	} else {
		m0 = uint32(mask[0][1])
		m1 = uint32(mask[1][1])
		m2 = uint32(mask[2][1])
	}
	vm := m0 | m1 | m2
	// Walk set bits directly (ctz), skipping empty 4x4 positions, instead of
	// stepping through every position. Same offsets in the same ascending order.
	for vm != 0 {
		offset := bits.TrailingZeros32(vm)
		y := uint32(1) << uint(offset)
		widthClass := 0
		if m2&y != 0 {
			widthClass = 2
		} else if m1&y != 0 {
			widthClass = 1
		}
		visit(offset, widthClass)
		vm &= vm - 1
	}
}

// ScanChroma is the two-strength-lane counterpart for one chroma 4x4 position
// (one entry of filter_uv[dir][pos]). widthClass is 0/1 selecting chroma filter
// widths 4/6. laneBits is the lo/hi split (dav1d 16 >> ss for the scanned
// direction): the hi lane holds positions >= laneBits and is only reached when
// the window extends past it. laneStart is the window's 4x4 start position within
// the chroma region (starty4 >> ss for the scanned direction).
func ScanChroma(mask *[2][2]uint16, laneStart, laneEnd, laneBits int, visit EdgeVisitor) {
	var m0, m1 uint32
	if laneStart < laneBits {
		m0 = uint32(mask[0][0])
		m1 = uint32(mask[1][0])
		if laneEnd > laneBits {
			m0 |= uint32(mask[0][1]) << uint(laneBits)
			m1 |= uint32(mask[1][1]) << uint(laneBits)
		}
	} else {
		m0 = uint32(mask[0][1])
		m1 = uint32(mask[1][1])
	}
	vm := m0 | m1
	for vm != 0 {
		offset := bits.TrailingZeros32(vm)
		y := uint32(1) << uint(offset)
		widthClass := 0
		if m1&y != 0 {
			widthClass = 1
		}
		visit(offset, widthClass)
		vm &= vm - 1
	}
}
