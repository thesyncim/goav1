// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

// 8-bit directional zone 1 and zone 3 kernels for amd64 in the form of
// dav1d's ipred_z1 / ipred_z3 (ipred_avx2.asm): the edge is copied into a
// padded buffer whose entries at and past max_base replicate edge[max_base],
// so the per-sample clamp of the scalar code becomes an ordinary two-tap
// interpolation (p*(32-s) + p*s + 16) >> 5 == p, and an upsampled edge is
// split into its even and odd samples so both taps are contiguous loads.
// They take the shapes the existing row kernels leave scalar: width-4
// blocks and upsampled edges.

// dirPadLen covers max_base (at most 2*(16+16-1) for the 16x16 limit of
// these kernels) plus two block extents and a vector.
const dirPadLen = 128

// dirPadEdge is a padded, optionally even/odd-split copy of one directional
// edge.
type dirPadEdge struct {
	pad  [dirPadLen]uint16
	even [dirPadLen / 2]uint16
	odd  [dirPadLen / 2]uint16
}

// load copies edge[0..maxBase] and replicates edge[maxBase] up to n
// entries; with up set it also splits the copy into even and odd samples.
func (d *dirPadEdge) load(edge []uint16, maxBase, n int, up bool) {
	copy(d.pad[:maxBase+1], edge[:maxBase+1])
	last := edge[maxBase]
	for i := maxBase + 1; i < n; i++ {
		d.pad[i] = last
	}
	if up {
		for i := 0; i < n/2; i++ {
			d.even[i] = d.pad[2*i]
			d.odd[i] = d.pad[2*i+1]
		}
	}
}

// taps returns the two interpolation taps of eight consecutive outputs
// starting at edge index base, stepping by 1 (up false) or 2 (up true).
func (d *dirPadEdge) taps(base int, up bool) (p0, p1 archsimd.Uint16x8) {
	if !up {
		return archsimd.LoadUint16x8(d.pad[base:]), archsimd.LoadUint16x8(d.pad[base+1:])
	}
	m := base >> 1
	if base&1 == 0 {
		return archsimd.LoadUint16x8(d.even[m:]), archsimd.LoadUint16x8(d.odd[m:])
	}
	return archsimd.LoadUint16x8(d.odd[m:]), archsimd.LoadUint16x8(d.even[m+1:])
}

// dirInterp is (p0*(32-s) + p1*s + 16) >> 5 on eight lanes.
func dirInterp(p0, p1 archsimd.Uint16x8, s int) archsimd.Uint16x8 {
	w1 := archsimd.BroadcastUint16x8(uint16(s))
	w0 := archsimd.BroadcastUint16x8(uint16(32 - s))
	return p0.Mul(w0).Add(p1.Mul(w1)).Add(archsimd.BroadcastUint16x8(16)).ShiftAllRight(5)
}

// predictDirectionalZ1Arch is the 8-bit zone-1 predictor for width-4 blocks
// and upsampled edges; it reports false for the shapes it leaves to the
// scalar and row-kernel paths. above is the edge from above[0], validated
// through above[maxBaseX].
func predictDirectionalZ1Arch(block planeBlock, above []uint16, upsampleAbove int, dx int, maxBaseX int) bool {
	w, h := block.width, block.height
	if !intraEdgeAVX2 || (upsampleAbove == 0 && w != 4) || w&3 != 0 || w > 16 || h > 16 ||
		maxBaseX+2*w+24 > dirPadLen || len(above) <= maxBaseX {
		return false
	}
	up := upsampleAbove == 1
	var d dirPadEdge
	d.load(above, maxBaseX, maxBaseX+2*w+24, up)
	fracBits := uint(6 - upsampleAbove)
	x := dx
	for row := 0; row < h; row++ {
		base := x >> fracBits
		line := block.pix[row*block.stride : row*block.stride+w]
		if base >= maxBaseX {
			fill := byte(d.pad[maxBaseX])
			for r := row; r < h; r++ {
				rl := block.pix[r*block.stride : r*block.stride+w]
				for c := range rl {
					rl[c] = fill
				}
			}
			return true
		}
		s := ((x << upsampleAbove) & 0x3f) >> 1
		for col := 0; col < w; col += 8 {
			p0, p1 := d.taps(base+(col<<upsampleAbove), up)
			v := narrowU16ToU8(dirInterp(p0, p1, s)).ReshapeToUint64s().GetElem(0)
			if w-col >= 8 {
				binary.LittleEndian.PutUint64(line[col:], v)
			} else {
				binary.LittleEndian.PutUint32(line[col:], uint32(v))
			}
		}
		x += dx
	}
	return true
}

// predictDirectionalZ3Arch is the 8-bit zone-3 predictor for height-4
// blocks and upsampled edges, with predictDirectionalZ1Arch's contract on
// the left edge. Each column is interpolated with its rows in the lanes and
// transposed into the block.
func predictDirectionalZ3Arch(block planeBlock, left []uint16, upsampleLeft int, dy int, maxBaseY int) bool {
	w, h := block.width, block.height
	if !intraEdgeAVX2 || (upsampleLeft == 0 && h != 4) || h&3 != 0 || w > 16 || h > 16 ||
		maxBaseY+2*h+24 > dirPadLen || len(left) <= maxBaseY {
		return false
	}
	up := upsampleLeft == 1
	var d dirPadEdge
	d.load(left, maxBaseY, maxBaseY+2*h+24, up)
	fracBits := uint(6 - upsampleLeft)
	fill := byte(d.pad[maxBaseY])
	// t holds the block transposed: t[col*16+row].
	var t [16 * 16]byte
	y := dy
	for col := 0; col < w; col++ {
		base := y >> fracBits
		tc := t[col*16 : col*16+16]
		if base >= maxBaseY {
			for r := 0; r < h; r++ {
				tc[r] = fill
			}
		} else {
			s := ((y << upsampleLeft) & 0x3f) >> 1
			for row := 0; row < h; row += 8 {
				p0, p1 := d.taps(base+(row<<upsampleLeft), up)
				binary.LittleEndian.PutUint64(tc[row:], narrowU16ToU8(dirInterp(p0, p1, s)).ReshapeToUint64s().GetElem(0))
			}
		}
		y += dy
	}
	if w == 4 && h == 4 {
		var c [4]uint32
		for i := range c {
			c[i] = binary.LittleEndian.Uint32(t[i*16:])
		}
		for r := 0; r < 4; r++ {
			sh := 8 * uint(r)
			v := (c[0]>>sh)&0xff | ((c[1]>>sh)&0xff)<<8 | ((c[2]>>sh)&0xff)<<16 | ((c[3]>>sh)&0xff)<<24
			binary.LittleEndian.PutUint32(block.pix[r*block.stride:], v)
		}
		return true
	}
	for r := 0; r < h; r++ {
		line := block.pix[r*block.stride : r*block.stride+w]
		for c := range line {
			line[c] = t[c*16+r]
		}
	}
	return true
}

// dirTaps returns the two interpolation taps of eight consecutive outputs
// read straight from edge at index start, stepping by 1 (up false) or 2 (up
// true). The caller has checked dirTapsFit; lanes past its validated span
// read whatever the edge holds there and are discarded.
func dirTaps(edge []uint16, start int, up bool) (p0, p1 archsimd.Uint16x8) {
	if !up {
		return archsimd.LoadUint16x8(edge[start:]), archsimd.LoadUint16x8(edge[start+1:])
	}
	a := archsimd.LoadUint16x8(edge[start:]).AsUint32x4()
	b := archsimd.LoadUint16x8(edge[start+8:]).AsUint32x4()
	m := archsimd.BroadcastUint32x4(0xffff)
	p0 = a.And(m).AsInt32x4().SaturateToUint16Concat(b.And(m).AsInt32x4())
	p1 = a.ShiftAllRight(16).AsInt32x4().SaturateToUint16Concat(b.ShiftAllRight(16).AsInt32x4())
	return p0, p1
}

// dirTapsFit reports whether dirTaps may read from first through last
// (chunk starts) in an edge of length n.
func dirTapsFit(first, last, n int, up bool) bool {
	if up {
		return first >= 0 && last+16 <= n
	}
	return first >= 0 && last+9 <= n
}

// predictDirectionalZ2Arch is the 8-bit zone-2 predictor for blocks up to
// 16x16 (dav1d's ipred_z2 form). Row r takes the left edge in columns
// 0..c0(r)-1, where (c<<6) - (r+1)*dx < -64, and the above edge after; the
// split is monotone in r. The left part is interpolated column by column
// (rows in the lanes, one shift per column) into a transposed buffer, the
// above part row by row (one shift per row), and each row is assembled from
// the two. It reports false, before writing anything, when a tap would
// leave the edge slices.
func predictDirectionalZ2Arch(block planeBlock, edges DirectionalEdges, upsampleAbove, upsampleLeft, dx, dy int) bool {
	w, h := block.width, block.height
	if !intraEdgeAVX2 || w&3 != 0 || h&3 != 0 || w > 16 || h > 16 {
		return false
	}
	fracX, fracY := uint(6-upsampleAbove), uint(6-upsampleLeft)
	upA, upL := upsampleAbove == 1, upsampleLeft == 1
	// Columns 0..cLeft-1 take the left edge in some row (the last row has
	// the most). Their chunk starts fall as c grows and rise with r.
	cLeft := min((h*dx-1)>>6, w)
	lastRowChunk := ((h - 1) &^ 7) << upsampleLeft
	if cLeft > 0 && !dirTapsFit(edges.LeftOrigin+((-cLeft*dy)>>fracY), edges.LeftOrigin+((-dy)>>fracY)+lastRowChunk, len(edges.Left), upL) {
		return false
	}
	// Above chunk starts of each row rise with the column and fall with r.
	for r := 0; r < h; r++ {
		ydx := (r + 1) * dx
		c0 := min((ydx-1)>>6, w)
		if c0 >= w {
			break
		}
		base := edges.AboveOrigin + (((c0 << 6) - ydx) >> fracX)
		if !dirTapsFit(base, base+(((w-1-c0)&^7)<<upsampleAbove), len(edges.Above), upA) {
			return false
		}
	}
	var t [16*16 + 8]byte
	for c := 0; c < cLeft; c++ {
		yy := -(c + 1) * dy
		base := edges.LeftOrigin + (yy >> fracY)
		s := ((yy << upsampleLeft) & 0x3f) >> 1
		for r := 0; r < h; r += 8 {
			p0, p1 := dirTaps(edges.Left, base+(r<<upsampleLeft), upL)
			binary.LittleEndian.PutUint64(t[c*16+r:], narrowU16ToU8(dirInterp(p0, p1, s)).ReshapeToUint64s().GetElem(0))
		}
	}
	for r := 0; r < h; r++ {
		ydx := (r + 1) * dx
		c0 := min((ydx-1)>>6, w)
		var row [16 + 8]byte
		for c := 0; c < c0; c++ {
			row[c] = t[c*16+r]
		}
		if c0 < w {
			x := (c0 << 6) - ydx
			base := edges.AboveOrigin + (x >> fracX)
			s := ((x << upsampleAbove) & 0x3f) >> 1
			for c := c0; c < w; c += 8 {
				p0, p1 := dirTaps(edges.Above, base+((c-c0)<<upsampleAbove), upA)
				binary.LittleEndian.PutUint64(row[c:], narrowU16ToU8(dirInterp(p0, p1, s)).ReshapeToUint64s().GetElem(0))
			}
		}
		copy(block.pix[r*block.stride:r*block.stride+w], row[:w])
	}
	return true
}
