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
