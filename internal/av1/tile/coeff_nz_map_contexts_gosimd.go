// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coeffNZMapPick4x2 gathers the two 4-row columns of a 16-lane vector loaded at
// column c: lanes 0..3 (column c) and lanes 8..11 (column c+1, one padded stride
// later) land in bytes 0..7. Index 0x80 yields zero.
var coeffNZMapPick4x2 = [16]uint8{0, 1, 2, 3, 8, 9, 10, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}

// coeffNZMapPickLo8 keeps the eight live rows of a column and zeroes the rest.
var coeffNZMapPickLo8 = [16]uint8{0, 1, 2, 3, 4, 5, 6, 7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}

// coeffNZMapWindowLen covers the levels read by one 4-row quad or 8-row pair
// group: the second column's start (at most 2*stride <= 72) plus the largest
// neighbour offset (4*stride <= 48) plus a 16-byte load, with room to spare.
const coeffNZMapWindowLen = 96

// coeffNZMapContextsSIMD is the Go SIMD slice of libaom's
// av1_get_nz_map_contexts_c full-map pass, adapted to goav1's column-major
// levels scratch. For every dense coefficient index col*height+row it computes
//
//	contexts = min((sum(min(levels[base+n_k], 3)) + 1) >> 1, 4) + offsets
//
// over the five neighbour byte offsets nb (base = col*stride + row). Height 4
// and 8 groups use 16-byte loads that may read unused lanes past the live rows;
// a group whose loads would leave the level scratch is staged through a
// zero-padded window instead, so the kernel never reads past the buffers.
//
// The 2D class of the square 16- and 32-row maps (the encoder's lower-context
// maps) reads each column through one checked window at constant neighbour
// offsets and writes through windows of the whole map; the other shapes take
// one checked load per neighbour.
func coeffNZMapContextsSIMD(levels []uint8, offsets []uint8, contexts []int8, scanWidth int, scanHeight int, stride int, nb [5]int, class transform.Class) {
	nMax := max(nb[0], nb[1], nb[2], nb[3], nb[4])
	levelsLen := len(levels)
	three := archsimd.BroadcastUint8x16(3)
	four := archsimd.BroadcastUint8x16(4)
	zero := archsimd.BroadcastUint8x16(0)
	c := 0
	switch {
	case scanWidth == 16 && scanHeight == 16 && stride == 16+txPadHorizontal && class == transform.Class2D &&
		len(contexts) >= 256 && len(offsets) >= 256:
		const s = 16 + txPadHorizontal
		ctxs, offs := (*[256]int8)(contexts[:256]), (*[256]uint8)(offsets[:256])
		for o := 0; o <= 256-16; o += 16 {
			at := (o >> 4) * s
			if at > levelsLen-coeffNZMap2DWin16 {
				break
			}
			w := (*[coeffNZMap2DWin16]uint8)(levels[at : at+coeffNZMap2DWin16])
			ctx := coeffNZMap2D16(w, three, four, zero)
			coeffNZMapStoreWin(ctxs[o:o+16], offs[o:o+16], ctx)
			c++
		}
	case scanWidth == 32 && scanHeight == 32 && stride == 32+txPadHorizontal && class == transform.Class2D &&
		len(contexts) >= 1024 && len(offsets) >= 1024:
		const s = 32 + txPadHorizontal
		ctxs, offs := (*[1024]int8)(contexts[:1024]), (*[1024]uint8)(offsets[:1024])
		for o := 0; o <= 1024-32; o += 32 {
			at := (o >> 5) * s
			if at > levelsLen-coeffNZMap2DWin32 {
				break
			}
			w := (*[coeffNZMap2DWin32]uint8)(levels[at : at+coeffNZMap2DWin32])
			cw, ow := (*[32]int8)(ctxs[o:o+32]), (*[32]uint8)(offs[o:o+32])
			coeffNZMapStoreWin(cw[0:16], ow[0:16], coeffNZMap2D32(w, 0, three, four, zero))
			coeffNZMapStoreWin(cw[16:32], ow[16:32], coeffNZMap2D32(w, 16, three, four, zero))
			c++
		}
	case scanHeight == 4:
		for ; c+4 <= scanWidth; c += 4 {
			var a, b archsimd.Uint8x16
			if (c+2)*stride+nMax+16 <= levelsLen {
				a = coeffNZMapCompress(coeffNZMapCtx16(levels, c*stride, nb, three, four, zero), &coeffNZMapPick4x2)
				b = coeffNZMapCompress(coeffNZMapCtx16(levels, (c+2)*stride, nb, three, four, zero), &coeffNZMapPick4x2)
			} else {
				var win [coeffNZMapWindowLen]uint8
				copy(win[:], levels[c*stride:])
				a = coeffNZMapCompress(coeffNZMapCtx16(win[:], 0, nb, three, four, zero), &coeffNZMapPick4x2)
				b = coeffNZMapCompress(coeffNZMapCtx16(win[:], 2*stride, nb, three, four, zero), &coeffNZMapPick4x2)
			}
			coeffNZMapStore16(contexts, offsets, c*4, coeffNZMapPack2(a, b))
		}
	case scanHeight == 8:
		for ; c+2 <= scanWidth; c += 2 {
			var a, b archsimd.Uint8x16
			if (c+1)*stride+nMax+16 <= levelsLen {
				a = coeffNZMapCompress(coeffNZMapCtx16(levels, c*stride, nb, three, four, zero), &coeffNZMapPickLo8)
				b = coeffNZMapCompress(coeffNZMapCtx16(levels, (c+1)*stride, nb, three, four, zero), &coeffNZMapPickLo8)
			} else {
				var win [coeffNZMapWindowLen]uint8
				copy(win[:], levels[c*stride:])
				a = coeffNZMapCompress(coeffNZMapCtx16(win[:], 0, nb, three, four, zero), &coeffNZMapPickLo8)
				b = coeffNZMapCompress(coeffNZMapCtx16(win[:], stride, nb, three, four, zero), &coeffNZMapPickLo8)
			}
			coeffNZMapStore16(contexts, offsets, c*8, coeffNZMapPack2(a, b))
		}
	case scanHeight == 16:
		for ; c < scanWidth && c*stride+nMax+16 <= levelsLen; c++ {
			coeffNZMapStore16(contexts, offsets, c*16, coeffNZMapCtx16(levels, c*stride, nb, three, four, zero))
		}
	case scanHeight == 32:
		for ; c < scanWidth && c*stride+nMax+32 <= levelsLen; c++ {
			base := c * stride
			coeffNZMapStore16(contexts, offsets, c*32, coeffNZMapCtx16(levels, base, nb, three, four, zero))
			coeffNZMapStore16(contexts, offsets, c*32+16, coeffNZMapCtx16(levels, base+16, nb, three, four, zero))
		}
	}
	for ; c < scanWidth; c++ {
		for row := range scanHeight {
			pos := c*scanHeight + row
			base := c*stride + row
			mag := int(min(levels[base+nb[0]], 3)) + int(min(levels[base+nb[1]], 3)) +
				int(min(levels[base+nb[2]], 3)) + int(min(levels[base+nb[3]], 3)) +
				int(min(levels[base+nb[4]], 3))
			contexts[pos] = int8(min((mag+1)>>1, 4) + int(offsets[pos]))
		}
	}
}

// The 2D neighbour offsets {1, s, s+1, 2s, 2} of the 16- and 32-row maps
// (s = height+TX_PAD_HOR) reach at most 2s past a column's first level; the
// windows add the rows one load group covers.
const (
	coeffNZMap2DWin16 = 2*(16+txPadHorizontal) + 16
	coeffNZMap2DWin32 = 2*(32+txPadHorizontal) + 32
)

// coeffNZMapClip3 loads sixteen levels and clamps each to 3.
func coeffNZMapClip3(p *[16]uint8, three archsimd.Uint8x16) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array(p).Min(three)
}

// coeffNZMapCtxSum returns min((sum of the five clipped neighbours + 1) >> 1,
// 4). Average with zero is the rounding half-sum (x+0+1)>>1, and the five
// clipped neighbours sum to at most 15 per lane.
func coeffNZMapCtxSum(a, b, c, d, e, four, zero archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Add(b).Add(c).Add(d).Add(e).Average(zero).Min(four)
}

// coeffNZMap2D16 is the 2D context of a 16-row map column window.
func coeffNZMap2D16(w *[coeffNZMap2DWin16]uint8, three, four, zero archsimd.Uint8x16) archsimd.Uint8x16 {
	const s = 16 + txPadHorizontal
	return coeffNZMapCtxSum(
		coeffNZMapClip3((*[16]uint8)(w[1:17]), three),
		coeffNZMapClip3((*[16]uint8)(w[s:s+16]), three),
		coeffNZMapClip3((*[16]uint8)(w[s+1:s+17]), three),
		coeffNZMapClip3((*[16]uint8)(w[2*s:2*s+16]), three),
		coeffNZMapClip3((*[16]uint8)(w[2:18]), three),
		four, zero)
}

// coeffNZMap2D32 is the 2D context of the sixteen rows at w[r:] (r is 0 or
// 16) of a 32-row map column window.
func coeffNZMap2D32(w *[coeffNZMap2DWin32]uint8, r int, three, four, zero archsimd.Uint8x16) archsimd.Uint8x16 {
	const s = 32 + txPadHorizontal
	r &= 16
	return coeffNZMapCtxSum(
		coeffNZMapClip3((*[16]uint8)(w[r+1:r+17]), three),
		coeffNZMapClip3((*[16]uint8)(w[r+s:r+s+16]), three),
		coeffNZMapClip3((*[16]uint8)(w[r+s+1:r+s+17]), three),
		coeffNZMapClip3((*[16]uint8)(w[r+2*s:r+2*s+16]), three),
		coeffNZMapClip3((*[16]uint8)(w[r+2:r+18]), three),
		four, zero)
}

// coeffNZMapCtx16 returns the context of sixteen consecutive rows at
// levels[base:], one checked load per neighbour.
func coeffNZMapCtx16(levels []uint8, base int, nb [5]int, three, four, zero archsimd.Uint8x16) archsimd.Uint8x16 {
	return coeffNZMapCtxSum(
		coeffNZMapClip3((*[16]uint8)(levels[base+nb[0]:base+nb[0]+16]), three),
		coeffNZMapClip3((*[16]uint8)(levels[base+nb[1]:base+nb[1]+16]), three),
		coeffNZMapClip3((*[16]uint8)(levels[base+nb[2]:base+nb[2]+16]), three),
		coeffNZMapClip3((*[16]uint8)(levels[base+nb[3]:base+nb[3]+16]), three),
		coeffNZMapClip3((*[16]uint8)(levels[base+nb[4]:base+nb[4]+16]), three),
		four, zero)
}

// coeffNZMapPack2 joins two vectors whose live bytes are in 0..7 (upper eight
// bytes zero) into one vector: lo in bytes 0..7, hi in bytes 8..15.
func coeffNZMapPack2(lo, hi archsimd.Uint8x16) archsimd.Uint8x16 {
	return lo.Or(hi.ConcatShiftBytesRight(lo, 8))
}

// coeffNZMapStoreWin adds the sixteen position offsets o to ctx and stores
// the contexts at c.
func coeffNZMapStoreWin(c []int8, o []uint8, ctx archsimd.Uint8x16) {
	ctx.Add(archsimd.LoadUint8x16Array((*[16]uint8)(o))).BitsToInt8().StoreArray((*[16]int8)(c))
}

// coeffNZMapStore16 adds the sixteen position offsets at off to ctx and
// stores the contexts there.
func coeffNZMapStore16(contexts []int8, offsets []uint8, off int, ctx archsimd.Uint8x16) {
	o := archsimd.LoadUint8x16Array((*[16]uint8)(offsets[off : off+16]))
	ctx.Add(o).BitsToInt8().StoreArray((*[16]int8)(contexts[off : off+16]))
}

// coeffNZMapContextsSIMDFull fills every dense context of size/class in
// contexts (len >= maxEOB) with the Go SIMD full-map pass. It returns false
// for unsupported geometry or class and leaves contexts untouched in that case.
func coeffNZMapContextsSIMDFull(levels []uint8, size TransformSize, class transform.Class, contexts []int8) bool {
	geo := coeffGeometryTable[size]
	maxEOB := int(geo.maxEOB)
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
		return false
	}
	switch geo.scanHeight {
	case 4, 8, 16, 32:
	default:
		return false
	}
	var offsets []uint8
	switch class {
	case transform.Class2D:
		offsets = coeffLower2DOffsetTable[size]
	case transform.ClassHoriz:
		offsets = coeffLowerHorizOffsetTable[size]
	case transform.ClassVert:
		offsets = coeffLowerVertOffsetTable[size]
	default:
		return false
	}
	if len(offsets) < maxEOB {
		return false
	}
	stride := int(geo.stride)
	var nb [5]int
	switch class {
	case transform.Class2D:
		nb = [5]int{1, stride, stride + 1, stride << 1, 2}
	case transform.ClassHoriz:
		nb = [5]int{1, stride, stride << 1, 3 * stride, 4 * stride}
	case transform.ClassVert:
		nb = [5]int{1, stride, 2, 3, 4}
	}
	coeffNZMapContextsSIMD(levels, offsets, contexts, int(geo.scanWidth), int(geo.scanHeight), stride, nb, class)
	if class == transform.Class2D {
		contexts[0] = 0
	}
	return true
}

func coeffNZMapContexts2DFullArch(levels []uint8, size TransformSize, contexts []int8) bool {
	if !coeffNZMapSIMDAvailable() {
		return false
	}
	return coeffNZMapContextsSIMDFull(levels, size, transform.Class2D, contexts)
}
