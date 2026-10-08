// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"simd/archsimd"
	"unsafe"

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
func coeffNZMapContextsSIMD(levels []uint8, offsets []uint8, contexts []int8, scanWidth int, scanHeight int, stride int, nb [5]int) {
	lv := unsafe.Pointer(unsafe.SliceData(levels))
	off := unsafe.Pointer(unsafe.SliceData(offsets))
	out := unsafe.Pointer(unsafe.SliceData(contexts))
	n0, n1, n2, n3, n4 := uintptr(nb[0]), uintptr(nb[1]), uintptr(nb[2]), uintptr(nb[3]), uintptr(nb[4])
	nMax := max(nb[0], nb[1], nb[2], nb[3], nb[4])
	levelsLen := len(levels)
	three := archsimd.BroadcastUint8x16(3)
	four := archsimd.BroadcastUint8x16(4)
	zero := archsimd.BroadcastUint8x16(0)
	c := 0
	switch scanHeight {
	case 4:
		for ; c+4 <= scanWidth; c += 4 {
			var a, b archsimd.Uint8x16
			if (c+2)*stride+nMax+16 <= levelsLen {
				a = coeffNZMapCompress(coeffNZMapCtx16(unsafe.Add(lv, c*stride), n0, n1, n2, n3, n4, three, four, zero), &coeffNZMapPick4x2)
				b = coeffNZMapCompress(coeffNZMapCtx16(unsafe.Add(lv, (c+2)*stride), n0, n1, n2, n3, n4, three, four, zero), &coeffNZMapPick4x2)
			} else {
				var win [coeffNZMapWindowLen]uint8
				copy(win[:], levels[c*stride:])
				a = coeffNZMapCompress(coeffNZMapCtx16Win(&win, 0, nb, three, four, zero), &coeffNZMapPick4x2)
				b = coeffNZMapCompress(coeffNZMapCtx16Win(&win, 2*stride, nb, three, four, zero), &coeffNZMapPick4x2)
			}
			coeffNZMapStore16(out, c*4, coeffNZMapPack2(a, b).Add(coeffNZMapLoad16(off, c*4)))
		}
	case 8:
		for ; c+2 <= scanWidth; c += 2 {
			var a, b archsimd.Uint8x16
			if (c+1)*stride+nMax+16 <= levelsLen {
				a = coeffNZMapCompress(coeffNZMapCtx16(unsafe.Add(lv, c*stride), n0, n1, n2, n3, n4, three, four, zero), &coeffNZMapPickLo8)
				b = coeffNZMapCompress(coeffNZMapCtx16(unsafe.Add(lv, (c+1)*stride), n0, n1, n2, n3, n4, three, four, zero), &coeffNZMapPickLo8)
			} else {
				var win [coeffNZMapWindowLen]uint8
				copy(win[:], levels[c*stride:])
				a = coeffNZMapCompress(coeffNZMapCtx16Win(&win, 0, nb, three, four, zero), &coeffNZMapPickLo8)
				b = coeffNZMapCompress(coeffNZMapCtx16Win(&win, stride, nb, three, four, zero), &coeffNZMapPickLo8)
			}
			coeffNZMapStore16(out, c*8, coeffNZMapPack2(a, b).Add(coeffNZMapLoad16(off, c*8)))
		}
	case 16:
		for ; c < scanWidth && c*stride+nMax+16 <= levelsLen; c++ {
			ctx := coeffNZMapCtx16(unsafe.Add(lv, c*stride), n0, n1, n2, n3, n4, three, four, zero)
			coeffNZMapStore16(out, c*16, ctx.Add(coeffNZMapLoad16(off, c*16)))
		}
	case 32:
		for ; c < scanWidth && c*stride+nMax+32 <= levelsLen; c++ {
			p := unsafe.Add(lv, c*stride)
			lo := coeffNZMapCtx16(p, n0, n1, n2, n3, n4, three, four, zero)
			hi := coeffNZMapCtx16(unsafe.Add(p, 16), n0, n1, n2, n3, n4, three, four, zero)
			coeffNZMapStore16(out, c*32, lo.Add(coeffNZMapLoad16(off, c*32)))
			coeffNZMapStore16(out, c*32+16, hi.Add(coeffNZMapLoad16(off, c*32+16)))
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

// coeffNZMapClip3 loads sixteen levels at p+n and clamps each to 3.
func coeffNZMapClip3(p unsafe.Pointer, n uintptr, three archsimd.Uint8x16) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, n))).Min(three)
}

// coeffNZMapCtx16 returns min((sum of the five clipped neighbours + 1) >> 1, 4)
// for sixteen consecutive rows at p. Average with zero is the rounding half-sum
// (x+0+1)>>1, and the five clipped neighbours sum to at most 15 per lane.
func coeffNZMapCtx16(p unsafe.Pointer, n0, n1, n2, n3, n4 uintptr, three, four, zero archsimd.Uint8x16) archsimd.Uint8x16 {
	sum := coeffNZMapClip3(p, n0, three).Add(coeffNZMapClip3(p, n1, three))
	sum = sum.Add(coeffNZMapClip3(p, n2, three)).Add(coeffNZMapClip3(p, n3, three)).Add(coeffNZMapClip3(p, n4, three))
	return sum.Average(zero).Min(four)
}

// coeffNZMapCtx16Win is coeffNZMapCtx16 over a staged window. It stays in slice
// form (no unsafe pointers) so the window never escapes under -d=checkptr.
func coeffNZMapCtx16Win(w *[coeffNZMapWindowLen]uint8, base int, nb [5]int, three, four, zero archsimd.Uint8x16) archsimd.Uint8x16 {
	sum := coeffNZMapLoadClip3(w, base+nb[0], three).Add(coeffNZMapLoadClip3(w, base+nb[1], three))
	sum = sum.Add(coeffNZMapLoadClip3(w, base+nb[2], three)).Add(coeffNZMapLoadClip3(w, base+nb[3], three))
	sum = sum.Add(coeffNZMapLoadClip3(w, base+nb[4], three))
	return sum.Average(zero).Min(four)
}

func coeffNZMapLoadClip3(w *[coeffNZMapWindowLen]uint8, at int, three archsimd.Uint8x16) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(w[at : at+16])).Min(three)
}

// coeffNZMapPack2 joins two vectors whose live bytes are in 0..7 (upper eight
// bytes zero) into one vector: lo in bytes 0..7, hi in bytes 8..15.
func coeffNZMapPack2(lo, hi archsimd.Uint8x16) archsimd.Uint8x16 {
	return lo.Or(hi.ConcatShiftBytesRight(lo, 8))
}

func coeffNZMapLoad16(p unsafe.Pointer, off int) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, off)))
}

func coeffNZMapStore16(p unsafe.Pointer, off int, v archsimd.Uint8x16) {
	v.Store(unsafe.Slice((*uint8)(unsafe.Add(p, off)), 16))
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
	coeffNZMapContextsSIMD(levels, offsets, contexts, int(geo.scanWidth), int(geo.scanHeight), stride, nb)
	if class == transform.Class2D {
		contexts[0] = 0
	}
	return true
}

// coeffNZMapContextsArch is the Go SIMD twin of the scalar nz-map context pass,
// gated on coeffNZMapSIMDAvailable. It handles the full-eob case directly and
// the partial case through a full map followed by a scan-ordered copy.
func coeffNZMapContextsArch(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8, maxEOB int) bool {
	if !coeffNZMapSIMDAvailable() || !class.Valid() || eob <= 1 {
		return false
	}
	geo := coeffGeometryTable[size]
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
		return false
	}
	switch geo.scanHeight {
	case 4, 8, 16, 32:
		if eob == maxEOB {
			if !coeffNZMapContextsSIMDFull(levels, size, class, contexts) {
				return false
			}
			coeffNZMapFinalizeFullClass(contexts, scan, eob, maxEOB, class)
			return true
		}
		var full [maxCoeffScanLen]int8
		if !coeffNZMapContextsSIMDFull(levels, size, class, full[:]) {
			return false
		}
		coeffNZMapCopyPartialClass(full[:], contexts, scan, eob, maxEOB, class)
		return true
	default:
		return false
	}
}

func coeffNZMapContexts2DFullArch(levels []uint8, size TransformSize, contexts []int8) bool {
	if !coeffNZMapSIMDAvailable() {
		return false
	}
	return coeffNZMapContextsSIMDFull(levels, size, transform.Class2D, contexts)
}

func coeffNZMapFinalizeFullClass(contexts []int8, scan []int16, eob int, maxEOB int, class transform.Class) {
	if class == transform.Class2D {
		contexts[0] = 0
	}
	lastPos := int(scan[eob-1])
	contexts[lastPos] = int8(coeffLowerLevelsCtxEOBFast(maxEOB, eob-1))
}

func coeffNZMapCopyPartialClass(full []int8, contexts []int8, scan []int16, eob int, maxEOB int, class transform.Class) {
	if class == transform.Class2D {
		full[0] = 0
	}
	lastPos := int(scan[eob-1])
	full[lastPos] = int8(coeffLowerLevelsCtxEOBFast(maxEOB, eob-1))
	for c := range eob {
		pos := int(scan[c])
		contexts[pos] = full[pos]
	}
}
