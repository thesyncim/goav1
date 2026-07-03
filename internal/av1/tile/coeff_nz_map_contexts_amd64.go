// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

package tile

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coeffNZMapAVX2Ctx is the asm calling context. Field order and sizes are part
// of the ABI shared with coeff_nz_map_contexts_avx2_amd64.s; do not reorder.
// n0..n4 are the five neighbour byte offsets (a + b*stride) for the transform
// class; rowBytes is the column height (4, 8, 16 or 32).
type coeffNZMapAVX2Ctx struct {
	levels   *uint8
	offsets  *uint8
	contexts *int8
	columns  uintptr
	stride   uintptr
	n0       uintptr
	n1       uintptr
	n2       uintptr
	n3       uintptr
	n4       uintptr
	rowBytes uintptr
}

//go:noescape
func coeffNZMapContextsAVX2Asm(ctx *coeffNZMapAVX2Ctx)

// coeffNZMapContextsAVX2Full runs the AVX2 full-map kernel unconditionally (no
// CPUID gate) so the differential test can exercise the assembly on every amd64
// host, including Rosetta where CPUID hides AVX2 from the dispatch gates below.
// It returns false when the geometry or class is unsupported.
func coeffNZMapContextsAVX2Full(levels []uint8, size TransformSize, class transform.Class, contexts []int8) bool {
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
	stride := uintptr(geo.stride)
	ctx := coeffNZMapAVX2Ctx{
		levels:   &levels[0],
		offsets:  &offsets[0],
		contexts: &contexts[0],
		columns:  uintptr(geo.scanWidth),
		stride:   stride,
		n0:       1,
		n1:       stride,
		rowBytes: uintptr(geo.scanHeight),
	}
	switch class {
	case transform.Class2D:
		ctx.n2 = stride + 1
		ctx.n3 = stride << 1
		ctx.n4 = 2
	case transform.ClassHoriz:
		ctx.n2 = stride << 1
		ctx.n3 = 3 * stride
		ctx.n4 = 4 * stride
	case transform.ClassVert:
		ctx.n2 = 2
		ctx.n3 = 3
		ctx.n4 = 4
	}
	coeffNZMapContextsAVX2Asm(&ctx)
	if class == transform.Class2D {
		contexts[0] = 0
	}
	return true
}

// coeffNZMapContextsFullArch is the amd64 AVX2 slice of libaom's
// av1_get_nz_map_contexts_c full-map pass, gated on OS-enabled AVX2.
func coeffNZMapContextsFullArch(levels []uint8, size TransformSize, class transform.Class, contexts []int8) bool {
	if !cpu.Detected.AVX2 {
		return false
	}
	return coeffNZMapContextsAVX2Full(levels, size, class, contexts)
}

func coeffNZMapContexts2DFullArch(levels []uint8, size TransformSize, contexts []int8) bool {
	if !cpu.Detected.AVX2 {
		return false
	}
	return coeffNZMapContextsAVX2Full(levels, size, transform.Class2D, contexts)
}

func coeffNZMapContextsArch(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8, maxEOB int) bool {
	if !cpu.Detected.AVX2 || !class.Valid() || eob <= 1 {
		return false
	}
	geo := coeffGeometryTable[size]
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
		return false
	}
	switch geo.scanHeight {
	case 4, 8, 16, 32:
		if eob == maxEOB {
			if !coeffNZMapContextsAVX2Full(levels, size, class, contexts) {
				return false
			}
			coeffNZMapFinalizeFullClass(contexts, scan, eob, maxEOB, class)
			return true
		}
		var full [maxCoeffScanLen]int8
		if !coeffNZMapContextsAVX2Full(levels, size, class, full[:]) {
			return false
		}
		coeffNZMapCopyPartialClass(full[:], contexts, scan, eob, maxEOB, class)
		return true
	default:
		return false
	}
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
