// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package tile

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

//go:noescape
func coeffNZMapContexts4Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts8Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts16Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts32Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts4RowsHorizNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts8RowsHorizNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts16RowsHorizNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts32RowsHorizNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts4RowsVertNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts8RowsVertNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts16RowsVertNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts32RowsVertNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

// coeffNZMapContextsArch is the arm64 slice of SVT-AV1's
// svt_av1_get_nz_map_contexts_neon. The NEON kernels compute the same
// five-neighbour min(level, 3) rounded half-sum used by SVT, adapted to
// goav1's column-major levels[col*(height+TX_PAD_HOR)+row] scratch ABI.
func coeffNZMapContextsArch(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8, maxEOB int) bool {
	if !cpu.Detected.NEON || !class.Valid() || eob <= 1 {
		return false
	}
	geo := coeffGeometryTable[size]
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
		return false
	}
	switch geo.scanHeight {
	case 4, 8, 16, 32:
		if eob == maxEOB {
			if !coeffNZMapContextsFullArch(levels, size, class, contexts) {
				return false
			}
			coeffNZMapFinalizeFullClass(contexts, scan, eob, maxEOB, class)
			return true
		}
		var full [maxCoeffScanLen]int8
		if !coeffNZMapContextsFullArch(levels, size, class, full[:]) {
			return false
		}
		coeffNZMapCopyPartialClass(full[:], contexts, scan, eob, maxEOB, class)
		return true
	default:
		return false
	}
}

func coeffNZMapContexts2DFullArch(levels []uint8, size TransformSize, contexts []int8) bool {
	return coeffNZMapContextsFullArch(levels, size, transform.Class2D, contexts)
}

func coeffNZMapContextsFullArch(levels []uint8, size TransformSize, class transform.Class, contexts []int8) bool {
	if !cpu.Detected.NEON {
		return false
	}
	geo := coeffGeometryTable[size]
	maxEOB := int(geo.maxEOB)
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
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
	columns := uintptr(geo.scanWidth)
	stride := uintptr(geo.stride)
	switch class {
	case transform.Class2D:
		switch geo.scanHeight {
		case 4:
			coeffNZMapContexts4Rows2DNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 8:
			coeffNZMapContexts8Rows2DNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 16:
			coeffNZMapContexts16Rows2DNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 32:
			coeffNZMapContexts32Rows2DNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		default:
			return false
		}
		contexts[0] = 0
	case transform.ClassHoriz:
		switch geo.scanHeight {
		case 4:
			coeffNZMapContexts4RowsHorizNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 8:
			coeffNZMapContexts8RowsHorizNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 16:
			coeffNZMapContexts16RowsHorizNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 32:
			coeffNZMapContexts32RowsHorizNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		default:
			return false
		}
	case transform.ClassVert:
		switch geo.scanHeight {
		case 4:
			coeffNZMapContexts4RowsVertNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 8:
			coeffNZMapContexts8RowsVertNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 16:
			coeffNZMapContexts16RowsVertNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		case 32:
			coeffNZMapContexts32RowsVertNEONAsm(&levels[0], &offsets[0], &contexts[0], columns, stride)
		default:
			return false
		}
	default:
		return false
	}
	return true
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
