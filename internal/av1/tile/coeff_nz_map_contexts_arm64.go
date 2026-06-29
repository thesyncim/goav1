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
func coeffNZMapContexts16Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

//go:noescape
func coeffNZMapContexts32Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)

// coeffNZMapContextsArch is the arm64 Class2D slice of SVT-AV1's
// svt_av1_get_nz_map_contexts_neon. The NEON kernel computes the same
// five-neighbour min(level, 3) rounded half-sum used by SVT, adapted to
// goav1's column-major levels[col*(height+TX_PAD_HOR)+row] scratch ABI.
func coeffNZMapContextsArch(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8, maxEOB int) bool {
	if !cpu.Detected.NEON || class != transform.Class2D || eob <= 1 {
		return false
	}
	geo := coeffGeometryTable[size]
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
		return false
	}
	offsets := coeffLower2DOffsetTable[size]
	if len(offsets) < maxEOB {
		return false
	}
	switch geo.scanHeight {
	case 16:
		if eob == maxEOB {
			coeffNZMapContexts16Rows2DNEONAsm(&levels[0], &offsets[0], &contexts[0], uintptr(geo.scanWidth), uintptr(geo.stride))
			coeffNZMapFinalizeFull(contexts, scan, eob, maxEOB)
			return true
		}
		var full [maxCoeffScanLen]int8
		coeffNZMapContexts16Rows2DNEONAsm(&levels[0], &offsets[0], &full[0], uintptr(geo.scanWidth), uintptr(geo.stride))
		coeffNZMapCopyPartial(full[:], contexts, scan, eob, maxEOB)
		return true
	case 32:
		if eob == maxEOB {
			coeffNZMapContexts32Rows2DNEONAsm(&levels[0], &offsets[0], &contexts[0], uintptr(geo.scanWidth), uintptr(geo.stride))
			coeffNZMapFinalizeFull(contexts, scan, eob, maxEOB)
			return true
		}
		var full [maxCoeffScanLen]int8
		coeffNZMapContexts32Rows2DNEONAsm(&levels[0], &offsets[0], &full[0], uintptr(geo.scanWidth), uintptr(geo.stride))
		coeffNZMapCopyPartial(full[:], contexts, scan, eob, maxEOB)
		return true
	default:
		return false
	}
}

func coeffNZMapFinalizeFull(contexts []int8, scan []int16, eob int, maxEOB int) {
	contexts[0] = 0
	lastPos := int(scan[eob-1])
	contexts[lastPos] = int8(coeffLowerLevelsCtxEOBFast(maxEOB, eob-1))
}

func coeffNZMapCopyPartial(full []int8, contexts []int8, scan []int16, eob int, maxEOB int) {
	full[0] = 0
	lastPos := int(scan[eob-1])
	full[lastPos] = int8(coeffLowerLevelsCtxEOBFast(maxEOB, eob-1))
	for c := range eob {
		pos := int(scan[c])
		contexts[pos] = full[pos]
	}
}
