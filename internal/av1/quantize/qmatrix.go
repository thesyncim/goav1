// Ported from libaom: av1/common/quant_common.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package quantize

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	QMLevelBits = 4
	QMLevels    = 1 << QMLevelBits

	DefaultQMFirst         = 5
	DefaultQMLast          = 9
	DefaultQMFirstAllIntra = 4
	DefaultQMLastAllIntra  = 10

	qmBits = 5
)

// QMLevel maps qindex to the inter quantization-matrix level range. This is
// the same increasing formula used by libaom's aom_get_qmlevel().
func QMLevel(qIndex int, first int, last int) int {
	return first + (qIndex*(last+1-first))/QIndexRange
}

// QMLevelAllIntra maps qindex to the all-intra quantization-matrix level range.
func QMLevelAllIntra(qIndex int, first int, last int) int {
	level := 4
	switch {
	case qIndex <= 40:
		level = 10
	case qIndex <= 100:
		level = 9
	case qIndex <= 160:
		level = 8
	case qIndex <= 200:
		level = 7
	case qIndex <= 220:
		level = 6
	case qIndex <= 240:
		level = 5
	}
	return clampInt(level, first, last)
}

func clampInt(v int, lo int, hi int) int {
	return min(max(v, lo), hi)
}

// InverseQMatrix returns libaom's inverse quantization matrix for a decoded
// transform block, or nil when AV1 uses the flat matrix path.
func InverseQMatrix(params parser.QuantizationParams, plane Plane, size transform.Size, typ transform.Type, lossless bool) ([]uint16, error) {
	if !params.UsingQMatrix || lossless || !usesInverseQMatrixTransform(typ) {
		return nil, nil
	}
	level, ok := qmatrixLevelForPlane(params, plane)
	if !ok {
		return nil, ErrInvalidQuantizer
	}
	if level >= QMLevels {
		return nil, ErrInvalidQuantizer
	}
	if level == QMLevels-1 {
		return nil, nil
	}
	offset, length, err := inverseQMatrixOffset(size)
	if err != nil {
		return nil, err
	}
	class := 0
	if plane != PlaneY {
		class = 1
	}
	return inverseQMatrixRef[level][class][offset : offset+length], nil
}

func usesInverseQMatrixTransform(typ transform.Type) bool {
	return typ < transform.TypeIDTX
}

func qmatrixLevelForPlane(params parser.QuantizationParams, plane Plane) (uint8, bool) {
	switch plane {
	case PlaneY:
		return params.QMatrixLevelY, true
	case PlaneU:
		return params.QMatrixLevelU, true
	case PlaneV:
		return params.QMatrixLevelV, true
	default:
		return 0, false
	}
}

func inverseQMatrixOffset(size transform.Size) (int, int, error) {
	scanSize, err := transform.ScanSize(size)
	if err != nil {
		return 0, 0, ErrInvalidQuantizer
	}
	switch scanSize {
	case transform.Size{Width: 4, Height: 4}:
		return 0, 16, nil
	case transform.Size{Width: 8, Height: 8}:
		return 16, 64, nil
	case transform.Size{Width: 16, Height: 16}:
		return 80, 256, nil
	case transform.Size{Width: 32, Height: 32}:
		return 336, 1024, nil
	case transform.Size{Width: 4, Height: 8}:
		return 1360, 32, nil
	case transform.Size{Width: 8, Height: 4}:
		return 1392, 32, nil
	case transform.Size{Width: 8, Height: 16}:
		return 1424, 128, nil
	case transform.Size{Width: 16, Height: 8}:
		return 1552, 128, nil
	case transform.Size{Width: 16, Height: 32}:
		return 1680, 512, nil
	case transform.Size{Width: 32, Height: 16}:
		return 2192, 512, nil
	case transform.Size{Width: 4, Height: 16}:
		return 2704, 64, nil
	case transform.Size{Width: 16, Height: 4}:
		return 2768, 64, nil
	case transform.Size{Width: 8, Height: 32}:
		return 2832, 256, nil
	case transform.Size{Width: 32, Height: 8}:
		return 3088, 256, nil
	default:
		return 0, 0, ErrInvalidQuantizer
	}
}
