// Ported from libaom: av1/common/quant_common.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package quantize

import (
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	QMLevelBits = 4
	QMLevels    = 1 << QMLevelBits

	DefaultQMFirst         = 5
	DefaultQMLast          = 9
	DefaultQMFirstAllIntra = 4
	DefaultQMLastAllIntra  = 10
)

// QMatrix is libaom's inverse quantization matrix of level for a luma (chroma
// false) or chroma transform block of size, in coefficient order
// (col * scanHeight + row, the decoder's cf order). It is nil for the flat
// level 15 and for sizes without a matrix.
func QMatrix(level uint8, chroma bool, size transform.Size) []uint16 {
	if level >= QMLevels-1 {
		return nil
	}
	offset, length, err := inverseQMatrixOffset(size)
	if err != nil {
		return nil
	}
	class := 0
	if chroma {
		class = 1
	}
	return inverseQMatrixRef[level][class][offset : offset+length]
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
