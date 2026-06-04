// Ported from libaom:
//   av1/common/txb_common.h
//   av1/decoder/decodetxb.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

const (
	txPadHorLog2 = 2
)

// MaxEOB returns libaom's av1_get_max_eob() value for size.
func MaxEOB(size Size) (int, error) {
	scanSize, err := ScanSize(size)
	if err != nil {
		return 0, err
	}
	return int(scanSize.Width) * int(scanSize.Height), nil
}

// PaddedCoeffIndex returns libaom's get_padded_idx() coefficient index for
// level buffers that carry four extra horizontal padding slots per column.
func PaddedCoeffIndex(size Size, coeffIndex int) (int, error) {
	maxEOB, err := MaxEOB(size)
	if err != nil {
		return 0, err
	}
	if coeffIndex < 0 || coeffIndex >= maxEOB {
		return 0, ErrInvalidTransform
	}
	scanSize, _ := ScanSize(size)
	bhl, ok := log2PowerOfTwo(int(scanSize.Height))
	if !ok {
		return 0, ErrInvalidTransform
	}
	return coeffIndex + ((coeffIndex >> bhl) << txPadHorLog2), nil
}

// LowerLevelsCtxEOB returns libaom's get_lower_levels_ctx_eob() context.
func LowerLevelsCtxEOB(size Size, scanIndex int) (int, error) {
	maxEOB, err := MaxEOB(size)
	if err != nil {
		return 0, err
	}
	if scanIndex < 0 || scanIndex >= maxEOB {
		return 0, ErrInvalidTransform
	}
	if scanIndex == 0 {
		return 0, nil
	}
	if scanIndex <= maxEOB/8 {
		return 1, nil
	}
	if scanIndex <= maxEOB/4 {
		return 2, nil
	}
	return 3, nil
}

func log2PowerOfTwo(v int) (int, bool) {
	switch v {
	case 4:
		return 2, true
	case 8:
		return 3, true
	case 16:
		return 4, true
	case 32:
		return 5, true
	default:
		return 0, false
	}
}
