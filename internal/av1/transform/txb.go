// Ported from libaom:
//   av1/common/txb_common.h
//   av1/decoder/decodetxb.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

const (
	txPadHorLog2              = 2
	txb8x8LevelStride         = 8 + (1 << txPadHorLog2)
	txb8x8PrepCoeffCount      = 64
	txbCoeffContextBits       = 3
	txbCoeffContextMask       = (1 << txbCoeffContextBits) - 1
	txbCoeffContextNegativeDC = 1 << txbCoeffContextBits
	txbCoeffContextPositiveDC = 2 << txbCoeffContextBits
)

// TXB8x8PrepResult is the coefficient-prep summary used by trusted 8x8 TXB
// writers. The bit masks are indexed by scan order, while MaxScanLine is the
// largest raster coefficient position reached by a non-zero coefficient.
type TXB8x8PrepResult struct {
	NonZeroBits uint64
	SignBits    uint64
	MaxScanLine uint16
	CulLevel    uint8
}

var txb8x8Scan2D = [...]uint8{
	0, 8, 1, 2, 9, 16, 24, 17,
	10, 3, 4, 11, 18, 25, 32, 40,
	33, 26, 19, 12, 5, 6, 13, 20,
	27, 34, 41, 48, 56, 49, 42, 35,
	28, 21, 14, 7, 15, 22, 29, 36,
	43, 50, 57, 58, 51, 44, 37, 30,
	23, 31, 38, 45, 52, 59, 60, 53,
	46, 39, 47, 54, 61, 62, 55, 63,
}

func txbPrep8x8Levels2DPureGo(coeffs *[64]int16, levels *[256]uint8, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	txbInitLevels8x8PureGo(coeffs, levels)
	return txbPrep8x8Summary(coeffs, absLevels, eob)
}

func txbInitLevels8x8PureGo(coeffs *[64]int16, levels *[256]uint8) {
	for col := range 8 {
		src := col * 8
		dst := col * txb8x8LevelStride
		for row := range 8 {
			level := txbAbsInt16(coeffs[src+row])
			level = min(level, 127)
			levels[dst+row] = uint8(level)
		}
	}
}

func txbPrep8x8Summary(coeffs *[64]int16, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	var result TXB8x8PrepResult
	eob = min(eob, txb8x8PrepCoeffCount)
	if eob <= 0 {
		return result
	}
	if eob == 1 {
		cv := coeffs[0]
		level := txbAbsInt16(cv)
		absLevels[0] = uint16(level)
		if cv == 0 {
			return result
		}
		result.NonZeroBits = 1
		if cv < 0 {
			result.SignBits = 1
		}
		level = min(level, txbCoeffContextMask)
		switch {
		case cv < 0:
			level |= txbCoeffContextNegativeDC
		case cv > 0:
			level += txbCoeffContextPositiveDC
		}
		result.CulLevel = uint8(level)
		return result
	}
	culLevel := 0
	for c := range eob {
		pos := txb8x8Scan2D[c]
		cv := coeffs[pos]
		if cv == 0 {
			absLevels[c] = 0
			continue
		}
		level := txbAbsInt16(cv)
		absLevels[c] = uint16(level)
		if uint16(pos) > result.MaxScanLine {
			result.MaxScanLine = uint16(pos)
		}
		result.NonZeroBits |= 1 << uint(c)
		if cv < 0 {
			result.SignBits |= 1 << uint(c)
		}
		culLevel += level
	}
	culLevel = min(culLevel, txbCoeffContextMask)
	switch {
	case coeffs[0] < 0:
		culLevel |= txbCoeffContextNegativeDC
	case coeffs[0] > 0:
		culLevel += txbCoeffContextPositiveDC
	}
	result.CulLevel = uint8(culLevel)
	return result
}

func txbAbsInt16(v int16) int {
	if v < 0 {
		return -int(v)
	}
	return int(v)
}

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
