// Ported from libaom: aom_dsp/sad.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package dsp

// minMaxAbsDiff8x8PureGo is the portable reference implementation of the
// MinMaxAbsDiff8x8 kernel. It is the canonical pure-Go path: every SIMD
// variant the dispatcher selects MUST produce bit-exact output relative to
// this function. The dispatch layer in minmax.go selects this implementation
// on architectures without a tuned variant.
func minMaxAbsDiff8x8PureGo(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return 0, 0, ErrInvalidBlock
	}
	rowBytes := 8 * bytesPerSample
	if aStride < rowBytes || bStride < rowBytes ||
		!byteBlockFits(len(a), aStride, rowBytes, 8) ||
		!byteBlockFits(len(b), bStride, rowBytes, 8) {
		return 0, 0, ErrInvalidBlock
	}

	minDiff := uint16(^uint16(0))
	var maxDiff uint16
	switch bytesPerSample {
	case 1:
		for row := range 8 {
			aLine := a[row*aStride : row*aStride+8]
			bLine := b[row*bStride : row*bStride+8]
			for col := range 8 {
				diff := absDiff8(aLine[col], bLine[col])
				if diff < minDiff {
					minDiff = diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}
	case 2:
		for row := range 8 {
			aLine := a[row*aStride : row*aStride+rowBytes]
			bLine := b[row*bStride : row*bStride+rowBytes]
			for col := range 8 {
				i := col * 2
				av := uint16(aLine[i]) | uint16(aLine[i+1])<<8
				bv := uint16(bLine[i]) | uint16(bLine[i+1])<<8
				diff := absDiff16(av, bv)
				if diff < minDiff {
					minDiff = diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}
	}
	return minDiff, maxDiff, nil
}

func byteBlockFits(length int, stride int, rowBytes int, height int) bool {
	if stride <= 0 || rowBytes <= 0 || height <= 0 {
		return false
	}
	lastRowOffset, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastRowOffset, rowBytes)
	return ok && needed <= length
}

func absDiff8(a byte, b byte) uint16 {
	if a > b {
		return uint16(a - b)
	}
	return uint16(b - a)
}

func absDiff16(a uint16, b uint16) uint16 {
	if a > b {
		return a - b
	}
	return b - a
}
