package dsp

// MinMaxAbsDiff8x8 returns the minimum and maximum absolute sample difference
// over an 8x8 block. Samples are 8-bit when bytesPerSample is 1 and little-endian
// unsigned 16-bit when bytesPerSample is 2.
func MinMaxAbsDiff8x8(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
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
