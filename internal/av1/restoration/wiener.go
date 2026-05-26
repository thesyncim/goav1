package restoration

const (
	WienerHalfwin    = 3
	WienerWin        = 2*WienerHalfwin + 1
	WienerSubpelTaps = WienerWin + 1
	WienerFilterBits = 7
	WienerFilterStep = 1 << WienerFilterBits
	WienerRound0Bits = 3
	WienerTap0Mid    = 3
	WienerTap1Mid    = -7
	WienerTap2Mid    = 15
	WienerTap0Bits   = 4
	WienerTap1Bits   = 5
	WienerTap2Bits   = 6
	WienerTap0Min    = WienerTap0Mid - ((1 << WienerTap0Bits) / 2)
	WienerTap1Min    = WienerTap1Mid - ((1 << WienerTap1Bits) / 2)
	WienerTap2Min    = WienerTap2Mid - ((1 << WienerTap2Bits) / 2)
	WienerTap0Max    = WienerTap0Mid - 1 + ((1 << WienerTap0Bits) / 2)
	WienerTap1Max    = WienerTap1Mid - 1 + ((1 << WienerTap1Bits) / 2)
	WienerTap2Max    = WienerTap2Mid - 1 + ((1 << WienerTap2Bits) / 2)
)

type WienerFilter [WienerSubpelTaps]int16

type WienerInfo struct {
	VFilter WienerFilter
	HFilter WienerFilter
}

func DefaultWienerInfo() WienerInfo {
	filter := NewWienerFilter(WienerTap0Mid, WienerTap1Mid, WienerTap2Mid)
	return WienerInfo{VFilter: filter, HFilter: filter}
}

func NewWienerFilter(tap0 int16, tap1 int16, tap2 int16) WienerFilter {
	tap3 := -2 * (tap0 + tap1 + tap2)
	return WienerFilter{tap0, tap1, tap2, tap3, tap2, tap1, tap0, 0}
}

func WienerScratchLen(width int, height int) (int, error) {
	if !validRestorationBlock(width, height) {
		return 0, ErrInvalidRestoration
	}
	return width * (height + 2*WienerHalfwin), nil
}

// ApplyWienerRestoration ports av1_highbd_wiener_convolve_add_src_c for
// unscaled restoration units. srcOrigin identifies the top-left pixel and src
// must include a WienerHalfwin-pixel border around that origin.
func ApplyWienerRestoration(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, info WienerInfo, bitDepth uint8, scratch []uint16) error {
	max, err := maxSample(bitDepth)
	if err != nil {
		return err
	}
	if !validRestorationBlock(width, height) ||
		!validWienerInfo(info) ||
		!borderedBlockFits(len(src), srcStride, srcOrigin, width, height, WienerHalfwin, WienerHalfwin) ||
		!blockFits(len(dst), dstStride, width, height) {
		return ErrInvalidRestoration
	}
	need, err := WienerScratchLen(width, height)
	if err != nil || len(scratch) < need {
		return ErrInvalidRestoration
	}

	round0, round1 := wienerRounds(int(bitDepth))
	temp := scratch[:need]
	if !wienerHorizontal(src, srcStride, srcOrigin, width, height, info.HFilter, int(bitDepth), round0, max, temp) {
		return ErrInvalidRestoration
	}
	wienerVertical(temp, width, dst, dstStride, width, height, info.VFilter, int(bitDepth), round1, max)
	return nil
}

func wienerHorizontal(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) bool {
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		dstRow := (row + WienerHalfwin) * width
		srcRow := srcOrigin + row*srcStride
		for col := range width {
			center := src[srcRow+col]
			if center > max {
				return false
			}
			sum := int32(center)<<WienerFilterBits + offset
			for k := range WienerWin {
				sample := src[srcRow+col-WienerHalfwin+k]
				if sample > max {
					return false
				}
				sum += int32(sample) * int32(filter[k])
			}
			temp[dstRow+col] = uint16(clampInt32(roundPowerOfTwo(sum, round0), 0, limit-1))
		}
	}
	return true
}

func wienerVertical(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	offset := int32(1 << (bitDepth + round1 - 1))
	for row := range height {
		for col := range width {
			center := temp[(row+WienerHalfwin)*tempStride+col]
			sum := int32(center)<<WienerFilterBits - offset
			for k := range WienerWin {
				sum += int32(temp[(row+k)*tempStride+col]) * int32(filter[k])
			}
			dst[row*dstStride+col] = uint16(clampInt32(roundPowerOfTwo(sum, round1), 0, int32(max)))
		}
	}
}

func validRestorationBlock(width int, height int) bool {
	return width > 0 && height > 0 && width <= ProcUnitSize && height <= ProcUnitSize
}

func wienerRounds(bitDepth int) (int, int) {
	round0 := WienerRound0Bits
	round1 := 2*WienerFilterBits - round0
	intBufRange := bitDepth + WienerFilterBits - round0 + 2
	if intBufRange > 16 {
		round0 += intBufRange - 16
		round1 -= intBufRange - 16
	}
	return round0, round1
}

func validWienerInfo(info WienerInfo) bool {
	return validWienerFilter(info.VFilter) && validWienerFilter(info.HFilter)
}

func validWienerFilter(filter WienerFilter) bool {
	if filter[7] != 0 ||
		filter[0] != filter[6] ||
		filter[1] != filter[5] ||
		filter[2] != filter[4] {
		return false
	}
	if int(filter[0]) < WienerTap0Min || int(filter[0]) > WienerTap0Max ||
		int(filter[1]) < WienerTap1Min || int(filter[1]) > WienerTap1Max ||
		int(filter[2]) < WienerTap2Min || int(filter[2]) > WienerTap2Max {
		return false
	}
	sum := int32(0)
	for i := range WienerWin {
		sum += int32(filter[i])
	}
	return sum == 0
}
