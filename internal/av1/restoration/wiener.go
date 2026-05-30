// Ported from libaom:
//   av1/common/restoration.c
//   av1/common/restoration.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

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
	if !wienerHorizontalImpl(src, srcStride, srcOrigin, width, height, info.HFilter, int(bitDepth), round0, max, temp) {
		return ErrInvalidRestoration
	}
	wienerVerticalImpl(temp, width, dst, dstStride, width, height, info.VFilter, int(bitDepth), round1, max)
	return nil
}

// wienerHorizontalImpl and wienerVerticalImpl are the dispatch slots for the two
// separable Wiener passes. They are resolved exactly once, at package init (see
// wiener_dispatch_*.go), so the per-call cost is a single indirect call with no
// feature-detection branches. wienerHorizontal / wienerVertical below are the
// canonical bit-exact references; every tuned variant MUST match them sample for
// sample (the rounding term, the round0/round1 shifts, and the [0,maxClamp] /
// [0,max] clamps). Tests and benchmarks must not mutate these concurrently with
// live decoding.
var (
	wienerHorizontalImpl = wienerHorizontal
	wienerVerticalImpl   = wienerVertical
)

func wienerHorizontal(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) bool {
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	maxClamp := limit - 1
	// Hoist the seven distinct taps. filter[3] is the window center; it is the
	// same sample read as `center`, so libaom counts it via both the shifted
	// center term and the tap product.
	f0 := int32(filter[0])
	f1 := int32(filter[1])
	f2 := int32(filter[2])
	f3 := int32(filter[3])
	f4 := int32(filter[4])
	f5 := int32(filter[5])
	f6 := int32(filter[6])
	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		dstRow := (row + WienerHalfwin) * width
		srcStart := srcOrigin + row*srcStride - WienerHalfwin
		// Reslice to the exact window the inner loop touches:
		// columns [col-3 .. col+3] for col in [0, width). This drops the
		// per-tap bounds checks on src.
		srcRow := src[srcStart : srcStart+width+2*WienerHalfwin]
		dstSlice := temp[dstRow : dstRow+width]
		for col := range width {
			w := srcRow[col : col+WienerWin : col+WienerWin]
			s0, s1, s2, s3, s4, s5, s6 := w[0], w[1], w[2], w[3], w[4], w[5], w[6]
			if s0 > max || s1 > max || s2 > max || s3 > max || s4 > max || s5 > max || s6 > max {
				return false
			}
			sum := int32(s3)<<WienerFilterBits + offset
			sum += int32(s0)*f0 + int32(s1)*f1 + int32(s2)*f2 + int32(s3)*f3 +
				int32(s4)*f4 + int32(s5)*f5 + int32(s6)*f6
			dstSlice[col] = uint16(clampInt32(roundPowerOfTwo(sum, round0), 0, maxClamp))
		}
	}
	return true
}

func wienerVertical(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	offset := int32(1 << (bitDepth + round1 - 1))
	maxI := int32(max)
	f0 := int32(filter[0])
	f1 := int32(filter[1])
	f2 := int32(filter[2])
	f3 := int32(filter[3])
	f4 := int32(filter[4])
	f5 := int32(filter[5])
	f6 := int32(filter[6])
	for row := range height {
		// The seven vertical taps are rows [row .. row+6]; row+3 is the center.
		r0 := temp[(row+0)*tempStride : (row+0)*tempStride+width]
		r1 := temp[(row+1)*tempStride : (row+1)*tempStride+width]
		r2 := temp[(row+2)*tempStride : (row+2)*tempStride+width]
		r3 := temp[(row+3)*tempStride : (row+3)*tempStride+width]
		r4 := temp[(row+4)*tempStride : (row+4)*tempStride+width]
		r5 := temp[(row+5)*tempStride : (row+5)*tempStride+width]
		r6 := temp[(row+6)*tempStride : (row+6)*tempStride+width]
		dstSlice := dst[row*dstStride : row*dstStride+width]
		// Length hints: each rN has len width, so indexing by col (range over
		// r0) is provably in bounds and the compiler drops the per-tap checks.
		r1 = r1[:width]
		r2 = r2[:width]
		r3 = r3[:width]
		r4 = r4[:width]
		r5 = r5[:width]
		r6 = r6[:width]
		dstSlice = dstSlice[:width]
		for col, c0 := range r0 {
			c3 := int32(r3[col])
			sum := c3<<WienerFilterBits - offset
			sum += int32(c0)*f0 + int32(r1[col])*f1 + int32(r2[col])*f2 +
				c3*f3 + int32(r4[col])*f4 + int32(r5[col])*f5 + int32(r6[col])*f6
			dstSlice[col] = uint16(clampInt32(roundPowerOfTwo(sum, round1), 0, maxI))
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
