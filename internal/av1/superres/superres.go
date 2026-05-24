package superres

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	Num          = 8
	DenomMin     = 9
	DenomMax     = 16
	FilterBits   = 6
	FilterShifts = 1 << FilterBits
	FilterTaps   = 8
	FilterOffset = 3
	ScaleBits    = 14
	ScaleMask    = (1 << ScaleBits) - 1
	ExtraBits    = ScaleBits - FilterBits

	filterRoundBits = 7
)

// UpscalePlane horizontally upscales src into dst using AV1's normative
// superres filter. The caller supplies already-decimated luma or chroma planes.
func UpscalePlane(src frame.SamplePlane, dst frame.SamplePlane, bitDepth uint8) error {
	if err := validatePlanes(src, dst, bitDepth); err != nil {
		return err
	}
	stepX := ((src.Width << ScaleBits) + (dst.Width / 2)) / dst.Width
	err := (dst.Width * stepX) - (src.Width << ScaleBits)
	initialSubpelX := ((-((dst.Width - src.Width) << (ScaleBits - 1))) + dst.Width/2) / dst.Width
	initialSubpelX += (1 << (ExtraBits - 1)) - err/2
	initialSubpelX &= ScaleMask

	maxValue := (1 << bitDepth) - 1
	for y := 0; y < src.Height; y++ {
		srcRow := src.Pix[y*src.Stride:]
		dstRow := dst.Pix[y*dst.Stride:]
		for x := 0; x < dst.Width; x++ {
			srcX := -(1 << ScaleBits) + initialSubpelX + x*stepX
			srcXPx := srcX >> ScaleBits
			srcXSubpel := (srcX & ScaleMask) >> ExtraBits
			filter := upscaleFilter[srcXSubpel]
			sum := 0
			for k := 0; k < FilterTaps; k++ {
				sampleX := clipInt(srcXPx+k-FilterOffset, 0, src.Width-1)
				sum += int(srcRow[sampleX]) * int(filter[k])
			}
			dstRow[x] = uint16(clipInt(roundPowerOfTwo(sum, filterRoundBits), 0, maxValue))
		}
	}
	return nil
}

func validatePlanes(src frame.SamplePlane, dst frame.SamplePlane, bitDepth uint8) error {
	if bitDepth < 8 || bitDepth > 12 ||
		!samplePlaneFits(src) || !samplePlaneFits(dst) ||
		dst.Height != src.Height ||
		dst.Width <= src.Width ||
		src.Width > int(^uint(0)>>1)>>ScaleBits {
		return frame.ErrInvalidPlane
	}
	return nil
}

func samplePlaneFits(plane frame.SamplePlane) bool {
	if plane.Width <= 0 || plane.Height <= 0 || plane.Stride < plane.Width {
		return false
	}
	need := (plane.Height-1)*plane.Stride + plane.Width
	return need >= 0 && len(plane.Pix) >= need
}

func roundPowerOfTwo(value int, bits int) int {
	if bits <= 0 {
		return value
	}
	return (value + (1 << (bits - 1))) >> bits
}

func clipInt(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

var upscaleFilter = [FilterShifts][FilterTaps]int16{
	{0, 0, 0, 128, 0, 0, 0, 0}, {0, 0, -1, 128, 2, -1, 0, 0},
	{0, 1, -3, 127, 4, -2, 1, 0}, {0, 1, -4, 127, 6, -3, 1, 0},
	{0, 2, -6, 126, 8, -3, 1, 0}, {0, 2, -7, 125, 11, -4, 1, 0},
	{-1, 2, -8, 125, 13, -5, 2, 0}, {-1, 3, -9, 124, 15, -6, 2, 0},
	{-1, 3, -10, 123, 18, -6, 2, -1}, {-1, 3, -11, 122, 20, -7, 3, -1},
	{-1, 4, -12, 121, 22, -8, 3, -1}, {-1, 4, -13, 120, 25, -9, 3, -1},
	{-1, 4, -14, 118, 28, -9, 3, -1}, {-1, 4, -15, 117, 30, -10, 4, -1},
	{-1, 5, -16, 116, 32, -11, 4, -1}, {-1, 5, -16, 114, 35, -12, 4, -1},
	{-1, 5, -17, 112, 38, -12, 4, -1}, {-1, 5, -18, 111, 40, -13, 5, -1},
	{-1, 5, -18, 109, 43, -14, 5, -1}, {-1, 6, -19, 107, 45, -14, 5, -1},
	{-1, 6, -19, 105, 48, -15, 5, -1}, {-1, 6, -19, 103, 51, -16, 5, -1},
	{-1, 6, -20, 101, 53, -16, 6, -1}, {-1, 6, -20, 99, 56, -17, 6, -1},
	{-1, 6, -20, 97, 58, -17, 6, -1}, {-1, 6, -20, 95, 61, -18, 6, -1},
	{-2, 7, -20, 93, 64, -18, 6, -2}, {-2, 7, -20, 91, 66, -19, 6, -1},
	{-2, 7, -20, 88, 69, -19, 6, -1}, {-2, 7, -20, 86, 71, -19, 6, -1},
	{-2, 7, -20, 84, 74, -20, 7, -2}, {-2, 7, -20, 81, 76, -20, 7, -1},
	{-2, 7, -20, 79, 79, -20, 7, -2}, {-1, 7, -20, 76, 81, -20, 7, -2},
	{-2, 7, -20, 74, 84, -20, 7, -2}, {-1, 6, -19, 71, 86, -20, 7, -2},
	{-1, 6, -19, 69, 88, -20, 7, -2}, {-1, 6, -19, 66, 91, -20, 7, -2},
	{-2, 6, -18, 64, 93, -20, 7, -2}, {-1, 6, -18, 61, 95, -20, 6, -1},
	{-1, 6, -17, 58, 97, -20, 6, -1}, {-1, 6, -17, 56, 99, -20, 6, -1},
	{-1, 6, -16, 53, 101, -20, 6, -1}, {-1, 5, -16, 51, 103, -19, 6, -1},
	{-1, 5, -15, 48, 105, -19, 6, -1}, {-1, 5, -14, 45, 107, -19, 6, -1},
	{-1, 5, -14, 43, 109, -18, 5, -1}, {-1, 5, -13, 40, 111, -18, 5, -1},
	{-1, 4, -12, 38, 112, -17, 5, -1}, {-1, 4, -12, 35, 114, -16, 5, -1},
	{-1, 4, -11, 32, 116, -16, 5, -1}, {-1, 4, -10, 30, 117, -15, 4, -1},
	{-1, 3, -9, 28, 118, -14, 4, -1}, {-1, 3, -9, 25, 120, -13, 4, -1},
	{-1, 3, -8, 22, 121, -12, 4, -1}, {-1, 3, -7, 20, 122, -11, 3, -1},
	{-1, 2, -6, 18, 123, -10, 3, -1}, {0, 2, -6, 15, 124, -9, 3, -1},
	{0, 2, -5, 13, 125, -8, 2, -1}, {0, 1, -4, 11, 125, -7, 2, 0},
	{0, 1, -3, 8, 126, -6, 2, 0}, {0, 1, -3, 6, 127, -4, 1, 0},
	{0, 1, -2, 4, 127, -3, 1, 0}, {0, 0, -1, 2, 128, -1, 0, 0},
}
