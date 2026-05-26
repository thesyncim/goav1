package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	warpedModelPrecBits       = 16
	warpedModelOne            = 1 << warpedModelPrecBits
	warpedPixelPrecBits       = 6
	warpedPixelPrecShifts     = 1 << warpedPixelPrecBits
	warpParamReduceBits       = 6
	warpedDiffPrecBits        = warpedModelPrecBits - warpedPixelPrecBits
	warpedIntermediateRows    = 15
	warpedIntermediateColumns = 8
)

// PredictWarpedPlaneBlockBitDepth predicts a single-reference affine warped
// block. The destination coordinates are absolute plane coordinates, matching
// libaom's p_col/p_row convention.
func PredictWarpedPlaneBlockBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	ssX := 0
	if subsamplingX {
		ssX = 1
	}
	ssY := 0
	if subsamplingY {
		ssY = 1
	}
	if bytesPerSample == 1 {
		warpAffine8(dst, ref, dstX, dstY, width, height, matrix, int(alpha), int(beta), int(gamma), int(delta), ssX, ssY)
		return nil
	}
	max, ok := highBDMax(bitDepth)
	if !ok {
		return ErrInvalidMotion
	}
	warpAffineHighBD(dst, ref, bitDepth, max, dstX, dstY, width, height, matrix, int(alpha), int(beta), int(gamma), int(delta), ssX, ssY)
	return nil
}

func warpAffine8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	const bd = 8
	reduceBitsHoriz := round0Bits
	reduceBitsVert := 2*filterBits - reduceBitsHoriz
	offsetBitsHoriz := bd + filterBits - 1
	offsetBitsVert := bd + 2*filterBits - reduceBitsHoriz
	for i := dstY; i < dstY+height; i += 8 {
		for j := dstX; j < dstX+width; j += 8 {
			warpHorizontal8(&tmp, ref, i, j, matrix, alpha, beta, gamma, delta, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			for k := -4; k < minWarpInt(4, dstY+height-i-4); k++ {
				sy := warpBlockSY(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY) + delta*(k+4)
				for l := -4; l < minWarpInt(4, dstX+width-j-4); l++ {
					offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
					if offs < 0 || offs >= len(warpedFilter) {
						continue
					}
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range filterTaps {
						sum += int(coeffs[m]) * int(tmp[(k+m+4)*warpedIntermediateColumns+(l+4)])
					}
					sum = roundPowerOfTwo(sum, reduceBitsVert)
					dst.Pix[(i+k+4)*dst.Stride+j+l+4] = byte(clipPixel(sum - (1 << (bd - 1)) - (1 << bd)))
					sy += gamma
				}
			}
		}
	}
}

func warpHorizontal8(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int, reduceBitsHoriz int, offsetBitsHoriz int) {
	ix4, sx4, iy4, _ := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	for k := -7; k < 8; k++ {
		iy := clampInt(iy4+k, 0, ref.Height-1)
		sx := sx4 + beta*(k+4)
		for l := -4; l < 4; l++ {
			ix := ix4 + l - 3
			offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				offs = clampInt(offs, 0, len(warpedFilter)-1)
			}
			coeffs := warpedFilter[offs]
			sum := 1 << offsetBitsHoriz
			for m := range filterTaps {
				sampleX := clampInt(ix+m, 0, ref.Width-1)
				sum += int(ref.Pix[iy*ref.Stride+sampleX]) * int(coeffs[m])
			}
			tmp[(k+7)*warpedIntermediateColumns+(l+4)] = int32(roundPowerOfTwo(sum, reduceBitsHoriz))
			sx += alpha
		}
	}
}

func warpAffineHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	reduceBitsHoriz := round0Bits
	reduceBitsVert := 2*filterBits - reduceBitsHoriz
	offsetBitsHoriz := int(bitDepth) + filterBits - 1
	offsetBitsVert := int(bitDepth) + 2*filterBits - reduceBitsHoriz
	for i := dstY; i < dstY+height; i += 8 {
		for j := dstX; j < dstX+width; j += 8 {
			warpHorizontalHighBD(&tmp, ref, i, j, matrix, alpha, beta, gamma, delta, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			for k := -4; k < minWarpInt(4, dstY+height-i-4); k++ {
				sy := warpBlockSY(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY) + delta*(k+4)
				for l := -4; l < minWarpInt(4, dstX+width-j-4); l++ {
					offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
					if offs < 0 || offs >= len(warpedFilter) {
						continue
					}
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range filterTaps {
						sum += int(coeffs[m]) * int(tmp[(k+m+4)*warpedIntermediateColumns+(l+4)])
					}
					sum = roundPowerOfTwo(sum, reduceBitsVert)
					storeHighBDSample(dst, j+l+4, i+k+4, clipPixelHighBD(sum-(1<<(bitDepth-1))-(1<<bitDepth), max))
					sy += gamma
				}
			}
		}
	}
}

func warpHorizontalHighBD(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int, reduceBitsHoriz int, offsetBitsHoriz int) {
	ix4, sx4, iy4, _ := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	for k := -7; k < 8; k++ {
		iy := clampInt(iy4+k, 0, ref.Height-1)
		sx := sx4 + beta*(k+4)
		for l := -4; l < 4; l++ {
			ix := ix4 + l - 3
			offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				offs = clampInt(offs, 0, len(warpedFilter)-1)
			}
			coeffs := warpedFilter[offs]
			sum := 1 << offsetBitsHoriz
			for m := range filterTaps {
				sampleX := clampInt(ix+m, 0, ref.Width-1)
				sum += int(loadHighBDSample(ref, sampleX, iy)) * int(coeffs[m])
			}
			tmp[(k+7)*warpedIntermediateColumns+(l+4)] = int32(roundPowerOfTwo(sum, reduceBitsHoriz))
			sx += alpha
		}
	}
}

func warpBlockSY(i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) int {
	_, _, _, sy4 := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	return sy4
}

func warpBlockOrigin(i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) (int, int, int, int) {
	srcX := int64((j + 4) << ssX)
	srcY := int64((i + 4) << ssY)
	dstX := int64(matrix[2])*srcX + int64(matrix[3])*srcY + int64(matrix[0])
	dstY := int64(matrix[4])*srcX + int64(matrix[5])*srcY + int64(matrix[1])
	x4 := dstX >> ssX
	y4 := dstY >> ssY
	ix4 := int(x4 >> warpedModelPrecBits)
	sx4 := int(x4 & (warpedModelOne - 1))
	iy4 := int(y4 >> warpedModelPrecBits)
	sy4 := int(y4 & (warpedModelOne - 1))
	sx4 += alpha*(-4) + beta*(-4)
	sy4 += gamma*(-4) + delta*(-4)
	sx4 &^= (1 << warpParamReduceBits) - 1
	sy4 &^= (1 << warpParamReduceBits) - 1
	return ix4, sx4, iy4, sy4
}

func minWarpInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}
