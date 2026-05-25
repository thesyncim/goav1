package goav1

import internaldsp "github.com/thesyncim/goav1/internal/av1/dsp"

var ErrDSPInvalidBlock = internaldsp.ErrInvalidBlock

func FillPlaneBlock(dst FramePlane, bytesPerSample int, x int, y int, width int, height int, value uint16) error {
	return internaldsp.FillPlaneBlock(dst, bytesPerSample, x, y, width, height, value)
}

func CopyPlaneBlock(dst FramePlane, src FramePlane, bytesPerSample int, dstX int, dstY int, srcX int, srcY int, width int, height int) error {
	return internaldsp.CopyPlaneBlock(dst, src, bytesPerSample, dstX, dstY, srcX, srcY, width, height)
}

func AddResidualPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, residual []int16, residualStride int) error {
	return internaldsp.AddResidualPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, residual, residualStride)
}

func BlendA64Mask(dst []uint16, dstStride int, src0 []uint16, src0Stride int, src1 []uint16, src1Stride int, mask []uint8, maskStride int, width int, height int, subX bool, subY bool, bitDepth uint8) error {
	return internaldsp.BlendA64Mask(dst, dstStride, src0, src0Stride, src1, src1Stride, mask, maskStride, width, height, subX, subY, bitDepth)
}

func MinMaxAbsDiff8x8(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	return internaldsp.MinMaxAbsDiff8x8(a, aStride, b, bStride, bytesPerSample)
}
