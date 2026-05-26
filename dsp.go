package goav1

import internaldsp "github.com/thesyncim/goav1/internal/av1/dsp"

// ErrDSPInvalidBlock is returned by DSP helpers when block geometry or buffer
// dimensions are inconsistent with the requested operation.
var ErrDSPInvalidBlock = internaldsp.ErrInvalidBlock

// FillPlaneBlock writes value into the (x, y, width, height) rectangle of dst.
// bytesPerSample selects 8-bit (1) or high-bit-depth (2) plane layout. The
// destination region must lie fully within dst; otherwise ErrDSPInvalidBlock
// is returned.
func FillPlaneBlock(dst FramePlane, bytesPerSample int, x int, y int, width int, height int, value uint16) error {
	return internaldsp.FillPlaneBlock(dst, bytesPerSample, x, y, width, height, value)
}

// CopyPlaneBlock copies a width-by-height block from src at (srcX, srcY) into
// dst at (dstX, dstY). The two planes may alias when the source and
// destination regions do not overlap. bytesPerSample selects the plane layout
// (1 for 8-bit, 2 for high-bit-depth).
func CopyPlaneBlock(dst FramePlane, src FramePlane, bytesPerSample int, dstX int, dstY int, srcX int, srcY int, width int, height int) error {
	return internaldsp.CopyPlaneBlock(dst, src, bytesPerSample, dstX, dstY, srcX, srcY, width, height)
}

// AddResidualPlaneBlock adds the residual block at residual[residualStride] to
// the (x, y, width, height) region of dst, clipping to bitDepth. residual is
// stored as int16 samples row-major with stride residualStride.
func AddResidualPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, residual []int16, residualStride int) error {
	return internaldsp.AddResidualPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, residual, residualStride)
}

// BlendA64Mask blends src0 and src1 into dst using a 64-entry compound mask.
// subX and subY enable horizontal and vertical mask subsampling to support
// chroma planes. All strides are in samples, not bytes; bitDepth clips the
// output samples.
func BlendA64Mask(dst []uint16, dstStride int, src0 []uint16, src0Stride int, src1 []uint16, src1Stride int, mask []uint8, maskStride int, width int, height int, subX bool, subY bool, bitDepth uint8) error {
	return internaldsp.BlendA64Mask(dst, dstStride, src0, src0Stride, src1, src1Stride, mask, maskStride, width, height, subX, subY, bitDepth)
}

// MinMaxAbsDiff8x8 returns the minimum and maximum absolute difference of
// corresponding samples across an 8x8 block in a and b. bytesPerSample selects
// the source layout. It is used as a building block by the loop-restoration
// pre-pass.
func MinMaxAbsDiff8x8(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	return internaldsp.MinMaxAbsDiff8x8(a, aStride, b, bStride, bytesPerSample)
}
