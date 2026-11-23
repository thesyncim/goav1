package goav1

import internalframe "github.com/thesyncim/goav1/internal/av1/frame"

// FrameFormat describes the geometry, bit depth, chroma subsampling, and
// alignment of a decoded AV1 frame buffer.
type FrameFormat = internalframe.Format

// FrameLayout reports the per-plane and total byte sizes for a FrameFormat,
// as computed by FrameRequiredSize.
type FrameLayout = internalframe.Layout

// FramePlane identifies one plane (Y, U, or V) of a Frame.
type FramePlane = internalframe.Plane

// Frame is a decoded AV1 picture stored in a caller-owned backing buffer.
// Instances are produced by BindFrame and acquired from a FramePool.
type Frame = internalframe.Frame

// Frame buffer binding errors. Each Err* value is returned by the frame and
// frame-pool helpers when the input buffer is too small, the format is
// invalid, the pool index is out of range, or the requested plane does not
// exist.
var (
	ErrFrameLayerPoolFull = internalframe.ErrLayerPoolFull
)

// framePoolRequiredSize returns the per-frame layout and the total bytes
// required to back a pool of count frames matching format. It validates that
// count is positive and that the resulting allocation does not overflow int.
func framePoolRequiredSize(format FrameFormat, count int) (FrameLayout, int, error) {
	if count <= 0 {
		return FrameLayout{}, 0, internalframe.ErrInvalidPool
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		return FrameLayout{}, 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if layout.Size != 0 && count > maxInt/layout.Size {
		return FrameLayout{}, 0, internalframe.ErrInvalidFormat
	}
	return layout, layout.Size * count, nil
}

// frameCodedFormatFromHeaders returns the FrameFormat sized to the coded
// (pre-super-res) frame width and the frame height taken from size, using the
// bit depth and chroma layout from sequence. align is propagated into the
// FrameFormat for downstream buffer sizing.
func frameCodedFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return frameFormatFromHeaderWidth(sequence, size.CodedWidth, size.Height, align)
}

// frameOutputFormatFromHeaders returns the FrameFormat sized to the upscaled
// (post-super-res) frame width and the frame height. It is the format
// applied to the final output surface presented to the caller.
func frameOutputFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return frameFormatFromHeaderWidth(sequence, size.UpscaledWidth, size.Height, align)
}

func frameFormatFromHeaderWidth(sequence SequenceHeader, width uint32, height uint32, align int) (FrameFormat, error) {
	if width == 0 || height == 0 || sequence.ColorConfig.BitDepth == 0 {
		return FrameFormat{}, internalframe.ErrInvalidFormat
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(width) > maxInt || uint64(height) > maxInt {
		return FrameFormat{}, internalframe.ErrInvalidFormat
	}
	// The buffer is allocated superblock-aligned so partial bottom/right
	// superblocks reconstruct their full transform extent into the padding;
	// the alignment must match the sequence's superblock size or the pool
	// surfaces will not match the format the decoder reconstructs into.
	sbSizeLog2 := uint8(6)
	if sequence.Use128x128Superblock {
		sbSizeLog2 = 7
	}
	return FrameFormat{
		Width:        int(width),
		Height:       int(height),
		BitDepth:     sequence.ColorConfig.BitDepth,
		MonoChrome:   sequence.ColorConfig.MonoChrome,
		SubsamplingX: sequence.ColorConfig.SubsamplingX,
		SubsamplingY: sequence.ColorConfig.SubsamplingY,
		SBSizeLog2:   sbSizeLog2,
		Align:        align,
	}, nil
}
