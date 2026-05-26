package goav1

import internalmotion "github.com/thesyncim/goav1/internal/av1/motion"

// MotionVector is a quarter-pel (or eighth-pel, when high precision is on)
// signed motion vector with Col/Row fields. It is the AV1 inter-prediction
// motion vector type.
type MotionVector = internalmotion.Vector

// MotionInterpFilter selects one of the AV1 8-tap subpel filters (regular,
// smooth, sharp) or the 4-tap bilinear filter for inter prediction.
type MotionInterpFilter = internalmotion.InterpFilter

// MotionInterpFilters bundles the horizontal and vertical
// MotionInterpFilter selections used to inter-predict one block.
type MotionInterpFilters = internalmotion.InterpFilters

// Motion subpel precision constants and interpolation-filter identifiers.
// MotionSubpelBits is the number of subpel fractional bits (3, i.e. 1/8 sample
// units); MotionSubpelScale is 1<<MotionSubpelBits.
//
// MotionInterpEightTapRegular / Smooth / MultiTapSharp / Bilinear identify the
// four AV1 interpolation filters used by inter prediction.
const (
	MotionSubpelBits  = internalmotion.SubpelBits
	MotionSubpelScale = internalmotion.SubpelScale

	MotionInterpEightTapRegular MotionInterpFilter = internalmotion.InterpEightTapRegular
	MotionInterpEightTapSmooth  MotionInterpFilter = internalmotion.InterpEightTapSmooth
	MotionInterpMultiTapSharp   MotionInterpFilter = internalmotion.InterpMultiTapSharp
	MotionInterpBilinear        MotionInterpFilter = internalmotion.InterpBilinear
)

// ErrMotionInvalidMotion is returned by the motion helpers when a vector or
// reference origin falls outside the AV1 spec range.
var ErrMotionInvalidMotion = internalmotion.ErrInvalidMotion

// FullpelMotionVector returns a MotionVector pointing to integer-pel
// coordinates (col, row). It validates that the inputs fit in the AV1 motion
// vector range.
func FullpelMotionVector(col int, row int) (MotionVector, error) {
	return internalmotion.FullpelVector(col, row)
}

// LowerMotionVectorPrecision applies the AV1 force_integer_mv / allow_high_
// precision_mv rounding rules to v, returning the rounded MotionVector.
func LowerMotionVectorPrecision(v MotionVector, allowHighPrecision bool, forceInteger bool) MotionVector {
	return internalmotion.LowerPrecision(v, allowHighPrecision, forceInteger)
}

// FullpelMotionReferenceOrigin returns the integer-pel reference coordinates
// for an output block at (dstX, dstY) referenced by the integer-pel mv.
func FullpelMotionReferenceOrigin(dstX int, dstY int, mv MotionVector) (int, int, error) {
	return internalmotion.FullpelReferenceOrigin(dstX, dstY, mv)
}

// MotionReferenceOrigin returns the integer-pel reference coordinates and the
// fractional subpel offsets implied by mv for an output block at (dstX, dstY).
func MotionReferenceOrigin(dstX int, dstY int, mv MotionVector) (refX int, refY int, subX int, subY int, err error) {
	return internalmotion.ReferenceOrigin(dstX, dstY, mv)
}

// MotionReferenceOriginSubsampled is MotionReferenceOrigin for chroma planes
// with the supplied horizontal and vertical subsampling.
func MotionReferenceOriginSubsampled(dstX int, dstY int, mv MotionVector, subsamplingX bool, subsamplingY bool) (refX int, refY int, subX int, subY int, err error) {
	return internalmotion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
}

// PredictInterPlaneBlock inter-predicts a width-by-height block in dst at
// (dstX, dstY) by sampling ref through mv with the default AV1 8-tap regular
// filter and default 8-bit clipping.
func PredictInterPlaneBlock(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, width int, height int, mv MotionVector) error {
	return internalmotion.PredictInterPlaneBlock(dst, ref, bytesPerSample, dstX, dstY, width, height, mv)
}

// PredictInterPlaneBlockBitDepth is PredictInterPlaneBlock with an explicit
// bit depth (8/10/12) controlling output clipping.
func PredictInterPlaneBlockBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv MotionVector) error {
	return internalmotion.PredictInterPlaneBlockBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, width, height, mv)
}

// PredictInterPlaneBlockWithFilter is PredictInterPlaneBlock with an explicit
// pair of horizontal/vertical interpolation filters.
func PredictInterPlaneBlockWithFilter(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, width int, height int, mv MotionVector, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockWithFilter(dst, ref, bytesPerSample, dstX, dstY, width, height, mv, filters)
}

// PredictInterPlaneBlockWithFilterBitDepth is PredictInterPlaneBlockWithFilter
// with an explicit bit depth (8/10/12) controlling output clipping.
func PredictInterPlaneBlockWithFilterBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv MotionVector, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, width, height, mv, filters)
}

// PredictInterPlaneBlockFromOrigin inter-predicts a block by sampling ref
// starting at the explicit reference origin (refX, refY) with subpel offsets
// (subX, subY). It is the form used when callers have already resolved the
// reference origin separately from the motion vector.
func PredictInterPlaneBlockFromOrigin(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int) error {
	return internalmotion.PredictInterPlaneBlockFromOrigin(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height, subX, subY)
}

// PredictInterPlaneBlockFromOriginBitDepth is PredictInterPlaneBlockFromOrigin
// with an explicit bit depth (8/10/12) controlling output clipping.
func PredictInterPlaneBlockFromOriginBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int) error {
	return internalmotion.PredictInterPlaneBlockFromOriginBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY)
}

// PredictInterPlaneBlockFromOriginWithFilter is
// PredictInterPlaneBlockFromOrigin with explicit interpolation filters.
func PredictInterPlaneBlockFromOriginWithFilter(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockFromOriginWithFilter(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}

// PredictInterPlaneBlockFromOriginWithFilterBitDepth is
// PredictInterPlaneBlockFromOriginWithFilter with an explicit bit depth
// (8/10/12) controlling output clipping.
func PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}
