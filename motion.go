package goav1

import internalmotion "github.com/thesyncim/goav1/internal/av1/motion"

type MotionVector = internalmotion.Vector
type MotionInterpFilter = internalmotion.InterpFilter
type MotionInterpFilters = internalmotion.InterpFilters

const (
	MotionSubpelBits  = internalmotion.SubpelBits
	MotionSubpelScale = internalmotion.SubpelScale

	MotionInterpEightTapRegular MotionInterpFilter = internalmotion.InterpEightTapRegular
	MotionInterpEightTapSmooth  MotionInterpFilter = internalmotion.InterpEightTapSmooth
	MotionInterpMultiTapSharp   MotionInterpFilter = internalmotion.InterpMultiTapSharp
	MotionInterpBilinear        MotionInterpFilter = internalmotion.InterpBilinear
)

var ErrMotionInvalidMotion = internalmotion.ErrInvalidMotion

func FullpelMotionVector(col int, row int) (MotionVector, error) {
	return internalmotion.FullpelVector(col, row)
}

func LowerMotionVectorPrecision(v MotionVector, allowHighPrecision bool, forceInteger bool) MotionVector {
	return internalmotion.LowerPrecision(v, allowHighPrecision, forceInteger)
}

func FullpelMotionReferenceOrigin(dstX int, dstY int, mv MotionVector) (int, int, error) {
	return internalmotion.FullpelReferenceOrigin(dstX, dstY, mv)
}

func MotionReferenceOrigin(dstX int, dstY int, mv MotionVector) (refX int, refY int, subX int, subY int, err error) {
	return internalmotion.ReferenceOrigin(dstX, dstY, mv)
}

func MotionReferenceOriginSubsampled(dstX int, dstY int, mv MotionVector, subsamplingX bool, subsamplingY bool) (refX int, refY int, subX int, subY int, err error) {
	return internalmotion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
}

func PredictInterPlaneBlock(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, width int, height int, mv MotionVector) error {
	return internalmotion.PredictInterPlaneBlock(dst, ref, bytesPerSample, dstX, dstY, width, height, mv)
}

func PredictInterPlaneBlockBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv MotionVector) error {
	return internalmotion.PredictInterPlaneBlockBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, width, height, mv)
}

func PredictInterPlaneBlockWithFilter(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, width int, height int, mv MotionVector, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockWithFilter(dst, ref, bytesPerSample, dstX, dstY, width, height, mv, filters)
}

func PredictInterPlaneBlockWithFilterBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv MotionVector, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, width, height, mv, filters)
}

func PredictInterPlaneBlockFromOrigin(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int) error {
	return internalmotion.PredictInterPlaneBlockFromOrigin(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height, subX, subY)
}

func PredictInterPlaneBlockFromOriginBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int) error {
	return internalmotion.PredictInterPlaneBlockFromOriginBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY)
}

func PredictInterPlaneBlockFromOriginWithFilter(dst FramePlane, ref FramePlane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockFromOriginWithFilter(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}

func PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst FramePlane, ref FramePlane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters MotionInterpFilters) error {
	return internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}
