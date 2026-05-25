package goav1

import internalprediction "github.com/thesyncim/goav1/internal/av1/prediction"

type PredictionIntraMode = internalprediction.IntraMode
type PredictionIntraEdges = internalprediction.IntraEdges
type PredictionDirectionalEdges = internalprediction.DirectionalEdges
type PredictionFilterIntraMode = internalprediction.FilterIntraMode
type PredictionCFLPredType = internalprediction.CFLPredType

const (
	PredictionIntraModeDC               PredictionIntraMode = internalprediction.IntraModeDC
	PredictionIntraModeVertical         PredictionIntraMode = internalprediction.IntraModeVertical
	PredictionIntraModeHorizontal       PredictionIntraMode = internalprediction.IntraModeHorizontal
	PredictionIntraModePaeth            PredictionIntraMode = internalprediction.IntraModePaeth
	PredictionIntraModeSmooth           PredictionIntraMode = internalprediction.IntraModeSmooth
	PredictionIntraModeSmoothVertical   PredictionIntraMode = internalprediction.IntraModeSmoothVertical
	PredictionIntraModeSmoothHorizontal PredictionIntraMode = internalprediction.IntraModeSmoothHorizontal

	PredictionFilterIntraModeDC         PredictionFilterIntraMode = internalprediction.FilterIntraModeDC
	PredictionFilterIntraModeVertical   PredictionFilterIntraMode = internalprediction.FilterIntraModeVertical
	PredictionFilterIntraModeHorizontal PredictionFilterIntraMode = internalprediction.FilterIntraModeHorizontal
	PredictionFilterIntraModeD157       PredictionFilterIntraMode = internalprediction.FilterIntraModeD157
	PredictionFilterIntraModePaeth      PredictionFilterIntraMode = internalprediction.FilterIntraModePaeth
	PredictionFilterIntraModes          PredictionFilterIntraMode = internalprediction.FilterIntraModes

	PredictionCFLBufLine   = internalprediction.CFLBufLine
	PredictionCFLBufSquare = internalprediction.CFLBufSquare

	PredictionCFLPredU PredictionCFLPredType = internalprediction.CFLPredU
	PredictionCFLPredV PredictionCFLPredType = internalprediction.CFLPredV
)

var ErrPredictionInvalidPrediction = internalprediction.ErrInvalidPrediction

func PredictIntraPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, mode PredictionIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictIntraPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, mode, edges)
}

func PredictIntraPlaneBlockWithExtent(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, predWidth int, predHeight int, mode PredictionIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictIntraPlaneBlockWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, predWidth, predHeight, mode, edges)
}

func PredictionDirectionalDX(angle int) int {
	return internalprediction.DirectionalDX(angle)
}

func PredictionDirectionalDY(angle int) int {
	return internalprediction.DirectionalDY(angle)
}

func PredictDirectionalIntraPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, angle int, edges PredictionDirectionalEdges) error {
	return internalprediction.PredictDirectionalIntraPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, angle, edges)
}

func PredictFilterIntraPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, mode PredictionFilterIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictFilterIntraPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, mode, edges)
}

func PredictFilterIntraPlaneBlockWithExtent(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, predWidth int, predHeight int, mode PredictionFilterIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictFilterIntraPlaneBlockWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, predWidth, predHeight, mode, edges)
}

func FilterIntraEdge(edge []uint16, scratch []uint16, strength uint8, bitDepth uint8) error {
	return internalprediction.FilterIntraEdge(edge, scratch, strength, bitDepth)
}

func UpsampleIntraEdge(edge []uint16, origin int, size int, scratch []uint16, bitDepth uint8) error {
	return internalprediction.UpsampleIntraEdge(edge, origin, size, scratch, bitDepth)
}

func FilterIntraEdgeCorner(above []uint16, aboveOrigin int, left []uint16, leftOrigin int, bitDepth uint8) error {
	return internalprediction.FilterIntraEdgeCorner(above, aboveOrigin, left, leftOrigin, bitDepth)
}

func IntraEdgeFilterStrength(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) uint8 {
	return internalprediction.IntraEdgeFilterStrength(blockSize0, blockSize1, delta, smoothNeighbor)
}

func UseIntraEdgeUpsample(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) bool {
	return internalprediction.UseIntraEdgeUpsample(blockSize0, blockSize1, delta, smoothNeighbor)
}

func CFLAlphaQ3(alphaIndex uint8, jointSign int8, predType PredictionCFLPredType) (int, error) {
	return internalprediction.CFLAlphaQ3(alphaIndex, jointSign, predType)
}

func SubsampleLuma8ToCFLQ3(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, subX bool, subY bool) error {
	return internalprediction.SubsampleLuma8ToQ3(outputQ3, input, inputStride, width, height, subX, subY)
}

func SubsampleLuma16ToCFLQ3(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, subX bool, subY bool, bitDepth uint8) error {
	return internalprediction.SubsampleLuma16ToQ3(outputQ3, input, inputStride, width, height, subX, subY, bitDepth)
}

func PadCFLReconQ3(reconQ3 []uint16, bufWidth int, bufHeight int, width int, height int) (int, int, error) {
	return internalprediction.PadCFLReconQ3(reconQ3, bufWidth, bufHeight, width, height)
}

func SubtractCFLAverage(srcQ3 []uint16, dstQ3 []int16, width int, height int) error {
	return internalprediction.SubtractCFLAverage(srcQ3, dstQ3, width, height)
}

func PredictCFLPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, acQ3 []int16, alphaQ3 int) error {
	return internalprediction.PredictCFLPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, acQ3, alphaQ3)
}
