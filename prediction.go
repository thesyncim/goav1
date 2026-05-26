package goav1

import internalprediction "github.com/thesyncim/goav1/internal/av1/prediction"

// PredictionIntraMode identifies one of the AV1 non-directional intra modes
// (DC, vertical, horizontal, Paeth, smooth, smooth-vertical, smooth-horizontal).
type PredictionIntraMode = internalprediction.IntraMode

// PredictionIntraEdges bundles the above/left/top-left edge samples used by
// an intra-prediction block.
type PredictionIntraEdges = internalprediction.IntraEdges

// PredictionDirectionalEdges extends PredictionIntraEdges with the additional
// top-right and bottom-left edge samples consumed by directional intra modes.
type PredictionDirectionalEdges = internalprediction.DirectionalEdges

// PredictionFilterIntraMode identifies one of the AV1 5-mode filter intra
// predictors (DC, vertical, horizontal, D157, Paeth).
type PredictionFilterIntraMode = internalprediction.FilterIntraMode

// PredictionCFLPredType selects between the U and V chroma planes when
// applying chroma-from-luma prediction.
type PredictionCFLPredType = internalprediction.CFLPredType

// PredictionIntraMode* values enumerate the non-directional AV1 intra modes.
// PredictionFilterIntraMode* values enumerate the AV1 filter-intra modes, and
// PredictionFilterIntraModes is the number of distinct filter modes.
//
// PredictionCFLBufLine / BufSquare are buffer-sizing constants for the
// chroma-from-luma intermediate buffers; PredictionCFLPredU / PredV identify
// the target chroma plane during CFL prediction.
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

// ErrPredictionInvalidPrediction is returned by the prediction helpers when
// the supplied mode, geometry, or edges are inconsistent with the AV1 spec.
var ErrPredictionInvalidPrediction = internalprediction.ErrInvalidPrediction

// PredictIntraPlaneBlock intra-predicts a width-by-height block in dst at
// (x, y) using the supplied non-directional mode and reference edges.
func PredictIntraPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, mode PredictionIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictIntraPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, mode, edges)
}

// PredictIntraPlaneBlockWithExtent intra-predicts the (predWidth, predHeight)
// sub-region of a width-by-height block in dst. It is used for chroma blocks
// whose prediction extent differs from the coded block size.
func PredictIntraPlaneBlockWithExtent(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, predWidth int, predHeight int, mode PredictionIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictIntraPlaneBlockWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, predWidth, predHeight, mode, edges)
}

// PredictionDirectionalDX returns the horizontal step (dx) for an AV1
// directional intra angle.
func PredictionDirectionalDX(angle int) int {
	return internalprediction.DirectionalDX(angle)
}

// PredictionDirectionalDY returns the vertical step (dy) for an AV1
// directional intra angle.
func PredictionDirectionalDY(angle int) int {
	return internalprediction.DirectionalDY(angle)
}

// PredictDirectionalIntraPlaneBlock intra-predicts a width-by-height block in
// dst using the directional mode identified by angle.
func PredictDirectionalIntraPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, angle int, edges PredictionDirectionalEdges) error {
	return internalprediction.PredictDirectionalIntraPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, angle, edges)
}

// PredictFilterIntraPlaneBlock intra-predicts a width-by-height block in dst
// using one of the AV1 filter-intra modes.
func PredictFilterIntraPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, mode PredictionFilterIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictFilterIntraPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, mode, edges)
}

// PredictFilterIntraPlaneBlockWithExtent applies filter-intra prediction to
// the (predWidth, predHeight) sub-region of a width-by-height block in dst.
func PredictFilterIntraPlaneBlockWithExtent(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, predWidth int, predHeight int, mode PredictionFilterIntraMode, edges PredictionIntraEdges) error {
	return internalprediction.PredictFilterIntraPlaneBlockWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, predWidth, predHeight, mode, edges)
}

// FilterIntraEdge applies the AV1 intra-edge smoothing filter to edge in
// place using scratch. strength selects the filter strength (0-3) and
// bitDepth clips the output samples.
func FilterIntraEdge(edge []uint16, scratch []uint16, strength uint8, bitDepth uint8) error {
	return internalprediction.FilterIntraEdge(edge, scratch, strength, bitDepth)
}

// UpsampleIntraEdge upsamples size samples of edge starting at origin in
// place using scratch.
func UpsampleIntraEdge(edge []uint16, origin int, size int, scratch []uint16, bitDepth uint8) error {
	return internalprediction.UpsampleIntraEdge(edge, origin, size, scratch, bitDepth)
}

// FilterIntraEdgeCorner applies the AV1 corner-edge filter to the samples at
// (aboveOrigin, leftOrigin) in the above and left edge buffers.
func FilterIntraEdgeCorner(above []uint16, aboveOrigin int, left []uint16, leftOrigin int, bitDepth uint8) error {
	return internalprediction.FilterIntraEdgeCorner(above, aboveOrigin, left, leftOrigin, bitDepth)
}

// IntraEdgeFilterStrength returns the AV1 intra-edge filter strength (0-3)
// for the given neighboring block sizes, directional delta, and smooth-
// neighbor flag.
func IntraEdgeFilterStrength(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) uint8 {
	return internalprediction.IntraEdgeFilterStrength(blockSize0, blockSize1, delta, smoothNeighbor)
}

// UseIntraEdgeUpsample reports whether AV1 enables intra-edge upsampling for
// the given neighboring block sizes, directional delta, and smooth-neighbor
// flag.
func UseIntraEdgeUpsample(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) bool {
	return internalprediction.UseIntraEdgeUpsample(blockSize0, blockSize1, delta, smoothNeighbor)
}

// CFLAlphaQ3 decodes a Q3 chroma-from-luma alpha for predType from the
// transmitted alphaIndex and joint sign.
func CFLAlphaQ3(alphaIndex uint8, jointSign int8, predType PredictionCFLPredType) (int, error) {
	return internalprediction.CFLAlphaQ3(alphaIndex, jointSign, predType)
}

// SubsampleLuma8ToCFLQ3 subsamples an 8-bit luma reconstruction into the Q3
// chroma-from-luma buffer.
func SubsampleLuma8ToCFLQ3(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, subX bool, subY bool) error {
	return internalprediction.SubsampleLuma8ToQ3(outputQ3, input, inputStride, width, height, subX, subY)
}

// SubsampleLuma16ToCFLQ3 subsamples a high-bit-depth luma reconstruction into
// the Q3 chroma-from-luma buffer.
func SubsampleLuma16ToCFLQ3(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, subX bool, subY bool, bitDepth uint8) error {
	return internalprediction.SubsampleLuma16ToQ3(outputQ3, input, inputStride, width, height, subX, subY, bitDepth)
}

// PadCFLReconQ3 extends the Q3 reconstruction buffer to fill (bufWidth,
// bufHeight) by replicating the last (width, height) row/column.
func PadCFLReconQ3(reconQ3 []uint16, bufWidth int, bufHeight int, width int, height int) (int, int, error) {
	return internalprediction.PadCFLReconQ3(reconQ3, bufWidth, bufHeight, width, height)
}

// SubtractCFLAverage subtracts the mean of srcQ3 and writes the AC residual
// into dstQ3, producing the AC-only Q3 input used by PredictCFLPlaneBlock.
func SubtractCFLAverage(srcQ3 []uint16, dstQ3 []int16, width int, height int) error {
	return internalprediction.SubtractCFLAverage(srcQ3, dstQ3, width, height)
}

// PredictCFLPlaneBlock applies a chroma-from-luma prediction to dst using the
// AC residual acQ3 and the per-block alpha alphaQ3.
func PredictCFLPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, acQ3 []int16, alphaQ3 int) error {
	return internalprediction.PredictCFLPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, acQ3, alphaQ3)
}
