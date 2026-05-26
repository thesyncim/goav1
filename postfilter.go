package goav1

import (
	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// CDEFPlane identifies a luma (Y) or chroma (U/V) plane for CDEF filtering.
type CDEFPlane = internalcdef.Plane

// CDEFBlockPosition is the (column, row, secondary-strength) position of one
// 8x8 CDEF block inside a frame's CDEF grid.
type CDEFBlockPosition = internalcdef.BlockPosition

// CDEFDirectionGrid stores the per-8x8 CDEF directions chosen for a frame.
type CDEFDirectionGrid = internalcdef.DirectionGrid

// CDEFVarianceGrid stores the per-8x8 CDEF variance measurements computed
// during direction search.
type CDEFVarianceGrid = internalcdef.VarianceGrid

// CDEFBlockFilterParams parameterizes one CDEF block filter call: damping,
// strengths, shifts, and direction.
type CDEFBlockFilterParams = internalcdef.BlockFilterParams

// CDEFFrameFilterParams is the frame-level version of CDEFBlockFilterParams,
// produced by CDEFFrameFilterParamsFromStrength.
type CDEFFrameFilterParams = internalcdef.FrameFilterParams

// CDEF and loop-restoration constants. They define plane identifiers, block
// geometry, border sizes, scratch buffer dimensions, and the per-tap ranges
// used by the Wiener and self-guided (SGR) restoration filters.
const (
	CDEFPlaneY CDEFPlane = internalcdef.PlaneY
	CDEFPlaneU CDEFPlane = internalcdef.PlaneU
	CDEFPlaneV CDEFPlane = internalcdef.PlaneV

	CDEFBlockSize         = internalcdef.BlockSize
	CDEFMaxSuperblockSize = internalcdef.MaxSuperblockSize
	CDEFNBlocks           = internalcdef.NBlocks
	CDEFVerticalBorder    = internalcdef.VerticalBorder
	CDEFHorizontalBorder  = internalcdef.HorizontalBorder
	CDEFBStride           = internalcdef.BStride
	CDEFVeryLarge         = internalcdef.VeryLarge
	CDEFInputBufferSize   = internalcdef.InputBufferSize

	RestorationProcUnitSize = internalrestoration.ProcUnitSize

	RestorationWienerHalfwin    = internalrestoration.WienerHalfwin
	RestorationWienerWin        = internalrestoration.WienerWin
	RestorationWienerSubpelTaps = internalrestoration.WienerSubpelTaps
	RestorationWienerFilterBits = internalrestoration.WienerFilterBits
	RestorationWienerFilterStep = internalrestoration.WienerFilterStep
	RestorationWienerRound0Bits = internalrestoration.WienerRound0Bits
	RestorationWienerTap0Mid    = internalrestoration.WienerTap0Mid
	RestorationWienerTap1Mid    = internalrestoration.WienerTap1Mid
	RestorationWienerTap2Mid    = internalrestoration.WienerTap2Mid
	RestorationWienerTap0Bits   = internalrestoration.WienerTap0Bits
	RestorationWienerTap1Bits   = internalrestoration.WienerTap1Bits
	RestorationWienerTap2Bits   = internalrestoration.WienerTap2Bits
	RestorationWienerTap0Min    = internalrestoration.WienerTap0Min
	RestorationWienerTap1Min    = internalrestoration.WienerTap1Min
	RestorationWienerTap2Min    = internalrestoration.WienerTap2Min
	RestorationWienerTap0Max    = internalrestoration.WienerTap0Max
	RestorationWienerTap1Max    = internalrestoration.WienerTap1Max
	RestorationWienerTap2Max    = internalrestoration.WienerTap2Max

	RestorationSGRProjBorderVert = internalrestoration.SGRProjBorderVert
	RestorationSGRProjBorderHorz = internalrestoration.SGRProjBorderHorz
	RestorationSGRProjParams     = internalrestoration.SGRProjParams
	RestorationSGRProjPrjBits    = internalrestoration.SGRProjPrjBits
	RestorationSGRProjRstBits    = internalrestoration.SGRProjRstBits
	RestorationSGRProjSgrBits    = internalrestoration.SGRProjSgrBits
	RestorationSGRProjSgr        = internalrestoration.SGRProjSgr
	RestorationSGRProjPrjMin0    = internalrestoration.SGRProjPrjMin0
	RestorationSGRProjPrjMax0    = internalrestoration.SGRProjPrjMax0
	RestorationSGRProjPrjMin1    = internalrestoration.SGRProjPrjMin1
	RestorationSGRProjPrjMax1    = internalrestoration.SGRProjPrjMax1
	RestorationSGRProjPrjSubexpK = internalrestoration.SGRProjPrjSubexpK
)

// Post-filter errors. ErrCDEFInvalidCDEF is returned by the CDEF helpers and
// ErrRestorationInvalidInput by the Wiener and SGR helpers when their inputs
// fall outside the AV1 spec.
var (
	ErrCDEFInvalidCDEF         = internalcdef.ErrInvalidCDEF
	ErrRestorationInvalidInput = internalrestoration.ErrInvalidRestoration
)

// DecodeCDEFStrength splits a packed CDEF strength byte into its primary
// (level) and secondary fields.
func DecodeCDEFStrength(packed uint8) (level int, secondary int, err error) {
	return internalcdef.DecodeStrength(packed)
}

// CDEFFrameFilterParamsFromStrength derives frame-level CDEF filter
// parameters from a packed strength byte, taking subsampling, damping, and
// the high-bit-depth coefficient shift into account.
func CDEFFrameFilterParamsFromStrength(plane CDEFPlane, xDec int, yDec int, packed uint8, damping int, coeffShift int) (CDEFFrameFilterParams, error) {
	return internalcdef.FrameFilterParamsFromStrength(plane, xDec, yDec, packed, damping, coeffShift)
}

// FindCDEFDirection performs the AV1 CDEF direction search on one 8x8 block
// and returns the best direction (0-7) together with its score.
func FindCDEFDirection(img []uint16, stride int, coeffShift int) (int, int32, error) {
	return internalcdef.FindDirection(img, stride, coeffShift)
}

// FindCDEFDirectionDual performs CDEF direction search on two 8x8 blocks
// simultaneously, returning each block's direction and score.
func FindCDEFDirectionDual(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32, error) {
	return internalcdef.FindDirectionDual(img1, img2, stride, coeffShift)
}

// FilterCDEFBlock applies CDEF to one 8x8 block, reading input at inputOrigin
// and writing into dst at dstOrigin.
func FilterCDEFBlock(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params CDEFBlockFilterParams) error {
	return internalcdef.FilterBlock(dst, dstStride, dstOrigin, input, inputOrigin, params)
}

// FilterCDEFFrameBlocks applies CDEF to a list of blocks within one frame,
// using the supplied direction and variance grids to look up per-block
// strengths.
func FilterCDEFFrameBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []CDEFBlockPosition, directions *CDEFDirectionGrid, variances *CDEFVarianceGrid, params CDEFFrameFilterParams) error {
	return internalcdef.FilterFrameBlocks(dst, dstStride, input, inputOrigin, blocks, directions, variances, params)
}

// DefaultRestorationWienerInfo returns the AV1 default RestorationWienerInfo
// (filter taps at their midpoints).
func DefaultRestorationWienerInfo() RestorationWienerInfo {
	return internalrestoration.DefaultWienerInfo()
}

// NewRestorationWienerFilter constructs a RestorationWienerFilter from the
// three transmitted half-filter taps.
func NewRestorationWienerFilter(tap0 int16, tap1 int16, tap2 int16) RestorationWienerFilter {
	return internalrestoration.NewWienerFilter(tap0, tap1, tap2)
}

// RestorationWienerScratchLen reports the number of uint16 scratch entries
// required by ApplyRestorationWiener for a (width, height) unit.
func RestorationWienerScratchLen(width int, height int) (int, error) {
	return internalrestoration.WienerScratchLen(width, height)
}

// ApplyRestorationWiener applies the AV1 Wiener loop-restoration filter
// described by info to src, writing into dst. scratch is caller-owned and
// must be sized via RestorationWienerScratchLen.
func ApplyRestorationWiener(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, info RestorationWienerInfo, bitDepth uint8, scratch []uint16) error {
	return internalrestoration.ApplyWienerRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, info, bitDepth, scratch)
}

// RestorationSelfguidedScratchLen reports the number of int32 scratch entries
// required by ApplyRestorationSelfguided for a (width, height) unit.
func RestorationSelfguidedScratchLen(width int, height int) (int, error) {
	return internalrestoration.SelfguidedScratchLen(width, height)
}

// RestorationSGRParamsByIndex returns the AV1 self-guided restoration
// parameter set at index (0..15).
func RestorationSGRParamsByIndex(index int) (RestorationSGRParams, error) {
	if index < 0 || index >= len(internalrestoration.SGRParameterTable) {
		return RestorationSGRParams{}, ErrRestorationInvalidInput
	}
	return internalrestoration.SGRParameterTable[index], nil
}

// DecodeRestorationSGRXQ derives the (xq0, xq1) projection coefficients used
// by the self-guided restoration filter from the transmitted xqd pair and
// the active SGR parameter set.
func DecodeRestorationSGRXQ(xqd [2]int, params RestorationSGRParams) [2]int {
	return internalrestoration.DecodeSGRXQ(xqd, params)
}

// ApplyRestorationSelfguided applies the AV1 self-guided restoration filter
// described by (paramsIndex, xqd) to src, writing into dst. scratch is
// caller-owned and must be sized via RestorationSelfguidedScratchLen.
func ApplyRestorationSelfguided(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, paramsIndex int, xqd [2]int, bitDepth uint8, scratch []int32) error {
	return internalrestoration.ApplySelfguidedRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, paramsIndex, xqd, bitDepth, scratch)
}
