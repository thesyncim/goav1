package goav1

import (
	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

type CDEFPlane = internalcdef.Plane
type CDEFBlockPosition = internalcdef.BlockPosition
type CDEFDirectionGrid = internalcdef.DirectionGrid
type CDEFVarianceGrid = internalcdef.VarianceGrid
type CDEFBlockFilterParams = internalcdef.BlockFilterParams
type CDEFFrameFilterParams = internalcdef.FrameFilterParams

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

var (
	ErrCDEFInvalidCDEF         = internalcdef.ErrInvalidCDEF
	ErrRestorationInvalidInput = internalrestoration.ErrInvalidRestoration
)

func DecodeCDEFStrength(packed uint8) (level int, secondary int, err error) {
	return internalcdef.DecodeStrength(packed)
}

func CDEFFrameFilterParamsFromStrength(plane CDEFPlane, xDec int, yDec int, packed uint8, damping int, coeffShift int) (CDEFFrameFilterParams, error) {
	return internalcdef.FrameFilterParamsFromStrength(plane, xDec, yDec, packed, damping, coeffShift)
}

func FindCDEFDirection(img []uint16, stride int, coeffShift int) (int, int32, error) {
	return internalcdef.FindDirection(img, stride, coeffShift)
}

func FindCDEFDirectionDual(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32, error) {
	return internalcdef.FindDirectionDual(img1, img2, stride, coeffShift)
}

func FilterCDEFBlock(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params CDEFBlockFilterParams) error {
	return internalcdef.FilterBlock(dst, dstStride, dstOrigin, input, inputOrigin, params)
}

func FilterCDEFFrameBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []CDEFBlockPosition, directions *CDEFDirectionGrid, variances *CDEFVarianceGrid, params CDEFFrameFilterParams) error {
	return internalcdef.FilterFrameBlocks(dst, dstStride, input, inputOrigin, blocks, directions, variances, params)
}

func DefaultRestorationWienerInfo() RestorationWienerInfo {
	return internalrestoration.DefaultWienerInfo()
}

func NewRestorationWienerFilter(tap0 int16, tap1 int16, tap2 int16) RestorationWienerFilter {
	return internalrestoration.NewWienerFilter(tap0, tap1, tap2)
}

func RestorationWienerScratchLen(width int, height int) (int, error) {
	return internalrestoration.WienerScratchLen(width, height)
}

func ApplyRestorationWiener(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, info RestorationWienerInfo, bitDepth uint8, scratch []uint16) error {
	return internalrestoration.ApplyWienerRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, info, bitDepth, scratch)
}

func RestorationSelfguidedScratchLen(width int, height int) (int, error) {
	return internalrestoration.SelfguidedScratchLen(width, height)
}

func RestorationSGRParamsByIndex(index int) (RestorationSGRParams, error) {
	if index < 0 || index >= len(internalrestoration.SGRParameterTable) {
		return RestorationSGRParams{}, ErrRestorationInvalidInput
	}
	return internalrestoration.SGRParameterTable[index], nil
}

func DecodeRestorationSGRXQ(xqd [2]int, params RestorationSGRParams) [2]int {
	return internalrestoration.DecodeSGRXQ(xqd, params)
}

func ApplyRestorationSelfguided(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, paramsIndex int, xqd [2]int, bitDepth uint8, scratch []int32) error {
	return internalrestoration.ApplySelfguidedRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, paramsIndex, xqd, bitDepth, scratch)
}
