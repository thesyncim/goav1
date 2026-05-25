package goav1

import internalsuperres "github.com/thesyncim/goav1/internal/av1/superres"

const (
	SuperResNum          = internalsuperres.Num
	SuperResDenomMin     = internalsuperres.DenomMin
	SuperResDenomMax     = internalsuperres.DenomMax
	SuperResFilterBits   = internalsuperres.FilterBits
	SuperResFilterShifts = internalsuperres.FilterShifts
	SuperResFilterTaps   = internalsuperres.FilterTaps
	SuperResFilterOffset = internalsuperres.FilterOffset
	SuperResScaleBits    = internalsuperres.ScaleBits
	SuperResScaleMask    = internalsuperres.ScaleMask
	SuperResExtraBits    = internalsuperres.ExtraBits
)

func UpscaleSuperResPlane(src FrameSamplePlane, dst FrameSamplePlane, bitDepth uint8) error {
	return internalsuperres.UpscalePlane(src, dst, bitDepth)
}
