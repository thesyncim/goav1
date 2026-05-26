package goav1

import internalsuperres "github.com/thesyncim/goav1/internal/av1/superres"

// Super-resolution upscaler constants. SuperResNum is the numerator used for
// the scale ratio; SuperResDenomMin and SuperResDenomMax bound the allowed
// denominator, and the remaining constants describe the 8-tap subpel filter
// (bit depth, shift count, tap count, offset, scale bits, scale mask, and
// extra fractional bits) the AV1 specification uses for horizontal upscaling.
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

// UpscaleSuperResPlane horizontally upscales src into dst using the AV1
// super-resolution 8-tap subpel filter. src and dst must reference buffers
// sized for the coded and upscaled widths respectively; bitDepth selects the
// 8/10/12-bit clipping range applied to the final samples.
func UpscaleSuperResPlane(src FrameSamplePlane, dst FrameSamplePlane, bitDepth uint8) error {
	return internalsuperres.UpscalePlane(src, dst, bitDepth)
}
