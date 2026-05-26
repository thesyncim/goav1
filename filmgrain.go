package goav1

import internalfilmgrain "github.com/thesyncim/goav1/internal/av1/filmgrain"

// FilmGrainRandom is the AV1 film-grain PRNG state used to seed grain
// generation for each frame, plane, and stripe.
type FilmGrainRandom = internalfilmgrain.Random

// FilmGrainScalingPoint is one (value, scale) entry of an AV1 film-grain
// piecewise-linear scaling function.
type FilmGrainScalingPoint = internalfilmgrain.ScalingPoint

// FilmGrainLumaGrainParams parameterizes one call to GenerateFilmGrainLuma:
// AR coefficients, AR shift, scaling shift, seed, and overlap flag.
type FilmGrainLumaGrainParams = internalfilmgrain.LumaGrainParams

// FilmGrainChromaGrainParams parameterizes one call to
// GenerateFilmGrainChroma: AR coefficients, AR shift, scaling shift, seed,
// subsampling, and overlap flag.
type FilmGrainChromaGrainParams = internalfilmgrain.ChromaGrainParams

// FilmGrainLumaRowParams parameterizes one call to ApplyFilmGrainLumaRow:
// row geometry, bit depth, scaling shift, and clip flags.
type FilmGrainLumaRowParams = internalfilmgrain.LumaRowParams

// FilmGrainChromaRowParams parameterizes one call to ApplyFilmGrainChromaRow:
// chroma row geometry, bit depth, scaling shift, subsampling, multipliers,
// and clip flags.
type FilmGrainChromaRowParams = internalfilmgrain.ChromaRowParams

// AV1 film-grain constants. These bound the grain block geometry, scaling LUT
// size, AR-coefficient counts, seed-XOR values, and legal sample ranges
// defined by section 7.20 of the AV1 specification.
const (
	FilmGrainGaussianBits        = internalfilmgrain.GaussianBits
	FilmGrainGaussianSequenceLen = internalfilmgrain.GaussianSequenceLen

	FilmGrainLumaGrainWidth   = internalfilmgrain.LumaGrainWidth
	FilmGrainLumaGrainHeight  = internalfilmgrain.LumaGrainHeight
	FilmGrainLumaGrainSamples = internalfilmgrain.LumaGrainSamples

	FilmGrainChromaGrainWidth            = internalfilmgrain.ChromaGrainWidth
	FilmGrainChromaGrainHeight           = internalfilmgrain.ChromaGrainHeight
	FilmGrainChromaGrainSamples          = internalfilmgrain.ChromaGrainSamples
	FilmGrainChromaSubsampledGrainWidth  = internalfilmgrain.ChromaSubsampledGrainWidth
	FilmGrainChromaSubsampledGrainHeight = internalfilmgrain.ChromaSubsampledGrainHeight
	FilmGrainChromaPlaneCb               = internalfilmgrain.ChromaPlaneCb
	FilmGrainChromaPlaneCr               = internalfilmgrain.ChromaPlaneCr
	FilmGrainMaxChromaARCoeffs           = internalfilmgrain.MaxChromaARCoeffs
	FilmGrainMaxLumaScalingPoints        = internalfilmgrain.MaxLumaScalingPoints
	FilmGrainMaxLumaARCoeffs             = internalfilmgrain.MaxLumaARCoeffs

	FilmGrainLumaBlockSize         = internalfilmgrain.LumaBlockSize
	FilmGrainNoiseStripeHeight     = internalfilmgrain.NoiseStripeHeight
	FilmGrainLumaOverlapSamples    = internalfilmgrain.LumaOverlapSamples
	FilmGrainLumaColumnScratchRows = internalfilmgrain.LumaColumnScratchRows

	FilmGrainCbSeedXor = internalfilmgrain.CbSeedXor
	FilmGrainCrSeedXor = internalfilmgrain.CrSeedXor

	FilmGrainScalingLUTSize = internalfilmgrain.ScalingLUTSize

	FilmGrainLumaLegalMin      = internalfilmgrain.LumaLegalMin
	FilmGrainLumaLegalMax      = internalfilmgrain.LumaLegalMax
	FilmGrainChromaLegalMax    = internalfilmgrain.ChromaLegalMax
	FilmGrainChromaIdentityMax = internalfilmgrain.ChromaIdentityMax
)

// ErrFilmGrainInvalidParams is returned by the film-grain helpers when their
// parameters fall outside the AV1 spec ranges.
var ErrFilmGrainInvalidParams = internalfilmgrain.ErrInvalidParams

// FilmGrainGaussian returns the AV1 Gaussian-sequence sample at index.
func FilmGrainGaussian(index uint16) int16 {
	return internalfilmgrain.Gaussian(index)
}

// NewFilmGrainRandom returns a FilmGrainRandom seeded with seed. It is the
// frame-level seed entry point.
func NewFilmGrainRandom(seed uint16) FilmGrainRandom {
	return internalfilmgrain.NewRandom(seed)
}

// NewFilmGrainPlaneRandom returns a FilmGrainRandom for a chroma plane,
// XOR-mixing the per-plane seed (FilmGrainCbSeedXor or FilmGrainCrSeedXor).
func NewFilmGrainPlaneRandom(seed uint16, plane int) (FilmGrainRandom, error) {
	return internalfilmgrain.NewPlaneRandom(seed, plane)
}

// NewFilmGrainStripeRandom returns a FilmGrainRandom seeded for the luma
// stripe starting at lumaLine.
func NewFilmGrainStripeRandom(seed uint16, lumaLine int) (FilmGrainRandom, error) {
	return internalfilmgrain.NewStripeRandom(seed, lumaLine)
}

// BuildFilmGrainScalingLUT fills dst with the AV1 piecewise-linear scaling
// LUT derived from points.
func BuildFilmGrainScalingLUT(dst []uint8, points []FilmGrainScalingPoint) error {
	return internalfilmgrain.BuildScalingLUT(dst, points)
}

// FilmGrainScaleLUT looks up the scaling value at index in lut, adjusting for
// bitDepth (8/10/12).
func FilmGrainScaleLUT(lut []uint8, index int, bitDepth uint8) (uint8, error) {
	return internalfilmgrain.ScaleLUT(lut, index, bitDepth)
}

// GenerateFilmGrainLuma fills dst with one frame's worth of luma grain
// samples using params.
func GenerateFilmGrainLuma(dst []int16, params FilmGrainLumaGrainParams) error {
	return internalfilmgrain.GenerateLumaGrain(dst, params)
}

// GenerateFilmGrainChroma fills dst with one frame's worth of chroma grain
// samples, optionally referencing the previously generated luma grain.
func GenerateFilmGrainChroma(dst []int16, luma []int16, params FilmGrainChromaGrainParams) error {
	return internalfilmgrain.GenerateChromaGrain(dst, luma, params)
}

// ApplyFilmGrainLumaRow applies one row of grain samples to src and writes
// the result into dst.
func ApplyFilmGrainLumaRow(dst []uint16, src []uint16, grain []int16, scaling []uint8, params FilmGrainLumaRowParams) error {
	return internalfilmgrain.ApplyLumaRow(dst, src, grain, scaling, params)
}

// ApplyFilmGrainChromaRow applies one row of chroma grain samples to src and
// writes the result into dst, using the matching luma row for scaling.
func ApplyFilmGrainChromaRow(dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, params FilmGrainChromaRowParams) error {
	return internalfilmgrain.ApplyChromaRow(dst, src, luma, grain, scaling, params)
}

// FilmGrainLumaSample returns the luma grain sample at (blockCol, blockRow,
// x, y) using the supplied per-block offset.
func FilmGrainLumaSample(grain []int16, offset uint8, blockCol int, blockRow int, x int, y int) (int16, error) {
	return internalfilmgrain.LumaGrainSample(grain, offset, blockCol, blockRow, x, y)
}

// FilmGrainChromaSample returns the chroma grain sample at (blockCol,
// blockRow, x, y), honoring the chroma subsampling flags.
func FilmGrainChromaSample(grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int) (int16, error) {
	return internalfilmgrain.ChromaGrainSample(grain, offset, subsamplingX, subsamplingY, blockCol, blockRow, x, y)
}

// BlendFilmGrainLumaOverlap blends a previously emitted grain sample with the
// current one at row offset, returning the AV1 weighted sum.
func BlendFilmGrainLumaOverlap(previous int16, current int16, offset int, bitDepth uint8) (int16, error) {
	return internalfilmgrain.BlendLumaOverlap(previous, current, offset, bitDepth)
}

// ApplyFilmGrainLumaSample combines one input sample with its grain
// contribution scaled through lut, clipping to bitDepth and (optionally) the
// AV1 restricted range.
func ApplyFilmGrainLumaSample(orig uint16, grain int16, lut []uint8, bitDepth uint8, scalingShift uint8, restrictedRange bool) (uint16, error) {
	return internalfilmgrain.ApplyLumaSample(orig, grain, lut, bitDepth, scalingShift, restrictedRange)
}
