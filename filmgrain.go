package goav1

import internalfilmgrain "github.com/thesyncim/goav1/internal/av1/filmgrain"

type FilmGrainRandom = internalfilmgrain.Random
type FilmGrainScalingPoint = internalfilmgrain.ScalingPoint
type FilmGrainLumaGrainParams = internalfilmgrain.LumaGrainParams
type FilmGrainChromaGrainParams = internalfilmgrain.ChromaGrainParams
type FilmGrainLumaRowParams = internalfilmgrain.LumaRowParams
type FilmGrainChromaRowParams = internalfilmgrain.ChromaRowParams

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

var ErrFilmGrainInvalidParams = internalfilmgrain.ErrInvalidParams

func FilmGrainGaussian(index uint16) int16 {
	return internalfilmgrain.Gaussian(index)
}

func NewFilmGrainRandom(seed uint16) FilmGrainRandom {
	return internalfilmgrain.NewRandom(seed)
}

func NewFilmGrainPlaneRandom(seed uint16, plane int) (FilmGrainRandom, error) {
	return internalfilmgrain.NewPlaneRandom(seed, plane)
}

func NewFilmGrainStripeRandom(seed uint16, lumaLine int) (FilmGrainRandom, error) {
	return internalfilmgrain.NewStripeRandom(seed, lumaLine)
}

func BuildFilmGrainScalingLUT(dst []uint8, points []FilmGrainScalingPoint) error {
	return internalfilmgrain.BuildScalingLUT(dst, points)
}

func FilmGrainScaleLUT(lut []uint8, index int, bitDepth uint8) (uint8, error) {
	return internalfilmgrain.ScaleLUT(lut, index, bitDepth)
}

func GenerateFilmGrainLuma(dst []int16, params FilmGrainLumaGrainParams) error {
	return internalfilmgrain.GenerateLumaGrain(dst, params)
}

func GenerateFilmGrainChroma(dst []int16, luma []int16, params FilmGrainChromaGrainParams) error {
	return internalfilmgrain.GenerateChromaGrain(dst, luma, params)
}

func ApplyFilmGrainLumaRow(dst []uint16, src []uint16, grain []int16, scaling []uint8, params FilmGrainLumaRowParams) error {
	return internalfilmgrain.ApplyLumaRow(dst, src, grain, scaling, params)
}

func ApplyFilmGrainChromaRow(dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, params FilmGrainChromaRowParams) error {
	return internalfilmgrain.ApplyChromaRow(dst, src, luma, grain, scaling, params)
}

func FilmGrainLumaSample(grain []int16, offset uint8, blockCol int, blockRow int, x int, y int) (int16, error) {
	return internalfilmgrain.LumaGrainSample(grain, offset, blockCol, blockRow, x, y)
}

func FilmGrainChromaSample(grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int) (int16, error) {
	return internalfilmgrain.ChromaGrainSample(grain, offset, subsamplingX, subsamplingY, blockCol, blockRow, x, y)
}

func BlendFilmGrainLumaOverlap(previous int16, current int16, offset int, bitDepth uint8) (int16, error) {
	return internalfilmgrain.BlendLumaOverlap(previous, current, offset, bitDepth)
}

func ApplyFilmGrainLumaSample(orig uint16, grain int16, lut []uint8, bitDepth uint8, scalingShift uint8, restrictedRange bool) (uint16, error) {
	return internalfilmgrain.ApplyLumaSample(orig, grain, lut, bitDepth, scalingShift, restrictedRange)
}
