// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

const (
	GaussianBits        = 11
	GaussianSequenceLen = 1 << GaussianBits

	LumaGrainWidth   = 82
	LumaGrainHeight  = 73
	LumaGrainSamples = LumaGrainWidth * LumaGrainHeight

	ChromaGrainWidth            = LumaGrainWidth
	ChromaGrainHeight           = LumaGrainHeight
	ChromaGrainSamples          = ChromaGrainWidth * ChromaGrainHeight
	ChromaSubsampledGrainWidth  = 44
	ChromaSubsampledGrainHeight = 38
	ChromaPlaneCb               = 1
	ChromaPlaneCr               = 2
	MaxChromaARCoeffs           = 25
	MaxLumaScalingPoints        = 14
	MaxLumaARCoeffs             = 24

	LumaBlockSize         = 32
	NoiseStripeHeight     = 34
	LumaOverlapSamples    = 2
	LumaColumnScratchRows = NoiseStripeHeight
)
