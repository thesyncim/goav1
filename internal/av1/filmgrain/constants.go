package filmgrain

const (
	GaussianBits        = 11
	GaussianSequenceLen = 1 << GaussianBits

	LumaGrainWidth   = 82
	LumaGrainHeight  = 73
	LumaGrainSamples = LumaGrainWidth * LumaGrainHeight

	LumaBlockSize         = 32
	NoiseStripeHeight     = 34
	LumaOverlapSamples    = 2
	LumaColumnScratchRows = NoiseStripeHeight
)
