package goav1

// DecoderFrameWorkCoeffReconstructionContext carries the per-block frame-work
// state needed to place a coefficient replay block into the output frame.
type DecoderFrameWorkCoeffReconstructionContext struct {
	Visit TileBlockVisit

	CurrentQIndex uint8
	SegmentID     uint8

	Int32Scratch    []int32
	ResidualScratch []int16
}

func DecoderFrameWorkLumaCoeffBlockReconstruction(ctx DecoderFrameWorkCoeffReconstructionContext, block TileLumaCoeffBlock) DecoderFrameWorkBlockCoeffReconstruction {
	return DecoderFrameWorkBlockCoeffReconstruction{
		Visit: ctx.Visit,
		Block: TileBlockCoeffBlock{
			Plane:     0,
			Block:     block.Block,
			Transform: block.Transform,
			Result:    block.Result,
			Coeffs:    block.Coeffs,
			Scan:      block.Scan,
		},
		Transform:       block.Transform,
		CurrentQIndex:   ctx.CurrentQIndex,
		SegmentID:       ctx.SegmentID,
		Int32Scratch:    ctx.Int32Scratch,
		ResidualScratch: ctx.ResidualScratch,
	}
}

func DecoderFrameWorkChromaCoeffBlockReconstruction(ctx DecoderFrameWorkCoeffReconstructionContext, block TileChromaCoeffBlock) DecoderFrameWorkBlockCoeffReconstruction {
	return DecoderFrameWorkBlockCoeffReconstruction{
		Visit: ctx.Visit,
		Block: TileBlockCoeffBlock{
			Plane:     block.Plane,
			Block:     block.Block,
			Transform: block.Transform,
			Result:    block.Result,
			Coeffs:    block.Coeffs,
			Scan:      block.Scan,
		},
		Transform:       block.Transform,
		CurrentQIndex:   ctx.CurrentQIndex,
		SegmentID:       ctx.SegmentID,
		Int32Scratch:    ctx.Int32Scratch,
		ResidualScratch: ctx.ResidualScratch,
	}
}

func ReconstructDecoderFrameWorkLumaCoeffBlock(batch DecoderFrameWorkBatch, index int, ctx DecoderFrameWorkCoeffReconstructionContext, block TileLumaCoeffBlock) error {
	return ReconstructDecoderFrameWorkBlockCoeff(batch, index, DecoderFrameWorkLumaCoeffBlockReconstruction(ctx, block))
}

func ReconstructDecoderFrameWorkChromaCoeffBlock(batch DecoderFrameWorkBatch, index int, ctx DecoderFrameWorkCoeffReconstructionContext, block TileChromaCoeffBlock) error {
	return ReconstructDecoderFrameWorkBlockCoeff(batch, index, DecoderFrameWorkChromaCoeffBlockReconstruction(ctx, block))
}
