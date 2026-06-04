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

// DecoderFrameWorkBlockCoeffDecodeRequest composes one block coefficient decode
// request with the frame-work reconstruction context for the same block.
type DecoderFrameWorkBlockCoeffDecodeRequest struct {
	Decode         TileBlockCoeffRequest
	Reconstruction DecoderFrameWorkCoeffReconstructionContext
}

// DecoderFrameWorkLumaCoeffBlockReconstruction packages a decoded luma
// coefficient block together with the reconstruction context required to
// apply it to the output frame.
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

// DecoderFrameWorkChromaCoeffBlockReconstruction packages a decoded chroma
// coefficient block together with the reconstruction context required to
// apply it to the output frame.
func DecoderFrameWorkChromaCoeffBlockReconstruction(ctx DecoderFrameWorkCoeffReconstructionContext, block TileChromaCoeffBlock) DecoderFrameWorkBlockCoeffReconstruction {
	return DecoderFrameWorkBlockCoeffReconstruction{
		Visit: ctx.Visit,
		Block: TileBlockCoeffBlock{
			Plane:     uint8(block.Plane),
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

// ReconstructDecoderFrameWorkLumaCoeffBlock reconstructs a single luma
// coefficient block into the output frame for the batch entry at index.
func ReconstructDecoderFrameWorkLumaCoeffBlock(batch DecoderFrameWorkBatch, index int, ctx DecoderFrameWorkCoeffReconstructionContext, block TileLumaCoeffBlock) error {
	return ReconstructDecoderFrameWorkBlockCoeff(batch, index, DecoderFrameWorkLumaCoeffBlockReconstruction(ctx, block))
}

// ReconstructDecoderFrameWorkChromaCoeffBlock reconstructs a single chroma
// coefficient block into the output frame for the batch entry at index.
func ReconstructDecoderFrameWorkChromaCoeffBlock(batch DecoderFrameWorkBatch, index int, ctx DecoderFrameWorkCoeffReconstructionContext, block TileChromaCoeffBlock) error {
	return ReconstructDecoderFrameWorkBlockCoeff(batch, index, DecoderFrameWorkChromaCoeffBlockReconstruction(ctx, block))
}

// DecodeAndReconstructDecoderFrameWorkBlockCoefficients decodes the block
// coefficients described by req.Decode and immediately reconstructs each
// emitted coefficient block into the output frame, then forwards the block
// to visit. It fuses the two decode/reconstruct steps for callers that do
// not need to separate them.
func DecodeAndReconstructDecoderFrameWorkBlockCoefficients(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, cdfs TileBlockCoeffCDFs, transformCtx *TileTransformContext, coeffCtx *TileCoeffEntropyContext, scratch *TileBlockCoeffScratch, req DecoderFrameWorkBlockCoeffDecodeRequest, visit TileBlockCoeffVisitor) (TileBlockCoeffResult, error) {
	return DecodeTileBlockCoefficients(state, cdfs, transformCtx, coeffCtx, scratch, req.Decode, func(block TileBlockCoeffBlock) error {
		if err := ReconstructDecoderFrameWorkBlockCoeff(batch, index, DecoderFrameWorkBlockCoeffReconstruction{
			Visit:           req.Reconstruction.Visit,
			Block:           block,
			Transform:       block.Transform,
			CurrentQIndex:   req.Reconstruction.CurrentQIndex,
			SegmentID:       req.Reconstruction.SegmentID,
			Int32Scratch:    req.Reconstruction.Int32Scratch,
			ResidualScratch: req.Reconstruction.ResidualScratch,
		}); err != nil {
			return err
		}
		if visit != nil {
			return visit(block)
		}
		return nil
	})
}
