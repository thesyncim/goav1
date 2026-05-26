package goav1

// ResetTileDecodeState rebinds state to the supplied tile payload, job, and
// options without freeing or reallocating its caller-owned scratches. It is
// the entry point used between back-to-back tile decodes.
func ResetTileDecodeState(state *TileDecodeState, payload []byte, job TileJob, options TileDecodeOptions) error {
	return state.Reset(payload, job, options)
}

// InitTileCoeffCDFsDefault initializes cdfs with the AV1 default coefficient
// CDFs for the given base quantizer index.
func InitTileCoeffCDFsDefault(cdfs *TileCoeffCDFs, baseQIndex uint8) error {
	if cdfs == nil {
		return ErrTileInvalidDecodeState
	}
	return cdfs.InitDefault(baseQIndex)
}

// InitTileTransformCDFsDefault initializes cdfs with the AV1 default
// transform-type CDFs.
func InitTileTransformCDFsDefault(cdfs *TileTransformCDFs) error {
	if cdfs == nil {
		return ErrTileInvalidDecodeState
	}
	return cdfs.InitDefault()
}

// DecodeTileBlockCoefficients decodes the coefficients for one transform
// block described by req. transformCtx and coeffCtx are the caller-owned
// per-superblock contexts; scratch is the caller-owned per-block scratch.
// The visit callback is invoked once per decoded non-zero coefficient.
func DecodeTileBlockCoefficients(state *TileDecodeState, cdfs TileBlockCoeffCDFs, transformCtx *TileTransformContext, coeffCtx *TileCoeffEntropyContext, scratch *TileBlockCoeffScratch, req TileBlockCoeffRequest, visit TileBlockCoeffVisitor) (TileBlockCoeffResult, error) {
	if transformCtx == nil || coeffCtx == nil {
		return TileBlockCoeffResult{}, ErrTileInvalidDecodeState
	}
	return state.DecodeBlockCoefficients(cdfs, &transformCtx.context, &coeffCtx.context, scratch, req, visit)
}

// DecodeTileLumaCoefficients walks the luma coefficient tree for one block,
// invoking visit for each emitted coefficient and returning aggregate stats.
func DecodeTileLumaCoefficients(state *TileDecodeState, cdfs *TileCoeffCDFs, ctx *TileCoeffEntropyContext, scratch *TileCoeffTreeScratch, req TileLumaCoeffTreeRequest, visit TileLumaCoeffVisitor) (TileLumaCoeffStats, error) {
	if ctx == nil {
		return TileLumaCoeffStats{}, ErrTileInvalidDecodeState
	}
	return state.DecodeLumaCoefficients(cdfs, &ctx.context, scratch, req, visit)
}

// DecodeTileChromaCoefficients walks the chroma coefficient tree for one
// block, invoking visit for each emitted coefficient and returning aggregate
// stats.
func DecodeTileChromaCoefficients(state *TileDecodeState, cdfs *TileCoeffCDFs, ctx *TileCoeffEntropyContext, scratch *TileCoeffTreeScratch, req TileChromaCoeffTreeRequest, visit TileChromaCoeffVisitor) (TileLumaCoeffStats, error) {
	if ctx == nil {
		return TileLumaCoeffStats{}, ErrTileInvalidDecodeState
	}
	return state.DecodeChromaCoefficients(cdfs, &ctx.context, scratch, req, visit)
}
