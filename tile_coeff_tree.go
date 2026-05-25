package goav1

func ResetTileDecodeState(state *TileDecodeState, payload []byte, job TileJob, options TileDecodeOptions) error {
	return state.Reset(payload, job, options)
}

func InitTileCoeffCDFsDefault(cdfs *TileCoeffCDFs, baseQIndex uint8) error {
	if cdfs == nil {
		return ErrTileInvalidDecodeState
	}
	return cdfs.InitDefault(baseQIndex)
}

func DecodeTileLumaCoefficients(state *TileDecodeState, cdfs *TileCoeffCDFs, ctx *TileCoeffEntropyContext, scratch *TileCoeffTreeScratch, req TileLumaCoeffTreeRequest, visit TileLumaCoeffVisitor) (TileLumaCoeffStats, error) {
	if ctx == nil {
		return TileLumaCoeffStats{}, ErrTileInvalidDecodeState
	}
	return state.DecodeLumaCoefficients(cdfs, &ctx.context, scratch, req, visit)
}

func DecodeTileChromaCoefficients(state *TileDecodeState, cdfs *TileCoeffCDFs, ctx *TileCoeffEntropyContext, scratch *TileCoeffTreeScratch, req TileChromaCoeffTreeRequest, visit TileChromaCoeffVisitor) (TileLumaCoeffStats, error) {
	if ctx == nil {
		return TileLumaCoeffStats{}, ErrTileInvalidDecodeState
	}
	return state.DecodeChromaCoefficients(cdfs, &ctx.context, scratch, req, visit)
}
