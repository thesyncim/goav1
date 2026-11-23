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
