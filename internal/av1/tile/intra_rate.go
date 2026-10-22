package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// IntraDCRateTables holds the frame-frozen symbol costs, in 512-per-bit
// units, that SVT-AV1's svt_aom_intra_fast_cost (Source/Lib/Codec/rd_cost.c)
// charges a DC_PRED candidate in an inter frame: the intra/inter flag
// (intra_inter_fac_bits), the inter-frame luma mode by size group
// (mb_mode_fac_bits) and the DC chroma mode (intra_uv_mode_fac_bits through
// svt_aom_get_intra_uv_fast_rate). As with InterModeRateTables the costs come
// from the frame-start CDFs (svt_aom_estimate_syntax_rate).
type IntraDCRateTables struct {
	IntraFlag [LumaIntraModeContexts][2]uint32
	YModeDC   [LumaIntraModeContexts]uint32
	UVModeDC  [2]uint32
}

// EstimateIntraDCRateTables freezes the DC-candidate symbol costs from the
// current CDF state.
func (c *IntraModeCDFs) EstimateIntraDCRateTables(t *IntraDCRateTables) error {
	for ctx := range t.IntraFlag {
		cdf, err := c.IntraCDF(ctx)
		if err != nil {
			return err
		}
		// WriteIntraFlag codes symbol 1 for inter, 0 for intra.
		t.IntraFlag[ctx][0] = entropy.CostSymbol(cdf, 0)
		t.IntraFlag[ctx][1] = entropy.CostSymbol(cdf, 1)
	}
	for ctx := range t.YModeDC {
		cdf, err := c.YModeCDF(ctx)
		if err != nil {
			return err
		}
		t.YModeDC[ctx] = entropy.CostSymbol(cdf, int(IntraModeDC))
	}
	for allowed := range t.UVModeDC {
		cdf, err := c.UVModeCDF(allowed == 1, IntraModeDC)
		if err != nil {
			return err
		}
		t.UVModeDC[allowed] = entropy.CostSymbol(cdf, int(ChromaIntraModeDC))
	}
	return nil
}

// LumaDCBits prices the inter-frame DC luma mode symbol for a block size.
func (t *IntraDCRateTables) LumaDCBits(size BlockSize) uint32 {
	if int(size) >= len(yModeSizeContext) {
		return 0
	}
	return t.YModeDC[yModeSizeContext[size]]
}
