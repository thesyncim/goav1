package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// inter_rate.go estimates the bit cost of the inter mode-symbol writers, in
// 512-per-bit units. It is the rate half of SVT-AV1's
// av1_inter_fast_cost_light (Source/Lib/Codec/rd_cost.c): the single-reference
// new/global/ref-mv cascade and the DRL continuation bits. As upstream does
// with md_rate_est_ctx (Source/Lib/Codec/md_rate_estimation.c
// svt_aom_estimate_syntax_rate), the costs are frozen into per-frame tables
// from the frame-start CDFs rather than read from the adapting CDFs, so the
// mode decision cannot feed back into its own pricing mid-frame.

// InterModeRateTables holds the frame-frozen mode and DRL symbol costs, the
// counterpart of SVT-AV1's new_mv_mode_fac_bits / zero_mv_mode_fac_bits /
// ref_mv_mode_fac_bits / drl_mode_fac_bits.
type InterModeRateTables struct {
	NewMV    [NewMVModeContexts][2]uint32
	GlobalMV [GlobalMVModeContexts][2]uint32
	RefMV    [RefMVModeContexts][2]uint32
	DRL      [DRLModeContexts][2]uint32
}

// EstimateRateTables freezes the mode and DRL symbol costs from the current
// CDF state, the port of svt_aom_estimate_syntax_rate's
// svt_aom_get_syntax_rate_from_cdf calls over these CDFs
// (Source/Lib/Codec/md_rate_estimation.c).
func (c *InterModeCDFs) EstimateRateTables(t *InterModeRateTables) {
	for i := range c.NewMV {
		for s := range 2 {
			t.NewMV[i][s] = entropy.CostSymbol(&c.NewMV[i], s)
		}
	}
	for i := range c.GlobalMV {
		for s := range 2 {
			t.GlobalMV[i][s] = entropy.CostSymbol(&c.GlobalMV[i], s)
		}
	}
	for i := range c.RefMV {
		for s := range 2 {
			t.RefMV[i][s] = entropy.CostSymbol(&c.RefMV[i], s)
		}
	}
	for i := range c.DRL {
		for s := range 2 {
			t.DRL[i][s] = entropy.CostSymbol(&c.DRL[i], s)
		}
	}
}

// SingleInterModeBits prices WriteSingleInterMode's cascade for mode under
// modeContext (the raw stack mode context, as WriteBlockInterMode receives
// it). Port of the single-ref mode cascade in SVT-AV1's
// av1_inter_fast_cost_light (Source/Lib/Codec/rd_cost.c).
func (t *InterModeRateTables) SingleInterModeBits(modeContext uint16, mode InterMode) uint32 {
	newCtx := int(modeContext & newMVMask)
	if newCtx >= NewMVModeContexts {
		newCtx = NewMVModeContexts - 1
	}
	total := t.NewMV[newCtx][boolToSym(mode != InterModeNewMV)]
	if mode == InterModeNewMV {
		return total
	}
	globalCtx := int((modeContext >> globalMVOffset) & globalMVMask)
	total += t.GlobalMV[globalCtx][boolToSym(mode != InterModeGlobalMV)]
	if mode == InterModeGlobalMV {
		return total
	}
	refCtx := int((modeContext >> refMVOffset) & refMVMask)
	if refCtx >= RefMVModeContexts {
		refCtx = RefMVModeContexts - 1
	}
	total += t.RefMV[refCtx][boolToSym(mode != InterModeNearestMV)]
	return total
}

// DRLIndexBits prices WriteDRLIndex's continuation bits for refMVIndex,
// mirroring that writer's slot walk symbol for symbol. Port of the
// drl_mode_fac_bits loops in SVT-AV1's av1_inter_fast_cost_light
// (Source/Lib/Codec/rd_cost.c).
func (t *InterModeRateTables) DRLIndexBits(req DRLRequest, refMVIndex int) uint32 {
	if refMVIndex < 0 || req.RefMVCount > MaxRefMVStackSize {
		return 0
	}
	refMVCount := int(req.RefMVCount)
	var total uint32
	if req.usesNewMV() {
		for idx := range 2 {
			if refMVCount <= idx+1 {
				return total
			}
			bit := refMVIndex > idx
			total += t.DRL[int(req.Contexts[idx])%DRLModeContexts][boolToSym(bit)]
			if !bit {
				return total
			}
		}
		return total
	}
	if req.usesNearMV() {
		for idx := 1; idx < 3; idx++ {
			if refMVCount <= idx+1 {
				return total
			}
			bit := refMVIndex > idx-1
			total += t.DRL[int(req.Contexts[idx])%DRLModeContexts][boolToSym(bit)]
			if !bit {
				return total
			}
		}
		return total
	}
	return total
}
