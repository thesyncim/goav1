package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// inter_filter_write.go holds the forward of the switchable interpolation
// filter reader plus the frame-frozen rate tables the encoder's filter search
// prices with. The writer mirrors ReadInterpFilters' no-symbol cases exactly
// (libaom read_mb_interp_filter; SVT-AV1 entropy_coding.c write_mb_interp_filter),
// and the rate tables are the counterpart of SVT-AV1's
// switchable_interp_fac_bitss (Source/Lib/Codec/md_rate_estimation.c
// svt_aom_estimate_syntax_rate).

// InterpFilterRateTables holds the frame-frozen switchable filter symbol
// costs, the counterpart of SVT-AV1's switchable_interp_fac_bitss
// (Source/Lib/Codec/md_rate_estimation.c).
type InterpFilterRateTables struct {
	Switchable [SwitchableFilterContexts][switchableFilterCount]uint32
}

// EstimateRateTables freezes the switchable filter symbol costs from the
// current CDF state, the port of svt_aom_estimate_syntax_rate's
// switchable_interp_fac_bitss fill (Source/Lib/Codec/md_rate_estimation.c).
func (c *InterpFilterCDFs) EstimateRateTables(t *InterpFilterRateTables) {
	for i := range c.Switchable {
		for s := range switchableFilterCount {
			t.Switchable[i][s] = entropy.CostSymbol(&c.Switchable[i], s)
		}
	}
}

// SwitchableBits prices one switchable filter symbol under symbol context
// ctx, the rate term of SVT-AV1's svt_aom_get_switchable_rate
// (Source/Lib/Codec/rd_cost.c).
func (t *InterpFilterRateTables) SwitchableBits(ctx int, filter motion.InterpFilter) uint32 {
	if ctx < 0 || ctx >= SwitchableFilterContexts || int(filter) >= switchableFilterCount {
		return 0
	}
	return t.Switchable[ctx][filter]
}
