package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// inter_filter_write.go holds the forward of the switchable interpolation
// filter reader plus the frame-frozen rate tables the encoder's filter search
// prices with. The writer mirrors ReadInterpFilters' no-symbol cases exactly
// (libaom read_mb_interp_filter; SVT-AV1 entropy_coding.c write_mb_interp_filter),
// and the rate tables are the counterpart of SVT-AV1's
// switchable_interp_fac_bitss (Source/Lib/Codec/md_rate_estimation.c
// svt_aom_estimate_syntax_rate).

// WriteInterpFilters codes the block's interpolation filter pair, the forward
// of ReadInterpFilters. Blocks whose filters the decoder infers (skip mode,
// warp, non-translational global motion, or a fixed frame-level filter) code
// no symbols; the caller-supplied filters must equal the decoder's inference
// so the mark stays in sync. It returns the number of switchable symbols
// written.
func WriteInterpFilters(w *entropy.Writer, cdfs *InterpFilterCDFs, ctx *BlockModeContext, req InterpFilterRequest, filters motion.InterpFilters) (int, error) {
	if w == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateInterpFilterRequest(req); err != nil {
		return 0, err
	}
	if !interpNeeded(req) {
		filter, err := unswitchableMotionFilter(req.FrameFilter)
		if err != nil {
			return 0, err
		}
		if filters.X != filter || filters.Y != filter {
			return 0, ErrInvalidDecodeState
		}
		return 0, nil
	}
	if req.FrameFilter != parser.InterpolationSwitchable {
		filter, err := fixedMotionFilter(req.FrameFilter)
		if err != nil {
			return 0, err
		}
		if filters.X != filter || filters.Y != filter {
			return 0, ErrInvalidDecodeState
		}
		return 0, nil
	}
	if !switchableMotionFilter(filters.Y) || !switchableMotionFilter(filters.X) {
		return 0, ErrInvalidDecodeState
	}
	if err := writeSwitchableInterpFilter(w, cdfs, ctx, req, 0, filters.Y); err != nil {
		return 0, err
	}
	writes := 1
	if req.EnableDualFilter {
		if err := writeSwitchableInterpFilter(w, cdfs, ctx, req, 1, filters.X); err != nil {
			return 0, err
		}
		writes++
	} else if filters.X != filters.Y {
		// With dual filter disabled the decoder copies the Y read into X.
		return 0, ErrInvalidDecodeState
	}
	return writes, nil
}

func writeSwitchableInterpFilter(w *entropy.Writer, cdfs *InterpFilterCDFs, ctx *BlockModeContext, req InterpFilterRequest, dir int, filter motion.InterpFilter) error {
	symbolCtx, err := ctx.SwitchableInterpContext(req, dir)
	if err != nil {
		return err
	}
	cdf, err := cdfs.SwitchableCDF(symbolCtx)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, int(filter))
	return nil
}

// InterpNeeded reports whether the decoder reads (or, at a fixed frame
// filter, applies) block-level interpolation for this request — the exported
// face of libaom's av1_is_interp_needed used by the encoder's filter-search
// gate.
func InterpNeeded(req InterpFilterRequest) bool {
	return interpNeeded(req)
}

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
