package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	SwitchableFilterContexts = 16

	switchableFilterCount   = 3
	switchableFilterMissing = switchableFilterCount
	interFilterCompOffset   = switchableFilterCount + 1
	interFilterDirOffset    = (switchableFilterCount + 1) * 2
)

// InterpFilterCDFs contains caller-owned CDFs for switchable inter
// interpolation filter syntax.
type InterpFilterCDFs struct {
	Switchable [SwitchableFilterContexts]entropy.CDF
}

// InterpFilterRequest describes libaom's read_mb_interp_filter() inputs after
// references, inter mode, motion vectors, compound blend, and motion mode are
// known for the current block.
type InterpFilterRequest struct {
	FrameFilter      parser.InterpolationFilter
	EnableDualFilter bool

	Size        BlockSize
	References  InterReferencesResult
	Mode        InterModeResult
	MotionMode  MotionMode
	GlobalTypes [2]parser.GlobalMotionType
	SkipMode    bool

	X4       uint8
	Y4       uint8
	HaveTop  bool
	HaveLeft bool
}

// InitDefault seeds c with libaom's default_switchable_interp_cdf table.
func (c *InterpFilterCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next InterpFilterCDFs
	defaults := [...][switchableFilterCount - 1]uint16{
		{31935, 32720}, {5568, 32719},
		{422, 2938}, {28244, 32608},
		{31206, 31953}, {4862, 32121},
		{770, 1152}, {20889, 25637},
		{31910, 32724}, {4120, 32712},
		{305, 2247}, {27403, 32636},
		{31022, 32009}, {2963, 32093},
		{601, 943}, {14969, 21398},
	}
	for i := range defaults {
		if err := next.Switchable[i].Init(defaults[i][:]); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

func (c *InterpFilterCDFs) SwitchableCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= SwitchableFilterContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.Switchable[ctx]
	if cdf.Symbols() != switchableFilterCount {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

// ReadInterpFilters ports libaom's read_mb_interp_filter(). It returns the
// decoded filters plus the number of switchable symbols consumed.
func (s *DecodeState) ReadInterpFilters(cdfs *InterpFilterCDFs, ctx *BlockModeContext, req InterpFilterRequest) (motion.InterpFilters, int, error) {
	if s == nil {
		return motion.InterpFilters{}, 0, ErrInvalidDecodeState
	}
	if err := validateInterpFilterRequest(req); err != nil {
		return motion.InterpFilters{}, 0, err
	}
	if !interpNeeded(req) {
		filter, err := unswitchableMotionFilter(req.FrameFilter)
		if err != nil {
			return motion.InterpFilters{}, 0, err
		}
		return motion.InterpFilters{X: filter, Y: filter}, 0, nil
	}
	if req.FrameFilter != parser.InterpolationSwitchable {
		filter, err := fixedMotionFilter(req.FrameFilter)
		if err != nil {
			return motion.InterpFilters{}, 0, err
		}
		return motion.InterpFilters{X: filter, Y: filter}, 0, nil
	}

	yFilter, err := s.readSwitchableInterpFilter(cdfs, ctx, req, 0)
	if err != nil {
		return motion.InterpFilters{}, 0, err
	}
	reads := 1
	xFilter := yFilter
	if req.EnableDualFilter {
		xFilter, err = s.readSwitchableInterpFilter(cdfs, ctx, req, 1)
		if err != nil {
			return motion.InterpFilters{}, 0, err
		}
		reads++
	}
	return motion.InterpFilters{X: xFilter, Y: yFilter}, reads, nil
}

func (s *DecodeState) readSwitchableInterpFilter(cdfs *InterpFilterCDFs, ctx *BlockModeContext, req InterpFilterRequest, dir int) (motion.InterpFilter, error) {
	symbolCtx, err := ctx.SwitchableInterpContext(req, dir)
	if err != nil {
		return 0, err
	}
	cdf, err := cdfs.SwitchableCDF(symbolCtx)
	if err != nil {
		return 0, err
	}
	symbol, err := s.ReadSymbol(cdf)
	if err != nil {
		return 0, err
	}
	filter := motion.InterpFilter(symbol)
	if !switchableMotionFilter(filter) {
		return 0, ErrInvalidDecodeState
	}
	return filter, nil
}

func validateInterpFilterRequest(req InterpFilterRequest) error {
	dims, ok := req.Size.Dimensions()
	if !ok ||
		int(req.X4)+int(dims.W4) > MaxBlockModeSlots ||
		int(req.Y4)+int(dims.H4) > MaxBlockModeSlots ||
		!req.References.Ref[0].Valid() {
		return ErrInvalidDecodeState
	}
	if req.References.Compound {
		if !req.References.Ref[1].Valid() || !req.Mode.CompoundMode.Valid() {
			return ErrInvalidDecodeState
		}
	} else if req.References.Ref[1] != ReferenceFrameNone || !req.Mode.Mode.Valid() {
		return ErrInvalidDecodeState
	}
	if req.MotionMode != MotionModeTranslation && req.MotionMode != MotionModeOBMC && req.MotionMode != MotionModeWarp {
		return ErrInvalidDecodeState
	}
	if !globalMotionTypeValid(req.GlobalTypes[0]) || !globalMotionTypeValid(req.GlobalTypes[1]) {
		return ErrInvalidDecodeState
	}
	if _, err := unswitchableMotionFilter(req.FrameFilter); err != nil {
		return err
	}
	return nil
}

// SwitchableInterpContext ports av1_get_pred_context_switchable_interp().
func (c *BlockModeContext) SwitchableInterpContext(req InterpFilterRequest, dir int) (int, error) {
	if c == nil || (dir != 0 && dir != 1) {
		return 0, ErrInvalidDecodeState
	}
	if err := validateInterpFilterRequest(req); err != nil {
		return 0, err
	}
	ctx := dir * interFilterDirOffset
	if req.References.Compound {
		ctx += interFilterCompOffset
	}
	ref := req.References.Ref[0]
	left := switchableFilterMissing
	above := switchableFilterMissing
	if req.HaveLeft {
		left = c.neighborSwitchableInterpFilter(false, int(req.Y4), dir, ref)
	}
	if req.HaveTop {
		above = c.neighborSwitchableInterpFilter(true, int(req.X4), dir, ref)
	}
	switch {
	case left == above:
		ctx += left
	case left == switchableFilterMissing:
		ctx += above
	case above == switchableFilterMissing:
		ctx += left
	default:
		ctx += switchableFilterMissing
	}
	if ctx < 0 || ctx >= SwitchableFilterContexts {
		return 0, ErrInvalidDecodeState
	}
	return ctx, nil
}

func (c *BlockModeContext) neighborSwitchableInterpFilter(above bool, slot int, dir int, ref ReferenceFrame) int {
	if slot < 0 || slot >= MaxBlockModeSlots {
		return switchableFilterMissing
	}
	var refs [2]ReferenceFrame
	var filters motion.InterpFilters
	valid := uint8(0)
	if above {
		refs[0] = c.AboveRef[0][slot]
		refs[1] = c.AboveRef[1][slot]
		filters = c.AboveInterp[slot]
		valid = c.AboveInterpValid[slot]
	} else {
		refs[0] = c.LeftRef[0][slot]
		refs[1] = c.LeftRef[1][slot]
		filters = c.LeftInterp[slot]
		valid = c.LeftInterpValid[slot]
	}
	if valid == 0 || (refs[0] != ref && refs[1] != ref) {
		return switchableFilterMissing
	}
	filter := filters.Y
	if dir == 1 {
		filter = filters.X
	}
	if !switchableMotionFilter(filter) {
		return switchableFilterMissing
	}
	return int(filter)
}

// MarkInterFilters records the decoded interpolation filter pair in the
// above/left mode context for later switchable-filter contexts.
func (c *BlockModeContext) MarkInterFilters(size BlockSize, x4 int, y4 int, refs InterReferencesResult, filters motion.InterpFilters) error {
	if c == nil || !refs.Ref[0].Valid() || !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidDecodeState
	}
	if refs.Compound {
		if !refs.Ref[1].Valid() {
			return ErrInvalidDecodeState
		}
	} else if refs.Ref[1] != ReferenceFrameNone {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok || x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	for i := 0; i < int(dims.W4); i++ {
		c.AboveInterp[x4+i] = filters
		c.AboveInterpValid[x4+i] = 1
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftInterp[y4+i] = filters
		c.LeftInterpValid[y4+i] = 1
	}
	// Record the per-MI filter pair so the sub8x8 chroma predictor can read
	// each covered luma sub-block's own filters (libaom's this_mbmi->
	// interp_filters), mirroring the GridInterMotion grid written by
	// MarkInterMotion.
	for r := 0; r < int(dims.H4); r++ {
		for col := 0; col < int(dims.W4); col++ {
			c.GridInterp[y4+r][x4+col] = filters
			c.GridInterpValid[y4+r][x4+col] = 1
		}
	}
	return nil
}

func interpNeeded(req InterpFilterRequest) bool {
	if req.SkipMode || req.MotionMode == MotionModeWarp {
		return false
	}
	return !nonTransGlobalMotion(req)
}

func nonTransGlobalMotion(req InterpFilterRequest) bool {
	dims, ok := req.Size.Dimensions()
	if !ok || dims.W4 < 2 || dims.H4 < 2 {
		return false
	}
	// libaom is_nontrans_global_motion() (av1/common/blockd.h): the block is
	// non-translational global motion when it is GLOBALMV/GLOBAL_GLOBALMV and the
	// global motion model for every reference is NOT TRANSLATION. This differs
	// from is_global_mv_block() (warp/ref_mv), which requires wmtype > TRANSLATION:
	// here an IDENTITY model also counts (wmtype != TRANSLATION), so a GLOBALMV
	// block over an identity model skips the interp-filter syntax read. Using
	// > TRANSLATION wrongly read the filter for identity-model GLOBALMV blocks,
	// desyncing the range coder.
	if !req.References.Compound {
		return req.Mode.Mode == InterModeGlobalMV && req.GlobalTypes[0] != parser.GlobalMotionTranslation
	}
	return req.Mode.CompoundMode == CompoundInterModeGlobalGlobal &&
		req.GlobalTypes[0] != parser.GlobalMotionTranslation &&
		req.GlobalTypes[1] != parser.GlobalMotionTranslation
}

func fixedMotionFilter(filter parser.InterpolationFilter) (motion.InterpFilter, error) {
	switch filter {
	case parser.InterpolationEightTap:
		return motion.InterpEightTapRegular, nil
	case parser.InterpolationSmooth:
		return motion.InterpEightTapSmooth, nil
	case parser.InterpolationSharp:
		return motion.InterpMultiTapSharp, nil
	case parser.InterpolationBilinear:
		return motion.InterpBilinear, nil
	default:
		return 0, ErrInvalidDecodeState
	}
}

func unswitchableMotionFilter(filter parser.InterpolationFilter) (motion.InterpFilter, error) {
	if filter == parser.InterpolationSwitchable {
		return motion.InterpEightTapRegular, nil
	}
	return fixedMotionFilter(filter)
}

func switchableMotionFilter(filter motion.InterpFilter) bool {
	return filter == motion.InterpEightTapRegular ||
		filter == motion.InterpEightTapSmooth ||
		filter == motion.InterpMultiTapSharp
}
