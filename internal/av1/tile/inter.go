package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	CompoundInterContexts         = 5
	ReferenceContexts             = 3
	CompoundReferenceTypeContexts = 5
	UniCompoundReferenceContexts  = 3
	SingleReferenceBits           = 6
	CompoundForwardReferenceBits  = 3
	CompoundBackwardReferenceBits = 2
	UniCompoundReferenceBits      = 3
)

// ReferenceFrame identifies one inter reference in dav1d order:
// LAST..ALTREF. ReferenceFrameNone is used for no second reference.
type ReferenceFrame int8

const (
	ReferenceFrameNone ReferenceFrame = -1

	ReferenceFrameLast    ReferenceFrame = 0
	ReferenceFrameLast2   ReferenceFrame = 1
	ReferenceFrameLast3   ReferenceFrame = 2
	ReferenceFrameGolden  ReferenceFrame = 3
	ReferenceFrameBWD     ReferenceFrame = 4
	ReferenceFrameAltref2 ReferenceFrame = 5
	ReferenceFrameAltref  ReferenceFrame = 6
	referenceFrameCount   ReferenceFrame = 7
)

// InterRefCDFs contains caller-owned CDFs for AV1 inter reference syntax.
type InterRefCDFs struct {
	CompInter   [CompoundInterContexts]entropy.CDF
	CompRefType [CompoundReferenceTypeContexts]entropy.CDF
	SingleRef   [SingleReferenceBits][ReferenceContexts]entropy.CDF
	CompFwdRef  [CompoundForwardReferenceBits][ReferenceContexts]entropy.CDF
	CompBwdRef  [CompoundBackwardReferenceBits][ReferenceContexts]entropy.CDF
	UniCompRef  [UniCompoundReferenceBits][UniCompoundReferenceContexts]entropy.CDF
}

// InterReferenceRequest describes the block/frame conditions for reference
// mode and reference-frame syntax.
type InterReferenceRequest struct {
	Size          BlockSize
	ReferenceMode parser.ReferenceMode
	SkipMode      bool
	SkipModeRefs  [2]ReferenceFrame

	SegmentationEnabled bool
	Segment             parser.SegmentData

	X4       uint8
	Y4       uint8
	HaveTop  bool
	HaveLeft bool
}

// InterReferencesResult is the entropy-decoded reference-frame pair.
type InterReferencesResult struct {
	Ref      [2]ReferenceFrame
	Compound bool
	Unidir   bool
}

func (ref ReferenceFrame) Valid() bool {
	return ref >= ReferenceFrameLast && ref < referenceFrameCount
}

func (ref ReferenceFrame) Backward() bool {
	return ref >= ReferenceFrameBWD && ref <= ReferenceFrameAltref
}

// InitDefault seeds c with dav1d/libaom's default inter-reference CDFs.
func (c *InterRefCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next InterRefCDFs
	for i, cumulative := range [...]uint16{26828, 24035, 12031, 10640, 2901} {
		if err := next.CompInter[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	for i, cumulative := range [...]uint16{1198, 2070, 9166, 7499, 22475} {
		if err := next.CompRefType[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	single := [...][ReferenceContexts]uint16{
		{4897, 16973, 29744},
		{1555, 16751, 30279},
		{4236, 19647, 31194},
		{8650, 24773, 31895},
		{904, 11014, 26875},
		{1444, 15087, 30304},
	}
	for bit := range single {
		for ctx := range single[bit] {
			if err := next.SingleRef[bit][ctx].Init([]uint16{single[bit][ctx]}); err != nil {
				return err
			}
		}
	}
	compFwd := [...][ReferenceContexts]uint16{
		{4946, 19891, 30731},
		{9468, 22441, 31059},
		{1503, 15160, 27544},
	}
	for bit := range compFwd {
		for ctx := range compFwd[bit] {
			if err := next.CompFwdRef[bit][ctx].Init([]uint16{compFwd[bit][ctx]}); err != nil {
				return err
			}
		}
	}
	compBwd := [...][ReferenceContexts]uint16{
		{2235, 17182, 30606},
		{1423, 15175, 30489},
	}
	for bit := range compBwd {
		for ctx := range compBwd[bit] {
			if err := next.CompBwdRef[bit][ctx].Init([]uint16{compBwd[bit][ctx]}); err != nil {
				return err
			}
		}
	}
	uni := [...][UniCompoundReferenceContexts]uint16{
		{5284, 23152, 31774},
		{3865, 14173, 25120},
		{3128, 15270, 26710},
	}
	for bit := range uni {
		for ctx := range uni[bit] {
			if err := next.UniCompRef[bit][ctx].Init([]uint16{uni[bit][ctx]}); err != nil {
				return err
			}
		}
	}
	*c = next
	return nil
}

func (c *InterRefCDFs) CompInterCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= CompoundInterContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompInter[ctx])
}

func (c *InterRefCDFs) CompRefTypeCDF(ctx int) (*entropy.CDF, error) {
	if c == nil || ctx < 0 || ctx >= CompoundReferenceTypeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompRefType[ctx])
}

func (c *InterRefCDFs) SingleRefCDF(bit int, ctx int) (*entropy.CDF, error) {
	if c == nil || bit < 0 || bit >= SingleReferenceBits || ctx < 0 || ctx >= ReferenceContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.SingleRef[bit][ctx])
}

func (c *InterRefCDFs) CompFwdRefCDF(bit int, ctx int) (*entropy.CDF, error) {
	if c == nil || bit < 0 || bit >= CompoundForwardReferenceBits || ctx < 0 || ctx >= ReferenceContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompFwdRef[bit][ctx])
}

func (c *InterRefCDFs) CompBwdRefCDF(bit int, ctx int) (*entropy.CDF, error) {
	if c == nil || bit < 0 || bit >= CompoundBackwardReferenceBits || ctx < 0 || ctx >= ReferenceContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.CompBwdRef[bit][ctx])
}

func (c *InterRefCDFs) UniCompRefCDF(bit int, ctx int) (*entropy.CDF, error) {
	if c == nil || bit < 0 || bit >= UniCompoundReferenceBits || ctx < 0 || ctx >= UniCompoundReferenceContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.UniCompRef[bit][ctx])
}

func binaryInterRefCDF(cdf *entropy.CDF) (*entropy.CDF, error) {
	if cdf == nil || cdf.Symbols() != 2 {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// ReferenceModeContext returns dav1d get_comp_ctx().
func (c *BlockModeContext) ReferenceModeContext(req InterReferenceRequest) (int, error) {
	if err := c.validateInterRefRequest(req); err != nil {
		return 0, err
	}
	if req.HaveTop {
		if req.HaveLeft {
			if c.AboveCompound[req.X4] != 0 {
				if c.LeftCompound[req.Y4] != 0 {
					return 4, nil
				}
				return 2 + boolInt(c.LeftRef[0][req.Y4].Backward() || c.LeftIntra[req.Y4] != 0), nil
			}
			if c.LeftCompound[req.Y4] != 0 {
				return 2 + boolInt(c.AboveRef[0][req.X4].Backward() || c.AboveIntra[req.X4] != 0), nil
			}
			return boolInt(c.LeftRef[0][req.Y4].Backward() != c.AboveRef[0][req.X4].Backward()), nil
		}
		if c.AboveCompound[req.X4] != 0 {
			return 3, nil
		}
		return boolInt(c.AboveRef[0][req.X4].Backward()), nil
	}
	if req.HaveLeft {
		if c.LeftCompound[req.Y4] != 0 {
			return 3, nil
		}
		return boolInt(c.LeftRef[0][req.Y4].Backward()), nil
	}
	return 1, nil
}

// CompoundReferenceTypeContext returns dav1d get_comp_dir_ctx().
func (c *BlockModeContext) CompoundReferenceTypeContext(req InterReferenceRequest) (int, error) {
	if err := c.validateInterRefRequest(req); err != nil {
		return 0, err
	}
	x4 := int(req.X4)
	y4 := int(req.Y4)
	if req.HaveTop && req.HaveLeft {
		aboveIntra := c.AboveIntra[req.X4] != 0
		leftIntra := c.LeftIntra[req.Y4] != 0
		if aboveIntra && leftIntra {
			return 2, nil
		}
		if aboveIntra || leftIntra {
			if aboveIntra {
				if c.LeftCompound[req.Y4] == 0 {
					return 2, nil
				}
				return 1 + 2*boolInt(c.leftUnidir(y4)), nil
			}
			if c.AboveCompound[req.X4] == 0 {
				return 2, nil
			}
			return 1 + 2*boolInt(c.aboveUnidir(x4)), nil
		}

		aboveComp := c.AboveCompound[req.X4] != 0
		leftComp := c.LeftCompound[req.Y4] != 0
		aboveRef0 := c.AboveRef[0][req.X4]
		leftRef0 := c.LeftRef[0][req.Y4]
		if !aboveComp && !leftComp {
			return 1 + 2*boolInt(aboveRef0.Backward() == leftRef0.Backward()), nil
		}
		if !aboveComp || !leftComp {
			if aboveComp {
				if !c.aboveUnidir(x4) {
					return 1, nil
				}
			} else if !c.leftUnidir(y4) {
				return 1, nil
			}
			return 3 + boolInt(aboveRef0.Backward() == leftRef0.Backward()), nil
		}
		aboveUni := c.aboveUnidir(x4)
		leftUni := c.leftUnidir(y4)
		if !aboveUni && !leftUni {
			return 0, nil
		}
		if !aboveUni || !leftUni {
			return 2, nil
		}
		return 3 + boolInt((aboveRef0 == ReferenceFrameBWD) == (leftRef0 == ReferenceFrameBWD)), nil
	}
	if req.HaveTop || req.HaveLeft {
		if req.HaveLeft {
			if c.LeftIntra[req.Y4] != 0 || c.LeftCompound[req.Y4] == 0 {
				return 2, nil
			}
			return 4 * boolInt(c.leftUnidir(y4)), nil
		}
		if c.AboveIntra[req.X4] != 0 || c.AboveCompound[req.X4] == 0 {
			return 2, nil
		}
		return 4 * boolInt(c.aboveUnidir(x4)), nil
	}
	return 2, nil
}

func (c *BlockModeContext) ReferenceContext(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefDirs(req)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0], counts[1]), nil
}

func (c *BlockModeContext) ForwardReferenceContext(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefs(req, 0, 4)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0]+counts[1], counts[2]+counts[3]), nil
}

func (c *BlockModeContext) ForwardReference1Context(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefs(req, 0, 2)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0], counts[1]), nil
}

func (c *BlockModeContext) ForwardReference2Context(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefs(req, 2, 4)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0], counts[1]), nil
}

func (c *BlockModeContext) BackwardReferenceContext(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefs(req, 4, 7)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0]+counts[1], counts[2]), nil
}

func (c *BlockModeContext) BackwardReference1Context(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefs(req, 4, 7)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0], counts[1]), nil
}

func (c *BlockModeContext) UniReference1Context(req InterReferenceRequest) (int, error) {
	counts, err := c.countRefs(req, 1, 4)
	if err != nil {
		return 0, err
	}
	return compareCounts(counts[0], counts[1]+counts[2]), nil
}

func (s *DecodeState) ReadInterReferences(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest) (InterReferencesResult, error) {
	if s == nil {
		return InterReferencesResult{}, ErrInvalidDecodeState
	}
	if err := ctx.validateInterRefRequest(req); err != nil {
		return InterReferencesResult{}, err
	}
	if req.SkipMode {
		if !req.SkipModeRefs[0].Valid() || !req.SkipModeRefs[1].Valid() {
			return InterReferencesResult{}, ErrInvalidDecodeState
		}
		return InterReferencesResult{Ref: req.SkipModeRefs, Compound: true}, nil
	}
	if req.SegmentationEnabled {
		if req.Segment.RefFrame > 0 {
			ref := ReferenceFrame(req.Segment.RefFrame - 1)
			if !ref.Valid() {
				return InterReferencesResult{}, ErrInvalidDecodeState
			}
			return InterReferencesResult{Ref: [2]ReferenceFrame{ref, ReferenceFrameNone}}, nil
		}
		if req.Segment.RefFrame == 0 {
			return InterReferencesResult{}, ErrInvalidDecodeState
		}
		if req.Segment.Skip || req.Segment.GlobalMV {
			return InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}, nil
		}
	}

	compound, err := s.ReadReferenceMode(cdfs, ctx, req)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if compound {
		return s.readCompoundReferences(cdfs, ctx, req)
	}
	ref, err := s.readSingleReference(cdfs, ctx, req)
	if err != nil {
		return InterReferencesResult{}, err
	}
	return InterReferencesResult{Ref: [2]ReferenceFrame{ref, ReferenceFrameNone}}, nil
}

func (s *DecodeState) ReadReferenceMode(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	if !compoundReferenceAllowed(req.Size) {
		return false, nil
	}
	switch req.ReferenceMode {
	case parser.ReferenceModeSingle:
		return false, nil
	case parser.ReferenceModeCompound:
		return true, nil
	case parser.ReferenceModeSelect:
	default:
		return false, ErrInvalidDecodeState
	}
	context, err := ctx.ReferenceModeContext(req)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.CompInterCDF(context)
	if err != nil {
		return false, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return false, err
	}
	return symbol != 0, nil
}

func (s *DecodeState) readCompoundReferences(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest) (InterReferencesResult, error) {
	context, err := ctx.CompoundReferenceTypeContext(req)
	if err != nil {
		return InterReferencesResult{}, err
	}
	cdf, err := cdfs.CompRefTypeCDF(context)
	if err != nil {
		return InterReferencesResult{}, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if symbol == 0 {
		return s.readUniCompoundReferences(cdfs, ctx, req)
	}
	return s.readBidirCompoundReferences(cdfs, ctx, req)
}

func (s *DecodeState) readUniCompoundReferences(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest) (InterReferencesResult, error) {
	bit, err := s.readUniCompRefBit(cdfs, ctx, req, 0)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if bit {
		return InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameBWD, ReferenceFrameAltref}, Compound: true, Unidir: true}, nil
	}
	bit, err = s.readUniCompRefBit(cdfs, ctx, req, 1)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if !bit {
		return InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast2}, Compound: true, Unidir: true}, nil
	}
	bit, err = s.readUniCompRefBit(cdfs, ctx, req, 2)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if bit {
		return InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameGolden}, Compound: true, Unidir: true}, nil
	}
	return InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast3}, Compound: true, Unidir: true}, nil
}

func (s *DecodeState) readBidirCompoundReferences(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest) (InterReferencesResult, error) {
	var result InterReferencesResult
	result.Compound = true

	bit, err := s.readCompFwdRefBit(cdfs, ctx, req, 0)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if !bit {
		bit, err = s.readCompFwdRefBit(cdfs, ctx, req, 1)
		if err != nil {
			return InterReferencesResult{}, err
		}
		if bit {
			result.Ref[0] = ReferenceFrameLast2
		} else {
			result.Ref[0] = ReferenceFrameLast
		}
	} else {
		bit, err = s.readCompFwdRefBit(cdfs, ctx, req, 2)
		if err != nil {
			return InterReferencesResult{}, err
		}
		if bit {
			result.Ref[0] = ReferenceFrameGolden
		} else {
			result.Ref[0] = ReferenceFrameLast3
		}
	}

	bit, err = s.readCompBwdRefBit(cdfs, ctx, req, 0)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if bit {
		result.Ref[1] = ReferenceFrameAltref
		return result, nil
	}
	bit, err = s.readCompBwdRefBit(cdfs, ctx, req, 1)
	if err != nil {
		return InterReferencesResult{}, err
	}
	if bit {
		result.Ref[1] = ReferenceFrameAltref2
	} else {
		result.Ref[1] = ReferenceFrameBWD
	}
	return result, nil
}

func (s *DecodeState) readSingleReference(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest) (ReferenceFrame, error) {
	bit, err := s.readSingleRefBit(cdfs, ctx, req, 0)
	if err != nil {
		return 0, err
	}
	if bit {
		bit, err = s.readSingleRefBit(cdfs, ctx, req, 1)
		if err != nil {
			return 0, err
		}
		if !bit {
			bit, err = s.readSingleRefBit(cdfs, ctx, req, 5)
			if err != nil {
				return 0, err
			}
			if bit {
				return ReferenceFrameAltref2, nil
			}
			return ReferenceFrameBWD, nil
		}
		return ReferenceFrameAltref, nil
	}

	bit, err = s.readSingleRefBit(cdfs, ctx, req, 2)
	if err != nil {
		return 0, err
	}
	if bit {
		bit, err = s.readSingleRefBit(cdfs, ctx, req, 4)
		if err != nil {
			return 0, err
		}
		if bit {
			return ReferenceFrameGolden, nil
		}
		return ReferenceFrameLast3, nil
	}
	bit, err = s.readSingleRefBit(cdfs, ctx, req, 3)
	if err != nil {
		return 0, err
	}
	if bit {
		return ReferenceFrameLast2, nil
	}
	return ReferenceFrameLast, nil
}

func (s *DecodeState) readSingleRefBit(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int) (bool, error) {
	context, err := singleRefContext(ctx, req, bit)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.SingleRefCDF(bit, context)
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

func (s *DecodeState) readCompFwdRefBit(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int) (bool, error) {
	context, err := compFwdRefContext(ctx, req, bit)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.CompFwdRefCDF(bit, context)
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

func (s *DecodeState) readCompBwdRefBit(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int) (bool, error) {
	context, err := compBwdRefContext(ctx, req, bit)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.CompBwdRefCDF(bit, context)
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

func (s *DecodeState) readUniCompRefBit(cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int) (bool, error) {
	context, err := uniCompRefContext(ctx, req, bit)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.UniCompRefCDF(bit, context)
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

func readBoolCDF(state *DecodeState, cdf *entropy.CDF) (bool, error) {
	if state == nil {
		return false, ErrInvalidDecodeState
	}
	symbol, err := state.Reader.ReadBinaryCDFTrusted(cdf)
	if err != nil {
		return false, err
	}
	return symbol != 0, nil
}

func singleRefContext(ctx *BlockModeContext, req InterReferenceRequest, bit int) (int, error) {
	switch bit {
	case 0:
		return ctx.ReferenceContext(req)
	case 1:
		return ctx.BackwardReferenceContext(req)
	case 2:
		return ctx.ForwardReferenceContext(req)
	case 3:
		return ctx.ForwardReference1Context(req)
	case 4:
		return ctx.ForwardReference2Context(req)
	case 5:
		return ctx.BackwardReference1Context(req)
	default:
		return 0, ErrInvalidDecodeState
	}
}

func compFwdRefContext(ctx *BlockModeContext, req InterReferenceRequest, bit int) (int, error) {
	switch bit {
	case 0:
		return ctx.ForwardReferenceContext(req)
	case 1:
		return ctx.ForwardReference1Context(req)
	case 2:
		return ctx.ForwardReference2Context(req)
	default:
		return 0, ErrInvalidDecodeState
	}
}

func compBwdRefContext(ctx *BlockModeContext, req InterReferenceRequest, bit int) (int, error) {
	switch bit {
	case 0:
		return ctx.BackwardReferenceContext(req)
	case 1:
		return ctx.BackwardReference1Context(req)
	default:
		return 0, ErrInvalidDecodeState
	}
}

func uniCompRefContext(ctx *BlockModeContext, req InterReferenceRequest, bit int) (int, error) {
	switch bit {
	case 0:
		return ctx.ReferenceContext(req)
	case 1:
		return ctx.UniReference1Context(req)
	case 2:
		return ctx.ForwardReference2Context(req)
	default:
		return 0, ErrInvalidDecodeState
	}
}

// MarkInter records an inter block into the top/left mode context. hasChroma
// gates the chroma-mode context fields: libaom locates the chroma smooth-filter
// neighbor via the chroma reference grid (set_mi_row_col chroma_above/left_mbmi),
// so a sub-8x8 luma block that carries no chroma must NOT overwrite the chroma
// mode context — leaving the covering chroma-reference block's mode in place.
// Clobbering it (e.g. an even-row 8x4 inter block dropping its parent's
// SMOOTH_V) flips get_intra_edge_filter_type and corrupts directional chroma
// prediction (av1-1-b8-00-quantizer-00 frame 1 mi(15,82) D135 chroma).
func (c *BlockModeContext) MarkInter(size BlockSize, x4 int, y4 int, result InterReferencesResult, hasChroma bool) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	if x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	if !result.Ref[0].Valid() {
		return ErrInvalidDecodeState
	}
	if result.Compound {
		if !result.Ref[1].Valid() {
			return ErrInvalidDecodeState
		}
	} else if result.Ref[1] != ReferenceFrameNone {
		return ErrInvalidDecodeState
	}

	compound := boolByte(result.Compound)
	compGroup := uint8(0)
	compIndex := uint8(1)
	for i := 0; i < int(dims.W4); i++ {
		c.AboveIntra[x4+i] = 0
		if hasChroma {
			c.AboveChromaIntra[x4+i] = 0
			c.AboveChromaMode[x4+i] = ChromaIntraModeDC
		}
		c.AboveRef[0][x4+i] = result.Ref[0]
		c.AboveRef[1][x4+i] = result.Ref[1]
		c.AboveCompound[x4+i] = compound
		c.AboveCompGroup[x4+i] = compGroup
		c.AboveCompIndex[x4+i] = compIndex
		c.AboveInterIntra[x4+i] = 0
		c.AboveInterMotion[x4+i] = InterMotionResult{}
		c.AboveMotionValid[x4+i] = 0
		c.AboveInterp[x4+i] = motion.InterpFilters{}
		c.AboveInterpValid[x4+i] = 0
		c.AboveBlockSize[x4+i] = size
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftIntra[y4+i] = 0
		if hasChroma {
			c.LeftChromaIntra[y4+i] = 0
			c.LeftChromaMode[y4+i] = ChromaIntraModeDC
		}
		c.LeftRef[0][y4+i] = result.Ref[0]
		c.LeftRef[1][y4+i] = result.Ref[1]
		c.LeftCompound[y4+i] = compound
		c.LeftCompGroup[y4+i] = compGroup
		c.LeftCompIndex[y4+i] = compIndex
		c.LeftInterIntra[y4+i] = 0
		c.LeftInterMotion[y4+i] = InterMotionResult{}
		c.LeftMotionValid[y4+i] = 0
		c.LeftInterp[y4+i] = motion.InterpFilters{}
		c.LeftInterpValid[y4+i] = 0
		c.LeftBlockSize[y4+i] = size
	}
	c.clearGridInterMotion(size, x4, y4, dims)
	return nil
}

// MarkInterIntra records that the block at (x4, y4) of `size` is an
// inter-intra block. libaom mirrors this by setting mbmi->ref_frame[1]
// = INTRA_FRAME, which makes ref_frame[1] != NONE_FRAME for any
// neighbor scan that filters on "has any second reference" (notably
// av1_findSamples for WARPED_CAUSAL projection). goav1 keeps Ref[1] =
// ReferenceFrameNone for inter-intra to preserve the
// has_second_ref(ref_frame[1] > INTRA_FRAME) semantics; this flag
// distinguishes the two cases so consumers can pick the right one.
func (c *BlockModeContext) MarkInterIntra(size BlockSize, x4 int, y4 int) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	if x4 < 0 || y4 < 0 ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	for i := 0; i < int(dims.W4); i++ {
		c.AboveInterIntra[x4+i] = 1
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftInterIntra[y4+i] = 1
	}
	return nil
}

func (c *BlockModeContext) validateInterRefRequest(req InterReferenceRequest) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return ErrInvalidDecodeState
	}
	return validateBlockModeSlot(int(req.X4), int(req.Y4))
}

func (c *BlockModeContext) countRefDirs(req InterReferenceRequest) ([2]int, error) {
	var counts [2]int
	err := c.visitNeighborRefs(req, func(ref ReferenceFrame) {
		counts[boolInt(ref.Backward())]++
	})
	return counts, err
}

func (c *BlockModeContext) countRefs(req InterReferenceRequest, from ReferenceFrame, to ReferenceFrame) ([4]int, error) {
	var counts [4]int
	if to <= from || int(to-from) > len(counts) {
		return counts, ErrInvalidDecodeState
	}
	err := c.visitNeighborRefs(req, func(ref ReferenceFrame) {
		if ref >= from && ref < to {
			counts[int(ref-from)]++
		}
	})
	return counts, err
}

func (c *BlockModeContext) visitNeighborRefs(req InterReferenceRequest, fn func(ReferenceFrame)) error {
	if err := c.validateInterRefRequest(req); err != nil {
		return err
	}
	if req.HaveTop && c.AboveIntra[req.X4] == 0 {
		c.visitRefPair(c.AboveRef[0][req.X4], c.AboveRef[1][req.X4], c.AboveCompound[req.X4] != 0, fn)
	}
	if req.HaveLeft && c.LeftIntra[req.Y4] == 0 {
		c.visitRefPair(c.LeftRef[0][req.Y4], c.LeftRef[1][req.Y4], c.LeftCompound[req.Y4] != 0, fn)
	}
	return nil
}

func (c *BlockModeContext) visitRefPair(ref0 ReferenceFrame, ref1 ReferenceFrame, compound bool, fn func(ReferenceFrame)) {
	if ref0.Valid() {
		fn(ref0)
	}
	if compound && ref1.Valid() {
		fn(ref1)
	}
}

func (c *BlockModeContext) aboveUnidir(x4 int) bool {
	return (c.AboveRef[0][x4] < ReferenceFrameBWD) == (c.AboveRef[1][x4] < ReferenceFrameBWD)
}

func (c *BlockModeContext) leftUnidir(y4 int) bool {
	return (c.LeftRef[0][y4] < ReferenceFrameBWD) == (c.LeftRef[1][y4] < ReferenceFrameBWD)
}

func compoundReferenceAllowed(size BlockSize) bool {
	dims, ok := size.Dimensions()
	return ok && dims.W4 > 1 && dims.H4 > 1
}

func compareCounts(a int, b int) int {
	if a == b {
		return 1
	}
	if a < b {
		return 0
	}
	return 2
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
