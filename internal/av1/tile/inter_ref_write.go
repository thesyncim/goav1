package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// inter_ref_write.go holds the forwards of the is-inter flag and the
// single-reference frame selection, mirroring ReadIntraFlagResult's inter path
// and ReadInterReferences/readSingleReference bit for bit.

// WriteIntraFlag codes the is-inter flag for one block on inter/switch frames,
// the forward of ReadIntraFlagResult: segment-driven and keyframe cases code
// nothing (the writer validates the implied value), otherwise one symbol from
// the neighbor-derived intra context (symbol 1 = inter).
func WriteIntraFlag(w *entropy.Writer, cdfs *IntraModeCDFs, ctx *BlockModeContext, req IntraFlagRequest, intra bool) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	if req.SkipMode {
		if intra {
			return ErrInvalidDecodeState
		}
		return nil
	}
	if frameTypeIsInterOrSwitch(req.FrameType) {
		if req.SegmentationEnabled && (req.Segment.RefFrame >= 0 || req.Segment.GlobalMV) {
			implied := req.Segment.RefFrame == 0 && !req.Segment.GlobalMV
			if intra != implied {
				return ErrInvalidDecodeState
			}
			return nil
		}
		context, err := ctx.IntraContext(int(req.X4), int(req.Y4), req.HaveTop, req.HaveLeft)
		if err != nil {
			return err
		}
		cdf, err := cdfs.IntraCDF(context)
		if err != nil {
			return err
		}
		w.WriteBinaryCDFTrusted(cdf, boolToSym(!intra))
		return nil
	}
	if req.AllowIntrabc {
		return ErrInvalidDecodeState // intrabc coding is not supported yet
	}
	if !intra {
		return ErrInvalidDecodeState
	}
	return nil
}

// WriteInterReferences codes the reference selection for one inter block, the
// forward of ReadInterReferences. Skip-mode and segment-driven blocks code
// nothing; otherwise the compound/single mode decision is written when the
// frame uses REFERENCE_MODE_SELECT, followed by the single or compound
// reference tree.
func WriteInterReferences(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, refs InterReferencesResult) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	if err := ctx.validateInterRefRequest(req); err != nil {
		return err
	}
	if req.SkipMode {
		if !refs.Compound || refs.Unidir || refs.Ref != req.SkipModeRefs {
			return ErrInvalidDecodeState
		}
		return nil
	}
	if req.SegmentationEnabled {
		if req.Segment.RefFrame > 0 {
			implied := ReferenceFrame(req.Segment.RefFrame - 1)
			if refs.Compound || refs.Unidir || refs.Ref[0] != implied || refs.Ref[1] != ReferenceFrameNone {
				return ErrInvalidDecodeState
			}
			return nil
		}
		if req.Segment.RefFrame == 0 {
			return ErrInvalidDecodeState
		}
		if req.Segment.Skip || req.Segment.GlobalMV {
			if refs.Compound || refs.Unidir || refs.Ref[0] != ReferenceFrameLast || refs.Ref[1] != ReferenceFrameNone {
				return ErrInvalidDecodeState
			}
			return nil
		}
	}
	if refs.Compound {
		if !refs.Ref[0].Valid() || !refs.Ref[1].Valid() {
			return ErrInvalidDecodeState
		}
	} else {
		if !refs.Ref[0].Valid() || refs.Ref[1] != ReferenceFrameNone || refs.Unidir {
			return ErrInvalidDecodeState
		}
	}

	if compoundReferenceAllowed(req.Size) {
		switch req.ReferenceMode {
		case parser.ReferenceModeSingle:
			if refs.Compound {
				return ErrInvalidDecodeState
			}
		case parser.ReferenceModeCompound:
			if !refs.Compound {
				return ErrInvalidDecodeState
			}
		case parser.ReferenceModeSelect:
			context, err := ctx.ReferenceModeContext(req)
			if err != nil {
				return err
			}
			cdf, err := cdfs.CompInterCDF(context)
			if err != nil {
				return err
			}
			w.WriteBinaryCDFTrusted(cdf, boolToSym(refs.Compound))
		default:
			return ErrInvalidDecodeState
		}
	} else if refs.Compound {
		return ErrInvalidDecodeState
	}
	if refs.Compound {
		return writeCompoundReferences(w, cdfs, ctx, req, refs)
	}
	return writeSingleReference(w, cdfs, ctx, req, refs.Ref[0])
}

// writeSingleReference codes the single_ref bit tree, the exact inverse of
// readSingleReference.
func writeSingleReference(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, ref ReferenceFrame) error {
	if !ref.Valid() {
		return ErrInvalidDecodeState
	}
	p := singleRefWritePatterns[ref]
	step0 := p.steps[0]
	context, err := singleRefContext(ctx, req, int(step0.bit))
	if err != nil {
		return err
	}
	cdf, err := cdfs.SingleRefCDF(int(step0.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step0.symbol))

	step1 := p.steps[1]
	context, err = singleRefContext(ctx, req, int(step1.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.SingleRefCDF(int(step1.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step1.symbol))
	if p.count == 2 {
		return nil
	}

	step2 := p.steps[2]
	context, err = singleRefContext(ctx, req, int(step2.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.SingleRefCDF(int(step2.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step2.symbol))
	return nil
}

type interRefWriteStep struct {
	bit    uint8
	symbol uint8
}

type interRefWritePattern struct {
	count uint8
	steps [3]interRefWriteStep
}

var singleRefWritePatterns = [referenceFrameCount]interRefWritePattern{
	ReferenceFrameLast:    {count: 3, steps: [3]interRefWriteStep{{0, 0}, {2, 0}, {3, 0}}},
	ReferenceFrameLast2:   {count: 3, steps: [3]interRefWriteStep{{0, 0}, {2, 0}, {3, 1}}},
	ReferenceFrameLast3:   {count: 3, steps: [3]interRefWriteStep{{0, 0}, {2, 1}, {4, 0}}},
	ReferenceFrameGolden:  {count: 3, steps: [3]interRefWriteStep{{0, 0}, {2, 1}, {4, 1}}},
	ReferenceFrameBWD:     {count: 3, steps: [3]interRefWriteStep{{0, 1}, {1, 0}, {5, 0}}},
	ReferenceFrameAltref2: {count: 3, steps: [3]interRefWriteStep{{0, 1}, {1, 0}, {5, 1}}},
	ReferenceFrameAltref:  {count: 2, steps: [3]interRefWriteStep{{0, 1}, {1, 1}}},
}

const referenceFramePairPatternCount = int(referenceFrameCount) * int(referenceFrameCount)

const (
	uniCompRefPatternLastLast2  = int(ReferenceFrameLast)*int(referenceFrameCount) + int(ReferenceFrameLast2)
	uniCompRefPatternLastLast3  = int(ReferenceFrameLast)*int(referenceFrameCount) + int(ReferenceFrameLast3)
	uniCompRefPatternLastGolden = int(ReferenceFrameLast)*int(referenceFrameCount) + int(ReferenceFrameGolden)
	uniCompRefPatternBWDAltref  = int(ReferenceFrameBWD)*int(referenceFrameCount) + int(ReferenceFrameAltref)
)

var uniCompRefWritePatterns = [referenceFramePairPatternCount]interRefWritePattern{
	uniCompRefPatternLastLast2:  {count: 2, steps: [3]interRefWriteStep{{0, 0}, {1, 0}}},
	uniCompRefPatternLastLast3:  {count: 3, steps: [3]interRefWriteStep{{0, 0}, {1, 1}, {2, 0}}},
	uniCompRefPatternLastGolden: {count: 3, steps: [3]interRefWriteStep{{0, 0}, {1, 1}, {2, 1}}},
	uniCompRefPatternBWDAltref:  {count: 1, steps: [3]interRefWriteStep{{0, 1}}},
}

var compFwdRefWritePatterns = [referenceFrameCount]interRefWritePattern{
	ReferenceFrameLast:   {count: 2, steps: [3]interRefWriteStep{{0, 0}, {1, 0}}},
	ReferenceFrameLast2:  {count: 2, steps: [3]interRefWriteStep{{0, 0}, {1, 1}}},
	ReferenceFrameLast3:  {count: 2, steps: [3]interRefWriteStep{{0, 1}, {2, 0}}},
	ReferenceFrameGolden: {count: 2, steps: [3]interRefWriteStep{{0, 1}, {2, 1}}},
}

var compBwdRefWritePatterns = [3]interRefWritePattern{
	{count: 2, steps: [3]interRefWriteStep{{0, 0}, {1, 0}}},
	{count: 2, steps: [3]interRefWriteStep{{0, 0}, {1, 1}}},
	{count: 1, steps: [3]interRefWriteStep{{0, 1}}},
}

func writeCompoundReferences(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, refs InterReferencesResult) error {
	unidir, err := compoundReferencesUnidir(refs)
	if err != nil {
		return err
	}
	if refs.Unidir != unidir {
		return ErrInvalidDecodeState
	}
	context, err := ctx.CompoundReferenceTypeContext(req)
	if err != nil {
		return err
	}
	cdf, err := cdfs.CompRefTypeCDF(context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, boolToSym(!unidir))
	if unidir {
		return writeUniCompoundReferences(w, cdfs, ctx, req, refs.Ref)
	}
	return writeBidirCompoundReferences(w, cdfs, ctx, req, refs.Ref)
}

func compoundReferencesUnidir(refs InterReferencesResult) (bool, error) {
	if !refs.Compound || !refs.Ref[0].Valid() || !refs.Ref[1].Valid() {
		return false, ErrInvalidDecodeState
	}
	switch refs.Ref {
	case [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast2},
		[2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast3},
		[2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameGolden},
		[2]ReferenceFrame{ReferenceFrameBWD, ReferenceFrameAltref}:
		return true, nil
	}
	if refs.Ref[0] >= ReferenceFrameLast && refs.Ref[0] <= ReferenceFrameGolden &&
		refs.Ref[1] >= ReferenceFrameBWD && refs.Ref[1] <= ReferenceFrameAltref {
		return false, nil
	}
	return false, ErrInvalidDecodeState
}

func writeUniCompoundReferences(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, refs [2]ReferenceFrame) error {
	if refs[0] < ReferenceFrameLast || refs[0] >= referenceFrameCount ||
		refs[1] < ReferenceFrameLast || refs[1] >= referenceFrameCount {
		return ErrInvalidDecodeState
	}
	p := uniCompRefWritePatterns[int(refs[0])*int(referenceFrameCount)+int(refs[1])]
	if p.count == 0 {
		return ErrInvalidDecodeState
	}

	step0 := p.steps[0]
	context, err := uniCompRefContext(ctx, req, int(step0.bit))
	if err != nil {
		return err
	}
	cdf, err := cdfs.UniCompRefCDF(int(step0.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step0.symbol))
	if p.count == 1 {
		return nil
	}

	step1 := p.steps[1]
	context, err = uniCompRefContext(ctx, req, int(step1.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.UniCompRefCDF(int(step1.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step1.symbol))
	if p.count == 2 {
		return nil
	}

	step2 := p.steps[2]
	context, err = uniCompRefContext(ctx, req, int(step2.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.UniCompRefCDF(int(step2.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step2.symbol))
	return nil
}

func writeBidirCompoundReferences(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, refs [2]ReferenceFrame) error {
	fwd := int(refs[0])
	bwd := int(refs[1]) - int(ReferenceFrameBWD)
	if uint(fwd) > uint(ReferenceFrameGolden) || uint(bwd) >= uint(len(compBwdRefWritePatterns)) {
		return ErrInvalidDecodeState
	}

	p := compFwdRefWritePatterns[fwd]
	step0 := p.steps[0]
	context, err := compFwdRefContext(ctx, req, int(step0.bit))
	if err != nil {
		return err
	}
	cdf, err := cdfs.CompFwdRefCDF(int(step0.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step0.symbol))

	step1 := p.steps[1]
	context, err = compFwdRefContext(ctx, req, int(step1.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.CompFwdRefCDF(int(step1.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step1.symbol))

	p = compBwdRefWritePatterns[bwd]
	step0 = p.steps[0]
	context, err = compBwdRefContext(ctx, req, int(step0.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.CompBwdRefCDF(int(step0.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step0.symbol))
	if p.count == 1 {
		return nil
	}

	step1 = p.steps[1]
	context, err = compBwdRefContext(ctx, req, int(step1.bit))
	if err != nil {
		return err
	}
	cdf, err = cdfs.CompBwdRefCDF(int(step1.bit), context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, int(step1.symbol))
	return nil
}
