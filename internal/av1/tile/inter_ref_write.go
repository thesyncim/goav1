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
	type refBit struct {
		bit int
		val bool
	}
	var bits []refBit
	switch ref {
	case ReferenceFrameLast:
		bits = []refBit{{0, false}, {2, false}, {3, false}}
	case ReferenceFrameLast2:
		bits = []refBit{{0, false}, {2, false}, {3, true}}
	case ReferenceFrameLast3:
		bits = []refBit{{0, false}, {2, true}, {4, false}}
	case ReferenceFrameGolden:
		bits = []refBit{{0, false}, {2, true}, {4, true}}
	case ReferenceFrameBWD:
		bits = []refBit{{0, true}, {1, false}, {5, false}}
	case ReferenceFrameAltref2:
		bits = []refBit{{0, true}, {1, false}, {5, true}}
	case ReferenceFrameAltref:
		bits = []refBit{{0, true}, {1, true}}
	default:
		return ErrInvalidDecodeState
	}
	for _, b := range bits {
		context, err := singleRefContext(ctx, req, b.bit)
		if err != nil {
			return err
		}
		cdf, err := cdfs.SingleRefCDF(b.bit, context)
		if err != nil {
			return err
		}
		w.WriteBinaryCDFTrusted(cdf, boolToSym(b.val))
	}
	return nil
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
	switch refs {
	case [2]ReferenceFrame{ReferenceFrameBWD, ReferenceFrameAltref}:
		return writeUniCompRefBit(w, cdfs, ctx, req, 0, true)
	case [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast2}:
		if err := writeUniCompRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		return writeUniCompRefBit(w, cdfs, ctx, req, 1, false)
	case [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast3}:
		if err := writeUniCompRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		if err := writeUniCompRefBit(w, cdfs, ctx, req, 1, true); err != nil {
			return err
		}
		return writeUniCompRefBit(w, cdfs, ctx, req, 2, false)
	case [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameGolden}:
		if err := writeUniCompRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		if err := writeUniCompRefBit(w, cdfs, ctx, req, 1, true); err != nil {
			return err
		}
		return writeUniCompRefBit(w, cdfs, ctx, req, 2, true)
	default:
		return ErrInvalidDecodeState
	}
}

func writeBidirCompoundReferences(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, refs [2]ReferenceFrame) error {
	switch refs[0] {
	case ReferenceFrameLast:
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 1, false); err != nil {
			return err
		}
	case ReferenceFrameLast2:
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 1, true); err != nil {
			return err
		}
	case ReferenceFrameLast3:
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 0, true); err != nil {
			return err
		}
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 2, false); err != nil {
			return err
		}
	case ReferenceFrameGolden:
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 0, true); err != nil {
			return err
		}
		if err := writeCompFwdRefBit(w, cdfs, ctx, req, 2, true); err != nil {
			return err
		}
	default:
		return ErrInvalidDecodeState
	}

	switch refs[1] {
	case ReferenceFrameBWD:
		if err := writeCompBwdRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		return writeCompBwdRefBit(w, cdfs, ctx, req, 1, false)
	case ReferenceFrameAltref2:
		if err := writeCompBwdRefBit(w, cdfs, ctx, req, 0, false); err != nil {
			return err
		}
		return writeCompBwdRefBit(w, cdfs, ctx, req, 1, true)
	case ReferenceFrameAltref:
		return writeCompBwdRefBit(w, cdfs, ctx, req, 0, true)
	default:
		return ErrInvalidDecodeState
	}
}

func writeCompFwdRefBit(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int, value bool) error {
	context, err := compFwdRefContext(ctx, req, bit)
	if err != nil {
		return err
	}
	cdf, err := cdfs.CompFwdRefCDF(bit, context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, boolToSym(value))
	return nil
}

func writeCompBwdRefBit(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int, value bool) error {
	context, err := compBwdRefContext(ctx, req, bit)
	if err != nil {
		return err
	}
	cdf, err := cdfs.CompBwdRefCDF(bit, context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, boolToSym(value))
	return nil
}

func writeUniCompRefBit(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, bit int, value bool) error {
	context, err := uniCompRefContext(ctx, req, bit)
	if err != nil {
		return err
	}
	cdf, err := cdfs.UniCompRefCDF(bit, context)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(cdf, boolToSym(value))
	return nil
}
