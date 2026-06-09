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
		w.WriteCDF(cdf, boolToSym(!intra))
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
// frame uses REFERENCE_MODE_SELECT, followed by the single-reference tree.
// Compound reference coding is not supported yet.
func WriteInterReferences(w *entropy.Writer, cdfs *InterRefCDFs, ctx *BlockModeContext, req InterReferenceRequest, refs InterReferencesResult) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	if err := ctx.validateInterRefRequest(req); err != nil {
		return err
	}
	if req.SkipMode {
		if !refs.Compound || refs.Ref != req.SkipModeRefs {
			return ErrInvalidDecodeState
		}
		return nil
	}
	if req.SegmentationEnabled {
		if req.Segment.RefFrame > 0 {
			implied := ReferenceFrame(req.Segment.RefFrame - 1)
			if refs.Compound || refs.Ref[0] != implied {
				return ErrInvalidDecodeState
			}
			return nil
		}
		if req.Segment.RefFrame == 0 {
			return ErrInvalidDecodeState
		}
		if req.Segment.Skip || req.Segment.GlobalMV {
			if refs.Compound || refs.Ref[0] != ReferenceFrameLast {
				return ErrInvalidDecodeState
			}
			return nil
		}
	}
	if refs.Compound {
		return ErrInvalidDecodeState // compound reference coding not supported yet
	}

	// reference mode (compound vs single) — only coded under SELECT.
	if compoundReferenceAllowed(req.Size) {
		switch req.ReferenceMode {
		case parser.ReferenceModeSingle:
		case parser.ReferenceModeCompound:
			return ErrInvalidDecodeState
		case parser.ReferenceModeSelect:
			context, err := ctx.ReferenceModeContext(req)
			if err != nil {
				return err
			}
			cdf, err := cdfs.CompInterCDF(context)
			if err != nil {
				return err
			}
			w.WriteCDF(cdf, 0) // single reference
		default:
			return ErrInvalidDecodeState
		}
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
		w.WriteCDF(cdf, boolToSym(b.val))
	}
	return nil
}
