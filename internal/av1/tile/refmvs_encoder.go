package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

// Encoder access to the dav1d refmvs rows. An encoder's block loop keeps the
// same tile-cached refmvs_block rows the decoder does: WalkBlockLoopWrite and
// WalkBlockLoopDecide run dav1d_refmvs_tile_sbrow_init per superblock row
// while the rows are bound, the encoder splats every block it commits
// (splat_oneref_mv / splat_tworef_mv / splat_intraref) and takes its MV
// candidate stacks from dav1d_refmvs_find. The candidate list is normative,
// so the stacks equal the libaom-shaped BuildReferenceMVStack ones.

// EncoderRefmvsFrame describes the frame and tile an encoder binds the rows
// for. The encoder codes no temporal MVs and no global motion, so those parts
// of refmvs_frame stay empty.
type EncoderRefmvsFrame struct {
	FrameMICols, FrameMIRows uint16
	// Tile is the tile's walk region.
	Tile BlockWalkRequest

	AllowHighPrecisionMV bool
	ForceIntegerMV       bool
	// RefSignBias is the frame's per-reference sign bias (index ref).
	RefSignBias [referenceFrameCount]bool
}

// BindEncoderRefmvs binds s's refmvs rows for one encoder tile walk; they
// stay bound until UnbindEncoderRefmvs.
func BindEncoderRefmvs(s *BlockLoopScratch, f EncoderRefmvsFrame) {
	if s == nil || f.FrameMICols == 0 || f.FrameMIRows == 0 {
		return
	}
	frame := refmvsFrame{
		iw4:                  int32(f.FrameMICols),
		ih4:                  int32(f.FrameMIRows),
		allowHighPrecisionMV: f.AllowHighPrecisionMV,
		forceIntegerMV:       f.ForceIntegerMV,
	}
	for i := range 7 {
		if f.RefSignBias[i] {
			frame.signBias[i] = 1
		}
	}
	s.Refmvs.bindFrame(frame, int32(f.FrameMICols)*4,
		int32(f.Tile.MIColStart), int32(f.Tile.MIColEnd),
		int32(f.Tile.MIRowStart), int32(f.Tile.MIRowEnd))
}

// UnbindEncoderRefmvs releases the binding made by BindEncoderRefmvs.
func UnbindEncoderRefmvs(s *BlockLoopScratch) {
	if s != nil {
		s.Refmvs.bound = false
	}
}

// EncoderRefmvsBound reports whether s's refmvs rows are bound.
func (s *BlockLoopScratch) EncoderRefmvsBound() bool {
	return s != nil && s.Refmvs.bound
}

// keepLegacyMotion reports whether s.Mode's libaom-shaped neighbor-motion
// arrays and grids must be maintained: they have no readers while refmvs rows
// are bound, except the goav1_refmvs_oracle build's legacy reference stack.
func (s *BlockLoopScratch) keepLegacyMotion() bool {
	return !s.Refmvs.bound || refmvsOracleEnabled
}

// EncoderReferenceMVStack runs dav1d_refmvs_find for req on the bound rows
// and returns the stack in the shape the mode writers and MV resolution
// consume: the candidates, the raw packed mode context and, for single
// references, the nearest/near pair (dav1d's first two stack entries,
// global-MV filled). The goav1_refmvs_oracle build diffs it against
// BuildReferenceMVStack(req).
func (s *BlockLoopScratch) EncoderReferenceMVStack(req *ReferenceMVStackRequest, block BlockVisit) ReferenceMVStackResult {
	refs := req.References
	var found refmvsResult
	target := [2]int8{int8(refs.Ref[0]) + 1, -1}
	if refs.Compound {
		target[1] = int8(refs.Ref[1]) + 1
	}
	s.Refmvs.find(&found, target, block.Size, block.Edge&edgeI444TopHasRight != 0, int32(block.MIRow), int32(block.MICol), [2]motion.Vector{})
	var out ReferenceMVStackResult
	out.ModeContext = uint16(found.rawCtx)
	out.Stack.Count = uint8(found.count)
	for i := range found.count {
		cand := &found.stack[i]
		out.Stack.Candidates[i] = ReferenceMVCandidate{
			This:     cand.mv[0].vector(),
			Compound: cand.mv[1].vector(),
			Weight:   uint16(min(cand.weight, 0xffff)),
		}
	}
	if !refs.Compound {
		out.Stack.SingleRefMVs[0] = found.stack[0].mv[0].vector()
		out.Stack.SingleRefMVs[1] = found.stack[1].mv[0].vector()
		out.Stack.SingleRefValid = true
	}
	if refmvsOracleEnabled {
		refmvsOracleEncoderStack(&s.Mode, req, refs, &out)
	}
	return out
}

// EncoderCommitInter marks a committed inter block in s.Mode
// (MarkInterMotionAndFilters) and, while refmvs rows are bound, splats it
// into them and skips the legacy neighbor-motion state.
func (s *BlockLoopScratch) EncoderCommitInter(block BlockVisit, m InterMotionResult, filters motion.InterpFilters, hasChroma bool) error {
	if !s.Refmvs.bound {
		return s.Mode.MarkInterMotionAndFilters(block.Size, int(block.X4), int(block.Y4), m, filters, hasChroma)
	}
	if err := s.Mode.markInterMotionAndFilters(block.Size, int(block.X4), int(block.Y4), m, filters, hasChroma, s.keepLegacyMotion()); err != nil {
		return err
	}
	s.Refmvs.splatInterBlock(block.Size, int32(block.MIRow), int32(block.MICol), m)
	return nil
}

// EncoderCommitIntra is EncoderCommitInter for an intra block (MarkIntra).
func (s *BlockLoopScratch) EncoderCommitIntra(block BlockVisit, mode IntraMode) error {
	if !s.Refmvs.bound {
		return s.Mode.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, mode)
	}
	if err := s.Mode.markIntra(block.Size, int(block.X4), int(block.Y4), true, mode, s.keepLegacyMotion()); err != nil {
		return err
	}
	s.Refmvs.splatIntraBlock(block.Size, int32(block.MIRow), int32(block.MICol))
	return nil
}
