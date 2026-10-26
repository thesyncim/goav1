//go:build goav1_refmvs_oracle

package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// Debug build: poison the refmvs row storage so any read of a cell that was
// never splatted in the current frame indexes blockDimensions out of range
// (bs = 0xff) and panics, and diff every dav1d refmvs_find result against the
// legacy libaom-shaped BuildReferenceMVStack while the legacy grids are still
// maintained.
const (
	refmvsPoisonEnabled = true
	refmvsOracleEnabled = true
)

func refmvsPoison(buf []refmvsBlock) {
	for i := range buf {
		buf[i] = refmvsBlock{mv: [2]refmvsMV{refmvsInvalidMV, refmvsInvalidMV}, ref: [2]int8{-128, -128}, bs: 0xff, mf: 0xff}
	}
}

func refmvsOracleCompare(ctx *BlockModeContext, rt *refmvsTile, found *refmvsResult, req *BlockLoopRequest, block *BlockVisit, refs InterReferencesResult, globalMVs [2]motion.Vector) {
	types := blockReferenceGlobalMotionTypes(refs, req.GlobalMotionTypes)
	stackReq := legacyReferenceMVStackRequest(*req, *block, refs, globalMVs, types)
	if stackReq.TemporalMVs == nil && rt.frame.projectTmvs {
		// Deferred projection: the legacy stack reads the oracle's own
		// whole-frame field.
		stackReq.TemporalMVs = rt.frame.temporal
	}
	legacy, err := ctx.BuildReferenceMVStack(stackReq)
	if err != nil {
		panic(err)
	}
	fail := func(what string) {
		msg := fmt.Sprintf("refmvs oracle: %s at mi_row=%d mi_col=%d size=%d refs=%v\n legacy count=%d ctx=%d stack=%+v\n dav1d  count=%d ctx=%d stack=",
			what, block.MIRow, block.MICol, block.Size, refs, legacy.Stack.Count, legacy.ModeContext, legacy.Stack.Candidates[:legacy.Stack.Count], found.count, found.ctx)
		for i := range found.count {
			c := found.stack[i]
			msg += fmt.Sprintf("{%v %v w=%d} ", c.mv[0].vector(), c.mv[1].vector(), c.weight)
		}
		by4, bx4 := int32(block.MIRow), int32(block.MICol)
		base := (by4 & 31) + 5
		msg += fmt.Sprintf("\n topHasRight=%v level=%d partition=%d", blockHasTopRight(req.SBSizeMIB, *block), block.Level, block.Partition)
		for dy := int32(-1); dy <= 1; dy++ {
			if rt.rows[base+dy] < 0 {
				continue
			}
			row := rt.row(base + dy)
			msg += fmt.Sprintf("\n row %d:", dy)
			for x := max(bx4-2, 0); x < bx4+4; x++ {
				c := row[x]
				msg += fmt.Sprintf(" [x=%d ref=%v mv=%v bs=%d mf=%d]", x, c.ref, c.mv[0].vector(), c.bs, c.mf)
			}
		}
		panic(msg)
	}
	if int32(legacy.Stack.Count) != found.count {
		fail("count")
	}
	for i := range found.count {
		l := legacy.Stack.Candidates[i]
		d := found.stack[i]
		if l.This != d.mv[0].vector() || (refs.Compound && l.Compound != d.mv[1].vector()) {
			fail(fmt.Sprintf("mv[%d]", i))
		}
		if (l.Weight >= 640) != (d.weight >= 640) {
			fail(fmt.Sprintf("weight class[%d]", i))
		}
	}
	if refs.Compound {
		want, _ := AnalyzeInterModeContext(legacy.ModeContext, true)
		if int32(want) != found.ctx {
			fail("compound ctx")
		}
	} else {
		if int32(legacy.ModeContext) != found.ctx {
			fail("ctx")
		}
		if legacy.Stack.SingleRefMVs[0] != found.stack[0].mv[0].vector() || legacy.Stack.SingleRefMVs[1] != found.stack[1].mv[0].vector() {
			fail("single ref mvs")
		}
	}
}

func refmvsOracleTopHasRight(sbSizeMIB uint8, block *BlockVisit, has bool) {
	// Blocks wider or taller than 64 never use the flag (refmvs_find and
	// find_matching_ref gate on size first) and blockHasTopRight rejects them.
	dims := blockDimensions[block.Size]
	if max(dims.W4, dims.H4) > 16 {
		return
	}
	if want := legacyRefmvsTopHasRight(sbSizeMIB, *block); want != has {
		panic(fmt.Sprintf("refmvs oracle: top-has-right edge=%#x want %v at mi_row=%d mi_col=%d size=%d level=%d partition=%d",
			block.Edge, want, block.MIRow, block.MICol, block.Size, block.Level, block.Partition))
	}
}

// refmvsOracleTmvs diffs one load_tmvs window against the whole-frame
// libaom-shaped projection when the caller also set that up.
func refmvsOracleTmvs(rt *refmvsTile, colStart8, colEnd8, rowStart8, rowEnd8 int32) {
	tmv := rt.frame.temporal
	if tmv == nil {
		return
	}
	stride := int32(tmv.Stride)
	for y := rowStart8; y < min(rowEnd8, rt.frame.mfmv.ih8); y++ {
		for x := colStart8; x < min(colEnd8, rt.frame.mfmv.iw8); x++ {
			got := rt.rpProj[(y&15)*rt.rpStride+x]
			want := tmv.Entries[y*stride+x]
			if got.Valid != want.Valid || (got.Valid && got != want) {
				panic(fmt.Sprintf("refmvs oracle: load_tmvs at y8=%d x8=%d got %+v want %+v (n_mfmvs=%d refs=%v)",
					y, x, got, want, rt.frame.mfmv.n, rt.frame.mfmv.ref))
			}
		}
	}
}

// refmvsOracleWholeField sets up the libaom-shaped whole-frame projection of
// the deferred references, which loadTmvs then diffs each window against.
func refmvsOracleWholeField(req *BlockLoopRequest) *TemporalMotionField {
	need, err := ReferenceMVFrameEntries(uint32(req.FrameMIRows), uint32(req.FrameMICols))
	if err != nil {
		panic(err)
	}
	field := new(TemporalMotionField)
	if err := field.Init(uint32(req.FrameMIRows), uint32(req.FrameMICols), make([]TemporalMotionEntry, need)); err != nil {
		panic(err)
	}
	if _, err := field.Setup(TemporalMotionSetupRequest{
		EnableOrderHint:  req.EnableOrderHint,
		OrderHintBits:    req.OrderHintBits,
		CurrentOrderHint: req.CurrentOrderHint,
		References:       *req.TemporalReferences,
	}); err != nil {
		panic(err)
	}
	return field
}

// refmvsOracleEncoderStack diffs an encoder refmvs_find stack against the
// legacy BuildReferenceMVStack one, field by field as the encoder consumes it.
func refmvsOracleEncoderStack(c *BlockModeContext, req *ReferenceMVStackRequest, refs InterReferencesResult, got *ReferenceMVStackResult) {
	if req == nil {
		return
	}
	want, err := c.BuildReferenceMVStack(*req)
	if err != nil {
		panic(err)
	}
	fail := func(what string) {
		panic(fmt.Sprintf("refmvs oracle: encoder %s at mi_row=%d mi_col=%d size=%d refs=%v\n legacy ctx=%#x %+v\n dav1d  ctx=%#x %+v",
			what, req.MIRow, req.MICol, req.Size, refs, want.ModeContext, want.Stack, got.ModeContext, got.Stack))
	}
	if want.Stack.Count != got.Stack.Count {
		fail("count")
	}
	if want.ModeContext != got.ModeContext {
		fail("mode context")
	}
	for i := range int(got.Stack.Count) {
		l, d := want.Stack.Candidates[i], got.Stack.Candidates[i]
		if l.This != d.This || (refs.Compound && l.Compound != d.Compound) {
			fail(fmt.Sprintf("mv[%d]", i))
		}
		if (l.Weight >= RefMVCategoryLevel) != (d.Weight >= RefMVCategoryLevel) {
			fail(fmt.Sprintf("weight class[%d]", i))
		}
	}
	if !refs.Compound && (want.Stack.SingleRefValid != got.Stack.SingleRefValid || want.Stack.SingleRefMVs != got.Stack.SingleRefMVs) {
		fail("single ref mvs")
	}
}
