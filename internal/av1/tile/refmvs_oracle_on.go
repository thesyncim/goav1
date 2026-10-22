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

func refmvsOracleCompare(ctx *BlockModeContext, rt *refmvsTile, found *refmvsResult, req *BlockLoopRequest, block BlockVisit, refs InterReferencesResult, globalMVs [2]motion.Vector) {
	types := blockReferenceGlobalMotionTypes(refs, req.GlobalMotionTypes)
	stackReq := legacyReferenceMVStackRequest(*req, block, refs, globalMVs, types)
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
		msg += fmt.Sprintf("\n topHasRight=%v level=%d partition=%d", blockHasTopRight(req.SBSizeMIB, block), block.Level, block.Partition)
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

func refmvsOracleTopHasRight(sbSizeMIB uint8, block BlockVisit, has bool) {
	// Blocks wider or taller than 64 never use the flag (refmvs_find and
	// find_matching_ref gate on size first) and blockHasTopRight rejects them.
	dims := blockDimensions[block.Size]
	if max(dims.W4, dims.H4) > 16 {
		return
	}
	if want := legacyRefmvsTopHasRight(sbSizeMIB, block); want != has {
		panic(fmt.Sprintf("refmvs oracle: top-has-right edge=%#x want %v at mi_row=%d mi_col=%d size=%d level=%d partition=%d",
			block.Edge, want, block.MIRow, block.MICol, block.Size, block.Level, block.Partition))
	}
}
