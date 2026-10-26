package encoder

import "github.com/thesyncim/goav1/internal/av1/tile"

// bindRefmvs binds the tile scratch's dav1d refmvs rows for one serial
// block-loop walk over walkReq. While bound, the scratch's mode context gets
// its MV candidate stacks from refmvs_find (referenceMVStack) and committed
// blocks are splatted into the rows (EncoderCommitInter/Intra).
func (st *lossyEncodeState) bindRefmvs(scratch *tile.BlockLoopScratch, walkReq tile.BlockWalkRequest, miCols, miRows uint16) {
	tile.BindEncoderRefmvs(scratch, tile.EncoderRefmvsFrame{
		FrameMICols:    miCols,
		FrameMIRows:    miRows,
		Tile:           walkReq,
		ForceIntegerMV: st.forceIntegerMV,
	})
	st.refmvsScratch = scratch
}

// unbindRefmvs releases bindRefmvs.
func (st *lossyEncodeState) unbindRefmvs(scratch *tile.BlockLoopScratch) {
	tile.UnbindEncoderRefmvs(scratch)
	st.refmvsScratch = nil
}

// referenceMVStack writes the MV candidate stack for refs at block into dst:
// dav1d_refmvs_find on the bound refmvs rows when modeCtx is the bound
// scratch's mode context, built in place as SVT's generate_av1_mvp_table
// fills ctx->ref_mv_stack; else the libaom-shaped BuildReferenceMVStack,
// whose request is only built on that path.
func (st *lossyEncodeState) referenceMVStack(dst *tile.ReferenceMVStackResult, modeCtx *tile.BlockModeContext, refs tile.InterReferencesResult, block *tile.BlockVisit, walkReq tile.BlockWalkRequest, miCols, miRows uint16) error {
	if s := st.refmvsScratch; s != nil && &s.Mode == modeCtx && s.EncoderRefmvsBound() {
		s.EncoderReferenceMVStack(dst, refs, block)
		return nil
	}
	var err error
	*dst, err = modeCtx.BuildReferenceMVStack(st.pblockStackRequest(*block, walkReq, miCols, miRows, refs))
	return err
}

// bindRefmvsShared binds a wavefront lane's scratch to the lanes' shared
// frame-tall refmvs store.
func (st *lossyEncodeState) bindRefmvsShared(scratch *tile.BlockLoopScratch, walkReq tile.BlockWalkRequest, miCols, miRows uint16, rows *tile.EncoderRefmvsRows) {
	tile.BindEncoderRefmvsShared(scratch, tile.EncoderRefmvsFrame{
		FrameMICols:    miCols,
		FrameMIRows:    miRows,
		Tile:           walkReq,
		ForceIntegerMV: st.forceIntegerMV,
	}, rows)
	st.refmvsScratch = scratch
}
