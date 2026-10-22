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

// referenceMVStack returns the MV candidate stack for req: dav1d_refmvs_find
// on the bound refmvs rows when modeCtx is the bound scratch's mode context,
// else the libaom-shaped BuildReferenceMVStack.
func (st *lossyEncodeState) referenceMVStack(modeCtx *tile.BlockModeContext, req tile.ReferenceMVStackRequest, block tile.BlockVisit) (tile.ReferenceMVStackResult, error) {
	if s := st.refmvsScratch; s != nil && &s.Mode == modeCtx && s.EncoderRefmvsBound() {
		return s.EncoderReferenceMVStack(&req, block), nil
	}
	return modeCtx.BuildReferenceMVStack(req)
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
