package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// Superblock-row post-filter pipeline, ported from dav1d's filter_sbrow
// ordering (src/recon_tmpl.c dav1d_filter_sbrow_deblock_cols/_rows,
// dav1d_filter_sbrow_cdef, dav1d_filter_sbrow_lr, driven per superblock row
// by src/decode.c): instead of a whole-frame deblock pass, then a whole-frame
// CDEF pass, then a whole-frame restoration pass, each superblock row is
// deblocked, then CDEF and loop restoration follow as soon as the rows they
// read are final.
//
// The stages run on goav1's units: deblock per 128-pixel mask region row R,
// CDEF per 64-pixel unit row u, restoration per processing stripe. After
// deblocking region R, every pixel above row 128(R+1)-6 is deblock-final (the
// next region's horizontal edge filter reaches at most six rows up), so CDEF
// unit rows u <= 2R, whose 2-row bottom halo ends at 64u+66, are ready. Per
// unit row u:
//
//   - the restoration context lines sourced in its rows are saved from the
//     deblocked pixels before CDEF overwrites them (dav1d copy_lpf in
//     filter_sbrow_deblock_rows); every such line lies inside one unit row;
//   - CDEF filters it in place, resuming the walk's pre-filter line backup
//     from the previous unit row and reading the not yet filtered rows below
//     straight from the frame;
//   - the post-CDEF frame-edge context lines sourced in it are saved;
//   - restoration stripe u, whose rows end at 64u+56, is filtered (dav1d
//     lr_sbrow), with the last unit row also flushing any final short stripe.
//
// Every stage therefore reads exactly the pixels the whole-frame order gives
// it, and the output is bit-identical. Phase 1 covers the serial 8-bit
// whole-frame chain (no pool lanes, no caller fan-out, no banding, no
// super-res, saved restoration boundaries); every other frame keeps the
// whole-frame stage order.

// sbrowPostFilterEligible reports whether this frame's deblock, CDEF and
// restoration can run as the per-superblock-row pipeline.
func (ctx *FrameWorkPostFilterContext) sbrowPostFilterEligible(req FrameWorkPostFilterRequest, banding FrameWorkPostFilterBanding, remaining FrameWorkPostFilterStage) bool {
	if ctx.Output == nil || ctx.Output.Layout.BytesPerSample != 1 || ctx.Output.Format.BitDepth != 8 {
		return false
	}
	need := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration
	if remaining&need != need || remaining.Has(FrameWorkPostFilterSuperRes) {
		return false
	}
	if banding.LoopFilterMIRows > 0 || banding.CDEFUnitRows > 0 || ctx.Parallel.workers() > 0 {
		return false
	}
	if ctx.pool != nil && ctx.postFilterWorkerCount() > 1 {
		return false
	}
	if !ctx.loopFilterMasksUsable() || ctx.superResActiveForRestorationSave() || req.Restoration.Optimized {
		return false
	}
	boundaries := req.Restoration.Boundaries
	if frameWorkRestorationBoundariesEmpty(boundaries) && ctx.RestorationFrameBuffers != nil {
		boundaries = ctx.RestorationFrameBuffers.Boundaries
	}
	return !frameWorkRestorationBoundariesEmpty(boundaries)
}

// applySbrowPostFilters runs deblock, CDEF and loop restoration as the
// superblock-row pipeline and marks the three stages complete.
func (ctx *FrameWorkPostFilterContext) applySbrowPostFilters(req FrameWorkPostFilterRequest, result *FrameWorkPostFilterResult) error {
	masks := ctx.LoopFilterMasks
	lfResult, levelCtx, maxPlane, err := ctx.beginLoopFilterEdgesFromMasks(masks, req.LoopFilter.Map)
	if err != nil {
		return err
	}

	// Restoration inputs, bound as ApplyLoopRestorationPostFilter binds them.
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		return err
	}
	lr := req.Restoration
	if ctx.RestorationFrameBuffers != nil {
		if frameWorkRestorationRecordsEmpty(lr.Records) {
			lr.Records = ctx.RestorationFrameBuffers.Records
		}
		if frameWorkRestorationBoundariesEmpty(lr.Boundaries) {
			lr.Boundaries = ctx.RestorationFrameBuffers.Boundaries
		}
	}
	lr.Boundaries = ctx.restorationPixelBoundaries(lr.Boundaries)
	applySamples, err := tile.RestorationFrameSampleScratchLen(plan, *ctx.Output, false)
	if err != nil {
		return err
	}
	dataScratch := lr.DataScratch
	if len(dataScratch) < applySamples.DataLen {
		return frame.ErrShortBuffer
	}
	dataScratch = dataScratch[:applySamples.DataLen]
	saveDeblockLines := plan.Active && ctx.shouldSaveRestorationDeblockBoundaries()

	_, cdefRows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}
	stripes := 0
	if plan.Active {
		for p := 0; p < int(plan.Planes); p++ {
			stripes = max(stripes, tile.RestorationTileStripeCount(plan.Grids[p]))
		}
	}

	var cdefResult FrameWorkCDEFPostFilterResult
	var lrResult tile.RestorationFrameApplyResult
	cdefResume := FrameWorkCDEFPostFilterU8BandBoundary{resume: true}
	const blockRows = 64
	unitRow := func(u int) error {
		y0, y1 := u*blockRows, (u+1)*blockRows
		if u+1 == cdefRows {
			y1 = int(^uint(0) >> 1)
		}
		if saveDeblockLines {
			if err := tile.SaveRestorationFrameBoundaryLinesFromFrameRows(plan, *ctx.Output, 1, lr.Boundaries, false, y0, y1); err != nil {
				return err
			}
		}
		res, err := ctx.applyCDEFPostFilterRows(req.CDEF, u, u+1, false, &cdefResume)
		if err != nil {
			return err
		}
		cdefResult.Units += res.Units
		cdefResult.Blocks += res.Blocks
		cdefResult.Planes = max(cdefResult.Planes, res.Planes)
		if !plan.Active {
			return nil
		}
		if err := tile.SaveRestorationFrameBoundaryLinesFromFrameRows(plan, *ctx.Output, 1, lr.Boundaries, true, y0, y1); err != nil {
			return err
		}
		sEnd := u + 1
		if u+1 == cdefRows {
			sEnd = stripes
		}
		return tile.ApplyRestorationFrameToFrameStripesU8(plan, *ctx.Output, lr.Records, lr.Boundaries, dataScratch, lr.Scratch, u, sEnd, &lrResult)
	}

	next := 0
	for region := 0; region < masks.SB128H; region++ {
		if err := ctx.applyLoopFilterEdgesFromMasksInPlanePassOrder(&lfResult, masks, &levelCtx, loopfilter.PlaneY, maxPlane, region, region+1); err != nil {
			return err
		}
		ready := min(2*region+1, cdefRows)
		if region+1 == masks.SB128H {
			ready = cdefRows
		}
		for ; next < ready; next++ {
			if err := unitRow(next); err != nil {
				return err
			}
		}
	}
	for ; next < cdefRows; next++ {
		if err := unitRow(next); err != nil {
			return err
		}
	}

	frameWorkMarkCompletedPostFilters(ctx, FrameWorkPostFilterLoopFilter|FrameWorkPostFilterCDEF|FrameWorkPostFilterLoopRestoration)
	result.Completed |= FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration
	result.LoopFilter = lfResult
	result.CDEF = cdefResult
	result.Restoration = lrResult
	return nil
}
