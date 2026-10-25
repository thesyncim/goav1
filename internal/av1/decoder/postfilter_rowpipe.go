// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import (
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// Frame-threaded superblock-row post-filter pipeline (dav1d's per-sbrow
// DAV1D_TASK_TYPE_DEBLOCK_COLS / DEBLOCK_ROWS / CDEF / LOOP_RESTORATION
// tasks, src/thread_task.c, running src/recon_tmpl.c filter_sbrow_*): the
// post-filter of a frame-threaded frame follows its reconstruction row by row
// instead of starting once every row is reconstructed, and publishes each
// finished row to the frames that predict from it.
//
// The stages run on the units of the serial sbrow pipeline
// (postfilter_sbrow.go): deblock per 128-row mask region, CDEF per 64-row
// unit row, loop restoration per 64-row processing stripe (rows 64s-8 to
// 64s+56). Deblocking region R starts once reconstruction reached the end
// of the superblock row below it: the vertical edges of R rewrite its last
// row, which that superblock row's intra prediction reads as its top edge
// (dav1d instead keeps a pre-filter copy of that line, ipred_edge).
//
// The calling goroutine deblocks the regions in order (dav1d's DEBLOCK_COLS /
// DEBLOCK_ROWS tasks wait on the previous row's deblock_progress). Once the
// rows a unit row reads are deblocked, the driver saves what its neighbours
// need from it before it changes (CDEF's two-row boundary snapshots, and the
// deblocked restoration stripe context lines sourced in it, dav1d copy_lpf)
// and hands it to the unit workers, which run its CDEF and save the post-CDEF
// frame-edge context lines. A restoration stripe runs as soon as the two unit
// rows it covers are done (dav1d's CDEF task falls through to the stripe's
// LOOP_RESTORATION). Every stage reads exactly the pixels the whole-frame
// order gives it, so the output is unchanged; the published rows are the
// prefix no later stage writes.

// BindReconProgress sets the reconstruction progress the next post-filter run
// follows, or clears it with nil.
func (p *FrameWorkPostFilterParallel) BindReconProgress(progress *threading.FrameProgress) {
	p.recon = progress
}

// BindParseProgress sets the entropy-pass progress of the frame the next
// post-filter run follows (FrameProgressDone once the pass finished), or
// clears it with nil.
func (p *FrameWorkPostFilterParallel) BindParseProgress(progress *threading.FrameProgress) {
	p.parsed = progress
}

// waitParsed blocks until the frame's entropy pass finished; it reports false
// when the pass failed.
func (ctx *FrameWorkPostFilterContext) waitParsed() bool {
	if ctx.Parallel != nil && ctx.Parallel.parsed != nil {
		return ctx.Parallel.parsed.Wait(threading.FrameProgressDone)
	}
	return true
}

// rowPipelineFollowsParse reports whether the row pipeline may start while
// the entropy pass still runs: every side datum it reads for a row is final
// once that row is reconstructed. A multi-tile frame's deblock masks are then
// fixed up across tile boundaries one region at a time, from the tile-edge
// backups (without them the fix needs every tile parsed), and levels not
// written at decode time would be derived from the whole record map.
func (ctx *FrameWorkPostFilterContext) rowPipelineFollowsParse(remaining FrameWorkPostFilterStage) bool {
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		m := ctx.LoopFilterMasks
		if !m.Valid() || !m.LevelsAtDecode() {
			return false
		}
		if ctx.multiTile() && !m.TileEdgesUsable() {
			return false
		}
	}
	return !threading.FrameWorkLoopFilterCompactStateActive(ctx.compactLoopFilterState)
}

// reconProgress is the bound reconstruction progress, or nil.
func (ctx *FrameWorkPostFilterContext) reconProgress() *threading.FrameProgress {
	if ctx.Parallel == nil {
		return nil
	}
	return ctx.Parallel.recon
}

// waitReconstructed blocks until every reconstruction row is final; it
// reports false when reconstruction failed.
func (ctx *FrameWorkPostFilterContext) waitReconstructed() bool {
	if p := ctx.reconProgress(); p != nil {
		return p.Wait(threading.FrameProgressDone)
	}
	return true
}

// rowPipelineCandidate reports whether the frame's remaining post-filter
// stages are ones the row pipeline runs, before the side data that
// rowPipelineEligible also checks is complete.
func (ctx *FrameWorkPostFilterContext) rowPipelineCandidate(remaining FrameWorkPostFilterStage) bool {
	if ctx.Output == nil || remaining.Has(FrameWorkPostFilterSuperRes) {
		return false
	}
	switch {
	case ctx.Output.Layout.BytesPerSample == 1 && ctx.Output.Format.BitDepth == 8:
	case ctx.Output.Layout.BytesPerSample == 2 && (ctx.Output.Format.BitDepth == 10 || ctx.Output.Format.BitDepth == 12):
	default:
		return false
	}
	return remaining&(FrameWorkPostFilterLoopFilter|FrameWorkPostFilterCDEF|FrameWorkPostFilterLoopRestoration) != 0 &&
		!threading.FrameWorkLoopFilterCompactStateActive(ctx.compactLoopFilterState)
}

// rowPipelineEligible reports whether this frame's remaining post-filter can
// run as the frame-threaded row pipeline.
func (ctx *FrameWorkPostFilterContext) rowPipelineEligible(req FrameWorkPostFilterRequest, banding FrameWorkPostFilterBanding, remaining FrameWorkPostFilterStage, follow bool) bool {
	p := ctx.Parallel
	if p == nil || p.recon == nil || p.rows.progress == nil {
		return false
	}
	if !ctx.rowPipelineCandidate(remaining) || banding.LoopFilterMIRows > 0 || banding.CDEFUnitRows > 0 {
		return false
	}
	if remaining.Has(FrameWorkPostFilterLoopFilter) && !ctx.loopFilterMasksUsable() {
		// A multi-tile frame followed during its parse fixes its tile
		// boundaries region by region.
		if !follow || !ctx.multiTile() || !ctx.LoopFilterMasks.TileEdgesUsable() {
			return false
		}
	}
	if remaining.Has(FrameWorkPostFilterLoopRestoration) {
		if req.Restoration.Optimized || ctx.superResActiveForRestorationSave() {
			return false
		}
		boundaries := req.Restoration.Boundaries
		if frameWorkRestorationBoundariesEmpty(boundaries) && ctx.RestorationFrameBuffers != nil {
			boundaries = ctx.RestorationFrameBuffers.Boundaries
		}
		if frameWorkRestorationBoundariesEmpty(boundaries) {
			return false
		}
	}
	return true
}

func (ctx *FrameWorkPostFilterContext) multiTile() bool {
	return ctx.Event.TileInfo.Cols > 1 || ctx.Event.TileInfo.Rows > 1
}

// frameWorkRowPipe is the unit-row side of the row pipeline: dav1d runs each
// superblock row's CDEF (and, falling through, its loop restoration) as its
// own task once that row is deblocked, so finished rows filter on other
// threads while the next rows deblock. Unit rows are filtered in place
// against two-row boundary snapshots taken before any neighbour is filtered
// (the parallel 8-bit band walk), on goroutines owned by the
// FrameWorkPostFilterParallel.
type frameWorkRowPipe struct {
	units   chan int
	started int
	wg      sync.WaitGroup
	err     atomic.Pointer[wavefrontPostFilterError]

	// The frame's unit-row job, written by the driver before it sends the
	// first unit row and read by the workers.
	ctx      FrameWorkPostFilterContext
	req      FrameWorkCDEFPostFilterRequest
	cdefOn   bool
	lrOn     bool
	unitRows int
	// hbd frames filter CDEF from the pre-CDEF snapshot in req.SampleScratch
	// (the uint16 band walk), which the driver fills one unit row at a time:
	// snap is each plane's bound snapshot and snapRows the rows loaded.
	hbd      bool
	snap     [3]frame.SamplePlane
	snapSrc  [3]frame.Plane
	snapRows [3]int
	snapOn   [3]bool

	// Loop restoration: the plan, the decoded records and stripe boundary
	// lines, each plane's data segment in a worker's band buffer, and the
	// stripe count over the planes.
	plan       tile.RestorationFramePlan
	records    [3][]tile.RestorationUnitRecord
	boundaries [3]tile.RestorationStripeBoundaries
	dataOff    [3]int
	dataLen    [3]int
	stripes    int

	// unitDone, stripeClaimed and stripeDone (guarded by
	// FrameWorkPostFilterParallel.rows.mu) order the restoration stripes
	// after the unit rows they cover.
	unitDone      []bool
	stripeClaimed []bool
	stripeDone    []bool
	stripePrefix  int
}

type wavefrontPostFilterError struct{ err error }

// startRowPipeWorkers makes sure n unit workers run.
func (p *FrameWorkPostFilterParallel) startRowPipeWorkers(n int) {
	pp := &p.pipe
	if pp.units == nil {
		pp.units = make(chan int, 64)
	}
	for ; pp.started < n; pp.started++ {
		go p.rowPipeWorker(pp.started, pp.units)
	}
}

func (p *FrameWorkPostFilterParallel) rowPipeWorker(w int, units <-chan int) {
	for u := range units {
		p.rowPipeUnit(w, u)
	}
}

// closeRowPipe stops the unit workers.
func (p *FrameWorkPostFilterParallel) closeRowPipe() {
	if p.pipe.units != nil {
		close(p.pipe.units)
		p.pipe.units = nil
		p.pipe.started = 0
	}
}

func (pp *frameWorkRowPipe) fail(err error) {
	pp.err.CompareAndSwap(nil, &wavefrontPostFilterError{err: err})
}

// rowPipeUnit runs unit row u on worker w: its CDEF, the post-CDEF
// restoration lines sourced in it, and every restoration stripe it completes,
// then publishes the finished prefix.
func (p *FrameWorkPostFilterParallel) rowPipeUnit(w, u int) {
	pp := &p.pipe
	defer pp.wg.Done()
	if pp.err.Load() != nil {
		return
	}
	out := pp.ctx.Output
	if pp.cdefOn {
		request := pp.req
		request.InputScratch = p.cdef[w].input
		var res FrameWorkCDEFPostFilterResult
		var err error
		if pp.hbd {
			request.UnitDstScratch = p.cdef[w].unitDst
			res, err = pp.ctx.applyCDEFPostFilterRows(request, u, u+1, false, nil)
		} else {
			request.SampleScratch = p.cdef[w].line
			res, err = pp.ctx.applyCDEFPostFilterRows(request, u, u+1, false, &p.cdefU8[u])
		}
		if err != nil {
			pp.fail(err)
			return
		}
		p.accumulateCDEFResult(w, res)
	}
	if !pp.lrOn {
		p.rows.finishUnit(out, u, pp.unitRows)
		return
	}
	y0, y1 := u*cdef.BlockSize, (u+1)*cdef.BlockSize
	if u+1 == pp.unitRows {
		y1 = int(^uint(0) >> 1)
	}
	if err := tile.SaveRestorationFrameBoundaryLinesFromFrameRows(pp.plan, *out, out.Layout.BytesPerSample, pp.boundaries, true, y0, y1); err != nil {
		pp.fail(err)
		return
	}
	// Stripe s covers unit rows s-1 and s (the last stripes stop at the last
	// unit row), so u completes at most stripes u and u+1.
	var run [2]int
	n := 0
	p.rows.mu.Lock()
	pp.unitDone[u] = true
	for s := u; s <= u+1 && s < pp.stripes; s++ {
		if !pp.stripeClaimed[s] && (s == 0 || pp.unitDone[s-1]) && pp.unitDone[min(s, pp.unitRows-1)] {
			pp.stripeClaimed[s] = true
			run[n] = s
			n++
		}
	}
	p.rows.mu.Unlock()
	for _, s := range run[:n] {
		if err := p.rowPipeStripe(w, s); err != nil {
			pp.fail(err)
			return
		}
		p.rows.mu.Lock()
		pp.stripeDone[s] = true
		for pp.stripePrefix < pp.stripes && pp.stripeDone[pp.stripePrefix] {
			pp.stripePrefix++
		}
		if pp.stripePrefix < pp.stripes {
			// Stripe s ends 8 rows above unit row s's end.
			p.rows.publish(out, pp.stripePrefix*cdef.BlockSize-8)
		}
		p.rows.mu.Unlock()
	}
}

// rowPipeStripe restores processing stripe s of every plane in place with
// worker w's scratch (dav1d lr_sbrow).
func (p *FrameWorkPostFilterParallel) rowPipeStripe(w, s int) error {
	pp := &p.pipe
	sc := &p.lr[w]
	scratch := tile.RestorationUnitRecordBoundaryScratch{
		Unit: tile.RestorationUnitScratch{Wiener: sc.wiener, SGRProj: sc.sgr},
	}
	for plane := 0; plane < int(pp.plan.Planes); plane++ {
		grid := pp.plan.Grids[plane]
		if grid.Type == parser.RestorationNone || s >= tile.RestorationTileStripeCount(grid) {
			continue
		}
		buffer, ok := frameWorkCDEFPlane(*pp.ctx.Output, plane)
		if !ok {
			return frame.ErrInvalidPlane
		}
		seg := sc.data[pp.dataOff[plane] : pp.dataOff[plane]+pp.dataLen[plane]]
		var r tile.RestorationPlaneApplyResult
		var err error
		if out := pp.ctx.Output; out.Layout.BytesPerSample == 2 {
			r, err = tile.ApplyRestorationFramePlaneInPlaceU16StripesFollowing(grid, pp.records[plane], pp.boundaries[plane], buffer, out.Format.BitDepth, seg, scratch, s, s+1)
		} else {
			r, err = tile.ApplyRestorationFramePlaneInPlaceU8StripesFollowing(grid, pp.records[plane], pp.boundaries[plane], buffer, seg, scratch, s, s+1)
		}
		if err != nil {
			return err
		}
		pr := &sc.frame.PlaneResults[plane]
		pr.Records += r.Records
		pr.FilteredRecords += r.FilteredRecords
		pr.Stripes += r.Stripes
		pr.ProcessingUnits += r.ProcessingUnits
	}
	return nil
}

// finishUnit records CDEF unit row u as filtered and publishes the rows of the
// new finished prefix with their right padding. The last unit row is left to
// the final publish, after the bottom padding is filled.
func (r *frameWorkPostFilterRowProgress) finishUnit(out *frame.Frame, u, units int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.done[u] = true
	for r.prefix < units && r.done[r.prefix] {
		r.prefix++
	}
	if r.prefix >= units {
		return
	}
	r.publish(out, r.prefix*cdef.BlockSize)
}

// loadCDEFRowBoundary snapshots, before any neighbour of unit row u is
// filtered, the deblocked rows u reads across its edge: the two rows above
// it (top) or the two rows below it (bottom).
func (ctx *FrameWorkPostFilterContext) loadCDEFRowBoundary(boundary *FrameWorkCDEFPostFilterU8BandBoundary, u, rows int, top bool) error {
	rowStart, rowEnd := 0, u+1
	if top {
		rowStart, rowEnd = u, rows
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		if err := loadCDEFU8BandBoundaryPlane(boundary, plane, planeFrame, rowStart, rowEnd, rows, yDec); err != nil {
			return err
		}
	}
	return nil
}

// applyRowPipelinedPostFilters runs deblock, CDEF and loop restoration behind
// reconstruction and publishes the rows they finish.
func (ctx *FrameWorkPostFilterContext) applyRowPipelinedPostFilters(req FrameWorkPostFilterRequest, remaining FrameWorkPostFilterStage, result *FrameWorkPostFilterResult) error {
	p := ctx.Parallel
	recon := p.recon
	rows := &p.rows
	out := ctx.Output
	sbPx := 64
	if ctx.Event.SequenceHeader.Use128x128Superblock {
		sbPx = 128
	}
	waitRecon := func(n int) error {
		if !recon.Wait(int32(min(n, int(threading.FrameProgressDone)))) {
			return threading.ErrFrameThreadReference
		}
		return nil
	}

	lfOn := remaining.Has(FrameWorkPostFilterLoopFilter)
	cdefOn := remaining.Has(FrameWorkPostFilterCDEF) && frameWorkCDEFHasFiltering(ctx.Event.CDEF, out.Format.MonoChrome)

	var lfResult FrameWorkLoopFilterPostFilterApplyResult
	var levelCtx frameWorkLoopFilterLevelContext
	maxPlane := loopfilter.PlaneV
	masks := ctx.LoopFilterMasks
	// A multi-tile frame still being parsed fixes its masks' tile boundaries
	// one region at a time, just before deblocking it.
	fixRegions := lfOn && !ctx.loopFilterMasksUsable()
	if lfOn {
		var err error
		lfResult, levelCtx, maxPlane, err = ctx.beginLoopFilterEdgesFromMasks(masks, req.LoopFilter.Map)
		if err != nil {
			return err
		}
	}
	_, unitRows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}

	pp := &p.pipe
	pp.cdefOn = cdefOn
	pp.lrOn = false
	pp.unitRows = unitRows
	saveDeblockLines := false
	if remaining.Has(FrameWorkPostFilterLoopRestoration) {
		plan, err := ctx.LoopRestorationPostFilterPlan()
		if err != nil {
			return err
		}
		if plan.Active {
			if err := ctx.bindRowPipeRestoration(req.Restoration, plan); err != nil {
				return err
			}
			saveDeblockLines = ctx.shouldSaveRestorationDeblockBoundaries()
		}
	}
	units := cdefOn || pp.lrOn
	workers := 0
	next := 0
	if units {
		if cdefOn {
			if err := ctx.validateCDEFPostFilterRequest(req.CDEF); err != nil {
				return err
			}
		}
		// Every started unit worker indexes its own scratch, so the scratch
		// covers all of them, not just the ones this frame needs.
		workers = max(1, p.Workers, pp.started)
		pp.hbd = out.Layout.BytesPerSample == 2
		if cdefOn {
			lumaWidth, chromaWidth := frameWorkCDEFParallelPlaneWidths(ctx)
			p.ensureCDEFScratch(workers, lumaWidth, chromaWidth, !pp.hbd, unitRows)
			p.resetCDEFResults(workers)
			if pp.hbd {
				if err := ctx.bindRowPipeCDEFSnapshot(req.CDEF); err != nil {
					return err
				}
			}
		}
		if pp.lrOn {
			p.ensureRestorationScratch(workers, pp.dataOff[2]+pp.dataLen[2], len(req.Restoration.Scratch.Unit.Wiener), len(req.Restoration.Scratch.Unit.SGRProj))
			for i := range workers {
				p.lr[i].frame = tile.RestorationFrameApplyResult{}
			}
			pp.unitDone = resetBools(pp.unitDone, unitRows)
			pp.stripeClaimed = resetBools(pp.stripeClaimed, pp.stripes)
			pp.stripeDone = resetBools(pp.stripeDone, pp.stripes)
			pp.stripePrefix = 0
		}
		rows.done = resetBools(rows.done, unitRows)
		rows.prefix = 0
		pp.ctx = *ctx
		// The unit rows run in band mode: their deblocked inputs are final.
		frameWorkMarkCompletedPostFilters(&pp.ctx, FrameWorkPostFilterLoopFilter)
		pp.req = req.CDEF
		pp.err.Store(nil)
		p.startRowPipeWorkers(workers)
	}
	// dispatch hands unit rows [next, end) to the workers once what their
	// neighbours read from them is saved.
	dispatch := func(end int) error {
		for u := next; u < end; u++ {
			if cdefOn && pp.hbd {
				if err := pp.loadSnapshotRows(u); err != nil {
					return err
				}
			} else if cdefOn {
				if err := ctx.loadCDEFRowBoundary(&p.cdefU8[u], u, unitRows, false); err != nil {
					return err
				}
				if u+1 < unitRows {
					if err := ctx.loadCDEFRowBoundary(&p.cdefU8[u+1], u+1, unitRows, true); err != nil {
						return err
					}
				}
			}
			if saveDeblockLines {
				y0, y1 := u*cdef.BlockSize, (u+1)*cdef.BlockSize
				if u+1 == unitRows {
					y1 = int(^uint(0) >> 1)
				}
				if err := tile.SaveRestorationFrameBoundaryLinesFromFrameRows(pp.plan, *out, out.Layout.BytesPerSample, pp.boundaries, false, y0, y1); err != nil {
					return err
				}
			}
		}
		for ; next < end; next++ {
			if pp.err.Load() != nil {
				return nil
			}
			pp.wg.Add(1)
			pp.units <- next
		}
		return nil
	}
	finish := func(err error) error {
		if units {
			pp.wg.Wait()
			if e := pp.err.Load(); e != nil && err == nil {
				err = e.err
			}
			pp.ctx = FrameWorkPostFilterContext{}
			pp.req = FrameWorkCDEFPostFilterRequest{}
			pp.records = [3][]tile.RestorationUnitRecord{}
			pp.boundaries = [3]tile.RestorationStripeBoundaries{}
			pp.snap = [3]frame.SamplePlane{}
			pp.snapSrc = [3]frame.Plane{}
		}
		return err
	}

	if lfOn {
		for region := 0; region < masks.SB128H; region++ {
			last := region+1 == masks.SB128H
			need := 128*(region+1) + sbPx
			if last {
				need = int(threading.FrameProgressDone)
			}
			if err := waitRecon(need); err != nil {
				return finish(err)
			}
			if fixRegions {
				if err := masks.FixTileBoundaryRegion(region); err != nil {
					return finish(err)
				}
			}
			if err := ctx.applyLoopFilterEdgesFromMasksInPlanePassOrder(&lfResult, masks, &levelCtx, loopfilter.PlaneY, maxPlane, region, region+1); err != nil {
				return finish(err)
			}
			if !units {
				if !last {
					// The next region's horizontal edges rewrite up to six
					// rows above it.
					rows.publish(out, 128*(region+1)-8)
				}
				continue
			}
			// Unit row u reads two rows below it, deblock-final once region
			// u/2 is done.
			ready := min(2*region+1, unitRows)
			if last {
				ready = unitRows
			}
			if err := dispatch(ready); err != nil {
				return finish(err)
			}
		}
	}
	if units {
		for u := next; u < unitRows; u++ {
			need := (u+1)*cdef.BlockSize + sbPx
			if u+1 == unitRows {
				need = int(threading.FrameProgressDone)
			}
			if err := waitRecon(need); err != nil {
				return finish(err)
			}
			if err := dispatch(u + 1); err != nil {
				return finish(err)
			}
		}
	}
	if err := finish(nil); err != nil {
		return err
	}
	// The caller pads the frame's borders next, below rows reconstruction
	// may still write when no stage waited for the last of them.
	if err := waitRecon(int(threading.FrameProgressDone)); err != nil {
		return err
	}
	if fixRegions {
		masks.MarkTileBoundariesFixed()
	}
	completed := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration
	frameWorkMarkCompletedPostFilters(ctx, completed)
	result.Completed |= completed & remaining
	if lfOn {
		result.LoopFilter = lfResult
	}
	if cdefOn {
		result.CDEF = p.mergeCDEFResults(workers)
	}
	if pp.lrOn {
		result.Restoration = p.mergeRowPipeRestorationResults(workers)
	}
	return nil
}

// bindRowPipeCDEFSnapshot binds each filtered plane's pre-CDEF snapshot in
// req.SampleScratch, as the whole-frame snapshot walk binds it; the driver
// then fills it one unit row at a time (loadSnapshotRows).
func (ctx *FrameWorkPostFilterContext) bindRowPipeCDEFSnapshot(req FrameWorkCDEFPostFilterRequest) error {
	pp := &ctx.Parallel.pipe
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	for plane := range 3 {
		pp.snapOn[plane], pp.snapRows[plane] = false, 0
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		snap, err := frame.BindSamplePlane(req.SampleScratch[plane], planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return err
		}
		pp.snap[plane], pp.snapSrc[plane], pp.snapOn[plane] = snap, planeFrame, true
	}
	return nil
}

// loadSnapshotRows copies the deblocked rows unit row u reads, down to two
// rows below it, into the snapshot before u or the row below it is filtered.
// Each row is loaded once, so no worker reads a row while it is written.
func (pp *frameWorkRowPipe) loadSnapshotRows(u int) error {
	out := pp.ctx.Output
	for plane := range 3 {
		if !pp.snapOn[plane] {
			continue
		}
		src := pp.snapSrc[plane]
		_, yDec := frameWorkCDEFPlaneDecimation(out.Format, plane)
		end := src.Height
		if u+1 < pp.unitRows {
			end = min(((u+1)*cdef.BlockSize)>>yDec+cdef.VerticalBorder, src.Height)
		}
		if end <= pp.snapRows[plane] {
			continue
		}
		if err := frame.LoadSamplePlaneRows(pp.snap[plane], src, out.Layout.BytesPerSample, pp.snapRows[plane], end); err != nil {
			return err
		}
		pp.snapRows[plane] = end
	}
	return nil
}

// bindRowPipeRestoration binds the frame's restoration inputs as
// ApplyLoopRestorationPostFilter binds them, and lays out each plane's data
// segment in a worker's band buffer.
func (ctx *FrameWorkPostFilterContext) bindRowPipeRestoration(lr FrameWorkRestorationPostFilterRequest, plan tile.RestorationFramePlan) error {
	pp := &ctx.Parallel.pipe
	if ctx.RestorationFrameBuffers != nil {
		if frameWorkRestorationRecordsEmpty(lr.Records) {
			lr.Records = ctx.RestorationFrameBuffers.Records
		}
		if frameWorkRestorationBoundariesEmpty(lr.Boundaries) {
			lr.Boundaries = ctx.RestorationFrameBuffers.Boundaries
		}
	}
	// The 8-bit stripes use the in-place segments of the frame sample
	// scratch layout; 10/12-bit stripes size their own band buffers.
	size, err := tile.RestorationFrameSampleScratchLen(plan, *ctx.Output, false)
	if err != nil {
		return err
	}
	pp.plan = plan
	pp.records = lr.Records
	pp.boundaries = ctx.restorationPixelBoundaries(lr.Boundaries)
	pp.stripes = 0
	off := 0
	for plane := range 3 {
		pp.dataOff[plane], pp.dataLen[plane] = off, 0
		if plane >= int(plan.Planes) || plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		pp.dataLen[plane] = size.Data[plane].Len
		if ctx.Output.Layout.BytesPerSample == 2 {
			if pp.dataLen[plane], err = tile.RestorationFramePlaneInPlaceU16ScratchLen(plan.Grids[plane]); err != nil {
				return err
			}
		}
		off += pp.dataLen[plane]
		pp.stripes = max(pp.stripes, tile.RestorationTileStripeCount(plan.Grids[plane]))
		if len(pp.records[plane]) != int(plan.Grids[plane].HorzUnits)*int(plan.Grids[plane].VertUnits) {
			return tile.ErrInvalidPlan
		}
	}
	pp.lrOn = true
	return nil
}

// mergeRowPipeRestorationResults sums the workers' restoration counters.
func (p *FrameWorkPostFilterParallel) mergeRowPipeRestorationResults(workers int) tile.RestorationFrameApplyResult {
	var out tile.RestorationFrameApplyResult
	plan := &p.pipe.plan
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		var pr tile.RestorationPlaneApplyResult
		for i := 0; i < workers && i < len(p.lr); i++ {
			r := &p.lr[i].frame.PlaneResults[plane]
			pr.Records += r.Records
			pr.FilteredRecords += r.FilteredRecords
			pr.Stripes += r.Stripes
			pr.ProcessingUnits += r.ProcessingUnits
		}
		out.PlaneResults[plane] = pr
		out.Planes++
		out.Records += pr.Records
		out.FilteredRecords += pr.FilteredRecords
		out.Stripes += pr.Stripes
		out.ProcessingUnits += pr.ProcessingUnits
	}
	return out
}

func resetBools(b []bool, n int) []bool {
	if cap(b) < n {
		b = make([]bool, n)
	}
	b = b[:n]
	clear(b)
	return b
}
