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
	"github.com/thesyncim/goav1/internal/av1/threading"
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
// unit row. Deblocking region R starts once reconstruction reached the end
// of the superblock row below it: the vertical edges of R rewrite its last
// row, which that superblock row's intra prediction reads as its top edge
// (dav1d instead keeps a pre-filter copy of that line, ipred_edge).

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
	if ctx.Output == nil || ctx.Output.Layout.BytesPerSample != 1 || ctx.Output.Format.BitDepth != 8 {
		return false
	}
	if remaining.Has(FrameWorkPostFilterSuperRes) || remaining.Has(FrameWorkPostFilterLoopRestoration) {
		return false
	}
	return remaining&(FrameWorkPostFilterLoopFilter|FrameWorkPostFilterCDEF) != 0 &&
		!threading.FrameWorkLoopFilterCompactStateActive(ctx.compactLoopFilterState)
}

// rowPipelineEligible reports whether this frame's remaining post-filter can
// run as the frame-threaded row pipeline.
func (ctx *FrameWorkPostFilterContext) rowPipelineEligible(banding FrameWorkPostFilterBanding, remaining FrameWorkPostFilterStage, follow bool) bool {
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
	return true
}

func (ctx *FrameWorkPostFilterContext) multiTile() bool {
	return ctx.Event.TileInfo.Cols > 1 || ctx.Event.TileInfo.Rows > 1
}

// frameWorkRowPipe is the CDEF side of the row pipeline: dav1d runs each
// superblock row's CDEF as its own task once that row is deblocked, so the
// CDEF of finished rows runs on other threads while the next rows deblock.
// Unit rows are filtered in place against two-row boundary snapshots taken
// before any neighbour is filtered (the parallel 8-bit band walk), on
// goroutines owned by the FrameWorkPostFilterParallel.
type frameWorkRowPipe struct {
	units   chan int
	started int
	wg      sync.WaitGroup
	err     atomic.Pointer[wavefrontPostFilterError]

	// ctx and req are the frame's CDEF band context and request, written by
	// the driver before it sends the first unit and read by the workers.
	ctx      FrameWorkPostFilterContext
	req      FrameWorkCDEFPostFilterRequest
	cdefRows int
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

// rowPipeUnit filters CDEF unit row u and publishes the finished prefix.
func (p *FrameWorkPostFilterParallel) rowPipeUnit(w, u int) {
	pp := &p.pipe
	defer pp.wg.Done()
	if pp.err.Load() != nil {
		return
	}
	request := pp.req
	request.SampleScratch = p.cdef[w].line
	request.InputScratch = p.cdef[w].input
	res, err := pp.ctx.applyCDEFPostFilterRows(request, u, u+1, false, &p.cdefU8[u])
	if err != nil {
		pp.err.CompareAndSwap(nil, &wavefrontPostFilterError{err: err})
		return
	}
	p.accumulateCDEFResult(w, res)
	p.rows.finishUnit(pp.ctx.Output, u, pp.cdefRows)
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

// applyRowPipelinedPostFilters runs deblock and CDEF behind reconstruction and
// publishes the rows they finish. The calling goroutine deblocks one mask
// region at a time, in order (dav1d's DEBLOCK_COLS / DEBLOCK_ROWS tasks, which
// wait on the previous row's deblock_progress); each CDEF unit row is handed
// to the unit workers as soon as the rows it reads are deblocked.
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
	_, cdefRows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}

	pp := &p.pipe
	workers := 0
	next := 0
	if cdefOn {
		if err := ctx.validateCDEFPostFilterRequest(req.CDEF); err != nil {
			return err
		}
		// Every started unit worker indexes its own scratch, so the scratch
		// covers all of them, not just the ones this frame needs.
		workers = max(1, p.Workers, pp.started)
		lumaWidth, chromaWidth := frameWorkCDEFParallelPlaneWidths(ctx)
		p.ensureCDEFScratch(workers, lumaWidth, chromaWidth, true, cdefRows)
		p.resetCDEFResults(workers)
		if cap(rows.done) < cdefRows {
			rows.done = make([]bool, cdefRows)
		}
		rows.done = rows.done[:cdefRows]
		clear(rows.done)
		rows.prefix = 0
		pp.ctx = *ctx
		// The units run in band mode: their deblocked inputs are final.
		frameWorkMarkCompletedPostFilters(&pp.ctx, FrameWorkPostFilterLoopFilter)
		pp.req = req.CDEF
		pp.cdefRows = cdefRows
		pp.err.Store(nil)
		p.startRowPipeWorkers(workers)
	}
	// dispatch hands unit rows [next, end) to the workers once their
	// boundary rows are saved.
	dispatch := func(end int) error {
		for u := next; u < end; u++ {
			if err := ctx.loadCDEFRowBoundary(&p.cdefU8[u], u, cdefRows, false); err != nil {
				return err
			}
			if u+1 < cdefRows {
				if err := ctx.loadCDEFRowBoundary(&p.cdefU8[u+1], u+1, cdefRows, true); err != nil {
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
		if cdefOn {
			pp.wg.Wait()
			if e := pp.err.Load(); e != nil && err == nil {
				err = e.err
			}
			pp.ctx = FrameWorkPostFilterContext{}
			pp.req = FrameWorkCDEFPostFilterRequest{}
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
			if !cdefOn {
				if !last {
					// The next region's horizontal edges rewrite up to six
					// rows above it.
					rows.publish(out, 128*(region+1)-8)
				}
				continue
			}
			ready := min(2*region+1, cdefRows)
			if last {
				ready = cdefRows
			}
			if err := dispatch(ready); err != nil {
				return finish(err)
			}
		}
	}
	if cdefOn {
		for u := next; u < cdefRows; u++ {
			need := 64*(u+1) + sbPx
			if u+1 == cdefRows {
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
	completed := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF
	frameWorkMarkCompletedPostFilters(ctx, completed)
	result.Completed |= completed & remaining
	if lfOn {
		result.LoopFilter = lfResult
	}
	if cdefOn {
		result.CDEF = p.mergeCDEFResults(workers)
	}
	return nil
}
