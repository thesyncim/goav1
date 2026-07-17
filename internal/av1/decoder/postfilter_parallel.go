package decoder

import (
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// FrameWorkPostFilterParallel is caller-owned, reusable scratch that lets the
// supported post-filter stages fan their independent bands out across worker
// goroutines. It carries the worker count plus per-worker private scratch so the
// fan-out is allocation-free once warmed up.
//
// Byte-exactness: the parallel path is a pure scheduling change. Stages run in
// sequence (loop filter fully completes before CDEF), so cross-stage order is
// unchanged. Within CDEF, every band reads immutable pre-CDEF inputs — a shared
// read-only whole-frame snapshot on the uint16 path, or per-band two-row
// top/bottom boundary snapshots on the 8-bit in-place path — and writes disjoint
// output rows. Within the mask loop filter (applyLoopFilterMaskBandsParallel),
// vertical edges are region-ROW banded (write disjoint rows) and horizontal
// edges region-COLUMN banded (write disjoint columns), with a barrier between the
// two directions and even-aligned populate bands for the chroma level cache.
// Band order therefore never changes any pixel.
//
// Restoration is still applied serially (it lacks a row-range apply entry point
// in the tile package).
type FrameWorkPostFilterParallel struct {
	// Workers is the number of goroutines available to the post-filter fan-out.
	// One or zero keeps the serial path.
	Workers int

	cdef   []frameWorkParallelCDEFScratch
	cdefU8 []FrameWorkCDEFPostFilterU8BandBoundary
	pool   *threading.Pool
	runner frameWorkParallelPoolRunner
}

// BindFrameWorkPostFilterParallelPool lets a high-level decoder reuse its
// already-live tile worker lanes for postfilter bands. Tile execution has
// completed before the postfilter callback runs, so the pool is idle here.
func BindFrameWorkPostFilterParallelPool(p *FrameWorkPostFilterParallel, pool *threading.Pool) {
	if p != nil {
		p.pool = pool
	}
}

// frameWorkParallelCDEFScratch is one worker's private CDEF band scratch. It
// mirrors the per-band buffers the serial banded CDEF apply threads through a
// single request: the uint16 tap ring (InputScratch), the uint16 unit dst
// (UnitDstScratch, unused by the 8-bit in-place walk), and the byte line-backup
// arena the 8-bit walk carves from SampleScratch.
type frameWorkParallelCDEFScratch struct {
	input   []uint16
	unitDst []uint16
	line    [3][]uint16
	result  FrameWorkCDEFPostFilterResult
}

// workers reports how many goroutines the parallel path may use for the frame.
// It returns 0 when the serial path must run (nil scratch or <=1 worker).
func (p *FrameWorkPostFilterParallel) workers() int {
	if p == nil || p.Workers <= 1 {
		return 0
	}
	return p.Workers
}

// ensureCDEFScratch grows the per-worker CDEF scratch to the current worker
// count and plane width. lineWidth is the luma plane width in samples; the U8
// line-backup arena needs 2*width uint16 for luma and width for each chroma
// plane (matching testCDEFU8LineScratch / the 4*width byte view the walk uses).
func (p *FrameWorkPostFilterParallel) ensureCDEFScratch(workers, lumaWidth, chromaWidth int, wantU8 bool, bands int) {
	if cap(p.cdef) < workers {
		p.cdef = make([]frameWorkParallelCDEFScratch, workers)
	}
	p.cdef = p.cdef[:workers]
	for i := range p.cdef {
		s := &p.cdef[i]
		if len(s.input) < cdef.InputBufferSize {
			s.input = make([]uint16, cdef.InputBufferSize)
		}
		if !wantU8 {
			if len(s.unitDst) < cdef.InputBufferSize {
				s.unitDst = make([]uint16, cdef.InputBufferSize)
			}
		} else {
			// The 8-bit walk carves a 4*plane-width byte line arena from each
			// SampleScratch[plane] (cdefByteScratchView needs 4*width <= 2*len(seg)
			// uint16 => len >= 2*width). Every plane needs 2*plane-width uint16.
			need0 := 2 * lumaWidth
			needC := 2 * chromaWidth
			if len(s.line[0]) < need0 {
				s.line[0] = make([]uint16, need0)
			}
			if len(s.line[1]) < needC {
				s.line[1] = make([]uint16, needC)
			}
			if len(s.line[2]) < needC {
				s.line[2] = make([]uint16, needC)
			}
		}
	}
	if wantU8 {
		if cap(p.cdefU8) < bands {
			p.cdefU8 = make([]FrameWorkCDEFPostFilterU8BandBoundary, bands)
		}
		p.cdefU8 = p.cdefU8[:bands]
		for i := range p.cdefU8 {
			p.cdefU8[i].ensure(lumaWidth, chromaWidth)
		}
	}
}

// ensure sizes the two-row top/bottom boundary strips for a U8 band. Top/Bottom
// hold cdef.VerticalBorder rows of plane width per plane.
func (b *FrameWorkCDEFPostFilterU8BandBoundary) ensure(lumaWidth, chromaWidth int) {
	widths := [3]int{lumaWidth, chromaWidth, chromaWidth}
	for plane, w := range widths {
		need := cdef.VerticalBorder * w
		if cap(b.Top[plane]) < need {
			b.Top[plane] = make([]byte, need)
		}
		b.Top[plane] = b.Top[plane][:need]
		if cap(b.Bottom[plane]) < need {
			b.Bottom[plane] = make([]byte, need)
		}
		b.Bottom[plane] = b.Bottom[plane][:need]
	}
}

// frameWorkParallelRun is the fallback for callers that did not bind a reusable
// worker pool. The concrete runner keeps the large postfilter context out of a
// goroutine closure; only the small runner pointer and lane id are captured.
func frameWorkParallelRun(workers, count int, runner *frameWorkParallelPoolRunner) error {
	if count <= 0 {
		return nil
	}
	if workers > count {
		workers = count
	}
	if workers <= 1 {
		return runner.RunLane(0)
	}
	var firstErr atomic.Pointer[frameWorkParallelError]
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(worker int) {
			defer wg.Done()
			if firstErr.Load() != nil {
				return
			}
			if err := runner.RunLane(worker); err != nil {
				firstErr.CompareAndSwap(nil, &frameWorkParallelError{err: err})
			}
		}(w)
	}
	wg.Wait()
	if e := firstErr.Load(); e != nil {
		return e.err
	}
	return nil
}

type frameWorkParallelError struct{ err error }

type frameWorkParallelPoolJob uint8

const (
	frameWorkParallelPoolNone frameWorkParallelPoolJob = iota
	frameWorkParallelPoolCDEFSnapshot
	frameWorkParallelPoolCDEFU8
	frameWorkParallelPoolLFPopulate
	frameWorkParallelPoolLFVertical
	frameWorkParallelPoolLFHorizontal
)

// frameWorkParallelPoolRunner is reused across stages and frames. Keeping the
// concrete job state here avoids an escaping closure (and its large copied
// FrameWorkPostFilterContext) on every frame.
type frameWorkParallelPoolRunner struct {
	job         frameWorkParallelPoolJob
	count       int
	parallel    *FrameWorkPostFilterParallel
	ctx         FrameWorkPostFilterContext
	cdefReq     FrameWorkCDEFPostFilterRequest
	unitRows    int
	rows        int
	loopBands   FrameWorkLoopFilterMaskBands
	loopJobSpan int
	next        atomic.Int64
	failed      atomic.Bool
}

func (r *frameWorkParallelPoolRunner) resetCommon(job frameWorkParallelPoolJob, count int, parallel *FrameWorkPostFilterParallel) {
	r.job = job
	r.count = count
	r.parallel = parallel
	r.next.Store(0)
	r.failed.Store(false)
}

func (r *frameWorkParallelPoolRunner) clear() {
	r.job = frameWorkParallelPoolNone
	r.count = 0
	r.parallel = nil
	r.ctx = FrameWorkPostFilterContext{}
	r.cdefReq = FrameWorkCDEFPostFilterRequest{}
	r.unitRows = 0
	r.rows = 0
	r.loopBands = FrameWorkLoopFilterMaskBands{}
	r.loopJobSpan = 0
}

func (r *frameWorkParallelPoolRunner) RunLane(worker int) error {
	for {
		band := int(r.next.Add(1)) - 1
		if band >= r.count || r.failed.Load() {
			return nil
		}
		var err error
		switch r.job {
		case frameWorkParallelPoolCDEFSnapshot:
			err = frameWorkRunCDEFBand(r.parallel, &r.ctx, &r.cdefReq, r.unitRows, r.rows, worker, band, false)
		case frameWorkParallelPoolCDEFU8:
			err = frameWorkRunCDEFBand(r.parallel, &r.ctx, &r.cdefReq, r.unitRows, r.rows, worker, band, true)
		case frameWorkParallelPoolLFPopulate, frameWorkParallelPoolLFVertical, frameWorkParallelPoolLFHorizontal:
			err = frameWorkRunLoopFilterBand(&r.loopBands, r.job, r.loopJobSpan, band)
		default:
			return nil
		}
		if err != nil {
			r.failed.Store(true)
			return err
		}
	}
}

func (p *FrameWorkPostFilterParallel) pooledLanes(workers, count int) int {
	if p == nil || p.pool == nil || count <= 0 {
		return 0
	}
	if workers > count {
		workers = count
	}
	if workers <= 1 || p.pool.WorkerCount() < workers {
		return 0
	}
	return workers
}

func (p *FrameWorkPostFilterParallel) runCDEFBands(ctx FrameWorkPostFilterContext, req FrameWorkCDEFPostFilterRequest, unitRows, rows, count, workers int, useU8 bool) error {
	job := frameWorkParallelPoolCDEFSnapshot
	if useU8 {
		job = frameWorkParallelPoolCDEFU8
	}
	p.runner.resetCommon(job, count, p)
	p.runner.ctx = ctx
	p.runner.cdefReq = req
	p.runner.unitRows = unitRows
	p.runner.rows = rows
	var err error
	if lanes := p.pooledLanes(workers, count); lanes > 0 {
		err = p.pool.ExecuteLanes(lanes, &p.runner)
	} else {
		err = frameWorkParallelRun(workers, count, &p.runner)
	}
	p.runner.clear()
	return err
}

func frameWorkRunCDEFBand(parallel *FrameWorkPostFilterParallel, ctx *FrameWorkPostFilterContext, req *FrameWorkCDEFPostFilterRequest, unitRows, rows, worker, band int, useU8 bool) error {
	rowStart := band * unitRows
	rowEnd := min(rowStart+unitRows, rows)
	bandReq := *req
	bandReq.InputScratch = parallel.cdef[worker].input
	if useU8 {
		bandReq.SampleScratch = parallel.cdef[worker].line
		r, err := ctx.ApplyCDEFPostFilterUnitRowsU8(bandReq, parallel.cdefU8[band], rowStart, rowEnd)
		if err == nil {
			parallel.accumulateCDEFResult(worker, r)
		}
		return err
	}
	bandReq.UnitDstScratch = parallel.cdef[worker].unitDst
	r, err := ctx.ApplyCDEFPostFilterUnitRows(bandReq, rowStart, rowEnd)
	if err == nil {
		parallel.accumulateCDEFResult(worker, r)
	}
	return err
}

func (p *FrameWorkPostFilterParallel) runLoopFilterBands(bands FrameWorkLoopFilterMaskBands, workers, count int, job frameWorkParallelPoolJob, span int) error {
	p.runner.resetCommon(job, count, p)
	p.runner.loopBands = bands
	p.runner.loopJobSpan = span
	var err error
	if lanes := p.pooledLanes(workers, count); lanes > 0 {
		err = p.pool.ExecuteLanes(lanes, &p.runner)
	} else {
		err = frameWorkParallelRun(workers, count, &p.runner)
	}
	p.runner.clear()
	return err
}

func frameWorkRunLoopFilterBand(bands *FrameWorkLoopFilterMaskBands, job frameWorkParallelPoolJob, span, band int) error {
	switch job {
	case frameWorkParallelPoolLFPopulate:
		rowStart := band * span
		return bands.PopulateBand(rowStart, min(rowStart+span, bands.MIRows()))
	case frameWorkParallelPoolLFVertical:
		plane := loopfilter.Plane(band / span)
		rr := band % span
		return bands.ApplyBand(plane, loopfilter.EdgeVertical, rr, rr+1)
	case frameWorkParallelPoolLFHorizontal:
		plane := loopfilter.Plane(band / span)
		rc := band % span
		return bands.ApplyBandCols(plane, loopfilter.EdgeHorizontal, rc, rc+1)
	default:
		return nil
	}
}

// applyCDEFPostFilterParallel runs CDEF in 64x64 unit-row bands across the
// parallel worker set. It reproduces ApplyCDEFPostFilterBanded byte-for-byte:
// the same snapshot / boundary loads happen before any band runs, then each
// band filters its disjoint unit rows against immutable inputs. It returns
// (result, true, nil) when it handled the stage, or (_, false, nil) when the
// caller should fall back to the serial path.
func (ctx FrameWorkPostFilterContext) applyCDEFPostFilterParallel(req FrameWorkCDEFPostFilterRequest, unitRowsPerBand int) (FrameWorkCDEFPostFilterResult, bool, error) {
	workers := ctx.Parallel.workers()
	if workers <= 0 || unitRowsPerBand <= 0 {
		return FrameWorkCDEFPostFilterResult{}, false, nil
	}
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkCDEFPostFilterResult{}, false, nil
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterResult{}, true, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterResult{}, false, frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return FrameWorkCDEFPostFilterResult{}, true, nil
	}
	if err := ctx.validateCDEFPostFilterRequest(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, false, err
	}
	_, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, false, err
	}

	// Compute the bands.
	bandCount := (rows + unitRowsPerBand - 1) / unitRowsPerBand
	if bandCount <= 1 {
		// One band is just the serial apply; no fan-out benefit.
		return FrameWorkCDEFPostFilterResult{}, false, nil
	}

	// Decide whether the 8-bit in-place walk applies (matches applyCDEFPostFilterRows).
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	useU8 := ctx.Output.Layout.BytesPerSample == 1 && coeffShift == 0

	lumaWidth, chromaWidth := frameWorkCDEFParallelPlaneWidths(ctx)
	ctx.Parallel.ensureCDEFScratch(workers, lumaWidth, chromaWidth, useU8, bandCount)

	if useU8 {
		return ctx.applyCDEFPostFilterParallelU8(req, unitRowsPerBand, rows, bandCount, workers)
	}
	return ctx.applyCDEFPostFilterParallelSnapshot(req, unitRowsPerBand, rows, bandCount, workers)
}

func frameWorkCDEFParallelPlaneWidths(ctx FrameWorkPostFilterContext) (int, int) {
	lumaWidth := 0
	chromaWidth := 0
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		if plane == 0 {
			lumaWidth = planeFrame.Width
		} else if planeFrame.Width > chromaWidth {
			chromaWidth = planeFrame.Width
		}
	}
	return lumaWidth, chromaWidth
}

// applyCDEFPostFilterParallelSnapshot is the 10/12-bit (or non-u8) parallel
// path. It loads the whole-frame pre-CDEF snapshot once (shared, read-only),
// then filters disjoint unit-row bands against it. Each worker binds its own
// InputScratch and UnitDstScratch; the shared snapshot, direction grid, and
// variance grid are read/written at disjoint band offsets.
func (ctx FrameWorkPostFilterContext) applyCDEFPostFilterParallelSnapshot(req FrameWorkCDEFPostFilterRequest, unitRowsPerBand, rows, bandCount, workers int) (FrameWorkCDEFPostFilterResult, bool, error) {
	if err := ctx.LoadCDEFPostFilterSamples(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, false, err
	}
	ctx.Parallel.resetCDEFResults(workers)
	err := ctx.Parallel.runCDEFBands(ctx, req, unitRowsPerBand, rows, bandCount, workers, false)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, false, err
	}
	return ctx.Parallel.mergeCDEFResults(workers), true, nil
}

// applyCDEFPostFilterParallelU8 is the 8-bit in-place parallel path. It snapshots
// each band's immutable two-row top/bottom halos first (serial, cheap), then
// filters the disjoint bands in place. Each worker binds its own byte line-backup
// arena (SampleScratch) and InputScratch.
func (ctx FrameWorkPostFilterContext) applyCDEFPostFilterParallelU8(req FrameWorkCDEFPostFilterRequest, unitRowsPerBand, rows, bandCount, workers int) (FrameWorkCDEFPostFilterResult, bool, error) {
	// Snapshot all band boundaries before any band mutates the frame in place.
	for band := 0; band < bandCount; band++ {
		rowStart := band * unitRowsPerBand
		rowEnd := rowStart + unitRowsPerBand
		if rowEnd > rows {
			rowEnd = rows
		}
		if err := ctx.LoadCDEFPostFilterU8BandBoundary(&ctx.Parallel.cdefU8[band], rowStart, rowEnd); err != nil {
			return FrameWorkCDEFPostFilterResult{}, false, err
		}
	}
	ctx.Parallel.resetCDEFResults(workers)
	err := ctx.Parallel.runCDEFBands(ctx, req, unitRowsPerBand, rows, bandCount, workers, true)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, false, err
	}
	return ctx.Parallel.mergeCDEFResults(workers), true, nil
}

func (p *FrameWorkPostFilterParallel) resetCDEFResults(workers int) {
	for i := 0; i < workers && i < len(p.cdef); i++ {
		p.cdef[i].result = FrameWorkCDEFPostFilterResult{}
	}
}

// accumulateCDEFResult folds a band result into the owning goroutine's private
// accumulator. Each worker id is owned by exactly one goroutine for the run, so
// this needs no synchronization.
func (p *FrameWorkPostFilterParallel) accumulateCDEFResult(worker int, r FrameWorkCDEFPostFilterResult) {
	acc := &p.cdef[worker].result
	acc.Units += r.Units
	acc.Blocks += r.Blocks
	if r.Planes > acc.Planes {
		acc.Planes = r.Planes
	}
}

func (p *FrameWorkPostFilterParallel) mergeCDEFResults(workers int) FrameWorkCDEFPostFilterResult {
	var out FrameWorkCDEFPostFilterResult
	for i := 0; i < workers && i < len(p.cdef); i++ {
		r := p.cdef[i].result
		out.Units += r.Units
		out.Blocks += r.Blocks
		if r.Planes > out.Planes {
			out.Planes = r.Planes
		}
	}
	return out
}

// applyLoopFilterMaskBandsParallel runs the mask-driven loop filter across the
// parallel worker set. It reproduces ApplyLoopFilterEdgesFromMasks byte-for-byte:
// PrepareLoopFilterMaskBands clears the shared per-4x4 level cache, PopulateBand
// jobs refill it across disjoint MI-row bands (barrier), then two ApplyBand
// phases -- every (plane, region-row) VERTICAL-edge job, a barrier, then every
// (plane, region-COLUMN) HORIZONTAL-edge job.
//
// Why it stays byte-exact and race-free: after Prepare the level cache and masks
// are read-only, and planes are independent surfaces. A vertical-edge filter
// modifies pixels left/right of the edge within its own rows, so region-ROW
// bands write disjoint rows; a horizontal-edge filter modifies pixels above/below
// the edge within its own columns, so region-COLUMN bands write disjoint columns
// (row-banding horizontal would straddle the 128px seam -- verified: -race +
// strict-MD5 both fail). The vertical->horizontal barrier preserves the
// single-shot's per-plane "vertical scan then horizontal scan" dependency. The
// populate bands are EVEN-aligned so a 4:2:0 chroma level-cache cell (shared by
// luma rows 2k/2k+1) is never split across two bands (a cell[2]/cell[3] race).
// Returns (result, true, nil) when it handled the stage, or (_, false, nil) when
// the caller should fall back to the serial apply.
func (ctx FrameWorkPostFilterContext) applyLoopFilterMaskBandsParallel(filterMap FrameWorkLoopFilterMap) (FrameWorkLoopFilterPostFilterApplyResult, bool, error) {
	workers := ctx.Parallel.workers()
	if workers <= 0 {
		return FrameWorkLoopFilterPostFilterApplyResult{}, false, nil
	}
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkLoopFilterPostFilterApplyResult{}, true, nil
	}
	if !ctx.loopFilterMasksUsable() {
		return FrameWorkLoopFilterPostFilterApplyResult{}, false, nil
	}
	masks := ctx.LoopFilterMasks
	bands, err := ctx.PrepareLoopFilterMaskBands(masks, filterMap)
	if err != nil {
		return FrameWorkLoopFilterPostFilterApplyResult{}, false, err
	}
	if !bands.Active() {
		return FrameWorkLoopFilterPostFilterApplyResult{}, true, nil
	}

	result := FrameWorkLoopFilterPostFilterApplyResult{Active: true}
	result.Plan.Active = true
	result.Plan.MICols = uint16(masks.Cols)
	result.Plan.MIRows = uint16(bands.MIRows())

	// Phase 0 (barrier): refill the per-4x4 level cache. Luma cells partition by
	// block coverage, but a vertically-subsampled (4:2:0) chroma cell is shared by
	// luma rows 2k and 2k+1, so a band boundary BETWEEN them would let two bands
	// write cell[2]/cell[3] (a race, and it breaks last-writer-wins). Align band
	// boundaries to even MI rows so each shared chroma cell's two luma rows fall
	// in one band.
	if !masks.LevelsFromDecode {
		miRows := bands.MIRows()
		populateRows := (frameWorkParallelBandRows(miRows, workers) + 1) &^ 1
		populateBands := (miRows + populateRows - 1) / populateRows
		if err := ctx.Parallel.runLoopFilterBands(bands, workers, populateBands, frameWorkParallelPoolLFPopulate, populateRows); err != nil {
			return FrameWorkLoopFilterPostFilterApplyResult{}, false, err
		}
	}

	regionRows := bands.RegionRows()
	regionCols := bands.RegionCols()
	if regionRows <= 0 || regionCols <= 0 {
		return result, true, nil
	}
	nPlanes := int(bands.MaxPlane()) + 1

	// Phase 1 (barrier): vertical edges, one (plane, region-row) job each.
	vJobs := nPlanes * regionRows
	if err := ctx.Parallel.runLoopFilterBands(bands, workers, vJobs, frameWorkParallelPoolLFVertical, regionRows); err != nil {
		return FrameWorkLoopFilterPostFilterApplyResult{}, false, err
	}

	// Phase 2 (barrier): horizontal edges, one (plane, region-COLUMN) job each.
	hJobs := nPlanes * regionCols
	if err := ctx.Parallel.runLoopFilterBands(bands, workers, hJobs, frameWorkParallelPoolLFHorizontal, regionCols); err != nil {
		return FrameWorkLoopFilterPostFilterApplyResult{}, false, err
	}
	return result, true, nil
}

// frameWorkParallelBandRows picks a row-band height giving the worker set several
// bands to steal from (~4 per worker) without over-fragmenting, at least one row.
func frameWorkParallelBandRows(rows, workers int) int {
	if rows <= 0 || workers <= 0 {
		return 1
	}
	perBand := (rows + workers*4 - 1) / (workers * 4)
	if perBand < 1 {
		perBand = 1
	}
	return perBand
}
