package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// frameWorkParallelRestorationJob is the plane the restoration stage is
// filtering. Bands split its restoration-unit rows.
type frameWorkParallelRestorationJob struct {
	grid       tile.RestorationPlaneGrid
	records    []tile.RestorationUnitRecord
	boundaries tile.RestorationStripeBoundaries
	plane      frame.Plane
}

// frameWorkParallelRestorationScratch is one worker's private scratch for the
// 8-bit in-place restoration walk: the lr_sbrow band buffer with its left-halo
// backups, one unit's Wiener/SGRProj scratch, and the worker's result counters.
type frameWorkParallelRestorationScratch struct {
	data   []uint16
	wiener []uint16
	sgr    []int32
	result tile.RestorationPlaneApplyResult
}

// applyLoopRestorationPostFilterParallel runs the 8-bit in-place loop
// restoration as restoration-unit-row bands across the parallel worker set.
// Each band binds its own band buffer and unit scratch, writes only its own
// units' rows, and reads cross-band context from the stripe boundary lines
// saved before CDEF, so the output matches the serial walk for any band count
// (see tile.ApplyRestorationFramePlaneInPlaceU8Rows). It returns handled=false
// when the frame needs the serial path (high bit depth, optimized walk, or a
// single band).
func (ctx *FrameWorkPostFilterContext) applyLoopRestorationPostFilterParallel(req FrameWorkRestorationPostFilterRequest) (tile.RestorationFrameApplyResult, bool, error) {
	workers := ctx.Parallel.workers()
	if workers <= 0 {
		return tile.RestorationFrameApplyResult{}, false, nil
	}
	remaining := ctx.RemainingPostFilters()
	if remaining&(FrameWorkPostFilterLoopFilter|FrameWorkPostFilterCDEF|FrameWorkPostFilterSuperRes) != 0 {
		return tile.RestorationFrameApplyResult{}, false, nil
	}
	if !remaining.Has(FrameWorkPostFilterLoopRestoration) {
		return tile.RestorationFrameApplyResult{}, true, nil
	}
	if ctx.Output == nil {
		return tile.RestorationFrameApplyResult{}, false, frame.ErrInvalidSlot
	}
	if req.Optimized || ctx.Output.Layout.BytesPerSample != 1 || ctx.Output.Format.BitDepth != 8 {
		return tile.RestorationFrameApplyResult{}, false, nil
	}
	if ctx.RestorationFrameBuffers != nil {
		if frameWorkRestorationRecordsEmpty(req.Records) {
			req.Records = ctx.RestorationFrameBuffers.Records
		}
		if frameWorkRestorationBoundariesEmpty(req.Boundaries) {
			req.Boundaries = ctx.RestorationFrameBuffers.Boundaries
		}
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		return tile.RestorationFrameApplyResult{}, false, err
	}
	if !plan.Active {
		return tile.RestorationFrameApplyResult{}, true, nil
	}
	maxRows, dataLen := 0, 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == parser.RestorationNone {
			continue
		}
		maxRows = max(maxRows, int(grid.VertUnits))
		seg, err := tile.RestorationFramePlaneInPlaceU8ScratchLen(grid)
		if err != nil {
			return tile.RestorationFrameApplyResult{}, false, err
		}
		dataLen = max(dataLen, seg)
	}
	if maxRows <= 1 {
		return tile.RestorationFrameApplyResult{}, false, nil
	}
	if err := ctx.validateLoopRestorationPostFilterRequest(req); err != nil {
		return tile.RestorationFrameApplyResult{}, false, err
	}
	pool, workers, err := ctx.Parallel.workerPool(ctx, workers)
	if err != nil {
		return tile.RestorationFrameApplyResult{}, false, err
	}
	workers = min(workers, maxRows)
	ctx.Parallel.ensureRestorationScratch(workers, dataLen, len(req.Scratch.Unit.Wiener), len(req.Scratch.Unit.SGRProj))

	var result tile.RestorationFrameApplyResult
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == parser.RestorationNone {
			continue
		}
		buffer, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			return tile.RestorationFrameApplyResult{}, false, frame.ErrInvalidPlane
		}
		rows := int(grid.VertUnits)
		if rows <= 0 {
			return tile.RestorationFrameApplyResult{}, false, tile.ErrInvalidPlan
		}
		bands := min(workers, rows)
		for i := range bands {
			ctx.Parallel.lr[i].result = tile.RestorationPlaneApplyResult{}
		}
		if err := ctx.Parallel.runRanges(pool, frameWorkPostFilterParallelJob{
			stage: frameWorkPostFilterStageRestorationU8,
			lr: frameWorkParallelRestorationJob{
				grid:       grid,
				records:    req.Records[plane],
				boundaries: req.Boundaries[plane],
				plane:      buffer,
			},
		}, rows, bands); err != nil {
			return tile.RestorationFrameApplyResult{}, false, err
		}
		var planeResult tile.RestorationPlaneApplyResult
		for i := range bands {
			r := &ctx.Parallel.lr[i].result
			planeResult.Records += r.Records
			planeResult.FilteredRecords += r.FilteredRecords
			planeResult.Stripes += r.Stripes
			planeResult.ProcessingUnits += r.ProcessingUnits
		}
		result.PlaneResults[plane] = planeResult
		result.Planes++
		result.Records += planeResult.Records
		result.FilteredRecords += planeResult.FilteredRecords
		result.Stripes += planeResult.Stripes
		result.ProcessingUnits += planeResult.ProcessingUnits
	}
	return result, true, nil
}

// runRestorationU8Range filters restoration-unit rows [lo, hi) of the job's
// plane with worker's private scratch.
func (p *FrameWorkPostFilterParallel) runRestorationU8Range(job *frameWorkParallelRestorationJob, worker, lo, hi int) error {
	s := &p.lr[worker]
	scratch := tile.RestorationUnitRecordBoundaryScratch{
		Unit: tile.RestorationUnitScratch{Wiener: s.wiener, SGRProj: s.sgr},
	}
	r, err := tile.ApplyRestorationFramePlaneInPlaceU8Rows(job.grid, job.records, job.boundaries, job.plane, s.data, scratch, lo, hi)
	if err != nil {
		return err
	}
	s.result = r
	return nil
}

// ensureRestorationScratch grows the per-worker restoration scratch. It only
// allocates while a stream's frame geometry is still growing.
func (p *FrameWorkPostFilterParallel) ensureRestorationScratch(workers, dataLen, wienerLen, sgrLen int) {
	if len(p.lr) < workers {
		p.lr = append(p.lr, make([]frameWorkParallelRestorationScratch, workers-len(p.lr))...)
	}
	for i := range workers {
		s := &p.lr[i]
		if len(s.data) < dataLen {
			s.data = make([]uint16, dataLen)
		}
		if len(s.wiener) < wienerLen {
			s.wiener = make([]uint16, wienerLen)
		}
		if len(s.sgr) < sgrLen {
			s.sgr = make([]int32, sgrLen)
		}
	}
}
