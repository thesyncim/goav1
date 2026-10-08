package decoder

import (
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// applyLoopRestorationPostFilterMaybePooled runs loop restoration across the
// frame's idle worker lanes when a multi-lane pool was threaded in and the frame
// is the 8-bit in-place walk, and otherwise runs the byte-identical serial apply.
// Only the 8-bit non-optimized in-place lr_sbrow walk bands cleanly over
// restoration-unit rows: its vertical stripe-seam context is read from the
// pre-saved deblock/CDEF boundary lines (never a neighbour band's surface) and
// its horizontal left-halo backup resets at each RU-row start, so RU-row bands
// write disjoint plane rows and read only their own rows plus the shared read-only
// boundary buffers. The 10/12-bit wide path (whole-plane snapshot + store-back)
// and the libaom optimized_lr mode fall back to the serial apply.
//
// The pooled dispatch lives in applyLoopRestorationPostFilterPooledBands, a
// separate function, so its band closure (which escapes to the worker goroutines)
// never lands here and the serial fall-through stays allocation-free, mirroring
// applyCDEFPostFilterMaybePooled.
func (ctx FrameWorkPostFilterContext) applyLoopRestorationPostFilterMaybePooled(req FrameWorkRestorationPostFilterRequest) (tile.RestorationFrameApplyResult, error) {
	pool := ctx.pool
	workers := ctx.postFilterWorkerCount()
	if pool == nil || workers <= 1 {
		return ctx.ApplyLoopRestorationPostFilter(req)
	}
	remaining := ctx.RemainingPostFilters()
	preRestoration := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes
	if remaining&preRestoration != 0 {
		return tile.RestorationFrameApplyResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterLoopRestoration) {
		return tile.RestorationFrameApplyResult{}, nil
	}
	if ctx.Output == nil {
		return tile.RestorationFrameApplyResult{}, frame.ErrInvalidSlot
	}
	if req.Optimized || ctx.Output.Layout.BytesPerSample != 1 || ctx.Output.Format.BitDepth != 8 {
		return ctx.ApplyLoopRestorationPostFilter(req)
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
		return tile.RestorationFrameApplyResult{}, err
	}
	if !plan.Active {
		return tile.RestorationFrameApplyResult{}, nil
	}
	// Without a band arena (single-lane sizing, or a caller that did not thread
	// pool scratch) there is nothing to fan; run the serial apply.
	if req.Pool.Bands < 2 || len(req.Pool.BandData) < req.Pool.Bands {
		return ctx.ApplyLoopRestorationPostFilter(req)
	}
	if err := ctx.validateLoopRestorationPostFilterRequest(req); err != nil {
		return tile.RestorationFrameApplyResult{}, err
	}
	return ctx.applyLoopRestorationPostFilterPooledBands(plan, req, pool)
}

// applyLoopRestorationPostFilterPooledBands filters each active plane's
// restoration-unit rows as deterministic contiguous bands across the pool lanes.
// Planes run in the outer (serial) loop, so all lanes share the frame's widest
// per-band arena slot; within a plane, band ordinal w binds slot w of the arena
// (its own lr_sbrow band buffer and Wiener/SGRProj scratch) and filters disjoint
// RU rows, so the restored plane is bit-identical to the serial whole-plane walk
// regardless of band count. Result counters sum the same per-record work the
// serial apply reports.
func (ctx FrameWorkPostFilterContext) applyLoopRestorationPostFilterPooledBands(plan tile.RestorationFramePlan, req FrameWorkRestorationPostFilterRequest, pool *threading.Pool) (tile.RestorationFrameApplyResult, error) {
	bands := req.Pool.Bands
	if bands <= 0 {
		return tile.RestorationFrameApplyResult{}, tile.ErrInvalidPlan
	}
	segLen := len(req.Pool.BandData) / bands
	wLen := len(req.Pool.BandWiener) / bands
	sgrLen := len(req.Pool.BandSGR) / bands
	if segLen <= 0 {
		return tile.RestorationFrameApplyResult{}, tile.ErrInvalidPlan
	}
	var result tile.RestorationFrameApplyResult
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == parser.RestorationNone {
			continue
		}
		bufPlane, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			return tile.RestorationFrameApplyResult{}, frame.ErrInvalidPlane
		}
		records := req.Records[plane]
		boundaries := req.Boundaries[plane]
		vertUnits := int(grid.VertUnits)
		if vertUnits <= 0 {
			return tile.RestorationFrameApplyResult{}, tile.ErrInvalidPlan
		}
		planeBands := bands
		if planeBands > vertUnits {
			planeBands = vertUnits
		}
		var recAcc, filtAcc, stripeAcc, puAcc atomic.Uint32
		runErr := pool.RunRanges(vertUnits, planeBands, func(band, lo, hi int) error {
			bandData := req.Pool.BandData[band*segLen : band*segLen+segLen]
			bandScratch := tile.RestorationUnitRecordBoundaryScratch{
				Unit: tile.RestorationUnitScratch{
					Wiener:  req.Pool.BandWiener[band*wLen : band*wLen+wLen],
					SGRProj: req.Pool.BandSGR[band*sgrLen : band*sgrLen+sgrLen],
				},
			}
			bandResult, err := tile.ApplyRestorationFramePlaneInPlaceU8Rows(grid, records, boundaries, bufPlane, bandData, bandScratch, lo, hi)
			if err != nil {
				return err
			}
			recAcc.Add(bandResult.Records)
			filtAcc.Add(bandResult.FilteredRecords)
			stripeAcc.Add(bandResult.Stripes)
			puAcc.Add(bandResult.ProcessingUnits)
			return nil
		})
		if runErr != nil {
			return tile.RestorationFrameApplyResult{}, runErr
		}
		planeResult := tile.RestorationPlaneApplyResult{
			Records:         recAcc.Load(),
			FilteredRecords: filtAcc.Load(),
			Stripes:         stripeAcc.Load(),
			ProcessingUnits: puAcc.Load(),
		}
		result.PlaneResults[plane] = planeResult
		result.Planes++
		result.Records += planeResult.Records
		result.FilteredRecords += planeResult.FilteredRecords
		result.Stripes += planeResult.Stripes
		result.ProcessingUnits += planeResult.ProcessingUnits
	}
	return result, nil
}
