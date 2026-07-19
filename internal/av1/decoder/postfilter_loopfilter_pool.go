package decoder

import (
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// lfMaskMinRegionRowsPerBand is the minimum number of 128x128 mask region-rows a
// pooled vertical band must own before the loop-filter apply fans out across the
// worker lanes. Each region-row is 128 pixel rows of edge filtering, so one per
// band already amortizes the dispatch; the gate keeps tiny frames on the serial
// whole-frame apply (no dispatch, no per-frame goroutine escape), matching the
// CDEF cdefMinUnitRowsPerBand gate.
const lfMaskMinRegionRowsPerBand = 1

// applyLoopFilterEdgesMaybePooled runs the loop filter across the frame's idle
// worker lanes when a multi-lane pool was threaded into the postfilter and the
// decode-time deblocking bitmasks can drive the apply, and otherwise runs the
// byte-identical serial apply. The mask apply is required because it is the only
// path that exposes the dav1d-style banded scan (vertical then horizontal region
// bands) the pool fans out; multi-tile frames stay on the serial edge-list apply
// (see loopFilterMasksUsable). The pooled dispatch lives in a separate function,
// applyLoopFilterEdgesFromMasksPooledBands, so its band closures (which escape to
// the worker goroutines) never land in this function's frame and the serial
// fall-through stays allocation-free, mirroring applyCDEFPostFilterMaybePooled.
func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesMaybePooled(req FrameWorkLoopFilterPostFilterRequest) (FrameWorkLoopFilterPostFilterApplyResult, error) {
	pool := ctx.pool
	workers := ctx.postFilterWorkerCount()
	if pool == nil || workers <= 1 {
		return ctx.ApplyLoopFilterEdges(req)
	}
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkLoopFilterPostFilterApplyResult{}, nil
	}
	if !ctx.loopFilterMasksUsable() {
		return ctx.ApplyLoopFilterEdges(req)
	}
	filterMap := req.Map
	if frameWorkLoopFilterMapEmpty(filterMap) && ctx.LoopFilterMap != nil {
		filterMap = *ctx.LoopFilterMap
	}
	return ctx.applyLoopFilterEdgesFromMasksPooledBands(pool, workers, ctx.LoopFilterMasks, filterMap)
}

// applyLoopFilterEdgesFromMasksPooledBands is the pooled counterpart of
// ApplyLoopFilterEdgesFromMasks. It reuses the same three barrier-separated
// phases as the encoder's applyParallelMasks (which is proven byte-identical to
// the single-shot mask apply) and the dav1d lf_apply ordering:
//
//   - Populate the frame-wide per-4x4 level cache. This is O(frame) light writes
//     (the expensive work is the edge scan+filter kernels), so it runs serially
//     on the calling goroutine and fills the plan stats the single-shot apply
//     reports.
//   - Vertical edge pass: fan the 128x128 mask region-rows out as contiguous
//     bands across the lanes. A vertical edge filters perpendicular to itself
//     (across columns) within its own rows and never crosses a region-row
//     boundary, so region-row bands write disjoint pixel rows for every plane and
//     the result is bit-identical to the serial scan regardless of band count.
//   - Horizontal edge pass: it reads pixels the vertical pass wrote, so it runs
//     only after the whole vertical pass (the vertical RunRanges has joined).
//     A horizontal edge at a region-row boundary reads/writes a few pixel rows
//     into the band above (the read-after-write halo), so horizontal region-row
//     bands within one plane are NOT independent. We keep each plane's horizontal
//     pass serial (whole frame, top-to-bottom, no halo) and run the planes
//     concurrently instead: they write disjoint surfaces, so plane-parallel
//     horizontal is race-free and worker-count invariant. Luma stays the serial
//     critical path (see the report; the byte-exact luma-horizontal split needs
//     dav1d's per-sb-row wavefront, which RunRanges' fork-join cannot express).
func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesFromMasksPooledBands(pool *threading.Pool, workers int, masks *threading.FrameWorkLoopFilterMasks, filterMap FrameWorkLoopFilterMap) (FrameWorkLoopFilterPostFilterApplyResult, error) {
	bands, err := ctx.PrepareLoopFilterMaskBands(masks, filterMap)
	if err != nil {
		return FrameWorkLoopFilterPostFilterApplyResult{}, err
	}
	var result FrameWorkLoopFilterPostFilterApplyResult
	if !bands.Active() {
		return result, nil
	}
	result.Active = true
	result.Plan.Active = true
	result.Plan.MICols = uint16(bands.masks.Cols)
	result.Plan.MIRows = uint16(bands.masks.Rows)

	// Populate the per-4x4 level cache serially. PrepareLoopFilterMaskBands already
	// cleared it; this fills the stats (Blocks/Cells/level maxes) into result.Plan.
	levelCtx := frameWorkLoopFilterLevelContextFor(&ctx.Event)
	if err := ctx.populateLoopFilterLevelCacheRange(bands.masks, bands.filterMap, levelCtx, &result.Plan, 0, bands.MIRows()); err != nil {
		return result, err
	}

	regionRows := bands.RegionRows()
	if regionRows <= 0 {
		return result, nil
	}
	lc := lfmask.LevelCache{Cells: bands.masks.LevelCache, Stride: bands.masks.Cols}
	sharpness := bands.sharpness
	minPlane, maxPlane := bands.minPlane, bands.maxPlane

	var edges, applied atomic.Uint32
	var planeEdges, planeApplied [3]atomic.Uint32
	var maxLevel atomic.Uint32
	var planeMaxLevel [3]atomic.Uint32
	accumulate := func(local *FrameWorkLoopFilterPostFilterApplyResult) {
		edges.Add(local.Edges)
		applied.Add(local.Applied)
		for p := range 3 {
			planeEdges[p].Add(local.PlaneEdges[p])
			planeApplied[p].Add(local.PlaneApplied[p])
			frameWorkLoopFilterAtomicMax(&planeMaxLevel[p], uint32(local.PlaneMaxLevel[p]))
		}
		frameWorkLoopFilterAtomicMax(&maxLevel, uint32(local.MaxLevel))
	}

	// Vertical pass: region-row bands, all active planes per band (disjoint rows).
	maxBands := regionRows / lfMaskMinRegionRowsPerBand
	vbands := workers
	if vbands > maxBands {
		vbands = maxBands
	}
	if vbands < 1 {
		vbands = 1
	}
	runErr := pool.RunRanges(regionRows, vbands, func(_ int, lo, hi int) error {
		var local FrameWorkLoopFilterPostFilterApplyResult
		for plane := minPlane; plane <= maxPlane; plane++ {
			if !bands.PlaneActive(plane) {
				continue
			}
			if err := ctx.applyLoopFilterMaskPlaneRange(&local, bands.masks, lc, sharpness, plane, maskDirVertical, lo, hi); err != nil {
				return err
			}
		}
		accumulate(&local)
		return nil
	})
	if runErr != nil {
		return result, runErr
	}

	// Horizontal pass: one whole-frame job per plane, planes fanned across the
	// lanes (disjoint surfaces). RunRanges partitions the plane index range, so a
	// band may own more than one plane when workers < plane count; each plane is
	// still filtered whole-frame and serially, keeping the output invariant.
	nPlanes := int(maxPlane) + 1
	hErr := pool.RunRanges(nPlanes, nPlanes, func(_ int, lo, hi int) error {
		var local FrameWorkLoopFilterPostFilterApplyResult
		for p := lo; p < hi; p++ {
			plane := loopfilter.Plane(p)
			if !bands.PlaneActive(plane) {
				continue
			}
			if err := ctx.applyLoopFilterMaskPlaneRange(&local, bands.masks, lc, sharpness, plane, maskDirHorizontal, 0, regionRows); err != nil {
				return err
			}
		}
		accumulate(&local)
		return nil
	})
	if hErr != nil {
		return result, hErr
	}

	result.Edges = edges.Load()
	result.Applied = applied.Load()
	for p := range 3 {
		result.PlaneEdges[p] = planeEdges[p].Load()
		result.PlaneApplied[p] = planeApplied[p].Load()
		result.PlaneMaxLevel[p] = uint8(planeMaxLevel[p].Load())
	}
	result.MaxLevel = uint8(maxLevel.Load())
	return result, nil
}

// frameWorkLoopFilterAtomicMax raises a to max(a, v) with a lock-free CAS loop,
// so the pooled bands can fold their per-band max levels into a shared maximum
// without a mutex. The fold is order-independent, keeping the merged result
// identical regardless of how many lanes ran.
func frameWorkLoopFilterAtomicMax(a *atomic.Uint32, v uint32) {
	for {
		old := a.Load()
		if v <= old || a.CompareAndSwap(old, v) {
			return
		}
	}
}
