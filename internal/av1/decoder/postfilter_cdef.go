package decoder

import (
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// FrameWorkCDEFIndexMap is the decoded frame-level CDEF unit index map.
type FrameWorkCDEFIndexMap = threading.FrameWorkCDEFIndexMap

// cdefMinUnitRowsPerBand is the minimum number of 64x64 CDEF unit rows a pooled
// band must own before applyCDEFPostFilterMaybePooled fans CDEF out across the
// worker lanes. Below it the serial whole-frame apply wins (no snapshot copy, no
// dispatch, no per-frame goroutine escape), matching the SB-row reconstruction
// wavefront's minimum-rows-per-worker gate.
const cdefMinUnitRowsPerBand = 2

// FrameWorkCDEFPostFilterScratchSize reports caller-owned scratch needed by
// ApplyCDEFPostFilter.
type FrameWorkCDEFPostFilterScratchSize struct {
	Samples       [3]int
	Dst           [3]int
	DirectionGrid int
	VarianceGrid  int
	Input         int
	UnitDst       int
}

// BindRequest validates and slices caller-owned scratch for CDEF postfiltering.
func (s FrameWorkCDEFPostFilterScratchSize) BindRequest(indexMap FrameWorkCDEFIndexMap, samples [3][]uint16, dst [3][]uint16, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, input []uint16, unitDst []uint16) (FrameWorkCDEFPostFilterRequest, error) {
	if len(directionGrid) < s.DirectionGrid || len(varianceGrid) < s.VarianceGrid ||
		len(input) < s.Input || len(unitDst) < s.UnitDst ||
		s.DirectionGrid < 0 || s.VarianceGrid < 0 || s.Input < 0 || s.UnitDst < 0 {
		return FrameWorkCDEFPostFilterRequest{}, frame.ErrShortBuffer
	}
	req := FrameWorkCDEFPostFilterRequest{
		IndexMap:       indexMap,
		DirectionGrid:  directionGrid[:s.DirectionGrid],
		VarianceGrid:   varianceGrid[:s.VarianceGrid],
		InputScratch:   input[:s.Input],
		UnitDstScratch: unitDst[:s.UnitDst],
	}
	for plane := range len(req.SampleScratch) {
		if s.Samples[plane] < 0 || s.Dst[plane] < 0 ||
			len(samples[plane]) < s.Samples[plane] || len(dst[plane]) < s.Dst[plane] {
			return FrameWorkCDEFPostFilterRequest{}, frame.ErrShortBuffer
		}
		req.SampleScratch[plane] = samples[plane][:s.Samples[plane]]
		req.DstScratch[plane] = dst[plane][:s.Dst[plane]]
	}
	return req, nil
}

// Max returns the per-field maximum CDEF scratch size.
func (s FrameWorkCDEFPostFilterScratchSize) Max(other FrameWorkCDEFPostFilterScratchSize) FrameWorkCDEFPostFilterScratchSize {
	result := FrameWorkCDEFPostFilterScratchSize{
		DirectionGrid: maxInt(s.DirectionGrid, other.DirectionGrid),
		VarianceGrid:  maxInt(s.VarianceGrid, other.VarianceGrid),
		Input:         maxInt(s.Input, other.Input),
		UnitDst:       maxInt(s.UnitDst, other.UnitDst),
	}
	for plane := range len(result.Samples) {
		result.Samples[plane] = maxInt(s.Samples[plane], other.Samples[plane])
		result.Dst[plane] = maxInt(s.Dst[plane], other.Dst[plane])
	}
	return result
}

// FrameWorkCDEFPostFilterRequest carries decoded CDEF state and caller-owned
// scratch for ApplyCDEFPostFilter.
type FrameWorkCDEFPostFilterRequest struct {
	IndexMap FrameWorkCDEFIndexMap

	SampleScratch  [3][]uint16
	DstScratch     [3][]uint16
	DirectionGrid  []cdef.DirectionGrid
	VarianceGrid   []cdef.VarianceGrid
	InputScratch   []uint16
	UnitDstScratch []uint16
}

// FrameWorkCDEFPostFilterU8BandBoundary holds immutable pre-CDEF boundary rows
// for an in-place 8-bit unit-row band. Top is the two rows immediately above
// rowStart; Bottom is the two rows immediately below rowEnd. Edge bands leave the
// corresponding plane slices nil.
type FrameWorkCDEFPostFilterU8BandBoundary struct {
	Top    [3][]byte
	Bottom [3][]byte
}

// FrameWorkCDEFPostFilterResult summarizes CDEF frame filtering work.
type FrameWorkCDEFPostFilterResult struct {
	Units  uint32
	Blocks uint32
	Planes uint8
}

// CDEFPostFilterScratchLen reports scratch lengths needed to apply CDEF to
// ctx.Output.
func (ctx FrameWorkPostFilterContext) CDEFPostFilterScratchLen() (FrameWorkCDEFPostFilterScratchSize, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterScratchSize{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterScratchSize{}, frame.ErrInvalidSlot
	}
	var size FrameWorkCDEFPostFilterScratchSize
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		need, err := frame.SamplePlaneLen(planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return FrameWorkCDEFPostFilterScratchSize{}, err
		}
		size.Samples[plane] = need
	}
	if !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF) {
		cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
		if err != nil {
			return FrameWorkCDEFPostFilterScratchSize{}, err
		}
		size.DirectionGrid = cols * rows
		size.VarianceGrid = cols * rows
	}
	// The pooled row-band apply gives each of the W worker lanes its own
	// InputBufferSize-sized Input and UnitDst scratch (they cannot share the
	// per-unit filter scratch while filtering different unit rows concurrently),
	// so size the two buffers for W bands. W is 1 on the serial path, which keeps
	// the single-band scratch and the whole-frame apply byte-for-byte unchanged.
	// SampleScratch stays a single full-frame snapshot: every band reads it
	// read-only, so it is not per-lane.
	workers := ctx.postFilterWorkerCount()
	size.Input = cdef.InputBufferSize * workers
	size.UnitDst = cdef.InputBufferSize * workers
	return size, nil
}

// ApplyCDEFPostFilter applies CDEF to ctx.Output. Loop filter must already be
// inactive or marked complete with WithCompletedPostFilters.
func (ctx FrameWorkPostFilterContext) ApplyCDEFPostFilter(req FrameWorkCDEFPostFilterRequest) (FrameWorkCDEFPostFilterResult, error) {
	return ctx.applyCDEFPostFilterRows(req, 0, -1, true, nil)
}

// ApplyCDEFPostFilterBanded applies CDEF in deterministic 64x64 unit-row
// bands after loading the pre-CDEF frame snapshot once. The output regions for
// each unit-row band are disjoint, while every band reads the same immutable
// snapshot. This is the source-shaped split needed by dav1d-style row postfilter
// scheduling without changing the CDEF decisions or filter kernels.
func (ctx FrameWorkPostFilterContext) ApplyCDEFPostFilterBanded(req FrameWorkCDEFPostFilterRequest, unitRowsPerBand int) (FrameWorkCDEFPostFilterResult, error) {
	if unitRowsPerBand <= 0 {
		return ctx.ApplyCDEFPostFilter(req)
	}
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkCDEFPostFilterResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if err := ctx.validateCDEFPostFilterRequest(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	_, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	if err := ctx.LoadCDEFPostFilterSamples(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	var result FrameWorkCDEFPostFilterResult
	for rowStart := 0; rowStart < rows; rowStart += unitRowsPerBand {
		rowEnd := rowStart + unitRowsPerBand
		if rowEnd > rows {
			rowEnd = rows
		}
		bandResult, err := ctx.ApplyCDEFPostFilterUnitRows(req, rowStart, rowEnd)
		if err != nil {
			return result, err
		}
		result.Units += bandResult.Units
		result.Blocks += bandResult.Blocks
		if bandResult.Planes > result.Planes {
			result.Planes = bandResult.Planes
		}
	}
	return result, nil
}

// applyCDEFPostFilterMaybePooled runs CDEF across the frame's worker lanes when
// the postfilter was handed a multi-lane pool, and otherwise runs the serial
// whole-frame apply. The pooled path snapshots the pre-CDEF planes once and fans
// the 64x64 CDEF unit rows out as deterministic contiguous bands: every band
// reads the same immutable snapshot and writes disjoint output rows, so the
// result is bit-identical to the serial apply regardless of lane count (dav1d
// cdef_brow processes independent superblock rows the same way). Each band binds
// its own InputBufferSize-sized Input and UnitDst scratch out of the arena that
// CDEFPostFilterScratchLen sized to W bands; the snapshot, direction grid, and
// variance grid are shared (bands touch disjoint unit rows). The result counters
// sum the same per-unit work the serial apply would report.
func (ctx FrameWorkPostFilterContext) applyCDEFPostFilterMaybePooled(req FrameWorkCDEFPostFilterRequest) (FrameWorkCDEFPostFilterResult, error) {
	pool := ctx.pool
	workers := ctx.postFilterWorkerCount()
	if pool == nil || workers <= 1 {
		return ctx.ApplyCDEFPostFilter(req)
	}
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkCDEFPostFilterResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if err := ctx.validateCDEFPostFilterRequest(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	_, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	// Keep at least cdefMinUnitRowsPerBand 64x64 unit rows per band so the
	// snapshot-load and dispatch overhead is amortized, mirroring the SB-row
	// reconstruction wavefront's minimum-rows-per-worker gate. Small frames (few
	// CDEF unit rows) fall through to the serial whole-frame apply, which keeps
	// the reusable-decoder hot path allocation-free the way the serial recon path
	// is; only frames with enough unit rows to win pay the per-frame dispatch.
	maxBands := rows / cdefMinUnitRowsPerBand
	bands := workers
	if bands > maxBands {
		bands = maxBands
	}
	ibs := cdef.InputBufferSize
	// Fall back to the serial whole-frame apply when there is only one band or the
	// caller did not size the per-band Input/UnitDst scratch. The serial apply uses
	// only the first InputBufferSize chunk, so a larger arena is harmless. The
	// pooled dispatch lives in applyCDEFPostFilterPooledBands, a separate function,
	// so its band closure (which escapes to the worker goroutines) never lands in
	// this function's frame — the serial fall-through path stays allocation-free.
	if bands <= 1 || len(req.InputScratch) < bands*ibs || len(req.UnitDstScratch) < bands*ibs {
		return ctx.ApplyCDEFPostFilter(req)
	}
	return ctx.applyCDEFPostFilterPooledBands(req, pool, rows, bands, ibs)
}

// applyCDEFPostFilterPooledBands snapshots the pre-CDEF planes once and fans the
// [0, rows) CDEF unit rows across bands worker lanes. Each band binds its own
// InputBufferSize-sized Input/UnitDst chunk (band ordinal w owns [w*ibs,(w+1)*ibs))
// and reconstructs disjoint output rows from the shared read-only snapshot, so
// the result is bit-identical to the serial apply. It is deliberately a distinct
// function: its band closure escapes to the goroutines, so keeping it out of the
// maybe-pooled entry point lets the serial fall-through stay allocation-free.
func (ctx FrameWorkPostFilterContext) applyCDEFPostFilterPooledBands(req FrameWorkCDEFPostFilterRequest, pool *threading.Pool, rows int, bands int, ibs int) (FrameWorkCDEFPostFilterResult, error) {
	if err := ctx.LoadCDEFPostFilterSamples(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	var units, blocks, planes atomic.Uint32
	runErr := pool.RunRanges(rows, bands, func(band, lo, hi int) error {
		bandReq := req
		bandReq.InputScratch = req.InputScratch[band*ibs : (band+1)*ibs]
		bandReq.UnitDstScratch = req.UnitDstScratch[band*ibs : (band+1)*ibs]
		bandResult, err := ctx.ApplyCDEFPostFilterUnitRows(bandReq, lo, hi)
		if err != nil {
			return err
		}
		units.Add(bandResult.Units)
		blocks.Add(bandResult.Blocks)
		for {
			old := planes.Load()
			if uint32(bandResult.Planes) <= old || planes.CompareAndSwap(old, uint32(bandResult.Planes)) {
				break
			}
		}
		return nil
	})
	if runErr != nil {
		return FrameWorkCDEFPostFilterResult{}, runErr
	}
	return FrameWorkCDEFPostFilterResult{
		Units:  units.Load(),
		Blocks: blocks.Load(),
		Planes: uint8(planes.Load()),
	}, nil
}

// LoadCDEFPostFilterSamples snapshots the pre-CDEF output planes into the
// request's sample scratch. Callers that band the application across unit
// rows load once and then run ApplyCDEFPostFilterUnitRows per band against
// the shared read-only snapshot.
func (ctx FrameWorkPostFilterContext) LoadCDEFPostFilterSamples(req FrameWorkCDEFPostFilterRequest) error {
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		if _, _, err := frame.LoadSamplePlaneFull(req.SampleScratch[plane], planeFrame, ctx.Output.Layout.BytesPerSample); err != nil {
			return err
		}
	}
	return nil
}

// ApplyCDEFPostFilterUnitRows applies CDEF to the 64x64 unit rows in
// [rowStart, rowEnd) against the snapshot a prior LoadCDEFPostFilterSamples
// call placed in the request's sample scratch. Units read only the snapshot
// and write disjoint output regions, so bands over disjoint row ranges run
// concurrently when each band binds its own input and unit scratch (the
// sample scratch, direction grid, and variance grid are shared; the chroma
// passes of a band reuse the luma directions of the same band's rows).
func (ctx FrameWorkPostFilterContext) ApplyCDEFPostFilterUnitRows(req FrameWorkCDEFPostFilterRequest, rowStart, rowEnd int) (FrameWorkCDEFPostFilterResult, error) {
	return ctx.applyCDEFPostFilterRows(req, rowStart, rowEnd, false, nil)
}

// LoadCDEFPostFilterU8BandBoundary snapshots only the rows an in-place 8-bit
// unit-row band may otherwise race with neighbouring bands. It must run before
// any parallel ApplyCDEFPostFilterUnitRowsU8 calls for the frame.
func (ctx FrameWorkPostFilterContext) LoadCDEFPostFilterU8BandBoundary(boundary *FrameWorkCDEFPostFilterU8BandBoundary, rowStart, rowEnd int) error {
	if boundary == nil {
		return frame.ErrShortBuffer
	}
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return nil
	}
	if ctx.Output.Layout.BytesPerSample != 1 || ctx.Output.Format.BitDepth != 8 {
		return frame.ErrInvalidFormat
	}
	_, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}
	if rowEnd < 0 || rowEnd > rows {
		rowEnd = rows
	}
	if rowStart < 0 || rowStart >= rowEnd {
		if rowStart == rowEnd {
			return nil
		}
		return threading.ErrInvalidBatch
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

// ApplyCDEFPostFilterUnitRowsU8 applies an 8-bit unit-row band in place using
// immutable boundary rows loaded by LoadCDEFPostFilterU8BandBoundary. Unlike
// ApplyCDEFPostFilterUnitRows, it does not read req.SampleScratch as a full-frame
// snapshot; each worker must provide private SampleScratch line buffers plus
// private InputScratch.
func (ctx FrameWorkPostFilterContext) ApplyCDEFPostFilterUnitRowsU8(req FrameWorkCDEFPostFilterRequest, boundary FrameWorkCDEFPostFilterU8BandBoundary, rowStart, rowEnd int) (FrameWorkCDEFPostFilterResult, error) {
	return ctx.applyCDEFPostFilterRows(req, rowStart, rowEnd, false, &boundary)
}

func loadCDEFU8BandBoundaryPlane(boundary *FrameWorkCDEFPostFilterU8BandBoundary, plane int, planeFrame frame.Plane, rowStart int, rowEnd int, rows int, yDec int) error {
	width := planeFrame.Width
	height := planeFrame.Height
	stride := planeFrame.Stride
	if width <= 0 || height <= 0 || stride < width ||
		(height-1)*stride+width > len(planeFrame.Pix) {
		return frame.ErrInvalidPlane
	}
	need := cdef.VerticalBorder * width
	if rowStart > 0 {
		top := boundary.Top[plane]
		if len(top) < need {
			return frame.ErrShortBuffer
		}
		y0 := (rowStart*cdef.BlockSize)>>yDec - cdef.VerticalBorder
		for r := 0; r < cdef.VerticalBorder; r++ {
			y := y0 + r
			if y < 0 || y >= height {
				return frame.ErrInvalidPlane
			}
			copy(top[r*width:(r+1)*width], planeFrame.Pix[y*stride:y*stride+width])
		}
	}
	if rowEnd < rows {
		bottom := boundary.Bottom[plane]
		if len(bottom) < need {
			return frame.ErrShortBuffer
		}
		y0 := (rowEnd * cdef.BlockSize) >> yDec
		for r := 0; r < cdef.VerticalBorder; r++ {
			y := y0 + r
			if y < 0 || y >= height {
				return frame.ErrInvalidPlane
			}
			copy(bottom[r*width:(r+1)*width], planeFrame.Pix[y*stride:y*stride+width])
		}
	}
	return nil
}

func (ctx FrameWorkPostFilterContext) applyCDEFPostFilterRows(req FrameWorkCDEFPostFilterRequest, rowStart, rowEnd int, loadSamples bool, u8Boundary *FrameWorkCDEFPostFilterU8BandBoundary) (FrameWorkCDEFPostFilterResult, error) {
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkCDEFPostFilterResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if u8Boundary != nil {
		if err := ctx.validateCDEFPostFilterU8BandRequest(req); err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
	} else if err := ctx.validateCDEFPostFilterRequest(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	if rowEnd < 0 || rowEnd > rows {
		rowEnd = rows
	}
	if rowStart < 0 || rowStart >= rowEnd {
		if rowStart == rowEnd {
			return FrameWorkCDEFPostFilterResult{}, nil
		}
		return FrameWorkCDEFPostFilterResult{}, threading.ErrInvalidBatch
	}
	indexMap := req.IndexMap
	if frameWorkCDEFIndexMapEmpty(indexMap) && ctx.CDEFIndexMap != nil {
		indexMap = *ctx.CDEFIndexMap
	}
	if err := frameWorkValidateCDEFIndexMap(indexMap, cols, rows); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	skipMap := ctx.LoopFilterMap
	var skipLevels *threading.FrameWorkLoopFilterMasks
	if ctx.LoopFilterMasks != nil && ctx.LoopFilterMasks.LevelsFromDecode {
		skipMap = nil
		skipLevels = ctx.LoopFilterMasks
	}
	// 8-bit frames take the in-place uint8 walk when a caller can provide the
	// pre-filter samples that in-place filtering would otherwise overwrite:
	// whole-frame serial apply owns the row walk, while banded apply must pass
	// immutable two-row boundaries for cross-band top/bottom halos. 10/12-bit
	// frames keep the uint16 snapshot walk.
	useU8 := ctx.Output.Layout.BytesPerSample == 1 && coeffShift == 0 &&
		((loadSamples && rowStart == 0 && rowEnd == rows) || (!loadSamples && u8Boundary != nil))

	var result FrameWorkCDEFPostFilterResult
	var directions cdef.DirectionGrid
	var variances cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		// libaom's cdef_prepare_fb processes the MI-aligned plane extent
		// (vsize = nvb << mi_high_l2, hsize = nhb << mi_wide_l2), reading and
		// writing the bottom/right partial-superblock padding rows/cols that
		// reconstruction populated. Expand the cropped output plane to that
		// MI-aligned extent (capped to the allocated buffer) so the final
		// unit row/col filters with the real padding pixels rather than the
		// VeryLarge sentinel.
		xDec0, yDec0 := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec0, yDec0, ctx.Output.Layout.BytesPerSample)
		if useU8 {
			boundary := FrameWorkCDEFPostFilterU8BandBoundary{}
			if u8Boundary != nil {
				boundary = *u8Boundary
			}
			planeUnits, planeBlocks, err := frameWorkApplyCDEFPlaneRowsU8(ctx.Event.CDEF, indexMap, skipMap, skipLevels, cols, rows, rowStart, rowEnd, planeFrame, req.SampleScratch[plane], boundary, req.InputScratch[:cdef.InputBufferSize], blockStorage[:], &directions, &variances, req.DirectionGrid, req.VarianceGrid, plane, xDec0, yDec0, chromaFiltering)
			if err != nil {
				return FrameWorkCDEFPostFilterResult{}, err
			}
			result.Units += planeUnits
			result.Blocks += planeBlocks
			result.Planes++
			continue
		}
		var src frame.SamplePlane
		if loadSamples {
			var err error
			src, _, err = frame.LoadSamplePlaneFull(req.SampleScratch[plane], planeFrame, ctx.Output.Layout.BytesPerSample)
			if err != nil {
				return FrameWorkCDEFPostFilterResult{}, err
			}
		} else {
			var err error
			src, err = frame.BindSamplePlane(req.SampleScratch[plane], planeFrame, ctx.Output.Layout.BytesPerSample)
			if err != nil {
				return FrameWorkCDEFPostFilterResult{}, err
			}
		}
		xDec, yDec := xDec0, yDec0
		planeUnits, planeBlocks, err := frameWorkApplyCDEFPlaneRows(ctx.Event.CDEF, indexMap, skipMap, skipLevels, cols, rows, rowStart, rowEnd, src, planeFrame, ctx.Output.Layout.BytesPerSample, req.InputScratch[:cdef.InputBufferSize], req.UnitDstScratch[:cdef.InputBufferSize], blockStorage[:], &directions, &variances, req.DirectionGrid, req.VarianceGrid, plane, xDec, yDec, coeffShift, chromaFiltering)
		if err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
		result.Units += planeUnits
		result.Blocks += planeBlocks
		result.Planes++
	}
	return result, nil
}

func (ctx FrameWorkPostFilterContext) validateCDEFPostFilterRequest(req FrameWorkCDEFPostFilterRequest) error {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF) {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return nil
	}
	cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}
	indexMap := req.IndexMap
	if frameWorkCDEFIndexMapEmpty(indexMap) && ctx.CDEFIndexMap != nil {
		indexMap = *ctx.CDEFIndexMap
	}
	if err := frameWorkValidateCDEFIndexMap(indexMap, cols, rows); err != nil {
		return err
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	unitCount := cols * rows
	if chromaFiltering && (len(req.DirectionGrid) < unitCount || len(req.VarianceGrid) < unitCount) {
		return frame.ErrShortBuffer
	}
	if len(req.InputScratch) < cdef.InputBufferSize || len(req.UnitDstScratch) < cdef.InputBufferSize {
		return frame.ErrShortBuffer
	}
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	if coeffShift < 0 || coeffShift > 4 {
		return frame.ErrInvalidFormat
	}
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
		need, err := frame.SamplePlaneLen(planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return err
		}
		if len(req.SampleScratch[plane]) < need {
			return frame.ErrShortBuffer
		}
	}
	return nil
}

func (ctx FrameWorkPostFilterContext) validateCDEFPostFilterU8BandRequest(req FrameWorkCDEFPostFilterRequest) error {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF) {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return nil
	}
	if ctx.Output.Layout.BytesPerSample != 1 || ctx.Output.Format.BitDepth != 8 {
		return frame.ErrInvalidFormat
	}
	cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}
	indexMap := req.IndexMap
	if frameWorkCDEFIndexMapEmpty(indexMap) && ctx.CDEFIndexMap != nil {
		indexMap = *ctx.CDEFIndexMap
	}
	if err := frameWorkValidateCDEFIndexMap(indexMap, cols, rows); err != nil {
		return err
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	unitCount := cols * rows
	if chromaFiltering && (len(req.DirectionGrid) < unitCount || len(req.VarianceGrid) < unitCount) {
		return frame.ErrShortBuffer
	}
	if len(req.InputScratch) < cdef.InputBufferSize {
		return frame.ErrShortBuffer
	}
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
		if _, ok := cdefByteScratchView(req.SampleScratch[plane], 4*planeFrame.Width); !ok {
			return frame.ErrShortBuffer
		}
	}
	return nil
}

func frameWorkCDEFIndexMapEmpty(indexMap FrameWorkCDEFIndexMap) bool {
	return indexMap.Stride == 0 && indexMap.Rows == 0 &&
		len(indexMap.Index) == 0 && len(indexMap.Read) == 0
}

func frameWorkApplyCDEFPlane(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, skipLevels *threading.FrameWorkLoopFilterMasks, cols int, rows int, src frame.SamplePlane, dst frame.Plane, bytesPerSample int, input []uint16, unitDst []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, coeffShift int, forceLumaDirections bool) (uint32, uint32, error) {
	return frameWorkApplyCDEFPlaneRows(params, indexMap, skipMap, skipLevels, cols, rows, 0, rows, src, dst, bytesPerSample, input, unitDst, blockStorage, directions, variances, directionGrid, varianceGrid, plane, xDec, yDec, coeffShift, forceLumaDirections)
}

func frameWorkApplyCDEFPlaneRows(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, skipLevels *threading.FrameWorkLoopFilterMasks, cols int, rows int, rowStart int, rowEnd int, src frame.SamplePlane, dst frame.Plane, bytesPerSample int, input []uint16, unitDst []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, coeffShift int, forceLumaDirections bool) (uint32, uint32, error) {
	var units uint32
	var blocksTotal uint32
	unitSizeX := cdef.BlockSize >> xDec
	unitSizeY := cdef.BlockSize >> yDec
	blockWidth := 8 >> xDec
	blockHeight := 8 >> yDec
	indexStride := int(indexMap.Stride)
	for unitRow := rowStart; unitRow < rowEnd; unitRow++ {
		for unitCol := range cols {
			mapOffset := unitRow*indexStride + unitCol
			if !indexMap.Read[mapOffset] {
				continue
			}
			strengthIndex := indexMap.Index[mapOffset]
			if int(strengthIndex) >= int(params.StrengthCount) || int(strengthIndex) >= parser.MaxCDEFStrengths {
				return units, blocksTotal, threading.ErrInvalidBatch
			}
			index := int(strengthIndex)
			packed := params.YStrength[index]
			if plane != 0 {
				packed = params.UVStrength[index]
			}
			directionOnly := plane == 0 && forceLumaDirections && packed == 0 && params.UVStrength[index] != 0
			if packed == 0 && !directionOnly {
				continue
			}
			unitX := (unitCol * cdef.BlockSize) >> xDec
			unitY := (unitRow * cdef.BlockSize) >> yDec
			unitW := minInt(unitSizeX, src.Width-unitX)
			unitH := minInt(unitSizeY, src.Height-unitY)
			if unitW <= 0 || unitH <= 0 {
				continue
			}
			blocks := frameWorkCDEFBlockPositionsFilteredSources(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, skipLevels, unitRow, unitCol)
			if len(blocks) == 0 {
				continue
			}
			if err := frameWorkFillCDEFInputSentinels(input, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}
			if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}
			unitIndex := unitRow*cols + unitCol
			unitDirections := directions
			unitVariances := variances
			if unitIndex < len(directionGrid) && unitIndex < len(varianceGrid) {
				unitDirections = &directionGrid[unitIndex]
				unitVariances = &varianceGrid[unitIndex]
			}
			cdefPlane := cdef.PlaneY
			if plane == 1 {
				cdefPlane = cdef.PlaneU
			} else if plane == 2 {
				cdefPlane = cdef.PlaneV
			}
			filterParams, err := cdef.FrameFilterParamsFromStrength(cdefPlane, xDec, yDec, packed, int(params.Damping), coeffShift)
			if err != nil {
				return units, blocksTotal, err
			}
			if cdefDebugUnit(plane, unitRow, unitCol) {
				cdefDebugLogUnit(plane, unitRow, unitCol, packed, filterParams, *unitDirections, *unitVariances, input, len(blocks))
			}
			// When some 8x8 blocks within the unit are skip-transform, they are
			// absent from blocks and FilterFrameBlocks will not write their
			// positions in unitDst. If unitDst is zero or holds stale data from
			// a previous unit, CopyRect16To16 would later copy those zeros over
			// the valid reconstructed pixels in dst. Pre-fill unitDst with the
			// source (reconstructed) pixels for the whole unit so that
			// FilterFrameBlocks can overwrite the non-skip positions; skipped
			// positions then already carry the correct values.
			blockCols := (unitW + blockWidth - 1) / blockWidth
			blockRows := (unitH + blockHeight - 1) / blockHeight
			if !directionOnly && len(blocks) < blockCols*blockRows {
				srcOffset := unitY*src.Stride + unitX
				if err := cdef.CopyRect16To16(unitDst, cdef.BStride, src.Pix[srcOffset:], src.Stride, unitW, unitH); err != nil {
					return units, blocksTotal, err
				}
			}
			if err := cdef.FilterFrameBlocksTrusted(unitDst, cdef.BStride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams); err != nil {
				return units, blocksTotal, err
			}
			cdefDebugLogUnitDst(plane, unitRow, unitCol, unitDst)
			if directionOnly {
				continue
			}
			if err := frameWorkStoreCDEFUnit(dst, bytesPerSample, unitX, unitY, unitW, unitH, unitDst); err != nil {
				return units, blocksTotal, err
			}
			units++
			blocksTotal += uint32(len(blocks))
		}
	}
	return units, blocksTotal, nil
}

func frameWorkStoreCDEFUnit(dst frame.Plane, bytesPerSample int, x int, y int, width int, height int, src []uint16) error {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return frame.ErrInvalidPlane
	}
	if x < 0 || y < 0 || width <= 0 || height <= 0 ||
		x+width > dst.Width || y+height > dst.Height ||
		dst.Stride < dst.Width*bytesPerSample ||
		len(src) < (height-1)*cdef.BStride+width {
		return frame.ErrInvalidPlane
	}
	lastRow := (y + height - 1) * dst.Stride
	rowBytes := width * bytesPerSample
	rowStart := lastRow + x*bytesPerSample
	if rowStart < 0 || rowStart+rowBytes > len(dst.Pix) {
		return frame.ErrInvalidPlane
	}
	switch bytesPerSample {
	case 1:
		for row := range height {
			dstOff := (y+row)*dst.Stride + x
			srcOff := row * cdef.BStride
			dstRow := dst.Pix[dstOff : dstOff+width]
			srcRow := src[srcOff : srcOff+width]
			col := 0
			for ; col+8 <= width; col += 8 {
				dstRow[col+0] = byte(srcRow[col+0])
				dstRow[col+1] = byte(srcRow[col+1])
				dstRow[col+2] = byte(srcRow[col+2])
				dstRow[col+3] = byte(srcRow[col+3])
				dstRow[col+4] = byte(srcRow[col+4])
				dstRow[col+5] = byte(srcRow[col+5])
				dstRow[col+6] = byte(srcRow[col+6])
				dstRow[col+7] = byte(srcRow[col+7])
			}
			for ; col < width; col++ {
				dstRow[col] = byte(srcRow[col])
			}
		}
	case 2:
		for row := range height {
			dstOff := (y+row)*dst.Stride + x*2
			srcOff := row * cdef.BStride
			for col, sample := range src[srcOff : srcOff+width] {
				off := dstOff + col*2
				dst.Pix[off] = byte(sample)
				dst.Pix[off+1] = byte(sample >> 8)
			}
		}
	}
	return nil
}

func frameWorkFillCDEFInputSentinels(input []uint16, unitW int, unitH int) error {
	fillW := unitW + 2*cdef.HorizontalBorder
	fillH := unitH + 2*cdef.VerticalBorder
	if unitW <= 0 || unitH <= 0 || fillW <= 0 || fillW > cdef.BStride || fillH <= 0 ||
		!cdefInputRectFits(len(input), fillW, fillH) {
		return threading.ErrInvalidBatch
	}

	for row := 0; row < cdef.VerticalBorder; row++ {
		frameWorkFillCDEFInputSentinelRow(input[row*cdef.BStride:], fillW)
	}
	midEnd := cdef.VerticalBorder + unitH
	rightStart := cdef.HorizontalBorder + unitW
	for row := cdef.VerticalBorder; row < midEnd; row++ {
		rowBuf := input[row*cdef.BStride:]
		frameWorkFillCDEFInputSentinelRow(rowBuf, cdef.HorizontalBorder)
		frameWorkFillCDEFInputSentinelRow(rowBuf[rightStart:], cdef.HorizontalBorder)
	}
	for row := midEnd; row < fillH; row++ {
		frameWorkFillCDEFInputSentinelRow(input[row*cdef.BStride:], fillW)
	}
	return nil
}

func frameWorkFillCDEFInputSentinelRow(row []uint16, width int) {
	row = row[:width]
	i := 0
	for ; i+8 <= width; i += 8 {
		row[i+0] = cdef.VeryLarge
		row[i+1] = cdef.VeryLarge
		row[i+2] = cdef.VeryLarge
		row[i+3] = cdef.VeryLarge
		row[i+4] = cdef.VeryLarge
		row[i+5] = cdef.VeryLarge
		row[i+6] = cdef.VeryLarge
		row[i+7] = cdef.VeryLarge
	}
	for ; i < width; i++ {
		row[i] = cdef.VeryLarge
	}
}

func cdefInputRectFits(length int, width int, height int) bool {
	if width <= 0 || height <= 0 || width > cdef.BStride {
		return false
	}
	last := (height-1)*cdef.BStride + width
	return last >= 0 && last <= length
}

func frameWorkCopyCDEFInput(input []uint16, src frame.SamplePlane, unitX int, unitY int, unitW int, unitH int) error {
	// Extend the read past the visible plane width into the row-stride padding
	// columns when src.Stride supplies them. libaom's cdef_prepare_fb reads
	// hsize+HBORDER cols from the YV12 buffer where hsize is rounded up to
	// MI_SIZE_LOG2*2 (8-pixel) alignment, not the visible crop, so the
	// past-visible MI-padding pixels written by the reconstruction stage
	// participate in the kernel's secondary taps for the visible cols.
	//
	// Cap the read width to src.Width. The production call site now passes the
	// MI-aligned plane extent (frameWorkCDEFAlignedPlane), so src.Width already
	// is mi_cols * MI_SIZE (the extent libaom's cdef_prepare_fb reads); the
	// reconstruction stage populated those past-crop MI-padding columns. Reading
	// further into the row-stride alignment padding (cols past the MI grid)
	// would pull in stale samples that no reconstruction wrote, so we stop at
	// src.Width and let the synthetic right-border / VeryLarge handling cover
	// the remainder, matching libaom's frame_boundary fill.
	readWidth := src.Width
	srcX0 := maxInt(unitX-cdef.HorizontalBorder, 0)
	srcY0 := maxInt(unitY-cdef.VerticalBorder, 0)
	srcX1 := minInt(unitX+unitW+cdef.HorizontalBorder, readWidth)
	srcY1 := minInt(unitY+unitH+cdef.VerticalBorder, src.Height)
	copyW := srcX1 - srcX0
	copyH := srcY1 - srcY0
	if copyW <= 0 || copyH <= 0 {
		return nil
	}
	dstOffset := cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder +
		(srcY0-unitY)*cdef.BStride + (srcX0 - unitX)
	srcOffset := srcY0*src.Stride + srcX0
	if err := cdef.CopyRect16To16(input[dstOffset:], cdef.BStride, src.Pix[srcOffset:], src.Stride, copyW, copyH); err != nil {
		return err
	}
	// libaom's CDEF bot_linebuf for non-final superblock rows is fed from the
	// frame buffer at rows past the current 64x64 superblock; for chroma planes
	// whose visible height isn't a multiple of the CDEF unit height that read
	// lands in the plane's aligned-but-out-of-crop padding (uv_height vs
	// uv_crop_height), which libaom populates with the last visible row via
	// implicit alignment/decode writes. Our SamplePlane is exactly the visible
	// window, so we synthesize the equivalent extension by replicating the last
	// visible row into the CDEF input bottom border whenever the unit is not the
	// final unit row but the visible plane ends inside the unit's
	// VerticalBorder reach. Without it the kernel's directional taps hit the
	// VeryLarge sentinel even when libaom had real data, producing the V plane
	// row-31 -1 divergence on 66x66 (chroma height 33, CDEF unit height 32).
	//
	// The final superblock unit row keeps the VeryLarge sentinel just like
	// libaom's fill_rect(...CDEF_VERY_LARGE) branch when fbr == nvfb - 1.
	wantSrcY1 := unitY + unitH + cdef.VerticalBorder
	rowExtend := srcY1 < wantSrcY1 && srcY1 > 0 && unitY+unitH < src.Height
	if rowExtend {
		lastRowOffset := dstOffset + (copyH-1)*cdef.BStride
		for row := srcY1; row < wantSrcY1; row++ {
			extOffset := dstOffset + (row-srcY0)*cdef.BStride
			copy(input[extOffset:extOffset+copyW], input[lastRowOffset:lastRowOffset+copyW])
		}
	}
	// Same scenario, but symmetric on the horizontal axis: libaom's central
	// av1_cdef_copy_sb8_16 read for a non-rightmost superblock pulls in
	// hsize + CDEF_HBORDER columns from the dst frame buffer. When the visible
	// plane width ends inside that read (e.g. 33-wide chroma in the 66x66
	// vector at SB(0,0): visible cols 0..32 vs the read of cols 0..39 for
	// hsize=32 + HBORDER=8), libaom reads from the YV12 buffer's
	// aligned-but-out-of-crop padding which holds the last visible column via
	// implicit alignment/decode writes. Our SamplePlane is exactly the visible
	// window, so the right border stays at VeryLarge and the kernel's
	// directional secondary taps at the right edge of the unit diverge —
	// producing the V plane col-31 -1 divergence on 66x66 once the bottom-edge
	// fix already covered the row direction. Y plane SBs whose hsize=64 only
	// reach +2 into the right halo with visible content, so the read never
	// hits the VeryLarge sentinel and luma stays unaffected by this branch.
	//
	// Synthesize the equivalent extension by replicating the last visible
	// column into the CDEF input right border whenever the unit is not the
	// final unit column but the visible plane ends inside the unit's
	// HorizontalBorder reach. The final superblock unit column keeps the
	// VeryLarge sentinel just like libaom's fill_rect(...CDEF_VERY_LARGE)
	// frame_boundary[RIGHT] overlay when fbc == nhfb - 1.
	//
	// Apply column extension after row extension so that the bottom-right
	// corner of the input buffer (which is doubly past-visible for chroma SBs
	// on 66x66) is also populated with the last visible pixel.
	wantSrcX1 := unitX + unitW + cdef.HorizontalBorder
	if srcX1 < wantSrcX1 && srcX1 > 0 && unitX+unitW < src.Width {
		extRows := copyH
		if rowExtend {
			extRows = wantSrcY1 - srcY0
		}
		for row := 0; row < extRows; row++ {
			rowOffset := dstOffset + row*cdef.BStride
			last := input[rowOffset+copyW-1]
			extStart := rowOffset + copyW
			extEnd := rowOffset + (wantSrcX1 - srcX0)
			for i := extStart; i < extEnd; i++ {
				input[i] = last
			}
		}
	}
	return nil
}

// frameWorkCDEFBlockPositionsFiltered enumerates 8x8 CDEF blocks inside the
// current 64x64 unit, skipping those whose luma MI cells all report
// SkipTransform=true. Block indexing follows libaom: by/bx are luma 8x8
// indices reused across planes, so the skip lookup is in luma MI coordinates
// regardless of plane subsampling. When skipMap is nil every block is kept,
// matching the previous behaviour.
func frameWorkCDEFBlockPositionsFilteredSources(storage []cdef.BlockPosition, unitW int, unitH int, blockW int, blockH int, skipMap *FrameWorkLoopFilterMap, skipLevels *threading.FrameWorkLoopFilterMasks, unitRow int, unitCol int) []cdef.BlockPosition {
	if skipLevels == nil {
		return frameWorkCDEFBlockPositionsFiltered(storage, unitW, unitH, blockW, blockH, skipMap, unitRow, unitCol)
	}
	cols := (unitW + blockW - 1) / blockW
	rows := (unitH + blockH - 1) / blockH
	out := storage[:0]
	for by := range rows {
		for bx := range cols {
			if skipLevels.CDEFBlockAllSkipped(unitRow, unitCol, by, bx) {
				continue
			}
			out = append(out, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	return out
}

func frameWorkCDEFBlockPositionsFiltered(storage []cdef.BlockPosition, unitW int, unitH int, blockW int, blockH int, skipMap *FrameWorkLoopFilterMap, unitRow int, unitCol int) []cdef.BlockPosition {
	cols := (unitW + blockW - 1) / blockW
	rows := (unitH + blockH - 1) / blockH
	out := storage[:0]
	for by := range rows {
		for bx := range cols {
			if frameWorkCDEFBlockAllSkipped(skipMap, unitRow, unitCol, by, bx) {
				continue
			}
			out = append(out, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	return out
}

// frameWorkCDEFBlockAllSkipped reports whether every luma MI cell covered by
// the 8x8 luma CDEF block at (by, bx) inside the unit at (unitRow, unitCol)
// has SkipTransform=true. Missing or invalid coverage falls back to false so
// CDEF still runs on those positions (matches the legacy unfiltered behaviour).
func frameWorkCDEFBlockAllSkipped(skipMap *FrameWorkLoopFilterMap, unitRow int, unitCol int, by int, bx int) bool {
	if skipMap == nil {
		return false
	}
	const miPerUnit = 16 // 64-pixel luma unit / 4-pixel MI cell
	const miPerBlock = 2 // 8-pixel block / 4-pixel MI cell
	miColStart := unitCol*miPerUnit + bx*miPerBlock
	miRowStart := unitRow*miPerUnit + by*miPerBlock
	if miColStart < 0 || miRowStart < 0 {
		return false
	}
	if miColStart+miPerBlock > int(skipMap.Stride) || miRowStart+miPerBlock > int(skipMap.Rows) {
		return false
	}
	stride := int(skipMap.Stride)
	for dy := range miPerBlock {
		row := (miRowStart + dy) * stride
		for dx := range miPerBlock {
			record := skipMap.Records[row+miColStart+dx]
			if !record.Valid {
				return false
			}
			if !record.SkipTransform {
				return false
			}
		}
	}
	return true
}

func frameWorkCDEFUnitGrid(size parser.FrameSize) (int, int, error) {
	if size.CodedWidth == 0 || size.Height == 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	cols := int((size.CodedWidth + cdef.BlockSize - 1) / cdef.BlockSize)
	rows := int((size.Height + cdef.BlockSize - 1) / cdef.BlockSize)
	if cols <= 0 || rows <= 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	return cols, rows, nil
}

func frameWorkValidateCDEFIndexMap(indexMap FrameWorkCDEFIndexMap, cols int, rows int) error {
	if indexMap.Stride <= 0 || indexMap.Rows <= 0 ||
		cols > int(indexMap.Stride) || rows > int(indexMap.Rows) {
		return threading.ErrInvalidBatch
	}
	limit := int(^uint(0) >> 1)
	if uint64(indexMap.Stride)*uint64(indexMap.Rows) > uint64(limit) {
		return threading.ErrInvalidBatch
	}
	length := int(indexMap.Stride) * int(indexMap.Rows)
	if len(indexMap.Index) < length || len(indexMap.Read) < length {
		return threading.ErrInvalidBatch
	}
	return nil
}

func frameWorkCDEFHasFiltering(params parser.CDEFParams, mono bool) bool {
	limit := min(int(params.StrengthCount), parser.MaxCDEFStrengths)
	for i := 0; i < limit; i++ {
		if params.YStrength[i] != 0 || (!mono && params.UVStrength[i] != 0) {
			return true
		}
	}
	return false
}

func frameWorkCDEFChromaHasFiltering(params parser.CDEFParams) bool {
	limit := min(int(params.StrengthCount), parser.MaxCDEFStrengths)
	for i := 0; i < limit; i++ {
		if params.UVStrength[i] != 0 {
			return true
		}
	}
	return false
}

func frameWorkCDEFPlaneHasFiltering(params parser.CDEFParams, plane int) bool {
	limit := min(int(params.StrengthCount), parser.MaxCDEFStrengths)
	for i := 0; i < limit; i++ {
		if plane == 0 && params.YStrength[i] != 0 {
			return true
		}
		if plane != 0 && params.UVStrength[i] != 0 {
			return true
		}
	}
	return false
}

func frameWorkCDEFPlane(output frame.Frame, plane int) (frame.Plane, bool) {
	switch plane {
	case 0:
		return output.Y, true
	case 1:
		return output.U, !output.Format.MonoChrome
	case 2:
		return output.V, !output.Format.MonoChrome
	default:
		return frame.Plane{}, false
	}
}

// frameWorkCDEFAlignedPlane returns a view of plane expanded from its cropped
// Width/Height to the MI-aligned extent libaom's cdef_prepare_fb operates on
// (mi_cols * MI_SIZE for luma, shifted by the plane subsampling for chroma).
// The byte buffer was allocated at the MI-aligned dimensions by
// frame.RequiredSize, so the expanded view shares the same Pix and Stride; only
// the reported Width/Height grow. This lets CDEF read and write the
// bottom/right partial-superblock padding rows/cols. It must return the same
// dimensions whether plane.Pix is the real allocation or a descriptor-only
// probe (used by the scratch-sizing pass), so the aligned extent is derived
// purely from the frame dimensions, not the Pix length.
func frameWorkCDEFAlignedPlane(plane frame.Plane, size parser.FrameSize, xDec int, yDec int, bytesPerSample int) frame.Plane {
	if bytesPerSample <= 0 || plane.Stride <= 0 {
		return plane
	}
	alignedLumaW := (int(size.CodedWidth) + 7) &^ 7
	alignedLumaH := (int(size.Height) + 7) &^ 7
	alignedW := alignedLumaW >> xDec
	alignedH := alignedLumaH >> yDec
	if alignedW < plane.Width {
		alignedW = plane.Width
	}
	if alignedH < plane.Height {
		alignedH = plane.Height
	}
	// The aligned width never exceeds the row stride (frame.RequiredSize aligns
	// the stride to at least the MI-aligned width).
	if w := plane.Stride / bytesPerSample; alignedW > w {
		alignedW = w
	}
	plane.Width = alignedW
	plane.Height = alignedH
	return plane
}

func frameWorkCDEFPlaneDecimation(format frame.Format, plane int) (int, int) {
	if plane == 0 {
		return 0, 0
	}
	xDec := 0
	yDec := 0
	if format.SubsamplingX {
		xDec = 1
	}
	if format.SubsamplingY {
		yDec = 1
	}
	return xDec, yDec
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
