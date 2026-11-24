package goav1

// This file exposes the caller-owned post-filter scratch-binding helpers used
// by the decoder frame-work pipeline.
//
// The helpers fall into four categories:
//
//   - Shape helpers (DecoderFrameWork*Shape, *ScratchLen,
//     *SideDataScratchLen): report the caller-owned slice lengths required
//     to back each post-filter map or scratch. They never allocate.
//   - Bind helpers (BindDecoderFrameWork*): wire caller-owned slices into
//     typed map/scratch/request views.
//   - Reset/Mark helpers (Reset*, Mark*): prepare a bound map for use during
//     one frame, or record per-block updates from inside the block-loop
//     visitor.
//   - Runner helpers (DecoderFrameWork{Supported,Caller}PostFilterScratch
//     Runner and setDecoderFrameWorkSideData): the convenience wrappers
//     that run the full post-filter chain from a single Apply call.
//
// The naming follows the pattern Shape -> Bind -> Reset -> Mark -> Bind
// Request -> Apply for each of the five post-filter stages (loop filter,
// CDEF, super-res, restoration, film grain) so callers can scan the file
// for the stage they want.

import (
	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internallfmask "github.com/thesyncim/goav1/internal/av1/lfmask"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// decoderFrameWorkCDEFIndexMapShape reports the column count, row count, and
// total entry count required to back the CDEF index map for (sequence, size).
func decoderFrameWorkCDEFIndexMapShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkCDEFIndexBatch(sequence, size, internalparser.CDEFParams{})
	return batch.CDEFIndexMapShape()
}

// decoderFrameWorkLoopFilterMapShape reports the column count, row count,
// and total entry count required to back the loop-filter map for
// (sequence, size).
func decoderFrameWorkLoopFilterMapShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.LoopFilterMapShape()
}

// bindDecoderFrameWorkCDEFIndexMap wires caller-owned index and read
// slices into a DecoderFrameWorkCDEFIndexMap sized for (sequence, size,
// cdef). The caller retains ownership of the backing storage.
func bindDecoderFrameWorkCDEFIndexMap(sequence SequenceHeader, size FrameSize, cdef internalparser.CDEFParams, index []uint8, read []bool) (internaldecoder.FrameWorkCDEFIndexMap, error) {
	batch := decoderFrameWorkCDEFIndexBatch(sequence, size, cdef)
	return batch.BindCDEFIndexMap(index, read)
}

// bindDecoderFrameWorkLoopFilterMap wires a caller-owned records slice
// into a DecoderFrameWorkLoopFilterMap sized for (sequence, size).
func bindDecoderFrameWorkLoopFilterMap(sequence SequenceHeader, size FrameSize, records []internalthreading.FrameWorkLoopFilterBlockRecord) (internaldecoder.FrameWorkLoopFilterMap, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindLoopFilterMap(records)
}

// bindDecoderFrameWorkLoopFilterMasks wires caller-owned edge-mask + level-cache
// storage into a DecoderFrameWorkLoopFilterMasks handle sized for (sequence,
// size). masks length must be >= the FrameWorkLoopFilterMaskShape mask count and
// levelCache >= the loop-filter map length. The handle is cleared; attach it via
// the side-data so the tile block walk builds it during decode.
func bindDecoderFrameWorkLoopFilterMasks(sequence SequenceHeader, size FrameSize, masks []internallfmask.FilterMask, levelCache [][4]uint8) (internalthreading.FrameWorkLoopFilterMasks, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindLoopFilterMasks(masks, levelCache)
}

// decoderFrameWorkPostFilterRequestBuffers groups caller-owned side data and
// scratch slices used to bind a full postfilter request.
type decoderFrameWorkPostFilterRequestBuffers struct {
	LoopFilterMap             internaldecoder.FrameWorkLoopFilterMap
	LoopFilterTrustedCoverage bool
	LoopFilterEdges           []internaldecoder.FrameWorkLoopFilterPostFilterEdge
	LoopFilterSchedule        []uint32

	CDEFIndexMap       internaldecoder.FrameWorkCDEFIndexMap
	CDEFSampleScratch  [3][]uint16
	CDEFSampleBytes    [3][]byte
	CDEFDstScratch     [3][]uint16
	CDEFDirectionGrid  []internalcdef.DirectionGrid
	CDEFVarianceGrid   []internalcdef.VarianceGrid
	CDEFInputScratch   []uint16
	CDEFUnitDstScratch []uint16

	SuperResOutputFrame   []byte
	SuperResCodedScratch  [3][]uint16
	SuperResOutputScratch [3][]uint16

	FilmGrainOutputFrame []byte

	RestorationRecords              [3][]internaltile.RestorationUnitRecord
	RestorationBoundaries           [3]internaltile.RestorationStripeBoundaries
	RestorationDataScratch          []uint16
	RestorationDstScratch           []uint16
	RestorationWienerScratch        []uint16
	RestorationSGRProjScratch       []int32
	RestorationBoundaryAboveScratch []uint16
	RestorationBoundaryBelowScratch []uint16
	RestorationOptimized            bool

	FilmGrainLumaGrain     []int16
	FilmGrainChromaGrain   [2][]int16
	FilmGrainLumaSamples   []uint16
	FilmGrainChromaSamples [2][]uint16
}

// decoderFrameWorkPostFilterRequestSideData groups postfilter side data that is
// not carved out of scratch arenas.
type decoderFrameWorkPostFilterRequestSideData struct {
	LoopFilterMap             internaldecoder.FrameWorkLoopFilterMap
	LoopFilterTrustedCoverage bool
	CDEFIndexMap              internaldecoder.FrameWorkCDEFIndexMap

	RestorationRecords    [3][]internaltile.RestorationUnitRecord
	RestorationBoundaries [3]internaltile.RestorationStripeBoundaries
	RestorationOptimized  bool
}

// decoderFrameWorkPostFilterRequestScratchSize reports typed arena lengths
// needed to bind a full postfilter request.
type decoderFrameWorkPostFilterRequestScratchSize struct {
	LoopFilterEdges    int
	LoopFilterSchedule int
	CDEFDirectionGrid  int
	CDEFVarianceGrid   int

	ByteScratch   int
	Uint16Scratch int
	Int16Scratch  int
	Int32Scratch  int
}

// decoderFrameWorkPostFilterRequestScratch carries typed arenas consumed by
// bindDecoderFrameWorkPostFilterRequestBuffersFromScratch.
type decoderFrameWorkPostFilterRequestScratch struct {
	LoopFilterEdges    []internaldecoder.FrameWorkLoopFilterPostFilterEdge
	LoopFilterSchedule []uint32
	CDEFDirectionGrid  []internalcdef.DirectionGrid
	CDEFVarianceGrid   []internalcdef.VarianceGrid

	ByteScratch   []byte
	Uint16Scratch []uint16
	Int16Scratch  []int16
	Int32Scratch  []int32
}

// Max returns per-arena maximum lengths for reusable postfilter request
// scratch.
func (s decoderFrameWorkPostFilterRequestScratchSize) Max(other decoderFrameWorkPostFilterRequestScratchSize) decoderFrameWorkPostFilterRequestScratchSize {
	return decoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:    max(s.LoopFilterEdges, other.LoopFilterEdges),
		LoopFilterSchedule: max(s.LoopFilterSchedule, other.LoopFilterSchedule),
		CDEFDirectionGrid:  max(s.CDEFDirectionGrid, other.CDEFDirectionGrid),
		CDEFVarianceGrid:   max(s.CDEFVarianceGrid, other.CDEFVarianceGrid),
		ByteScratch:        max(s.ByteScratch, other.ByteScratch),
		Uint16Scratch:      max(s.Uint16Scratch, other.Uint16Scratch),
		Int16Scratch:       max(s.Int16Scratch, other.Int16Scratch),
		Int32Scratch:       max(s.Int32Scratch, other.Int32Scratch),
	}
}

// decoderFrameWorkSideDataScratchSize reports caller-owned side-data backing
// lengths for tile residual decode and final postfilter planning.
type decoderFrameWorkSideDataScratchSize struct {
	CDEFIndexMap int
	CDEFReadMap  int

	LoopFilterMap        int
	LoopFilterMasks      int
	LoopFilterLevelCache int

	RestorationRecords       int
	RestorationBoundaryAbove int
	RestorationBoundaryBelow int
}

// decoderFrameWorkSideDataScratch carries typed caller-owned side-data backing
// for bindDecoderFrameWorkSideData.
type decoderFrameWorkSideDataScratch struct {
	CDEFIndexMap []uint8
	CDEFReadMap  []bool

	LoopFilterMap        []internalthreading.FrameWorkLoopFilterBlockRecord
	LoopFilterMasks      []internallfmask.FilterMask
	LoopFilterLevelCache [][4]uint8

	RestorationRecords       []internaltile.RestorationUnitRecord
	RestorationBoundaryAbove []uint16
	RestorationBoundaryBelow []uint16
}

// Max returns per-arena maximum lengths for reusable frame-work side data.
func (s decoderFrameWorkSideDataScratchSize) Max(other decoderFrameWorkSideDataScratchSize) decoderFrameWorkSideDataScratchSize {
	return decoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             max(s.CDEFIndexMap, other.CDEFIndexMap),
		CDEFReadMap:              max(s.CDEFReadMap, other.CDEFReadMap),
		LoopFilterMap:            max(s.LoopFilterMap, other.LoopFilterMap),
		LoopFilterMasks:          max(s.LoopFilterMasks, other.LoopFilterMasks),
		LoopFilterLevelCache:     max(s.LoopFilterLevelCache, other.LoopFilterLevelCache),
		RestorationRecords:       max(s.RestorationRecords, other.RestorationRecords),
		RestorationBoundaryAbove: max(s.RestorationBoundaryAbove, other.RestorationBoundaryAbove),
		RestorationBoundaryBelow: max(s.RestorationBoundaryBelow, other.RestorationBoundaryBelow),
	}
}

// decoderFrameWorkSideData groups frame-level side maps shared by tile
// residual decode and postfilter planning.
type decoderFrameWorkSideData struct {
	CDEFIndexMap            internaldecoder.FrameWorkCDEFIndexMap
	LoopFilterMap           internaldecoder.FrameWorkLoopFilterMap
	LoopFilterMasks         internalthreading.FrameWorkLoopFilterMasks
	RestorationFrameBuffers internalthreading.FrameWorkRestorationFrameBuffers
}

// decoderFrameWorkSupportedPostFilterScratchRunner binds exact postfilter
// request views from caller-owned scratch at final-frame callback time, after
// tile residual decode has filled frame side data.
type decoderFrameWorkSupportedPostFilterScratchRunner struct {
	Scratch              decoderFrameWorkPostFilterRequestScratch
	RestorationOptimized bool
	FilmGrainOutput      Frame

	// Parallel, when non-nil with more than one worker, fans the supported
	// post-filter chain's independent row bands out across goroutines. It is
	// caller-owned and reused across frames; decoded output is byte-identical to
	// the serial path.
	Parallel *internaldecoder.FrameWorkPostFilterParallel

	Size   internaldecoder.FrameWorkPostFilterScratchSize
	Result internaldecoder.FrameWorkPostFilterResult

	output *Frame
	// source is the last successful Apply's reconstructed (pre-grain)
	// output. Only this pointer is kept, not a copy of the whole context:
	// copying the pointerful context every frame runs bulk write barriers
	// while the application's GC is marking.
	source *Frame

	// work holds the in-flight context so the chain runs on it by pointer.
	work internaldecoder.FrameWorkPostFilterContext
}

// decoderFrameWorkPostFilterOutputProvider exposes the final display output
// selected by a direct postfilter runner.
type decoderFrameWorkPostFilterOutputProvider interface {
	PostFilterOutput() (*Frame, bool)
}

// ScratchLen reports the exact scratch size required for ctx using side data
// carried by the context.
func (r *decoderFrameWorkSupportedPostFilterScratchRunner) ScratchLen(ctx internaldecoder.FrameWorkPostFilterContext) (internaldecoder.FrameWorkPostFilterScratchSize, error) {
	if r == nil {
		return internaldecoder.FrameWorkPostFilterScratchSize{}, internaldecoder.ErrInvalidFrameWorkState
	}
	return ctx.SupportedPostFilterScratchLen(internaldecoder.FrameWorkPostFilterRequest{
		Restoration: internaldecoder.FrameWorkRestorationPostFilterRequest{Optimized: r.RestorationOptimized},
	})
}

func (r *decoderFrameWorkSupportedPostFilterScratchRunner) reusableScratchLen(ctx *internaldecoder.FrameWorkPostFilterContext) (internaldecoder.FrameWorkPostFilterScratchSize, error) {
	if r == nil {
		return internaldecoder.FrameWorkPostFilterScratchSize{}, internaldecoder.ErrInvalidFrameWorkState
	}
	remaining := ctx.RemainingPostFilters()
	supported := internaldecoder.FrameWorkPostFilterLoopFilter |
		internaldecoder.FrameWorkPostFilterCDEF |
		internaldecoder.FrameWorkPostFilterLoopRestoration |
		internaldecoder.FrameWorkPostFilterFilmGrain
	if remaining&^supported != 0 {
		return internaldecoder.FrameWorkPostFilterScratchSize{}, ErrDecoderUnsupportedPostFilter
	}
	var size internaldecoder.FrameWorkPostFilterScratchSize
	if remaining.Has(internaldecoder.FrameWorkPostFilterLoopFilter) {
		_, _, length, err := decoderFrameWorkLoopFilterMapShape(ctx.Event.SequenceHeader, ctx.Event.FrameSize)
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		maxInt := int(^uint(0) >> 1)
		if length > maxInt/8 {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, internalframe.ErrInvalidFormat
		}
		size.LoopFilter.Edges = length * 8
		size.LoopFilter.Schedule = length * 8
	}
	if remaining.Has(internaldecoder.FrameWorkPostFilterCDEF) {
		cdefSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.CDEF = cdefSize
	}
	if remaining.Has(internaldecoder.FrameWorkPostFilterLoopRestoration) {
		var records [3][]internaltile.RestorationUnitRecord
		if ctx.RestorationFrameBuffers != nil {
			records = ctx.RestorationFrameBuffers.Records
		}
		restorationSize, err := ctx.LoopRestorationPostFilterScratchLen(records, r.RestorationOptimized)
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.Restoration = restorationSize
	}
	if remaining.Has(internaldecoder.FrameWorkPostFilterFilmGrain) {
		filmGrainSize, err := ctx.FilmGrainPostFilterScratchLen()
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.FilmGrain = filmGrainSize
	}
	return size, nil
}

// Apply binds exact request views from Scratch, applies supported postfilters,
// and requires the resulting output to remain publishable by the frame pool.
func (r *decoderFrameWorkSupportedPostFilterScratchRunner) Apply(ctx internaldecoder.FrameWorkPostFilterContext) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	r.work = ctx
	size, err := r.work.SupportedPostFilterScratchLen(internaldecoder.FrameWorkPostFilterRequest{
		Restoration: internaldecoder.FrameWorkRestorationPostFilterRequest{Optimized: r.RestorationOptimized},
	})
	if err != nil {
		return err
	}
	return r.applyWithScratchSize(size)
}

// applyWithScratchSize runs the chain on r.work, which the caller has loaded.
func (r *decoderFrameWorkSupportedPostFilterScratchRunner) applyWithScratchSize(size internaldecoder.FrameWorkPostFilterScratchSize) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	r.output = nil
	ctx := &r.work
	side := postFilterRequestSideDataFromContext(ctx)
	side.RestorationOptimized = r.RestorationOptimized
	req, err := bindDecoderFrameWorkPostFilterRequestFromScratch(size, side, r.Scratch)
	if err != nil {
		return err
	}
	req.FilmGrain.OutputView = &r.FilmGrainOutput
	ctx.Parallel = r.Parallel
	output, result, err := internaldecoder.ApplySupportedPostFiltersForPublicationInPlace(ctx, req)
	if err != nil {
		return err
	}
	if err := ctx.RequirePublishablePostFilterOutput(); err != nil {
		return err
	}
	r.Size = size
	if r.source != ctx.Output {
		r.source = ctx.Output
	}
	r.Result = result
	r.output = output
	return nil
}

// PostFilterOutput returns the final display output after Apply.
func (r *decoderFrameWorkSupportedPostFilterScratchRunner) PostFilterOutput() (*Frame, bool) {
	if r == nil {
		return nil, false
	}
	if r.output != nil {
		return r.output, true
	}
	if r.source == nil {
		return nil, false
	}
	return r.source, true
}

// decoderFrameWorkReusableSupportedPostFilterRunner is a self-sizing
// post-filter runner for incremental decode loops that need publishable
// frame-pool references. Each Apply computes the exact scratch the supported
// post-filter chain (loop filter, CDEF, loop restoration, and display-only film
// grain) needs for the completed frame, grows its caller-owned arena when a
// frame needs more than any prior frame, then applies the chain. This type
// implements
// DecoderFrameWorkPostFilterRunner and
// decoderFrameWorkPostFilterOutputProvider, so it can be passed directly to
// RunLowOverheadIntoWithPostFilterRunner and the event runner will route the
// filtered display frame back to the caller.
//
// Without a post-filter runner the public path publishes the raw reconstructed
// frame, which does not match libaom for any stream whose frames signal loop
// filter, CDEF, loop restoration, super-res, or film grain. Allocations occur
// only on the first frame and whenever a later frame needs a larger arena;
// steady-state decode is allocation-free.
type decoderFrameWorkReusableSupportedPostFilterRunner struct {
	// RestorationOptimized selects the optimized loop-restoration apply path.
	RestorationOptimized bool

	// Parallel, when non-nil with more than one worker, fans the supported
	// post-filter chain's independent row bands out across goroutines. Set it
	// once (e.g. from the decoder worker count); it is reused across frames and
	// keeps decoded output byte-identical to the serial path.
	Parallel *internaldecoder.FrameWorkPostFilterParallel

	runner decoderFrameWorkSupportedPostFilterScratchRunner
	size   decoderFrameWorkPostFilterRequestScratchSize
}

// Apply sizes scratch for ctx, grows the caller-owned arena if needed, and runs
// the supported post-filter chain in place.
func (r *decoderFrameWorkReusableSupportedPostFilterRunner) Apply(ctx internaldecoder.FrameWorkPostFilterContext) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	r.runner.RestorationOptimized = r.RestorationOptimized
	r.runner.Parallel = r.Parallel
	r.runner.work = ctx
	exact, err := r.runner.reusableScratchLen(&r.runner.work)
	if err != nil {
		return err
	}
	arena := decoderFrameWorkPostFilterRequestScratchLen(exact)
	if r.decoderFrameWorkReusablePostFilterArenaTooSmall(arena) {
		r.size = r.size.Max(arena)
		r.runner.Scratch = decoderFrameWorkReusablePostFilterScratch(r.size)
	}
	return r.runner.applyWithScratchSize(exact)
}

// PostFilterOutput returns the final publishable output after Apply.
func (r *decoderFrameWorkReusableSupportedPostFilterRunner) PostFilterOutput() (*Frame, bool) {
	if r == nil {
		return nil, false
	}
	return r.runner.PostFilterOutput()
}

func (r *decoderFrameWorkReusableSupportedPostFilterRunner) decoderFrameWorkReusablePostFilterArenaTooSmall(arena decoderFrameWorkPostFilterRequestScratchSize) bool {
	return decoderFrameWorkPostFilterArenaTooSmall(r.runner.Scratch, arena)
}

func decoderFrameWorkPostFilterArenaTooSmall(s decoderFrameWorkPostFilterRequestScratch, arena decoderFrameWorkPostFilterRequestScratchSize) bool {
	return len(s.LoopFilterEdges) < arena.LoopFilterEdges ||
		len(s.LoopFilterSchedule) < arena.LoopFilterSchedule ||
		len(s.CDEFDirectionGrid) < arena.CDEFDirectionGrid ||
		len(s.CDEFVarianceGrid) < arena.CDEFVarianceGrid ||
		len(s.ByteScratch) < arena.ByteScratch ||
		len(s.Uint16Scratch) < arena.Uint16Scratch ||
		len(s.Int16Scratch) < arena.Int16Scratch ||
		len(s.Int32Scratch) < arena.Int32Scratch
}

func decoderFrameWorkReusablePostFilterScratch(size decoderFrameWorkPostFilterRequestScratchSize) decoderFrameWorkPostFilterRequestScratch {
	return decoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, size.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]internalcdef.DirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]internalcdef.VarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:        make([]byte, size.ByteScratch),
		Uint16Scratch:      make([]uint16, size.Uint16Scratch),
		Int16Scratch:       make([]int16, size.Int16Scratch),
		Int32Scratch:       make([]int32, size.Int32Scratch),
	}
}

// decoderFrameWorkPostFilterScratchContext builds a header-derived final-frame
// postfilter context for scratch sizing. output is filled with a geometry-only
// coded-frame view; its pixel slices are nil and must not be used to apply
// filters.
func decoderFrameWorkPostFilterScratchContext(sequence SequenceHeader, event *internaldecoder.Event, align int, side *decoderFrameWorkSideData, output *Frame) (internaldecoder.FrameWorkPostFilterContext, error) {
	if output == nil {
		return internaldecoder.FrameWorkPostFilterContext{}, internalframe.ErrInvalidSlot
	}
	format, err := frameCodedFormatFromHeaders(sequence, event.FrameSize, align)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterContext{}, err
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterContext{}, err
	}
	*output = decoderFrameWorkPostFilterScratchFrame(format, layout)
	ctx := internaldecoder.FrameWorkPostFilterContext{
		Event:  *event,
		Output: output,
	}
	ctx.Event.SequenceHeader = sequence
	if side != nil {
		ctx.CDEFIndexMap = &side.CDEFIndexMap
		ctx.LoopFilterMap = &side.LoopFilterMap
		ctx.RestorationFrameBuffers = &side.RestorationFrameBuffers
	}
	return ctx, nil
}

func decoderFrameWorkPostFilterScratchFrame(format FrameFormat, layout FrameLayout) Frame {
	return Frame{
		Format: format,
		Layout: layout,
		Y: FramePlane{
			Stride: layout.YStride,
			Width:  format.Width,
			Height: format.Height,
		},
		U: FramePlane{
			Stride: layout.UStride,
			Width:  layout.ChromaWidth,
			Height: layout.ChromaHeight,
		},
		V: FramePlane{
			Stride: layout.VStride,
			Width:  layout.ChromaWidth,
			Height: layout.ChromaHeight,
		},
	}
}

// decoderFrameWorkPostFilterRequestScratchLen reports the flat typed arena
// lengths needed for size.
func decoderFrameWorkPostFilterRequestScratchLen(size internaldecoder.FrameWorkPostFilterScratchSize) decoderFrameWorkPostFilterRequestScratchSize {
	var uint16Scratch int
	for plane := range 3 {
		uint16Scratch += size.CDEF.Samples[plane] + size.CDEF.Dst[plane]
		uint16Scratch += size.SuperRes.CodedSamples[plane] + size.SuperRes.OutputSamples[plane]
	}
	uint16Scratch += size.CDEF.Input + size.CDEF.UnitDst
	uint16Scratch += size.Restoration.Samples.DataLen + size.Restoration.Samples.DstLen
	uint16Scratch += size.Restoration.Apply.Unit.Wiener
	uint16Scratch += size.Restoration.Apply.Boundary.Above + size.Restoration.Apply.Boundary.Below
	uint16Scratch += size.FilmGrain.LumaSamples
	for plane := range 2 {
		uint16Scratch += size.FilmGrain.ChromaSamples[plane]
	}

	int16Scratch := size.FilmGrain.LumaGrain
	for plane := range 2 {
		int16Scratch += size.FilmGrain.ChromaGrain[plane]
	}

	return decoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:    size.LoopFilter.Edges,
		LoopFilterSchedule: size.LoopFilter.Schedule,
		CDEFDirectionGrid:  size.CDEF.DirectionGrid,
		CDEFVarianceGrid:   size.CDEF.VarianceGrid,
		ByteScratch:        size.SuperRes.OutputFrame + size.FilmGrain.OutputFrame + size.CDEF.SampleBytes[0] + size.CDEF.SampleBytes[1] + size.CDEF.SampleBytes[2],
		Uint16Scratch:      uint16Scratch,
		Int16Scratch:       int16Scratch,
		Int32Scratch:       size.Restoration.Apply.Unit.SGRProj,
	}
}

// decoderFrameWorkSideDataScratchLen reports the caller-owned scratch
// lengths required to back the full frame-work side data (CDEF index
// map, loop-filter map, restoration records, restoration boundaries)
// for (sequence, size, cdef, restoration).
func decoderFrameWorkSideDataScratchLen(sequence SequenceHeader, size FrameSize, cdef internalparser.CDEFParams, restoration internalparser.RestorationParams) (decoderFrameWorkSideDataScratchSize, error) {
	_, _, cdefLength, err := decoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		return decoderFrameWorkSideDataScratchSize{}, err
	}
	lfCols, lfRows, loopFilterLength, err := decoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		return decoderFrameWorkSideDataScratchSize{}, err
	}
	// The deblocking edge bitmasks are sized per 128x128 region; the per-4x4
	// level cache is padded to whole regions of rows plus one cell, as dav1d
	// allocates it, so the deblock drivers read whole groups of levels at the
	// bottom of the frame. Sized unconditionally (by
	// frame geometry) since the build gate is per-frame LoopFilter activity,
	// which the scratch sizing does not see -- the masks stay unbuilt/unused on
	// loop-filter-inactive frames.
	_, _, maskCount := internalthreading.FrameWorkLoopFilterMaskShape(lfCols, lfRows)
	restorationPlan, err := decoderFrameWorkRestorationFramePlan(sequence, size, restoration)
	if err != nil {
		return decoderFrameWorkSideDataScratchSize{}, err
	}
	boundaryLength := restorationPlan.BoundaryBufferLen()
	return decoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             cdefLength,
		CDEFReadMap:              cdefLength,
		LoopFilterMap:            loopFilterLength,
		LoopFilterMasks:          maskCount,
		LoopFilterLevelCache:     internalthreading.FrameWorkLoopFilterLevelCacheLen(lfCols, lfRows),
		RestorationRecords:       restorationPlan.UnitRecordLen(),
		RestorationBoundaryAbove: boundaryLength,
		RestorationBoundaryBelow: boundaryLength,
	}, nil
}

// bindDecoderFrameWorkSideData binds caller-owned scratch arenas into a
// reusable decoderFrameWorkSideData value containing the CDEF index map,
// loop-filter map, and loop-restoration buffers for one frame.
//
// See also: decoderFrameWorkSideDataScratchLen, setDecoderFrameWorkSideData.
func bindDecoderFrameWorkSideData(sequence SequenceHeader, size FrameSize, cdef internalparser.CDEFParams, restoration internalparser.RestorationParams, scratch decoderFrameWorkSideDataScratch) (decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkSideDataWith(sequence, size, cdef, restoration, scratch, nil, nil, nil, false)
}

func bindDecoderFrameWorkSideDataWith(sequence SequenceHeader, size FrameSize, cdef internalparser.CDEFParams, restoration internalparser.RestorationParams, scratch decoderFrameWorkSideDataScratch, compact *internalthreading.FrameWorkLoopFilterCompactState, noSkip []byte, event *internaldecoder.Event, maskOnlyLoopFilter bool) (decoderFrameWorkSideData, error) {
	scratchSize, err := decoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
	if err != nil {
		return decoderFrameWorkSideData{}, err
	}
	if decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFIndexMap, scratchSize.CDEFIndexMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFReadMap, scratchSize.CDEFReadMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.LoopFilterMap, scratchSize.LoopFilterMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationRecords, scratchSize.RestorationRecords) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryAbove, scratchSize.RestorationBoundaryAbove) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryBelow, scratchSize.RestorationBoundaryBelow) {
		return decoderFrameWorkSideData{}, internalframe.ErrShortBuffer
	}
	// The deblocking edge masks are OPTIONAL: a caller that provides mask +
	// level-cache scratch (the high-level Decoder does) gets the fast bitmask
	// loop-filter apply; a low-level caller that leaves them nil keeps the
	// byte-identical edge-list sweep. Only validate them when provided.
	buildMasks := !decoderFrameWorkPostFilterScratchTooShort(scratch.LoopFilterMasks, scratchSize.LoopFilterMasks) &&
		!decoderFrameWorkPostFilterScratchTooShort(scratch.LoopFilterLevelCache, scratchSize.LoopFilterLevelCache)
	// Clear the CDEF index/read scratch before binding. The scratch is reused
	// across frames, so it may still hold the previous frame's decoded cdef_idx
	// values. bindDecoderFrameWorkCDEFIndexMap validates each marked entry
	// against this frame's cdef_bits limit; a higher index left over from a
	// frame with more cdef_bits would otherwise be rejected against a smaller
	// limit. Per-frame CDEF state always starts empty, so clearing first makes
	// that validation operate on this frame's storage rather than stale data.
	cdefIndex := scratch.CDEFIndexMap[:scratchSize.CDEFIndexMap]
	cdefRead := scratch.CDEFReadMap[:scratchSize.CDEFReadMap]
	clear(cdefIndex)
	clear(cdefRead)
	cdefMap, err := bindDecoderFrameWorkCDEFIndexMap(
		sequence,
		size,
		cdef,
		cdefIndex,
		cdefRead,
	)
	if err != nil {
		return decoderFrameWorkSideData{}, err
	}
	// The per-MI record map is bound only for a caller that may read it. A
	// caller that binds the masks' level cache, tile-edge backups and noskip
	// (maskOnlyLoopFilter) deblocks from the masks alone, as dav1d keeps no
	// per-MI loop-filter record, so its map stays unbound and uncleared.
	var loopFilterMap internaldecoder.FrameWorkLoopFilterMap
	if compact == nil && !(maskOnlyLoopFilter && buildMasks) {
		loopFilterMap, err = bindDecoderFrameWorkLoopFilterMap(sequence, size, scratch.LoopFilterMap[:scratchSize.LoopFilterMap])
		if err != nil {
			return decoderFrameWorkSideData{}, err
		}
	}
	var loopFilterMasks internalthreading.FrameWorkLoopFilterMasks
	if buildMasks {
		loopFilterMasks, err = bindDecoderFrameWorkLoopFilterMasks(sequence, size, scratch.LoopFilterMasks[:scratchSize.LoopFilterMasks], scratch.LoopFilterLevelCache[:scratchSize.LoopFilterLevelCache])
		if err != nil {
			return decoderFrameWorkSideData{}, err
		}
		if compact != nil {
			if event == nil {
				return decoderFrameWorkSideData{}, internaldecoder.ErrInvalidFrameWorkState
			}
			if err := internalthreading.BindFrameWorkLoopFilterCompactState(&loopFilterMasks, compact, noSkip, event.LoopFilter, event.Segmentation, event.Delta, sequence.ColorConfig.MonoChrome); err != nil {
				return decoderFrameWorkSideData{}, err
			}
		}
	} else if compact != nil {
		return decoderFrameWorkSideData{}, internalframe.ErrShortBuffer
	}
	restorationBuffers, err := bindDecoderFrameWorkRestorationFrameBuffers(
		sequence,
		size,
		restoration,
		scratch.RestorationRecords[:scratchSize.RestorationRecords],
		scratch.RestorationBoundaryAbove[:scratchSize.RestorationBoundaryAbove],
		scratch.RestorationBoundaryBelow[:scratchSize.RestorationBoundaryBelow],
	)
	if err != nil {
		return decoderFrameWorkSideData{}, err
	}
	if err := restorationBuffers.ResetRecords(); err != nil {
		return decoderFrameWorkSideData{}, err
	}
	return decoderFrameWorkSideData{
		CDEFIndexMap:            cdefMap,
		LoopFilterMap:           loopFilterMap,
		LoopFilterMasks:         loopFilterMasks,
		RestorationFrameBuffers: restorationBuffers,
	}, nil
}

// decoderFrameWorkPostFilterRequestSideDataFromContext extracts postfilter
// request side-data views attached to a final-frame postfilter callback.
func decoderFrameWorkPostFilterRequestSideDataFromContext(ctx internaldecoder.FrameWorkPostFilterContext) decoderFrameWorkPostFilterRequestSideData {
	return postFilterRequestSideDataFromContext(&ctx)
}

func postFilterRequestSideDataFromContext(ctx *internaldecoder.FrameWorkPostFilterContext) decoderFrameWorkPostFilterRequestSideData {
	var side decoderFrameWorkPostFilterRequestSideData
	if ctx.LoopFilterMap != nil {
		side.LoopFilterMap = *ctx.LoopFilterMap
		side.LoopFilterTrustedCoverage = true
	}
	if ctx.CDEFIndexMap != nil {
		side.CDEFIndexMap = *ctx.CDEFIndexMap
	}
	if ctx.RestorationFrameBuffers != nil {
		side.RestorationRecords = ctx.RestorationFrameBuffers.Records
		side.RestorationBoundaries = ctx.RestorationFrameBuffers.Boundaries
	}
	return side
}

// setDecoderFrameWorkSideData attaches all bound frame-level side data to an
// active frame-work state as a single validated update.
func setDecoderFrameWorkSideData(state *internaldecoder.FrameWorkState, side decoderFrameWorkSideData) error {
	if err := state.SetSideData(side.CDEFIndexMap, side.LoopFilterMap, side.RestorationFrameBuffers); err != nil {
		return err
	}
	// Attach the deblocking edge masks so the supported post-filter's mask apply
	// (loopFilterMasksUsable) is byte-identical to and faster than the edge-list
	// sweep on single-tile frames. Harmless when the frame skips the loop filter.
	return state.SetLoopFilterMasks(side.LoopFilterMasks)
}

// bindDecoderFrameWorkPostFilterRequestBuffersFromScratch slices flat typed
// arenas into the component buffers used by bindDecoderFrameWorkPostFilterRequest.
func bindDecoderFrameWorkPostFilterRequestBuffersFromScratch(size internaldecoder.FrameWorkPostFilterScratchSize, side decoderFrameWorkPostFilterRequestSideData, scratch decoderFrameWorkPostFilterRequestScratch) (decoderFrameWorkPostFilterRequestBuffers, error) {
	buffers := decoderFrameWorkPostFilterRequestBuffers{
		LoopFilterMap:             side.LoopFilterMap,
		LoopFilterTrustedCoverage: side.LoopFilterTrustedCoverage,
		CDEFIndexMap:              side.CDEFIndexMap,

		RestorationRecords:    side.RestorationRecords,
		RestorationBoundaries: side.RestorationBoundaries,
		RestorationOptimized:  side.RestorationOptimized,
	}
	var err error
	buffers.LoopFilterEdges, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.LoopFilterEdges, size.LoopFilter.Edges)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.LoopFilterSchedule, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.LoopFilterSchedule, size.LoopFilter.Schedule)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.CDEFDirectionGrid, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.CDEFDirectionGrid, size.CDEF.DirectionGrid)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.CDEFVarianceGrid, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.CDEFVarianceGrid, size.CDEF.VarianceGrid)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}

	uint16Scratch := scratch.Uint16Scratch
	for plane := range 3 {
		buffers.CDEFSampleScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.Samples[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	for plane := range 3 {
		buffers.CDEFDstScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.Dst[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	buffers.CDEFInputScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.Input)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.CDEFUnitDstScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.UnitDst)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}

	byteScratch := scratch.ByteScratch
	buffers.SuperResOutputFrame, byteScratch, err = decoderFrameWorkPostFilterTakeScratch(byteScratch, size.SuperRes.OutputFrame)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	for plane := range 3 {
		buffers.CDEFSampleBytes[plane], byteScratch, err = decoderFrameWorkPostFilterTakeScratch(byteScratch, size.CDEF.SampleBytes[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	for plane := range 3 {
		buffers.SuperResCodedScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.SuperRes.CodedSamples[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	for plane := range 3 {
		buffers.SuperResOutputScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.SuperRes.OutputSamples[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}

	buffers.FilmGrainOutputFrame, _, err = decoderFrameWorkPostFilterTakeScratch(byteScratch, size.FilmGrain.OutputFrame)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}

	buffers.RestorationDataScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Samples.DataLen)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationDstScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Samples.DstLen)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationWienerScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Apply.Unit.Wiener)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationBoundaryAboveScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Apply.Boundary.Above)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationBoundaryBelowScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Apply.Boundary.Below)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}

	buffers.RestorationSGRProjScratch, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.Int32Scratch, size.Restoration.Apply.Unit.SGRProj)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}

	int16Scratch := scratch.Int16Scratch
	buffers.FilmGrainLumaGrain, int16Scratch, err = decoderFrameWorkPostFilterTakeScratch(int16Scratch, size.FilmGrain.LumaGrain)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	for plane := range 2 {
		buffers.FilmGrainChromaGrain[plane], int16Scratch, err = decoderFrameWorkPostFilterTakeScratch(int16Scratch, size.FilmGrain.ChromaGrain[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	buffers.FilmGrainLumaSamples, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.FilmGrain.LumaSamples)
	if err != nil {
		return decoderFrameWorkPostFilterRequestBuffers{}, err
	}
	for plane := range 2 {
		buffers.FilmGrainChromaSamples[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.FilmGrain.ChromaSamples[plane])
		if err != nil {
			return decoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	return buffers, nil
}

// bindDecoderFrameWorkPostFilterRequestFromScratch binds flat typed arenas
// directly into a full postfilter request.
func bindDecoderFrameWorkPostFilterRequestFromScratch(size internaldecoder.FrameWorkPostFilterScratchSize, side decoderFrameWorkPostFilterRequestSideData, scratch decoderFrameWorkPostFilterRequestScratch) (internaldecoder.FrameWorkPostFilterRequest, error) {
	buffers, err := bindDecoderFrameWorkPostFilterRequestBuffersFromScratch(size, side, scratch)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	return bindDecoderFrameWorkPostFilterRequest(size, buffers)
}

// bindDecoderFrameWorkPostFilterRequest binds all caller-owned postfilter
// scratch into one request. Zero-sized stages accept nil backing slices.
func bindDecoderFrameWorkPostFilterRequest(size internaldecoder.FrameWorkPostFilterScratchSize, buffers decoderFrameWorkPostFilterRequestBuffers) (internaldecoder.FrameWorkPostFilterRequest, error) {
	loopFilterReq, err := bindDecoderFrameWorkLoopFilterPostFilterRequest(size.LoopFilter, buffers.LoopFilterMap, buffers.LoopFilterEdges, buffers.LoopFilterSchedule)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	loopFilterReq.TrustedCoverage = buffers.LoopFilterTrustedCoverage
	cdefReq, err := bindDecoderFrameWorkCDEFPostFilterRequest(size.CDEF, buffers.CDEFIndexMap, buffers.CDEFSampleScratch, buffers.CDEFDstScratch, buffers.CDEFDirectionGrid, buffers.CDEFVarianceGrid, buffers.CDEFInputScratch, buffers.CDEFUnitDstScratch)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	if err := size.CDEF.BindSampleBytes(&cdefReq, buffers.CDEFSampleBytes); err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	superResReq, err := bindDecoderFrameWorkSuperResPostFilterRequest(size.SuperRes, buffers.SuperResOutputFrame, buffers.SuperResCodedScratch, buffers.SuperResOutputScratch)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	restorationReq, err := bindDecoderFrameWorkRestorationPostFilterRequest(size.Restoration, buffers.RestorationRecords, buffers.RestorationBoundaries, buffers.RestorationDataScratch, buffers.RestorationDstScratch, buffers.RestorationWienerScratch, buffers.RestorationSGRProjScratch, buffers.RestorationBoundaryAboveScratch, buffers.RestorationBoundaryBelowScratch, buffers.RestorationOptimized)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	filmGrainReq, err := bindDecoderFrameWorkFilmGrainPostFilterRequest(size.FilmGrain, buffers.FilmGrainOutputFrame, buffers.FilmGrainLumaGrain, buffers.FilmGrainChromaGrain, buffers.FilmGrainLumaSamples, buffers.FilmGrainChromaSamples)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterRequest{}, err
	}
	return internaldecoder.FrameWorkPostFilterRequest{
		LoopFilter:  loopFilterReq,
		CDEF:        cdefReq,
		SuperRes:    superResReq,
		Restoration: restorationReq,
		FilmGrain:   filmGrainReq,
	}, nil
}

// bindDecoderFrameWorkLoopFilterPostFilterRequest binds the caller-owned
// loop-filter map and edges slice into a
// DecoderFrameWorkLoopFilterPostFilterRequest sized for size.
func bindDecoderFrameWorkLoopFilterPostFilterRequest(size internaldecoder.FrameWorkLoopFilterPostFilterScratchSize, filterMap internaldecoder.FrameWorkLoopFilterMap, edges []internaldecoder.FrameWorkLoopFilterPostFilterEdge, schedule []uint32) (internaldecoder.FrameWorkLoopFilterPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(edges, size.Edges) {
		return internaldecoder.FrameWorkLoopFilterPostFilterRequest{}, internalframe.ErrShortBuffer
	}
	if decoderFrameWorkPostFilterScratchTooShort(schedule, size.Schedule) {
		return internaldecoder.FrameWorkLoopFilterPostFilterRequest{}, internalframe.ErrShortBuffer
	}
	return internaldecoder.FrameWorkLoopFilterPostFilterRequest{
		Map:      filterMap,
		Edges:    edges[:size.Edges],
		Schedule: schedule[:size.Schedule],
	}, nil
}

// decoderFrameWorkRestorationFramePlan returns the loop-restoration
// frame plan for (sequence, size, restoration). The plan reports the
// per-plane unit grid, stripe geometry, and boundary buffer sizing
// needed to bind a DecoderFrameWorkRestorationFrameBuffers value.
func decoderFrameWorkRestorationFramePlan(sequence SequenceHeader, size FrameSize, restoration internalparser.RestorationParams) (internaltile.RestorationFramePlan, error) {
	batch := decoderFrameWorkRestorationBatch(sequence, size, restoration)
	return batch.RestorationFramePlan()
}

// bindDecoderFrameWorkRestorationFrameBuffers binds caller-owned
// records, above-boundary, and below-boundary slices into the per-frame
// loop-restoration buffers used by the residual and post-filter
// pipelines.
//
// See also: decoderFrameWorkRestorationFramePlan.
func bindDecoderFrameWorkRestorationFrameBuffers(sequence SequenceHeader, size FrameSize, restoration internalparser.RestorationParams, records []internaltile.RestorationUnitRecord, above []uint16, below []uint16) (internalthreading.FrameWorkRestorationFrameBuffers, error) {
	batch := decoderFrameWorkRestorationBatch(sequence, size, restoration)
	return batch.BindRestorationFrameBuffers(records, above, below)
}

// bindDecoderFrameWorkCDEFPostFilterRequest binds the caller-owned CDEF
// scratch (per-plane samples, dst, direction grid, variance grid,
// input/unit-dst buffers) and indexMap into a CDEF apply request.
func bindDecoderFrameWorkCDEFPostFilterRequest(size internaldecoder.FrameWorkCDEFPostFilterScratchSize, indexMap internaldecoder.FrameWorkCDEFIndexMap, sampleScratch [3][]uint16, dstScratch [3][]uint16, directionGrid []internalcdef.DirectionGrid, varianceGrid []internalcdef.VarianceGrid, inputScratch []uint16, unitDstScratch []uint16) (internaldecoder.FrameWorkCDEFPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(directionGrid, size.DirectionGrid) ||
		decoderFrameWorkPostFilterScratchTooShort(varianceGrid, size.VarianceGrid) ||
		decoderFrameWorkPostFilterScratchTooShort(inputScratch, size.Input) ||
		decoderFrameWorkPostFilterScratchTooShort(unitDstScratch, size.UnitDst) {
		return internaldecoder.FrameWorkCDEFPostFilterRequest{}, internalframe.ErrShortBuffer
	}
	req := internaldecoder.FrameWorkCDEFPostFilterRequest{
		IndexMap:       indexMap,
		DirectionGrid:  directionGrid[:size.DirectionGrid],
		VarianceGrid:   varianceGrid[:size.VarianceGrid],
		InputScratch:   inputScratch[:size.Input],
		UnitDstScratch: unitDstScratch[:size.UnitDst],
	}
	for plane := range 3 {
		if decoderFrameWorkPostFilterScratchTooShort(sampleScratch[plane], size.Samples[plane]) ||
			decoderFrameWorkPostFilterScratchTooShort(dstScratch[plane], size.Dst[plane]) {
			return internaldecoder.FrameWorkCDEFPostFilterRequest{}, internalframe.ErrShortBuffer
		}
		req.SampleScratch[plane] = sampleScratch[plane][:size.Samples[plane]]
		req.DstScratch[plane] = dstScratch[plane][:size.Dst[plane]]
	}
	return req, nil
}

// bindDecoderFrameWorkSuperResPostFilterRequest binds the caller-owned
// super-res scratch (output frame backing plus per-plane coded and
// output scratch) into a super-res apply request.
func bindDecoderFrameWorkSuperResPostFilterRequest(size internaldecoder.FrameWorkSuperResPostFilterScratchSize, outputFrame []byte, codedScratch [3][]uint16, outputScratch [3][]uint16) (internaldecoder.FrameWorkSuperResPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(outputFrame, size.OutputFrame) {
		return internaldecoder.FrameWorkSuperResPostFilterRequest{}, internalframe.ErrShortBuffer
	}
	req := internaldecoder.FrameWorkSuperResPostFilterRequest{
		OutputFrame: outputFrame[:size.OutputFrame],
	}
	for plane := range 3 {
		if decoderFrameWorkPostFilterScratchTooShort(codedScratch[plane], size.CodedSamples[plane]) ||
			decoderFrameWorkPostFilterScratchTooShort(outputScratch[plane], size.OutputSamples[plane]) {
			return internaldecoder.FrameWorkSuperResPostFilterRequest{}, internalframe.ErrShortBuffer
		}
		req.CodedScratch[plane] = codedScratch[plane][:size.CodedSamples[plane]]
		req.OutputScratch[plane] = outputScratch[plane][:size.OutputSamples[plane]]
	}
	return req, nil
}

// bindDecoderFrameWorkRestorationPostFilterRequest binds caller-owned
// loop-restoration buffers (records, stripe boundaries, data/dst sample
// scratch, Wiener and SGR-proj per-unit scratch, above/below boundary
// scratch) into a loop-restoration apply request. The optimized flag
// selects the SIMD-friendly apply path when supported.
func bindDecoderFrameWorkRestorationPostFilterRequest(size internaldecoder.FrameWorkRestorationPostFilterScratchSize, records [3][]internaltile.RestorationUnitRecord, boundaries [3]internaltile.RestorationStripeBoundaries, dataScratch []uint16, dstScratch []uint16, wienerScratch []uint16, sgrProjScratch []int32, boundaryAboveScratch []uint16, boundaryBelowScratch []uint16, optimized bool) (internaldecoder.FrameWorkRestorationPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(dataScratch, size.Samples.DataLen) ||
		decoderFrameWorkPostFilterScratchTooShort(dstScratch, size.Samples.DstLen) ||
		decoderFrameWorkPostFilterScratchTooShort(wienerScratch, size.Apply.Unit.Wiener) ||
		decoderFrameWorkPostFilterScratchTooShort(sgrProjScratch, size.Apply.Unit.SGRProj) ||
		decoderFrameWorkPostFilterScratchTooShort(boundaryAboveScratch, size.Apply.Boundary.Above) ||
		decoderFrameWorkPostFilterScratchTooShort(boundaryBelowScratch, size.Apply.Boundary.Below) {
		return internaldecoder.FrameWorkRestorationPostFilterRequest{}, internalframe.ErrShortBuffer
	}
	return internaldecoder.FrameWorkRestorationPostFilterRequest{
		Records:     records,
		Boundaries:  boundaries,
		DataScratch: dataScratch[:size.Samples.DataLen],
		DstScratch:  dstScratch[:size.Samples.DstLen],
		Scratch: internaltile.RestorationUnitRecordBoundaryScratch{
			Unit: internaltile.RestorationUnitScratch{
				Wiener:  wienerScratch[:size.Apply.Unit.Wiener],
				SGRProj: sgrProjScratch[:size.Apply.Unit.SGRProj],
			},
			Boundary: internaltile.RestorationStripeBoundaryScratch{
				Above: boundaryAboveScratch[:size.Apply.Boundary.Above],
				Below: boundaryBelowScratch[:size.Apply.Boundary.Below],
			},
		},
		Optimized: optimized,
	}, nil
}

// bindDecoderFrameWorkFilmGrainPostFilterRequest binds caller-owned film-grain
// display, grain, and sample scratch into a film-grain apply request.
func bindDecoderFrameWorkFilmGrainPostFilterRequest(size internaldecoder.FrameWorkFilmGrainPostFilterScratchSize, outputFrame []byte, lumaGrain []int16, chromaGrain [2][]int16, lumaSamples []uint16, chromaSamples [2][]uint16) (internaldecoder.FrameWorkFilmGrainPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(outputFrame, size.OutputFrame) ||
		decoderFrameWorkPostFilterScratchTooShort(lumaGrain, size.LumaGrain) ||
		decoderFrameWorkPostFilterScratchTooShort(lumaSamples, size.LumaSamples) {
		return internaldecoder.FrameWorkFilmGrainPostFilterRequest{}, internalframe.ErrShortBuffer
	}
	req := internaldecoder.FrameWorkFilmGrainPostFilterRequest{
		OutputFrame: outputFrame[:size.OutputFrame],
		LumaGrain:   lumaGrain[:size.LumaGrain],
		LumaSamples: lumaSamples[:size.LumaSamples],
	}
	for plane := range 2 {
		if decoderFrameWorkPostFilterScratchTooShort(chromaGrain[plane], size.ChromaGrain[plane]) ||
			decoderFrameWorkPostFilterScratchTooShort(chromaSamples[plane], size.ChromaSamples[plane]) {
			return internaldecoder.FrameWorkFilmGrainPostFilterRequest{}, internalframe.ErrShortBuffer
		}
		req.ChromaGrain[plane] = chromaGrain[plane][:size.ChromaGrain[plane]]
		req.ChromaSamples[plane] = chromaSamples[plane][:size.ChromaSamples[plane]]
	}
	return req, nil
}

func decoderFrameWorkPostFilterScratchTooShort[T any](scratch []T, need int) bool {
	return need < 0 || len(scratch) < need
}

func decoderFrameWorkPostFilterTakeScratch[T any](scratch []T, need int) ([]T, []T, error) {
	if decoderFrameWorkPostFilterScratchTooShort(scratch, need) {
		return nil, nil, internalframe.ErrShortBuffer
	}
	return scratch[:need], scratch[need:], nil
}

func decoderFrameWorkCDEFIndexBatch(sequence SequenceHeader, size FrameSize, cdef internalparser.CDEFParams) internalthreading.FrameWorkBatch {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	batch.CDEF = cdef
	return batch
}

func decoderFrameWorkRestorationBatch(sequence SequenceHeader, size FrameSize, restoration internalparser.RestorationParams) internalthreading.FrameWorkBatch {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	batch.Restoration = restoration
	return batch
}

func decoderFrameWorkFrameBatch(sequence SequenceHeader, size FrameSize) internalthreading.FrameWorkBatch {
	return internalthreading.FrameWorkBatch{
		FrameWorkFrameContext: internalthreading.FrameWorkFrameContext{
			Sequence:  internalthreading.FrameWorkSequenceContextFromHeader(sequence),
			FrameSize: size,
		},
	}
}

// postFilterSource returns the reconstructed picture behind f when f is the
// runner's detached film-grain display view (dav1d applies grain into a new
// output picture while the source stays the reference), or f itself.
func (r *decoderFrameWorkReusableSupportedPostFilterRunner) postFilterSource(f *Frame) *Frame {
	if r != nil && f == &r.runner.FilmGrainOutput && r.runner.source != nil {
		return r.runner.source
	}
	return f
}
