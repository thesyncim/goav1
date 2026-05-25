package goav1

import internalthreading "github.com/thesyncim/goav1/internal/av1/threading"

func DecoderFrameWorkCDEFIndexMapShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkCDEFIndexBatch(sequence, size, CDEFParams{})
	return batch.CDEFIndexMapShape()
}

func DecoderFrameWorkLoopFilterMapShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.LoopFilterMapShape()
}

func BindDecoderFrameWorkCDEFIndexMap(sequence SequenceHeader, size FrameSize, cdef CDEFParams, index []uint8, read []bool) (DecoderFrameWorkCDEFIndexMap, error) {
	batch := decoderFrameWorkCDEFIndexBatch(sequence, size, cdef)
	return batch.BindCDEFIndexMap(index, read)
}

func BindDecoderFrameWorkLoopFilterMap(sequence SequenceHeader, size FrameSize, records []DecoderFrameWorkLoopFilterBlockRecord) (DecoderFrameWorkLoopFilterMap, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindLoopFilterMap(records)
}

func ResetDecoderFrameWorkCDEFIndexMap(indexMap DecoderFrameWorkCDEFIndexMap) error {
	return indexMap.Reset()
}

func MarkDecoderFrameWorkCDEFIndexMapBlock(indexMap DecoderFrameWorkCDEFIndexMap, cdef CDEFParams, visit TileBlockLoopVisit) error {
	return indexMap.MarkBlock(cdef, visit)
}

func ResetDecoderFrameWorkLoopFilterMap(filterMap DecoderFrameWorkLoopFilterMap) error {
	return filterMap.Reset()
}

func MarkDecoderFrameWorkLoopFilterMapBlock(filterMap DecoderFrameWorkLoopFilterMap, visit TileBlockLoopVisit, state *TileDecodeState) error {
	return filterMap.MarkBlock(visit, state)
}

// DecoderFrameWorkPostFilterRequestBuffers groups caller-owned side data and
// scratch slices used to bind a full postfilter request.
type DecoderFrameWorkPostFilterRequestBuffers struct {
	LoopFilterMap   DecoderFrameWorkLoopFilterMap
	LoopFilterEdges []DecoderFrameWorkLoopFilterPostFilterEdge

	CDEFIndexMap       DecoderFrameWorkCDEFIndexMap
	CDEFSampleScratch  [3][]uint16
	CDEFDstScratch     [3][]uint16
	CDEFDirectionGrid  []CDEFDirectionGrid
	CDEFVarianceGrid   []CDEFVarianceGrid
	CDEFInputScratch   []uint16
	CDEFUnitDstScratch []uint16

	SuperResOutputFrame   []byte
	SuperResCodedScratch  [3][]uint16
	SuperResOutputScratch [3][]uint16

	RestorationRecords              [3][]TileRestorationUnitRecord
	RestorationBoundaries           [3]TileRestorationStripeBoundaries
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

// DecoderFrameWorkPostFilterRequestSideData groups postfilter side data that is
// not carved out of scratch arenas.
type DecoderFrameWorkPostFilterRequestSideData struct {
	LoopFilterMap DecoderFrameWorkLoopFilterMap
	CDEFIndexMap  DecoderFrameWorkCDEFIndexMap

	RestorationRecords    [3][]TileRestorationUnitRecord
	RestorationBoundaries [3]TileRestorationStripeBoundaries
	RestorationOptimized  bool
}

// DecoderFrameWorkPostFilterRequestScratchSize reports typed arena lengths
// needed to bind a full postfilter request.
type DecoderFrameWorkPostFilterRequestScratchSize struct {
	LoopFilterEdges   int
	CDEFDirectionGrid int
	CDEFVarianceGrid  int

	ByteScratch   int
	Uint16Scratch int
	Int16Scratch  int
	Int32Scratch  int
}

// DecoderFrameWorkPostFilterRequestScratch carries typed arenas consumed by
// BindDecoderFrameWorkPostFilterRequestBuffersFromScratch.
type DecoderFrameWorkPostFilterRequestScratch struct {
	LoopFilterEdges   []DecoderFrameWorkLoopFilterPostFilterEdge
	CDEFDirectionGrid []CDEFDirectionGrid
	CDEFVarianceGrid  []CDEFVarianceGrid

	ByteScratch   []byte
	Uint16Scratch []uint16
	Int16Scratch  []int16
	Int32Scratch  []int32
}

// DecoderFrameWorkSideDataScratchSize reports caller-owned side-data backing
// lengths for tile residual decode and final postfilter planning.
type DecoderFrameWorkSideDataScratchSize struct {
	CDEFIndexMap int
	CDEFReadMap  int

	LoopFilterMap int

	RestorationRecords       int
	RestorationBoundaryAbove int
	RestorationBoundaryBelow int
}

// DecoderFrameWorkSideDataScratch carries typed caller-owned side-data backing
// for BindDecoderFrameWorkSideData.
type DecoderFrameWorkSideDataScratch struct {
	CDEFIndexMap []uint8
	CDEFReadMap  []bool

	LoopFilterMap []DecoderFrameWorkLoopFilterBlockRecord

	RestorationRecords       []TileRestorationUnitRecord
	RestorationBoundaryAbove []uint16
	RestorationBoundaryBelow []uint16
}

// DecoderFrameWorkSideData groups frame-level side maps shared by tile
// residual decode and postfilter planning.
type DecoderFrameWorkSideData struct {
	CDEFIndexMap            DecoderFrameWorkCDEFIndexMap
	LoopFilterMap           DecoderFrameWorkLoopFilterMap
	RestorationFrameBuffers DecoderFrameWorkRestorationFrameBuffers
}

// DecoderFrameWorkSupportedPostFilterScratchRunner binds exact postfilter
// request views from caller-owned scratch at final-frame callback time, after
// tile residual decode has filled frame side data.
type DecoderFrameWorkSupportedPostFilterScratchRunner struct {
	Scratch              DecoderFrameWorkPostFilterRequestScratch
	RestorationOptimized bool

	Size    DecoderFrameWorkPostFilterScratchSize
	Request DecoderFrameWorkPostFilterRequest
	Context DecoderFrameWorkPostFilterContext
	Result  DecoderFrameWorkPostFilterResult
}

// DecoderFrameWorkCallerPostFilterScratchRunner binds caller-owned full
// postfilter request views from typed scratch at final-frame callback time. It
// allows superres to switch Context.Output to detached caller-owned storage.
type DecoderFrameWorkCallerPostFilterScratchRunner struct {
	Scratch              DecoderFrameWorkPostFilterRequestScratch
	RestorationOptimized bool
	SuperResOutput       Frame

	Size    DecoderFrameWorkPostFilterScratchSize
	Request DecoderFrameWorkPostFilterRequest
	Context DecoderFrameWorkPostFilterContext
	Result  DecoderFrameWorkCallerPostFilterResult
}

// ScratchLen reports the exact scratch size required for ctx using side data
// carried by the context.
func (r *DecoderFrameWorkSupportedPostFilterScratchRunner) ScratchLen(ctx DecoderFrameWorkPostFilterContext) (DecoderFrameWorkPostFilterScratchSize, error) {
	if r == nil {
		return DecoderFrameWorkPostFilterScratchSize{}, ErrDecoderInvalidFrameWorkState
	}
	return ctx.SupportedPostFilterScratchLen(DecoderFrameWorkPostFilterRequest{
		Restoration: DecoderFrameWorkRestorationPostFilterRequest{Optimized: r.RestorationOptimized},
	})
}

// Apply binds exact request views from Scratch, applies supported postfilters,
// and requires the resulting output to remain publishable by the frame pool.
func (r *DecoderFrameWorkSupportedPostFilterScratchRunner) Apply(ctx DecoderFrameWorkPostFilterContext) error {
	if r == nil {
		return ErrDecoderInvalidFrameWorkState
	}
	size, err := r.ScratchLen(ctx)
	if err != nil {
		return err
	}
	side := DecoderFrameWorkPostFilterRequestSideDataFromContext(ctx)
	side.RestorationOptimized = r.RestorationOptimized
	req, err := BindDecoderFrameWorkPostFilterRequestFromScratch(size, side, r.Scratch)
	if err != nil {
		return err
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		return err
	}
	if err := next.RequirePublishablePostFilterOutput(); err != nil {
		return err
	}
	r.Size = size
	r.Request = req
	r.Context = next
	r.Result = result
	return nil
}

// ScratchLen reports caller-owned scratch for ctx. When superres is active,
// the first call may only report pre/post-superres bootstrap scratch; after
// Scratch.ByteScratch is sized for the superres output frame, this reports the
// full tail scratch for loop restoration and film grain.
func (r *DecoderFrameWorkCallerPostFilterScratchRunner) ScratchLen(ctx DecoderFrameWorkPostFilterContext) (DecoderFrameWorkPostFilterScratchSize, error) {
	if r == nil {
		return DecoderFrameWorkPostFilterScratchSize{}, ErrDecoderInvalidFrameWorkState
	}
	req := DecoderFrameWorkPostFilterRequest{
		Restoration: DecoderFrameWorkRestorationPostFilterRequest{Optimized: r.RestorationOptimized},
	}
	size, err := ctx.CallerPostFilterScratchLen(req)
	if err != nil {
		return DecoderFrameWorkPostFilterScratchSize{}, err
	}
	if size.SuperRes.OutputFrame != 0 && len(r.Scratch.ByteScratch) >= size.SuperRes.OutputFrame {
		req.SuperRes.OutputFrame = r.Scratch.ByteScratch[:size.SuperRes.OutputFrame]
		req.SuperRes.OutputView = &r.SuperResOutput
		size, err = ctx.CallerPostFilterScratchLen(req)
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
	}
	return size, nil
}

// Apply binds caller-owned request views from Scratch and runs the full
// postfilter chain, allowing Context.Output to become detached scratch.
func (r *DecoderFrameWorkCallerPostFilterScratchRunner) Apply(ctx DecoderFrameWorkPostFilterContext) error {
	if r == nil {
		return ErrDecoderInvalidFrameWorkState
	}
	size, err := r.ScratchLen(ctx)
	if err != nil {
		return err
	}
	side := DecoderFrameWorkPostFilterRequestSideDataFromContext(ctx)
	side.RestorationOptimized = r.RestorationOptimized
	req, err := BindDecoderFrameWorkPostFilterRequestFromScratch(size, side, r.Scratch)
	if err != nil {
		return err
	}
	req.SuperRes.OutputView = &r.SuperResOutput
	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		return err
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		return err
	}
	r.Size = size
	r.Request = req
	r.Context = next
	r.Result = result
	return nil
}

// DecoderFrameWorkPostFilterScratchContext builds a header-derived final-frame
// postfilter context for scratch sizing. output is filled with a geometry-only
// coded-frame view; its pixel slices are nil and must not be used to apply
// filters.
func DecoderFrameWorkPostFilterScratchContext(sequence SequenceHeader, event DecoderEvent, align int, side *DecoderFrameWorkSideData, output *Frame) (DecoderFrameWorkPostFilterContext, error) {
	if output == nil {
		return DecoderFrameWorkPostFilterContext{}, ErrFrameInvalidSlot
	}
	event.SequenceHeader = sequence
	format, err := FrameCodedFormatFromHeaders(sequence, event.FrameSize, align)
	if err != nil {
		return DecoderFrameWorkPostFilterContext{}, err
	}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		return DecoderFrameWorkPostFilterContext{}, err
	}
	*output = decoderFrameWorkPostFilterScratchFrame(format, layout)
	ctx := DecoderFrameWorkPostFilterContext{
		Event:  event,
		Output: output,
	}
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

// DecoderFrameWorkPostFilterRequestScratchLen reports the flat typed arena
// lengths needed for size.
func DecoderFrameWorkPostFilterRequestScratchLen(size DecoderFrameWorkPostFilterScratchSize) DecoderFrameWorkPostFilterRequestScratchSize {
	var uint16Scratch int
	for plane := 0; plane < 3; plane++ {
		uint16Scratch += size.CDEF.Samples[plane] + size.CDEF.Dst[plane]
		uint16Scratch += size.SuperRes.CodedSamples[plane] + size.SuperRes.OutputSamples[plane]
	}
	uint16Scratch += size.CDEF.Input + size.CDEF.UnitDst
	uint16Scratch += size.Restoration.Samples.DataLen + size.Restoration.Samples.DstLen
	uint16Scratch += size.Restoration.Apply.Unit.Wiener
	uint16Scratch += size.Restoration.Apply.Boundary.Above + size.Restoration.Apply.Boundary.Below
	uint16Scratch += size.FilmGrain.LumaSamples
	for plane := 0; plane < 2; plane++ {
		uint16Scratch += size.FilmGrain.ChromaSamples[plane]
	}

	int16Scratch := size.FilmGrain.LumaGrain
	for plane := 0; plane < 2; plane++ {
		int16Scratch += size.FilmGrain.ChromaGrain[plane]
	}

	return DecoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:   size.LoopFilter.Edges,
		CDEFDirectionGrid: size.CDEF.DirectionGrid,
		CDEFVarianceGrid:  size.CDEF.VarianceGrid,
		ByteScratch:       size.SuperRes.OutputFrame,
		Uint16Scratch:     uint16Scratch,
		Int16Scratch:      int16Scratch,
		Int32Scratch:      size.Restoration.Apply.Unit.SGRProj,
	}
}

func DecoderFrameWorkSideDataScratchLen(sequence SequenceHeader, size FrameSize, cdef CDEFParams, restoration RestorationParams) (DecoderFrameWorkSideDataScratchSize, error) {
	_, _, cdefLength, err := DecoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		return DecoderFrameWorkSideDataScratchSize{}, err
	}
	_, _, loopFilterLength, err := DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		return DecoderFrameWorkSideDataScratchSize{}, err
	}
	restorationPlan, err := DecoderFrameWorkRestorationFramePlan(sequence, size, restoration)
	if err != nil {
		return DecoderFrameWorkSideDataScratchSize{}, err
	}
	boundaryLength := restorationPlan.BoundaryBufferLen()
	return DecoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             cdefLength,
		CDEFReadMap:              cdefLength,
		LoopFilterMap:            loopFilterLength,
		RestorationRecords:       restorationPlan.UnitRecordLen(),
		RestorationBoundaryAbove: boundaryLength,
		RestorationBoundaryBelow: boundaryLength,
	}, nil
}

func BindDecoderFrameWorkSideData(sequence SequenceHeader, size FrameSize, cdef CDEFParams, restoration RestorationParams, scratch DecoderFrameWorkSideDataScratch) (DecoderFrameWorkSideData, error) {
	scratchSize, err := DecoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
	if err != nil {
		return DecoderFrameWorkSideData{}, err
	}
	if decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFIndexMap, scratchSize.CDEFIndexMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFReadMap, scratchSize.CDEFReadMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.LoopFilterMap, scratchSize.LoopFilterMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationRecords, scratchSize.RestorationRecords) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryAbove, scratchSize.RestorationBoundaryAbove) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryBelow, scratchSize.RestorationBoundaryBelow) {
		return DecoderFrameWorkSideData{}, ErrFrameShortBuffer
	}
	cdefMap, err := BindDecoderFrameWorkCDEFIndexMap(
		sequence,
		size,
		cdef,
		scratch.CDEFIndexMap[:scratchSize.CDEFIndexMap],
		scratch.CDEFReadMap[:scratchSize.CDEFReadMap],
	)
	if err != nil {
		return DecoderFrameWorkSideData{}, err
	}
	if err := cdefMap.Reset(); err != nil {
		return DecoderFrameWorkSideData{}, err
	}
	loopFilterMap, err := BindDecoderFrameWorkLoopFilterMap(sequence, size, scratch.LoopFilterMap[:scratchSize.LoopFilterMap])
	if err != nil {
		return DecoderFrameWorkSideData{}, err
	}
	restorationBuffers, err := BindDecoderFrameWorkRestorationFrameBuffers(
		sequence,
		size,
		restoration,
		scratch.RestorationRecords[:scratchSize.RestorationRecords],
		scratch.RestorationBoundaryAbove[:scratchSize.RestorationBoundaryAbove],
		scratch.RestorationBoundaryBelow[:scratchSize.RestorationBoundaryBelow],
	)
	if err != nil {
		return DecoderFrameWorkSideData{}, err
	}
	if err := restorationBuffers.ResetRecords(); err != nil {
		return DecoderFrameWorkSideData{}, err
	}
	return DecoderFrameWorkSideData{
		CDEFIndexMap:            cdefMap,
		LoopFilterMap:           loopFilterMap,
		RestorationFrameBuffers: restorationBuffers,
	}, nil
}

func DecoderFrameWorkPostFilterSideData(side DecoderFrameWorkSideData) DecoderFrameWorkPostFilterRequestSideData {
	return DecoderFrameWorkPostFilterRequestSideData{
		LoopFilterMap:         side.LoopFilterMap,
		CDEFIndexMap:          side.CDEFIndexMap,
		RestorationRecords:    side.RestorationFrameBuffers.Records,
		RestorationBoundaries: side.RestorationFrameBuffers.Boundaries,
	}
}

// DecoderFrameWorkPostFilterRequestSideDataFromContext extracts postfilter
// request side-data views attached to a final-frame postfilter callback.
func DecoderFrameWorkPostFilterRequestSideDataFromContext(ctx DecoderFrameWorkPostFilterContext) DecoderFrameWorkPostFilterRequestSideData {
	var side DecoderFrameWorkPostFilterRequestSideData
	if ctx.LoopFilterMap != nil {
		side.LoopFilterMap = *ctx.LoopFilterMap
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

// SetDecoderFrameWorkSideData attaches all bound frame-level side data to an
// active frame-work state as a single validated update.
func SetDecoderFrameWorkSideData(state *DecoderFrameWorkState, side DecoderFrameWorkSideData) error {
	return state.SetSideData(side.CDEFIndexMap, side.LoopFilterMap, side.RestorationFrameBuffers)
}

// BindDecoderFrameWorkPostFilterRequestBuffersFromScratch slices flat typed
// arenas into the component buffers used by BindDecoderFrameWorkPostFilterRequest.
func BindDecoderFrameWorkPostFilterRequestBuffersFromScratch(size DecoderFrameWorkPostFilterScratchSize, side DecoderFrameWorkPostFilterRequestSideData, scratch DecoderFrameWorkPostFilterRequestScratch) (DecoderFrameWorkPostFilterRequestBuffers, error) {
	buffers := DecoderFrameWorkPostFilterRequestBuffers{
		LoopFilterMap: side.LoopFilterMap,
		CDEFIndexMap:  side.CDEFIndexMap,

		RestorationRecords:    side.RestorationRecords,
		RestorationBoundaries: side.RestorationBoundaries,
		RestorationOptimized:  side.RestorationOptimized,
	}
	var err error
	buffers.LoopFilterEdges, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.LoopFilterEdges, size.LoopFilter.Edges)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.CDEFDirectionGrid, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.CDEFDirectionGrid, size.CDEF.DirectionGrid)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.CDEFVarianceGrid, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.CDEFVarianceGrid, size.CDEF.VarianceGrid)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}

	uint16Scratch := scratch.Uint16Scratch
	for plane := 0; plane < 3; plane++ {
		buffers.CDEFSampleScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.Samples[plane])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	for plane := 0; plane < 3; plane++ {
		buffers.CDEFDstScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.Dst[plane])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	buffers.CDEFInputScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.Input)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.CDEFUnitDstScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.CDEF.UnitDst)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}

	buffers.SuperResOutputFrame, _, err = decoderFrameWorkPostFilterTakeScratch(scratch.ByteScratch, size.SuperRes.OutputFrame)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	for plane := 0; plane < 3; plane++ {
		buffers.SuperResCodedScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.SuperRes.CodedSamples[plane])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	for plane := 0; plane < 3; plane++ {
		buffers.SuperResOutputScratch[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.SuperRes.OutputSamples[plane])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}

	buffers.RestorationDataScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Samples.DataLen)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationDstScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Samples.DstLen)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationWienerScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Apply.Unit.Wiener)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationBoundaryAboveScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Apply.Boundary.Above)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	buffers.RestorationBoundaryBelowScratch, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.Restoration.Apply.Boundary.Below)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}

	int32Scratch := scratch.Int32Scratch
	buffers.RestorationSGRProjScratch, int32Scratch, err = decoderFrameWorkPostFilterTakeScratch(int32Scratch, size.Restoration.Apply.Unit.SGRProj)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}

	int16Scratch := scratch.Int16Scratch
	buffers.FilmGrainLumaGrain, int16Scratch, err = decoderFrameWorkPostFilterTakeScratch(int16Scratch, size.FilmGrain.LumaGrain)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	for plane := 0; plane < 2; plane++ {
		buffers.FilmGrainChromaGrain[plane], int16Scratch, err = decoderFrameWorkPostFilterTakeScratch(int16Scratch, size.FilmGrain.ChromaGrain[plane])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	buffers.FilmGrainLumaSamples, uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.FilmGrain.LumaSamples)
	if err != nil {
		return DecoderFrameWorkPostFilterRequestBuffers{}, err
	}
	for plane := 0; plane < 2; plane++ {
		buffers.FilmGrainChromaSamples[plane], uint16Scratch, err = decoderFrameWorkPostFilterTakeScratch(uint16Scratch, size.FilmGrain.ChromaSamples[plane])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestBuffers{}, err
		}
	}
	return buffers, nil
}

// BindDecoderFrameWorkPostFilterRequestBuffersFromSideData slices flat typed
// arenas into postfilter buffers using frame-work side data directly.
func BindDecoderFrameWorkPostFilterRequestBuffersFromSideData(size DecoderFrameWorkPostFilterScratchSize, side DecoderFrameWorkSideData, scratch DecoderFrameWorkPostFilterRequestScratch) (DecoderFrameWorkPostFilterRequestBuffers, error) {
	return BindDecoderFrameWorkPostFilterRequestBuffersFromScratch(size, DecoderFrameWorkPostFilterSideData(side), scratch)
}

// BindDecoderFrameWorkPostFilterRequestFromScratch binds flat typed arenas
// directly into a full postfilter request.
func BindDecoderFrameWorkPostFilterRequestFromScratch(size DecoderFrameWorkPostFilterScratchSize, side DecoderFrameWorkPostFilterRequestSideData, scratch DecoderFrameWorkPostFilterRequestScratch) (DecoderFrameWorkPostFilterRequest, error) {
	buffers, err := BindDecoderFrameWorkPostFilterRequestBuffersFromScratch(size, side, scratch)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	return BindDecoderFrameWorkPostFilterRequest(size, buffers)
}

// BindDecoderFrameWorkPostFilterRequestFromSideData binds flat typed arenas
// and frame-work side data directly into a full postfilter request.
func BindDecoderFrameWorkPostFilterRequestFromSideData(size DecoderFrameWorkPostFilterScratchSize, side DecoderFrameWorkSideData, scratch DecoderFrameWorkPostFilterRequestScratch) (DecoderFrameWorkPostFilterRequest, error) {
	buffers, err := BindDecoderFrameWorkPostFilterRequestBuffersFromSideData(size, side, scratch)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	return BindDecoderFrameWorkPostFilterRequest(size, buffers)
}

// BindDecoderFrameWorkPostFilterRequest binds all caller-owned postfilter
// scratch into one request. Zero-sized stages accept nil backing slices.
func BindDecoderFrameWorkPostFilterRequest(size DecoderFrameWorkPostFilterScratchSize, buffers DecoderFrameWorkPostFilterRequestBuffers) (DecoderFrameWorkPostFilterRequest, error) {
	loopFilterReq, err := BindDecoderFrameWorkLoopFilterPostFilterRequest(size.LoopFilter, buffers.LoopFilterMap, buffers.LoopFilterEdges)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	cdefReq, err := BindDecoderFrameWorkCDEFPostFilterRequest(size.CDEF, buffers.CDEFIndexMap, buffers.CDEFSampleScratch, buffers.CDEFDstScratch, buffers.CDEFDirectionGrid, buffers.CDEFVarianceGrid, buffers.CDEFInputScratch, buffers.CDEFUnitDstScratch)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	superResReq, err := BindDecoderFrameWorkSuperResPostFilterRequest(size.SuperRes, buffers.SuperResOutputFrame, buffers.SuperResCodedScratch, buffers.SuperResOutputScratch)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	restorationReq, err := BindDecoderFrameWorkRestorationPostFilterRequest(size.Restoration, buffers.RestorationRecords, buffers.RestorationBoundaries, buffers.RestorationDataScratch, buffers.RestorationDstScratch, buffers.RestorationWienerScratch, buffers.RestorationSGRProjScratch, buffers.RestorationBoundaryAboveScratch, buffers.RestorationBoundaryBelowScratch, buffers.RestorationOptimized)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	filmGrainReq, err := BindDecoderFrameWorkFilmGrainPostFilterRequest(size.FilmGrain, buffers.FilmGrainLumaGrain, buffers.FilmGrainChromaGrain, buffers.FilmGrainLumaSamples, buffers.FilmGrainChromaSamples)
	if err != nil {
		return DecoderFrameWorkPostFilterRequest{}, err
	}
	return DecoderFrameWorkPostFilterRequest{
		LoopFilter:  loopFilterReq,
		CDEF:        cdefReq,
		SuperRes:    superResReq,
		Restoration: restorationReq,
		FilmGrain:   filmGrainReq,
	}, nil
}

func BindDecoderFrameWorkLoopFilterPostFilterRequest(size DecoderFrameWorkLoopFilterPostFilterScratchSize, filterMap DecoderFrameWorkLoopFilterMap, edges []DecoderFrameWorkLoopFilterPostFilterEdge) (DecoderFrameWorkLoopFilterPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(edges, size.Edges) {
		return DecoderFrameWorkLoopFilterPostFilterRequest{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges[:size.Edges],
	}, nil
}

func DecoderFrameWorkRestorationFramePlan(sequence SequenceHeader, size FrameSize, restoration RestorationParams) (TileRestorationFramePlan, error) {
	batch := decoderFrameWorkRestorationBatch(sequence, size, restoration)
	return batch.RestorationFramePlan()
}

func BindDecoderFrameWorkRestorationFrameBuffers(sequence SequenceHeader, size FrameSize, restoration RestorationParams, records []TileRestorationUnitRecord, above []uint16, below []uint16) (DecoderFrameWorkRestorationFrameBuffers, error) {
	batch := decoderFrameWorkRestorationBatch(sequence, size, restoration)
	return batch.BindRestorationFrameBuffers(records, above, below)
}

func BindDecoderFrameWorkCDEFPostFilterRequest(size DecoderFrameWorkCDEFPostFilterScratchSize, indexMap DecoderFrameWorkCDEFIndexMap, sampleScratch [3][]uint16, dstScratch [3][]uint16, directionGrid []CDEFDirectionGrid, varianceGrid []CDEFVarianceGrid, inputScratch []uint16, unitDstScratch []uint16) (DecoderFrameWorkCDEFPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(directionGrid, size.DirectionGrid) ||
		decoderFrameWorkPostFilterScratchTooShort(varianceGrid, size.VarianceGrid) ||
		decoderFrameWorkPostFilterScratchTooShort(inputScratch, size.Input) ||
		decoderFrameWorkPostFilterScratchTooShort(unitDstScratch, size.UnitDst) {
		return DecoderFrameWorkCDEFPostFilterRequest{}, ErrFrameShortBuffer
	}
	req := DecoderFrameWorkCDEFPostFilterRequest{
		IndexMap:       indexMap,
		DirectionGrid:  directionGrid[:size.DirectionGrid],
		VarianceGrid:   varianceGrid[:size.VarianceGrid],
		InputScratch:   inputScratch[:size.Input],
		UnitDstScratch: unitDstScratch[:size.UnitDst],
	}
	for plane := 0; plane < 3; plane++ {
		if decoderFrameWorkPostFilterScratchTooShort(sampleScratch[plane], size.Samples[plane]) ||
			decoderFrameWorkPostFilterScratchTooShort(dstScratch[plane], size.Dst[plane]) {
			return DecoderFrameWorkCDEFPostFilterRequest{}, ErrFrameShortBuffer
		}
		req.SampleScratch[plane] = sampleScratch[plane][:size.Samples[plane]]
		req.DstScratch[plane] = dstScratch[plane][:size.Dst[plane]]
	}
	return req, nil
}

func BindDecoderFrameWorkSuperResPostFilterRequest(size DecoderFrameWorkSuperResPostFilterScratchSize, outputFrame []byte, codedScratch [3][]uint16, outputScratch [3][]uint16) (DecoderFrameWorkSuperResPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(outputFrame, size.OutputFrame) {
		return DecoderFrameWorkSuperResPostFilterRequest{}, ErrFrameShortBuffer
	}
	req := DecoderFrameWorkSuperResPostFilterRequest{
		OutputFrame: outputFrame[:size.OutputFrame],
	}
	for plane := 0; plane < 3; plane++ {
		if decoderFrameWorkPostFilterScratchTooShort(codedScratch[plane], size.CodedSamples[plane]) ||
			decoderFrameWorkPostFilterScratchTooShort(outputScratch[plane], size.OutputSamples[plane]) {
			return DecoderFrameWorkSuperResPostFilterRequest{}, ErrFrameShortBuffer
		}
		req.CodedScratch[plane] = codedScratch[plane][:size.CodedSamples[plane]]
		req.OutputScratch[plane] = outputScratch[plane][:size.OutputSamples[plane]]
	}
	return req, nil
}

func BindDecoderFrameWorkRestorationPostFilterRequest(size DecoderFrameWorkRestorationPostFilterScratchSize, records [3][]TileRestorationUnitRecord, boundaries [3]TileRestorationStripeBoundaries, dataScratch []uint16, dstScratch []uint16, wienerScratch []uint16, sgrProjScratch []int32, boundaryAboveScratch []uint16, boundaryBelowScratch []uint16, optimized bool) (DecoderFrameWorkRestorationPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(dataScratch, size.Samples.DataLen) ||
		decoderFrameWorkPostFilterScratchTooShort(dstScratch, size.Samples.DstLen) ||
		decoderFrameWorkPostFilterScratchTooShort(wienerScratch, size.Apply.Unit.Wiener) ||
		decoderFrameWorkPostFilterScratchTooShort(sgrProjScratch, size.Apply.Unit.SGRProj) ||
		decoderFrameWorkPostFilterScratchTooShort(boundaryAboveScratch, size.Apply.Boundary.Above) ||
		decoderFrameWorkPostFilterScratchTooShort(boundaryBelowScratch, size.Apply.Boundary.Below) {
		return DecoderFrameWorkRestorationPostFilterRequest{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkRestorationPostFilterRequest{
		Records:     records,
		Boundaries:  boundaries,
		DataScratch: dataScratch[:size.Samples.DataLen],
		DstScratch:  dstScratch[:size.Samples.DstLen],
		Scratch: TileRestorationUnitRecordBoundaryScratch{
			Unit: TileRestorationUnitScratch{
				Wiener:  wienerScratch[:size.Apply.Unit.Wiener],
				SGRProj: sgrProjScratch[:size.Apply.Unit.SGRProj],
			},
			Boundary: TileRestorationStripeBoundaryScratch{
				Above: boundaryAboveScratch[:size.Apply.Boundary.Above],
				Below: boundaryBelowScratch[:size.Apply.Boundary.Below],
			},
		},
		Optimized: optimized,
	}, nil
}

func BindDecoderFrameWorkFilmGrainPostFilterRequest(size DecoderFrameWorkFilmGrainPostFilterScratchSize, lumaGrain []int16, chromaGrain [2][]int16, lumaSamples []uint16, chromaSamples [2][]uint16) (DecoderFrameWorkFilmGrainPostFilterRequest, error) {
	if decoderFrameWorkPostFilterScratchTooShort(lumaGrain, size.LumaGrain) ||
		decoderFrameWorkPostFilterScratchTooShort(lumaSamples, size.LumaSamples) {
		return DecoderFrameWorkFilmGrainPostFilterRequest{}, ErrFrameShortBuffer
	}
	req := DecoderFrameWorkFilmGrainPostFilterRequest{
		LumaGrain:   lumaGrain[:size.LumaGrain],
		LumaSamples: lumaSamples[:size.LumaSamples],
	}
	for plane := 0; plane < 2; plane++ {
		if decoderFrameWorkPostFilterScratchTooShort(chromaGrain[plane], size.ChromaGrain[plane]) ||
			decoderFrameWorkPostFilterScratchTooShort(chromaSamples[plane], size.ChromaSamples[plane]) {
			return DecoderFrameWorkFilmGrainPostFilterRequest{}, ErrFrameShortBuffer
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
		return nil, nil, ErrFrameShortBuffer
	}
	return scratch[:need], scratch[need:], nil
}

func decoderFrameWorkCDEFIndexBatch(sequence SequenceHeader, size FrameSize, cdef CDEFParams) internalthreading.FrameWorkBatch {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	batch.CDEF = cdef
	return batch
}

func decoderFrameWorkRestorationBatch(sequence SequenceHeader, size FrameSize, restoration RestorationParams) internalthreading.FrameWorkBatch {
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
