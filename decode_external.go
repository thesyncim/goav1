package goav1

const decoderExternalOutputSurfaceBase = 256

func decoderExternalOutputFormat(payloads [][]byte, align int) (bool, FrameFormat, error) {
	var stream DecoderStream
	var format FrameFormat
	have := false
	events := make([]DecoderEvent, 0)
	for _, payload := range payloads {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return false, FrameFormat{}, err
		}
		if len(events) < eventCount {
			events = make([]DecoderEvent, eventCount)
		}
		sequence, _ := stream.SequenceHeader()
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return false, FrameFormat{}, err
		}
		for i := range count {
			event := events[i]
			sequence = decoderFrameWorkResidualEventSequence(sequence, event)
			if !decoderFrameWorkResidualEventBindCandidate(event) ||
				!event.FrameSize.SuperResEnabled ||
				event.FrameSize.UpscaledWidth <= event.FrameSize.CodedWidth {
				continue
			}
			next, err := FrameOutputFormatFromHeaders(sequence, event.FrameSize, align)
			if err != nil {
				return false, FrameFormat{}, err
			}
			if have && next != format {
				return false, FrameFormat{}, ErrFrameInvalidFormat
			}
			format = next
			have = true
		}
	}
	return have, format, nil
}

func decoderPostFilterScratchArenaUpperBound(payloads [][]byte, align int, events []DecoderEvent) (DecoderFrameWorkPostFilterRequestScratchSize, error) {
	var stream DecoderStream
	var arena DecoderFrameWorkPostFilterRequestScratchSize
	for _, payload := range payloads {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return DecoderFrameWorkPostFilterRequestScratchSize{}, err
		}
		if len(events) < eventCount {
			return DecoderFrameWorkPostFilterRequestScratchSize{}, ErrDecoderEventBufferTooSmall
		}
		sequence, _ := stream.SequenceHeader()
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return DecoderFrameWorkPostFilterRequestScratchSize{}, err
		}
		for i := range count {
			event := events[i]
			sequence = decoderFrameWorkResidualEventSequence(sequence, event)
			if !decoderFrameWorkResidualEventBindCandidate(event) {
				continue
			}
			size, err := decoderPostFilterScratchSizeUpperBound(sequence, event, align)
			if err != nil {
				return DecoderFrameWorkPostFilterRequestScratchSize{}, err
			}
			arena = arena.Max(decoderExternalPostFilterScratchLen(size))
		}
	}
	return arena, nil
}

func decoderTemporalMotionScratchFrameSizeUpperBound(payloads [][]byte, events []DecoderEvent) (FrameSize, error) {
	var stream DecoderStream
	var maxSize FrameSize
	for _, payload := range payloads {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return FrameSize{}, err
		}
		if len(events) < eventCount {
			return FrameSize{}, ErrDecoderEventBufferTooSmall
		}
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return FrameSize{}, err
		}
		for i := range count {
			event := events[i]
			if !decoderFrameWorkResidualEventBindCandidate(event) ||
				event.FrameHeader.ShowExistingFrame {
				continue
			}
			if event.FrameSize.CodedWidth > maxSize.CodedWidth {
				maxSize.CodedWidth = event.FrameSize.CodedWidth
			}
			if event.FrameSize.Height > maxSize.Height {
				maxSize.Height = event.FrameSize.Height
			}
		}
	}
	return maxSize, nil
}

func decoderPostFilterScratchSizeUpperBound(sequence SequenceHeader, event DecoderEvent, align int) (DecoderFrameWorkPostFilterScratchSize, error) {
	var output Frame
	ctx, err := DecoderFrameWorkPostFilterScratchContext(sequence, event, align, nil, &output)
	if err != nil {
		return DecoderFrameWorkPostFilterScratchSize{}, err
	}
	remaining := ctx.RemainingPostFilters()
	var size DecoderFrameWorkPostFilterScratchSize
	if remaining.Has(DecoderFrameWorkPostFilterLoopFilter) {
		edges, err := decoderLoopFilterEdgeScratchUpperBound(sequence, event.FrameSize)
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		size.LoopFilter.Edges = edges
		size.LoopFilter.Schedule = edges
	}
	if remaining.Has(DecoderFrameWorkPostFilterCDEF) {
		cdefSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		size.CDEF = cdefSize
	}

	tailCtx := ctx.WithCompletedPostFilters(DecoderFrameWorkPostFilterLoopFilter | DecoderFrameWorkPostFilterCDEF)
	if tailCtx.RemainingPostFilters().Has(DecoderFrameWorkPostFilterSuperRes) {
		superPlan, err := tailCtx.SuperResPostFilterPlan()
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		superSize, err := tailCtx.SuperResPostFilterScratchLen()
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		size.SuperRes = superSize
		outputLayout, err := FrameRequiredSize(superPlan.OutputFormat)
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		superOutput := decoderFrameWorkPostFilterScratchFrame(superPlan.OutputFormat, outputLayout)
		tailCtx.Output = &superOutput
		tailCtx = tailCtx.WithCompletedPostFilters(DecoderFrameWorkPostFilterSuperRes)
	}
	if tailCtx.RemainingPostFilters().Has(DecoderFrameWorkPostFilterLoopRestoration) {
		restorationSize, err := decoderLoopRestorationScratchUpperBound(tailCtx)
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		size.Restoration = restorationSize
	}
	if tailCtx.RemainingPostFilters().Has(DecoderFrameWorkPostFilterFilmGrain) {
		filmGrainSize, err := tailCtx.FilmGrainPostFilterScratchLen()
		if err != nil {
			return DecoderFrameWorkPostFilterScratchSize{}, err
		}
		size.FilmGrain = filmGrainSize
	}
	return size, nil
}

func decoderLoopFilterEdgeScratchUpperBound(sequence SequenceHeader, size FrameSize) (int, error) {
	cols, rows, _, err := DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		return 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if cols > maxInt/rows || cols*rows > maxInt/8 {
		return 0, ErrFrameInvalidFormat
	}
	return cols * rows * 8, nil
}

func decoderLoopRestorationScratchUpperBound(ctx DecoderFrameWorkPostFilterContext) (DecoderFrameWorkRestorationPostFilterScratchSize, error) {
	plan, err := DecoderFrameWorkRestorationFramePlan(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Event.Restoration)
	if err != nil {
		return DecoderFrameWorkRestorationPostFilterScratchSize{}, err
	}
	if !plan.Active {
		return DecoderFrameWorkRestorationPostFilterScratchSize{}, nil
	}
	var out DecoderFrameWorkRestorationPostFilterScratchSize
	for _, typ := range [...]RestorationType{RestorationWiener, RestorationSGRProj} {
		records, err := decoderRestorationWorstCaseRecords(plan, typ)
		if err != nil {
			return DecoderFrameWorkRestorationPostFilterScratchSize{}, err
		}
		next, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
		if err != nil {
			return DecoderFrameWorkRestorationPostFilterScratchSize{}, err
		}
		out = out.Max(next)
	}
	return out, nil
}

func decoderRestorationWorstCaseRecords(plan TileRestorationFramePlan, typ RestorationType) ([3][]TileRestorationUnitRecord, error) {
	backing := make([]TileRestorationUnitRecord, plan.UnitRecordLen())
	records, err := BindTileRestorationFrameRecordBuffers(plan, backing)
	if err != nil {
		return [3][]TileRestorationUnitRecord{}, err
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == RestorationNone {
			continue
		}
		if err := ResetTileRestorationPlaneRecords(grid, records[plane]); err != nil {
			return [3][]TileRestorationUnitRecord{}, err
		}
		unitType := grid.Type
		if grid.Type == RestorationSwitchable {
			unitType = typ
		}
		for i := range records[plane] {
			records[plane][i].Unit.Type = unitType
		}
	}
	return records, nil
}

type decoderExternalSurfaceProvider struct {
	coded  *FramePool
	output *FramePool
}

func (p decoderExternalSurfaceProvider) FrameSurface(id int) (*Frame, error) {
	pool, local, err := p.resolve(id)
	if err != nil {
		return nil, err
	}
	return pool.Frame(local)
}

func (p decoderExternalSurfaceProvider) ReleaseFrameSurfaces(ids []int) error {
	for i, id := range ids {
		for j := range i {
			if ids[j] == id {
				return ErrFrameInvalidSlot
			}
		}
		pool, local, err := p.resolve(id)
		if err != nil {
			return err
		}
		if _, err := pool.Frame(local); err != nil {
			return err
		}
	}
	for _, id := range ids {
		pool, local, err := p.resolve(id)
		if err != nil {
			return err
		}
		if err := pool.Release(local); err != nil {
			return err
		}
	}
	return nil
}

func (p decoderExternalSurfaceProvider) resolve(id int) (*FramePool, int, error) {
	if id < 0 {
		return nil, -1, ErrDecoderInvalidSurfaceReference
	}
	if id >= decoderExternalOutputSurfaceBase {
		if p.output == nil {
			return nil, -1, ErrDecoderInvalidSurfaceReference
		}
		return p.output, id - decoderExternalOutputSurfaceBase, nil
	}
	if p.coded == nil {
		return nil, -1, ErrDecoderInvalidSurfaceReference
	}
	return p.coded, id, nil
}

type decoderExternalPostFilterRunner struct {
	supported  DecoderFrameWorkReusableSupportedPostFilterRunner
	outputPool *FramePool
	scratch    DecoderFrameWorkPostFilterRequestScratch
	size       DecoderFrameWorkPostFilterRequestScratchSize
	output     *Frame
	published  int
}

func (r *decoderExternalPostFilterRunner) Apply(ctx DecoderFrameWorkPostFilterContext) error {
	if r == nil {
		return ErrDecoderInvalidFrameWorkState
	}
	r.output = nil
	r.published = -1
	if !ctx.ActivePostFilters().Has(DecoderFrameWorkPostFilterSuperRes) {
		if err := r.supported.Apply(ctx); err != nil {
			return err
		}
		if out, ok := r.supported.PostFilterOutput(); ok {
			r.output = out
		}
		return nil
	}
	if r.outputPool == nil {
		return ErrFrameInvalidPool
	}
	format, err := FrameOutputFormatFromHeaders(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Output.Format.Align)
	if err != nil {
		return err
	}
	local, out, err := r.outputPool.AcquireFormat(format)
	if err != nil {
		return err
	}
	published := false
	defer func() {
		if !published {
			_ = r.outputPool.Release(local)
		}
	}()
	backing, err := decoderFrameBacking(out)
	if err != nil {
		return err
	}
	side := DecoderFrameWorkPostFilterRequestSideDataFromContext(ctx)
	first, err := ctx.CallerPostFilterScratchLen(DecoderFrameWorkPostFilterRequest{
		LoopFilter:  DecoderFrameWorkLoopFilterPostFilterRequest{Map: side.LoopFilterMap},
		CDEF:        DecoderFrameWorkCDEFPostFilterRequest{IndexMap: side.CDEFIndexMap},
		Restoration: DecoderFrameWorkRestorationPostFilterRequest{Records: side.RestorationRecords},
	})
	if err != nil {
		return err
	}
	probe := DecoderFrameWorkPostFilterRequest{
		LoopFilter:  DecoderFrameWorkLoopFilterPostFilterRequest{Map: side.LoopFilterMap},
		CDEF:        DecoderFrameWorkCDEFPostFilterRequest{IndexMap: side.CDEFIndexMap},
		Restoration: DecoderFrameWorkRestorationPostFilterRequest{Records: side.RestorationRecords},
		SuperRes: DecoderFrameWorkSuperResPostFilterRequest{
			OutputFrame: backing,
			OutputView:  out,
		},
	}
	full, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		return err
	}
	arena := decoderExternalPostFilterScratchLen(first).Max(
		decoderExternalPostFilterScratchLen(full))
	if decoderPostFilterScratchTooSmall(r.scratch, arena) {
		return ErrFrameShortBuffer
	}
	bindSize := decoderExternalPostFilterBindScratchSize(full)
	buffers, err := BindDecoderFrameWorkPostFilterRequestBuffersFromScratch(bindSize, side, r.scratch)
	if err != nil {
		return err
	}
	buffers.SuperResOutputFrame = backing
	req, err := BindDecoderFrameWorkPostFilterRequest(full, buffers)
	if err != nil {
		return err
	}
	req.SuperRes.OutputView = out
	next, _, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		return err
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		return err
	}
	r.output = next.Output
	r.published = decoderExternalOutputSurfaceBase + local
	published = true
	return nil
}

func (r *decoderExternalPostFilterRunner) PostFilterOutput() (*Frame, bool) {
	if r == nil || r.output == nil {
		return nil, false
	}
	return r.output, true
}

func (r *decoderExternalPostFilterRunner) PublishedFrameWorkGlobalSurface() (int, bool) {
	if r == nil || r.published < 0 {
		return -1, false
	}
	return r.published, true
}

func decoderExternalPostFilterScratchLen(size DecoderFrameWorkPostFilterScratchSize) DecoderFrameWorkPostFilterRequestScratchSize {
	return DecoderFrameWorkPostFilterRequestScratchLen(decoderExternalPostFilterBindScratchSize(size))
}

func decoderExternalPostFilterBindScratchSize(size DecoderFrameWorkPostFilterScratchSize) DecoderFrameWorkPostFilterScratchSize {
	size.SuperRes.OutputFrame = 0
	return size
}

func decoderFrameBacking(f *Frame) ([]byte, error) {
	if f == nil {
		return nil, ErrFrameInvalidSlot
	}
	if f.Layout.Size < 0 || cap(f.Y.Pix) < f.Layout.Size {
		return nil, ErrFrameShortBuffer
	}
	return f.Y.Pix[:f.Layout.Size], nil
}

func decoderPostFilterScratchTooSmall(s DecoderFrameWorkPostFilterRequestScratch, size DecoderFrameWorkPostFilterRequestScratchSize) bool {
	return len(s.LoopFilterEdges) < size.LoopFilterEdges ||
		len(s.LoopFilterSchedule) < size.LoopFilterSchedule ||
		len(s.CDEFDirectionGrid) < size.CDEFDirectionGrid ||
		len(s.CDEFVarianceGrid) < size.CDEFVarianceGrid ||
		len(s.ByteScratch) < size.ByteScratch ||
		len(s.Uint16Scratch) < size.Uint16Scratch ||
		len(s.Int16Scratch) < size.Int16Scratch ||
		len(s.Int32Scratch) < size.Int32Scratch
}

func decoderPostFilterScratch(size DecoderFrameWorkPostFilterRequestScratchSize) DecoderFrameWorkPostFilterRequestScratch {
	return DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]DecoderFrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, size.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]CDEFDirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]CDEFVarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:        make([]byte, size.ByteScratch),
		Uint16Scratch:      make([]uint16, size.Uint16Scratch),
		Int16Scratch:       make([]int16, size.Int16Scratch),
		Int32Scratch:       make([]int32, size.Int32Scratch),
	}
}
