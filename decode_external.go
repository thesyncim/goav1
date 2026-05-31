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
	arena := DecoderFrameWorkPostFilterRequestScratchLen(first).Max(
		DecoderFrameWorkPostFilterRequestScratchLen(full))
	if decoderPostFilterScratchTooSmall(r.scratch, arena) {
		r.size = r.size.Max(arena)
		r.scratch = decoderPostFilterScratch(r.size)
	}
	buffers, err := BindDecoderFrameWorkPostFilterRequestBuffersFromScratch(full, side, r.scratch)
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
		len(s.CDEFDirectionGrid) < size.CDEFDirectionGrid ||
		len(s.CDEFVarianceGrid) < size.CDEFVarianceGrid ||
		len(s.ByteScratch) < size.ByteScratch ||
		len(s.Uint16Scratch) < size.Uint16Scratch ||
		len(s.Int16Scratch) < size.Int16Scratch ||
		len(s.Int32Scratch) < size.Int32Scratch
}

func decoderPostFilterScratch(size DecoderFrameWorkPostFilterRequestScratchSize) DecoderFrameWorkPostFilterRequestScratch {
	return DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:   make([]DecoderFrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		CDEFDirectionGrid: make([]CDEFDirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:  make([]CDEFVarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:       make([]byte, size.ByteScratch),
		Uint16Scratch:     make([]uint16, size.Uint16Scratch),
		Int16Scratch:      make([]int16, size.Int16Scratch),
		Int32Scratch:      make([]int32, size.Int32Scratch),
	}
}
