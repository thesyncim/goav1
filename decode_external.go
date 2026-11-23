package goav1

import (
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
)

const decoderExternalOutputSurfaceBase = 256

// decoderUpperBoundScratch is reused across the per-frame upper-bound probes of a
// stream so sizing every frame of a long clip does not allocate per frame.
type decoderUpperBoundScratch struct {
	restorationRecords []internaltile.RestorationUnitRecord
}

func decoderPostFilterScratchSizeUpperBound(sequence SequenceHeader, event *internaldecoder.Event, align int, workerPool *internalthreading.Pool, scratch *decoderUpperBoundScratch) (internaldecoder.FrameWorkPostFilterScratchSize, error) {
	var output Frame
	ctx, err := decoderFrameWorkPostFilterScratchContext(sequence, event, align, nil, &output)
	if err != nil {
		return internaldecoder.FrameWorkPostFilterScratchSize{}, err
	}
	// Size for the pooled apply: the frame-work runner hands the post-filter
	// the tile worker pool, whose lanes each take their own band scratch.
	internaldecoder.BindFrameWorkPostFilterPool(&ctx, workerPool)
	remaining := ctx.RemainingPostFilters()
	var size internaldecoder.FrameWorkPostFilterScratchSize
	if remaining.Has(internaldecoder.FrameWorkPostFilterLoopFilter) {
		edges, err := decoderLoopFilterEdgeScratchUpperBound(sequence, event.FrameSize)
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.LoopFilter.Edges = edges
		size.LoopFilter.Schedule = edges
	}
	if remaining.Has(internaldecoder.FrameWorkPostFilterCDEF) {
		cdefSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.CDEF = cdefSize
	}

	tailCtx := ctx.WithCompletedPostFilters(internaldecoder.FrameWorkPostFilterLoopFilter | internaldecoder.FrameWorkPostFilterCDEF)
	if tailCtx.RemainingPostFilters().Has(internaldecoder.FrameWorkPostFilterSuperRes) {
		superPlan, err := tailCtx.SuperResPostFilterPlan()
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		superSize, err := tailCtx.SuperResPostFilterScratchLen()
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.SuperRes = superSize
		outputLayout, err := internalframe.RequiredSize(superPlan.OutputFormat)
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		superOutput := decoderFrameWorkPostFilterScratchFrame(superPlan.OutputFormat, outputLayout)
		tailCtx.Output = &superOutput
		tailCtx = tailCtx.WithCompletedPostFilters(internaldecoder.FrameWorkPostFilterSuperRes)
	}
	if tailCtx.RemainingPostFilters().Has(internaldecoder.FrameWorkPostFilterLoopRestoration) {
		restorationSize, err := decoderLoopRestorationScratchUpperBound(tailCtx, scratch)
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.Restoration = restorationSize
	}
	if tailCtx.RemainingPostFilters().Has(internaldecoder.FrameWorkPostFilterFilmGrain) {
		filmGrainSize, err := tailCtx.FilmGrainPostFilterScratchLen()
		if err != nil {
			return internaldecoder.FrameWorkPostFilterScratchSize{}, err
		}
		size.FilmGrain = filmGrainSize
	}
	return size, nil
}

func decoderLoopFilterEdgeScratchUpperBound(sequence SequenceHeader, size FrameSize) (int, error) {
	cols, rows, _, err := decoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		return 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if cols > maxInt/rows || cols*rows > maxInt/8 {
		return 0, internalframe.ErrInvalidFormat
	}
	return cols * rows * 8, nil
}

func decoderLoopRestorationScratchUpperBound(ctx internaldecoder.FrameWorkPostFilterContext, scratch *decoderUpperBoundScratch) (internaldecoder.FrameWorkRestorationPostFilterScratchSize, error) {
	plan, err := decoderFrameWorkRestorationFramePlan(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Event.Restoration)
	if err != nil {
		return internaldecoder.FrameWorkRestorationPostFilterScratchSize{}, err
	}
	if !plan.Active {
		return internaldecoder.FrameWorkRestorationPostFilterScratchSize{}, nil
	}
	var out internaldecoder.FrameWorkRestorationPostFilterScratchSize
	for _, typ := range [...]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj} {
		records, err := decoderRestorationWorstCaseRecords(plan, typ, scratch)
		if err != nil {
			return internaldecoder.FrameWorkRestorationPostFilterScratchSize{}, err
		}
		next, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
		if err != nil {
			return internaldecoder.FrameWorkRestorationPostFilterScratchSize{}, err
		}
		out = out.Max(next)
	}
	return out, nil
}

func decoderRestorationWorstCaseRecords(plan internaltile.RestorationFramePlan, typ internalparser.RestorationType, scratch *decoderUpperBoundScratch) ([3][]internaltile.RestorationUnitRecord, error) {
	n := plan.UnitRecordLen()
	if cap(scratch.restorationRecords) < n {
		scratch.restorationRecords = make([]internaltile.RestorationUnitRecord, n)
	}
	backing := scratch.restorationRecords[:n]
	clear(backing)
	records, err := internaltile.BindRestorationFrameRecordBuffers(plan, backing)
	if err != nil {
		return [3][]internaltile.RestorationUnitRecord{}, err
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == internalparser.RestorationNone {
			continue
		}
		if err := internaltile.ResetRestorationPlaneRecords(grid, records[plane]); err != nil {
			return [3][]internaltile.RestorationUnitRecord{}, err
		}
		unitType := grid.Type
		if grid.Type == internalparser.RestorationSwitchable {
			unitType = typ
		}
		for i := range records[plane] {
			records[plane][i].Unit.Type = unitType
		}
	}
	return records, nil
}

type decoderExternalSurfaceProvider struct {
	coded  *internalframe.Pool
	output *internalframe.Pool
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
				return internalframe.ErrInvalidSlot
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

func (p decoderExternalSurfaceProvider) resolve(id int) (*internalframe.Pool, int, error) {
	if id < 0 {
		return nil, -1, internaldecoder.ErrInvalidSurfaceReference
	}
	if id >= decoderExternalOutputSurfaceBase {
		if p.output == nil {
			return nil, -1, internaldecoder.ErrInvalidSurfaceReference
		}
		return p.output, id - decoderExternalOutputSurfaceBase, nil
	}
	if p.coded == nil {
		return nil, -1, internaldecoder.ErrInvalidSurfaceReference
	}
	return p.coded, id, nil
}

type decoderExternalPostFilterRunner struct {
	supported  decoderFrameWorkReusableSupportedPostFilterRunner
	outputPool *internalframe.Pool
	scratch    decoderFrameWorkPostFilterRequestScratch
	size       decoderFrameWorkPostFilterRequestScratchSize
	output     *Frame
	published  int
}

func (r *decoderExternalPostFilterRunner) Apply(ctx internaldecoder.FrameWorkPostFilterContext) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	r.output = nil
	r.published = -1
	if !ctx.ActivePostFilters().Has(internaldecoder.FrameWorkPostFilterSuperRes) {
		if err := r.supported.Apply(ctx); err != nil {
			return err
		}
		if out, ok := r.supported.PostFilterOutput(); ok {
			r.output = out
		}
		return nil
	}
	if r.outputPool == nil {
		return internalframe.ErrInvalidPool
	}
	format, err := frameOutputFormatFromHeaders(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Output.Format.Align)
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
	side := decoderFrameWorkPostFilterRequestSideDataFromContext(ctx)
	first, err := ctx.CallerPostFilterScratchLen(internaldecoder.FrameWorkPostFilterRequest{
		LoopFilter:  internaldecoder.FrameWorkLoopFilterPostFilterRequest{Map: side.LoopFilterMap},
		CDEF:        internaldecoder.FrameWorkCDEFPostFilterRequest{IndexMap: side.CDEFIndexMap},
		Restoration: internaldecoder.FrameWorkRestorationPostFilterRequest{Records: side.RestorationRecords},
	})
	if err != nil {
		return err
	}
	probe := internaldecoder.FrameWorkPostFilterRequest{
		LoopFilter:  internaldecoder.FrameWorkLoopFilterPostFilterRequest{Map: side.LoopFilterMap},
		CDEF:        internaldecoder.FrameWorkCDEFPostFilterRequest{IndexMap: side.CDEFIndexMap},
		Restoration: internaldecoder.FrameWorkRestorationPostFilterRequest{Records: side.RestorationRecords},
		SuperRes: internaldecoder.FrameWorkSuperResPostFilterRequest{
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
		return internalframe.ErrShortBuffer
	}
	bindSize := decoderExternalPostFilterBindScratchSize(full)
	buffers, err := bindDecoderFrameWorkPostFilterRequestBuffersFromScratch(bindSize, side, r.scratch)
	if err != nil {
		return err
	}
	buffers.SuperResOutputFrame = backing
	req, err := bindDecoderFrameWorkPostFilterRequest(full, buffers)
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

func decoderExternalPostFilterScratchLen(size internaldecoder.FrameWorkPostFilterScratchSize) decoderFrameWorkPostFilterRequestScratchSize {
	return decoderFrameWorkPostFilterRequestScratchLen(decoderExternalPostFilterBindScratchSize(size))
}

func decoderExternalPostFilterBindScratchSize(size internaldecoder.FrameWorkPostFilterScratchSize) internaldecoder.FrameWorkPostFilterScratchSize {
	size.SuperRes.OutputFrame = 0
	return size
}

func decoderFrameBacking(f *Frame) ([]byte, error) {
	if f == nil {
		return nil, internalframe.ErrInvalidSlot
	}
	if f.Layout.Size < 0 || cap(f.Y.Pix) < f.Layout.Size {
		return nil, internalframe.ErrShortBuffer
	}
	return f.Y.Pix[:f.Layout.Size], nil
}

func decoderPostFilterScratchTooSmall(s decoderFrameWorkPostFilterRequestScratch, size decoderFrameWorkPostFilterRequestScratchSize) bool {
	return len(s.LoopFilterEdges) < size.LoopFilterEdges ||
		len(s.LoopFilterSchedule) < size.LoopFilterSchedule ||
		len(s.CDEFDirectionGrid) < size.CDEFDirectionGrid ||
		len(s.CDEFVarianceGrid) < size.CDEFVarianceGrid ||
		len(s.ByteScratch) < size.ByteScratch ||
		len(s.Uint16Scratch) < size.Uint16Scratch ||
		len(s.Int16Scratch) < size.Int16Scratch ||
		len(s.Int32Scratch) < size.Int32Scratch
}
