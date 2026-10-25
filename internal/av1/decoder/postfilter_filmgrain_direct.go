package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// frameWorkApplyFilmGrainForPublication writes grain from the reconstructed
// source directly to a separate display surface, as dav1d's fg_apply_tmpl.c
// does. The source remains the reference picture, including the original luma
// used by both chroma planes. Public in-place operations, caller aliases,
// unusual geometry, and (on big-endian hosts) high bit depth keep the staged
// path.
func frameWorkApplyFilmGrainForPublication(ctx *FrameWorkPostFilterContext, req FrameWorkFilmGrainPostFilterRequest) (*frame.Frame, FrameWorkFilmGrainPostFilterResult, error) {
	if ctx.Output != nil && req.OutputView != nil && req.OutputView != ctx.Output && len(req.OutputFrame) >= ctx.Output.Layout.Size {
		plan, planErr := ctx.FilmGrainPostFilterPlan()
		output, bindErr := frame.Bind(req.OutputFrame[:ctx.Output.Layout.Size], ctx.Output.Format)
		if planErr == nil && plan.Active && !frameWorkFilmGrainNoOp(plan.Params) && bindErr == nil &&
			frameWorkFilmGrainDirectCompatible(&output, ctx.Output, plan) {
			result, ok, applyErr := frameWorkApplyFilmGrainDirect(ctx, &output, req, plan)
			if ok && applyErr == nil {
				*req.OutputView = output
				result.Output = req.OutputView
				return req.OutputView, result, nil
			}
			// Keep legacy error ordering and output-copy behavior for inputs
			// rejected by the existing grain generators or row validators.
		}
	}

	output, err := ctx.BindFilmGrainPostFilterOutput(req)
	if err != nil {
		return nil, FrameWorkFilmGrainPostFilterResult{}, err
	}
	displayCtx := *ctx
	if output != nil {
		displayCtx.Output = output
		displayCtx.detachedPostFilterOutput = true
	}
	result, err := displayCtx.ApplyFilmGrainPostFilter(req)
	return output, result, err
}

func frameWorkFilmGrainDirectCompatible(dst, src *frame.Frame, plan FrameWorkFilmGrainPostFilterPlan) bool {
	if dst.Format != src.Format || dst.Layout != src.Layout {
		return false
	}
	switch src.Layout.BytesPerSample {
	case 1:
		if src.Format.BitDepth != 8 {
			return false
		}
	case 2:
		if src.Format.BitDepth != 10 && src.Format.BitDepth != 12 {
			return false
		}
	default:
		return false
	}
	// The staged API's odd-width subsampled chroma reads the caller's luma
	// scratch padding. Preserve that contract rather than substituting the
	// reconstructed picture's extended right border.
	if plan.Format.SubsamplingX && src.Y.Width&1 != 0 && (plan.Planes[1].Active || plan.Planes[2].Active) {
		return false
	}
	dstPlanes := [3]frame.Plane{dst.Y, dst.U, dst.V}
	srcPlanes := [3]frame.Plane{src.Y, src.U, src.V}
	for i, s := range srcPlanes {
		d := dstPlanes[i]
		if d.Width != s.Width || d.Height != s.Height || d.Stride != s.Stride || len(d.Pix) != len(s.Pix) {
			return false
		}
		for _, other := range dstPlanes {
			if frameWorkFilmGrainBytesOverlap(s.Pix, other.Pix) {
				return false
			}
		}
	}
	return true
}

// Addresses are used only for a conservative overlap comparison; no pointer
// is reconstructed and no memory is accessed outside either live slice.
func frameWorkFilmGrainBytesOverlap(a, b []byte) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	ap := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bp := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if ap <= bp {
		return bp-ap < uintptr(len(a))
	}
	return ap-bp < uintptr(len(b))
}

// frameWorkFilmGrainDirectPlanes are one picture's planes as sample slices,
// with strides and widths in samples.
type frameWorkFilmGrainDirectPlanes[T uint8 | uint16] struct {
	pix    [3][]T
	stride [3]int
	width  [3]int
	height [3]int
}

// frameWorkFilmGrainRows is one picture's grain application split as dav1d's
// delayed film grain task (thread_task.c delayed_fg_task): prep runs once
// (dav1d_prep_grain: scaling tables, grain templates, the planes without
// grain copied), then every 32-luma-row stripe (dav1d_apply_grain_row: the
// luma rows and the chroma rows they cover) is independent, so stripes run
// across workers in any order.
type frameWorkFilmGrainRows[T uint8 | uint16] struct {
	plan        FrameWorkFilmGrainPostFilterPlan
	src, dst    frameWorkFilmGrainDirectPlanes[T]
	luts        FrameWorkFilmGrainPostFilterScalingLUTs
	lumaGrain   []int16
	chromaGrain [2][]int16
	rows        [3]int
	stripes     int
	// tables, for 10/12-bit planes, hold each plane's per-sample-value
	// scaling (dav1d's 16bpc generate_scaling); unused at 8 bits.
	tables [3][filmgrain.ScalingTableSize]uint8
}

// frameWorkApplyFilmGrainDirect binds the source and display planes at their
// sample width and runs the row kernels from one to the other. ok is false
// when the planes have no direct sample view (high bit depth on a big-endian
// host); the caller then takes the staged path.
func frameWorkApplyFilmGrainDirect(ctx *FrameWorkPostFilterContext, output *frame.Frame, req FrameWorkFilmGrainPostFilterRequest, plan FrameWorkFilmGrainPostFilterPlan) (FrameWorkFilmGrainPostFilterResult, bool, error) {
	srcPlanes := [3]frame.Plane{ctx.Output.Y, ctx.Output.U, ctx.Output.V}
	dstPlanes := [3]frame.Plane{output.Y, output.U, output.V}
	par := ctx.Parallel
	if par.workers() <= 1 {
		par = nil
	}
	if ctx.Output.Layout.BytesPerSample == 1 {
		var src, dst frameWorkFilmGrainDirectPlanes[uint8]
		for i := range srcPlanes {
			src.pix[i], dst.pix[i] = srcPlanes[i].Pix, dstPlanes[i].Pix
			src.stride[i], src.width[i], src.height[i] = srcPlanes[i].Stride, srcPlanes[i].Width, srcPlanes[i].Height
		}
		dst.stride, dst.width, dst.height = src.stride, src.width, src.height
		if par != nil {
			result, err := frameWorkApplyFilmGrainStripes(ctx, output, req, plan, &par.fgU8, src, dst, par, frameWorkPostFilterStageFilmGrainU8)
			return result, true, err
		}
		var rows frameWorkFilmGrainRows[uint8]
		result, err := frameWorkApplyFilmGrainStripes(ctx, output, req, plan, &rows, src, dst, nil, frameWorkPostFilterStageNone)
		return result, true, err
	}
	var src, dst frameWorkFilmGrainDirectPlanes[uint16]
	for i := range srcPlanes {
		s, ok := frameWorkFilmGrainPlaneU16(srcPlanes[i])
		if !ok {
			return FrameWorkFilmGrainPostFilterResult{}, false, nil
		}
		d, ok := frameWorkFilmGrainPlaneU16(dstPlanes[i])
		if !ok {
			return FrameWorkFilmGrainPostFilterResult{}, false, nil
		}
		src.pix[i], dst.pix[i] = s, d
		src.stride[i], src.width[i], src.height[i] = srcPlanes[i].Stride/2, srcPlanes[i].Width, srcPlanes[i].Height
	}
	dst.stride, dst.width, dst.height = src.stride, src.width, src.height
	if par != nil {
		result, err := frameWorkApplyFilmGrainStripes(ctx, output, req, plan, &par.fgU16, src, dst, par, frameWorkPostFilterStageFilmGrainU16)
		return result, true, err
	}
	var rows frameWorkFilmGrainRows[uint16]
	result, err := frameWorkApplyFilmGrainStripes(ctx, output, req, plan, &rows, src, dst, nil, frameWorkPostFilterStageNone)
	return result, true, err
}

// frameWorkApplyFilmGrainStripes preps rows and applies its stripes, across
// par's workers when par is non-nil (rows is then par's stage state, which
// RunRange reads) or on the calling goroutine.
func frameWorkApplyFilmGrainStripes[T uint8 | uint16](ctx *FrameWorkPostFilterContext, output *frame.Frame, req FrameWorkFilmGrainPostFilterRequest, plan FrameWorkFilmGrainPostFilterPlan, rows *frameWorkFilmGrainRows[T], src, dst frameWorkFilmGrainDirectPlanes[T], par *FrameWorkPostFilterParallel, stage frameWorkPostFilterParallelStage) (FrameWorkFilmGrainPostFilterResult, error) {
	rows.src, rows.dst = src, dst
	result, err := rows.prep(ctx, output, req, plan)
	if err == nil {
		if par == nil {
			err = rows.apply(0, rows.stripes)
		} else {
			var pool *threading.Pool
			var workers int
			pool, workers, err = par.workerPool(ctx, par.workers())
			if err == nil {
				err = par.runRanges(pool, frameWorkPostFilterParallelJob{stage: stage}, rows.stripes, workers)
			}
		}
	}
	// Drop the frame's planes and grain so the stage state retains nothing.
	rows.src, rows.dst = frameWorkFilmGrainDirectPlanes[T]{}, frameWorkFilmGrainDirectPlanes[T]{}
	rows.lumaGrain, rows.chromaGrain = nil, [2][]int16{}
	if err != nil {
		return FrameWorkFilmGrainPostFilterResult{}, err
	}
	return result, nil
}

// prep is dav1d_prep_grain: it builds the scaling lookups and the grain
// templates, copies the planes that get no grain and the row padding of the
// ones that do, and counts each plane's block rows.
func (r *frameWorkFilmGrainRows[T]) prep(ctx *FrameWorkPostFilterContext, output *frame.Frame, req FrameWorkFilmGrainPostFilterRequest, plan FrameWorkFilmGrainPostFilterPlan) (FrameWorkFilmGrainPostFilterResult, error) {
	r.plan = plan
	r.stripes = 0
	r.rows = [3]int{}
	var err error
	r.luts, err = ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		return FrameWorkFilmGrainPostFilterResult{}, err
	}
	var lumaGrain FrameWorkFilmGrainPostFilterLumaGrain
	if plan.Planes[0].Active {
		lumaGrain, err = ctx.GenerateFilmGrainLumaGrain(req.LumaGrain)
		if err != nil {
			return FrameWorkFilmGrainPostFilterResult{}, err
		}
	}
	r.lumaGrain = lumaGrain.Grain
	for plane := filmgrain.ChromaPlaneCb; plane <= filmgrain.ChromaPlaneCr; plane++ {
		r.chromaGrain[plane-1] = nil
		if plan.Planes[plane].Active {
			chromaGrain, err := ctx.GenerateFilmGrainChromaGrain(req.ChromaGrain[plane-1], lumaGrain.Grain, plane)
			if err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, err
			}
			r.chromaGrain[plane-1] = chromaGrain.Grain
		}
	}

	_, highBitDepth := any(r.src.pix[0]).([]uint16)
	result := FrameWorkFilmGrainPostFilterResult{Plan: plan, OutputSize: output.Layout.Size}
	for plane := range r.src.pix {
		s, d := r.src.pix[plane], r.dst.pix[plane]
		stride, width, height := r.src.stride[plane], r.src.width[plane], r.src.height[plane]
		if !plan.Planes[plane].Active {
			copy(d, s)
			continue
		}
		// StoreSamplePlane historically changed visible samples only. Copy
		// the extended row/bottom padding without pre-copying active pixels.
		for y := 0; y < height; y++ {
			start := y*stride + width
			end := (y + 1) * stride
			copy(d[start:end], s[start:end])
		}
		copy(d[height*stride:], s[height*stride:])
		if highBitDepth {
			if err := filmgrain.BuildScalingTable(&r.tables[plane], r.luts.LUTs[plane][:], plan.BitDepth); err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, frame.ErrInvalidFormat
			}
		}
		blockHeight := filmgrain.LumaBlockSize
		if plane != 0 {
			blockHeight = frameWorkFilmGrainChromaBlockHeight(plan.Format)
		}
		rows := (height + blockHeight - 1) / blockHeight
		rowCount, ok := frameWorkFilmGrainRows16(rows)
		if !ok {
			return FrameWorkFilmGrainPostFilterResult{}, frame.ErrInvalidFormat
		}
		r.rows[plane] = rows
		r.stripes = max(r.stripes, rows)
		if plane == 0 {
			result.LumaRows = rowCount
		} else {
			result.ChromaRows[plane-1] = rowCount
		}
	}
	return result, nil
}

// apply is dav1d_apply_grain_row for the stripes [lo, hi).
func (r *frameWorkFilmGrainRows[T]) apply(lo, hi int) error {
	_, highBitDepth := any(r.src.pix[0]).([]uint16)
	for row := lo; row < hi; row++ {
		for plane := range r.src.pix {
			if row >= r.rows[plane] {
				continue
			}
			s, d := r.src.pix[plane], r.dst.pix[plane]
			stride, height := r.src.stride[plane], r.src.height[plane]
			var table *[filmgrain.ScalingTableSize]uint8
			if highBitDepth {
				table = &r.tables[plane]
			}
			blockHeight := filmgrain.LumaBlockSize
			if plane != 0 {
				blockHeight = frameWorkFilmGrainChromaBlockHeight(r.plan.Format)
			}
			h := min(blockHeight, height-row*blockHeight)
			start := row * blockHeight * stride
			var err error
			if plane == 0 {
				params := frameWorkFilmGrainLumaRowParams(r.plan, row, h)
				err = frameWorkFilmGrainApplyLumaRow(d[start:], s[start:], r.lumaGrain, &r.luts.LUTs[plane], table, params)
			} else {
				lumaStart := row * filmgrain.LumaBlockSize * r.src.stride[0]
				params := frameWorkFilmGrainChromaRowParams(r.plan, plane, row, h, r.src.stride[0])
				err = frameWorkFilmGrainApplyChromaRow(d[start:], s[start:], r.src.pix[0][lumaStart:], r.chromaGrain[plane-1], &r.luts.LUTs[plane], table, params)
			}
			if err != nil {
				return frame.ErrInvalidFormat
			}
		}
	}
	return nil
}

// frameWorkFilmGrainApplyLumaRow dispatches on the sample type with concrete
// calls, so the scaling LUTs stay on the caller's stack.
func frameWorkFilmGrainApplyLumaRow[T uint8 | uint16](dst, src []T, grain []int16, scaling *[filmgrain.ScalingLUTSize]uint8, table *[filmgrain.ScalingTableSize]uint8, params filmgrain.LumaRowParams) error {
	if d, ok := any(dst).([]uint8); ok {
		return filmgrain.ApplyLumaRowU8(d, any(src).([]uint8), grain, scaling[:], params)
	}
	return filmgrain.ApplyLumaRowTable(any(dst).([]uint16), any(src).([]uint16), grain, scaling[:], table, params)
}

// frameWorkFilmGrainApplyChromaRow is frameWorkFilmGrainApplyLumaRow for a
// chroma row.
func frameWorkFilmGrainApplyChromaRow[T uint8 | uint16](dst, src, luma []T, grain []int16, scaling *[filmgrain.ScalingLUTSize]uint8, table *[filmgrain.ScalingTableSize]uint8, params filmgrain.ChromaRowParams) error {
	if d, ok := any(dst).([]uint8); ok {
		return filmgrain.ApplyChromaRowU8(d, any(src).([]uint8), any(luma).([]uint8), grain, scaling[:], params)
	}
	return filmgrain.ApplyChromaRowTable(any(dst).([]uint16), any(src).([]uint16), any(luma).([]uint16), grain, scaling[:], table, params)
}
