package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
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
	// tables, for 10/12-bit planes, hold each plane's per-sample-value
	// scaling (dav1d's 16bpc generate_scaling); nil at 8 bits.
	tables *[3][filmgrain.ScalingTableSize]uint8
}

// frameWorkApplyFilmGrainDirect binds the source and display planes at their
// sample width and runs the row kernels from one to the other. ok is false
// when the planes have no direct sample view (high bit depth on a big-endian
// host); the caller then takes the staged path.
func frameWorkApplyFilmGrainDirect(ctx *FrameWorkPostFilterContext, output *frame.Frame, req FrameWorkFilmGrainPostFilterRequest, plan FrameWorkFilmGrainPostFilterPlan) (FrameWorkFilmGrainPostFilterResult, bool, error) {
	srcPlanes := [3]frame.Plane{ctx.Output.Y, ctx.Output.U, ctx.Output.V}
	dstPlanes := [3]frame.Plane{output.Y, output.U, output.V}
	if ctx.Output.Layout.BytesPerSample == 1 {
		var src, dst frameWorkFilmGrainDirectPlanes[uint8]
		for i := range srcPlanes {
			src.pix[i], dst.pix[i] = srcPlanes[i].Pix, dstPlanes[i].Pix
			src.stride[i], src.width[i], src.height[i] = srcPlanes[i].Stride, srcPlanes[i].Width, srcPlanes[i].Height
		}
		dst.stride, dst.width, dst.height = src.stride, src.width, src.height
		result, err := frameWorkApplyFilmGrainDirectPlanes(ctx, output, req, plan, &src, &dst)
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
	var tables [3][filmgrain.ScalingTableSize]uint8
	src.tables = &tables
	dst.stride, dst.width, dst.height = src.stride, src.width, src.height
	result, err := frameWorkApplyFilmGrainDirectPlanes(ctx, output, req, plan, &src, &dst)
	return result, true, err
}

func frameWorkApplyFilmGrainDirectPlanes[T uint8 | uint16](ctx *FrameWorkPostFilterContext, output *frame.Frame, req FrameWorkFilmGrainPostFilterRequest, plan FrameWorkFilmGrainPostFilterPlan, src, dst *frameWorkFilmGrainDirectPlanes[T]) (FrameWorkFilmGrainPostFilterResult, error) {
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
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
	var chromaGrain [2]FrameWorkFilmGrainPostFilterChromaGrain
	for plane := filmgrain.ChromaPlaneCb; plane <= filmgrain.ChromaPlaneCr; plane++ {
		if plan.Planes[plane].Active {
			chromaGrain[plane-1], err = ctx.GenerateFilmGrainChromaGrain(req.ChromaGrain[plane-1], lumaGrain.Grain, plane)
			if err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, err
			}
		}
	}

	result := FrameWorkFilmGrainPostFilterResult{Plan: plan, OutputSize: output.Layout.Size}
	for plane := range src.pix {
		s, d := src.pix[plane], dst.pix[plane]
		stride, width, height := src.stride[plane], src.width[plane], src.height[plane]
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
		var table *[filmgrain.ScalingTableSize]uint8
		if src.tables != nil {
			table = &src.tables[plane]
			if err := filmgrain.BuildScalingTable(table, luts.LUTs[plane][:], plan.BitDepth); err != nil {
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
		for row := range rows {
			h := min(blockHeight, height-row*blockHeight)
			start := row * blockHeight * stride
			if plane == 0 {
				params := frameWorkFilmGrainLumaRowParams(plan, row, h)
				err = frameWorkFilmGrainApplyLumaRow(d[start:], s[start:], lumaGrain.Grain, &luts.LUTs[plane], table, params)
			} else {
				lumaStart := row * filmgrain.LumaBlockSize * src.stride[0]
				params := frameWorkFilmGrainChromaRowParams(plan, plane, row, h, src.stride[0])
				err = frameWorkFilmGrainApplyChromaRow(d[start:], s[start:], src.pix[0][lumaStart:], chromaGrain[plane-1].Grain, &luts.LUTs[plane], table, params)
			}
			if err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, frame.ErrInvalidFormat
			}
		}
		if plane == 0 {
			result.LumaRows = rowCount
		} else {
			result.ChromaRows[plane-1] = rowCount
		}
	}
	return result, nil
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
