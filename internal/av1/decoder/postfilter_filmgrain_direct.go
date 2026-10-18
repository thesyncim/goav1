package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// frameWorkApplyFilmGrainForPublication writes grain from the reconstructed
// source directly to a separate 8-bit display surface, as dav1d's
// fg_apply_tmpl.c does. The source remains the reference picture, including
// the original luma used by both chroma planes. Public in-place operations,
// caller aliases, unusual geometry, and high bit depth keep the staged path.
func frameWorkApplyFilmGrainForPublication(ctx *FrameWorkPostFilterContext, req FrameWorkFilmGrainPostFilterRequest) (*frame.Frame, FrameWorkFilmGrainPostFilterResult, error) {
	if frameWorkFilmGrainDirectU8Available && ctx.Output != nil && ctx.Output.Format.BitDepth == 8 &&
		req.OutputView != nil && req.OutputView != ctx.Output && len(req.OutputFrame) >= ctx.Output.Layout.Size {
		plan, planErr := ctx.FilmGrainPostFilterPlan()
		output, bindErr := frame.Bind(req.OutputFrame[:ctx.Output.Layout.Size], ctx.Output.Format)
		if planErr == nil && plan.Active && !frameWorkFilmGrainNoOp(plan.Params) && bindErr == nil &&
			frameWorkFilmGrainDirectCompatible(&output, ctx.Output, plan) {
			result, applyErr := frameWorkApplyFilmGrainDirectU8(ctx, &output, req, plan)
			if applyErr == nil {
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
	if dst.Format != src.Format || dst.Layout != src.Layout || src.Layout.BytesPerSample != 1 {
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

func frameWorkApplyFilmGrainDirectU8(ctx *FrameWorkPostFilterContext, output *frame.Frame, req FrameWorkFilmGrainPostFilterRequest, plan FrameWorkFilmGrainPostFilterPlan) (FrameWorkFilmGrainPostFilterResult, error) {
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
	srcPlanes := [3]frame.Plane{ctx.Output.Y, ctx.Output.U, ctx.Output.V}
	dstPlanes := [3]frame.Plane{output.Y, output.U, output.V}
	for plane, src := range srcPlanes {
		dst := dstPlanes[plane]
		if !plan.Planes[plane].Active {
			copy(dst.Pix, src.Pix)
			continue
		}
		// StoreSamplePlane historically changed visible samples only. Copy
		// the extended row/bottom padding without pre-copying active pixels.
		for y := 0; y < src.Height; y++ {
			start := y*src.Stride + src.Width
			end := (y + 1) * src.Stride
			copy(dst.Pix[start:end], src.Pix[start:end])
		}
		copy(dst.Pix[src.Height*src.Stride:], src.Pix[src.Height*src.Stride:])
		blockHeight := filmgrain.LumaBlockSize
		if plane != 0 {
			blockHeight = frameWorkFilmGrainChromaBlockHeight(plan.Format)
		}
		rows := (src.Height + blockHeight - 1) / blockHeight
		rowCount, ok := frameWorkFilmGrainRows16(rows)
		if !ok {
			return FrameWorkFilmGrainPostFilterResult{}, frame.ErrInvalidFormat
		}
		for row := range rows {
			height := min(blockHeight, src.Height-row*blockHeight)
			start := row * blockHeight * src.Stride
			if plane == 0 {
				params := frameWorkFilmGrainLumaRowParams(plan, row, height)
				err = filmgrain.ApplyLumaRowU8(dst.Pix[start:], src.Pix[start:], lumaGrain.Grain, luts.LUTs[plane][:], params)
			} else {
				lumaStart := row * filmgrain.LumaBlockSize * ctx.Output.Y.Stride
				params := frameWorkFilmGrainChromaRowParams(plan, plane, row, height, ctx.Output.Y.Stride)
				err = filmgrain.ApplyChromaRowU8(dst.Pix[start:], src.Pix[start:], ctx.Output.Y.Pix[lumaStart:], chromaGrain[plane-1].Grain, luts.LUTs[plane][:], params)
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
