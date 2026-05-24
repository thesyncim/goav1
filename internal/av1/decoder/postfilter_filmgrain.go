package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// FrameWorkFilmGrainPostFilterPlanePlan describes one plane's film-grain
// synthesis inputs.
type FrameWorkFilmGrainPostFilterPlanePlan struct {
	Active bool

	ScalingPoints int
	ARCoeffs      int

	Width  int
	Height int
	// Stride is measured in samples, not bytes.
	Stride int
}

// FrameWorkFilmGrainPostFilterPlan summarizes active film-grain parameters for
// the eventual synthesis pass. It does not generate noise or mutate ctx.Output.
type FrameWorkFilmGrainPostFilterPlan struct {
	Active bool

	Params parser.FilmGrainParams
	Format frame.Format

	Seed     uint16
	BitDepth uint8

	Planes [3]FrameWorkFilmGrainPostFilterPlanePlan

	ChromaScalingFromLuma bool
	Overlap               bool
	ClipToRestrictedRange bool
	MatrixIdentity        bool
}

// FrameWorkFilmGrainPostFilterScratchSize reports caller-owned scratch needed
// to stage scaling points and autoregressive coefficients for synthesis.
type FrameWorkFilmGrainPostFilterScratchSize struct {
	ScalingPoints [3]int
	ARCoeffs      [3]int

	LumaGrain     int
	ChromaGrain   [2]int
	LumaSamples   int
	ChromaSamples [2]int
	LumaLine      int
	LumaColumn    int
}

// FrameWorkFilmGrainPostFilterRequest carries caller-owned scratch for
// ApplyFilmGrainPostFilter.
type FrameWorkFilmGrainPostFilterRequest struct {
	LumaGrain     []int16
	ChromaGrain   [2][]int16
	LumaSamples   []uint16
	ChromaSamples [2][]uint16
}

// FrameWorkFilmGrainPostFilterResult summarizes film-grain postfilter work.
type FrameWorkFilmGrainPostFilterResult struct {
	Plan FrameWorkFilmGrainPostFilterPlan

	NoOp       bool
	LumaRows   int
	ChromaRows [2]int
}

// FrameWorkFilmGrainPostFilterScalingLUTs contains the 8-bit-domain scaling
// lookup tables used by film-grain synthesis.
type FrameWorkFilmGrainPostFilterScalingLUTs struct {
	Active bool

	LUTs [3][filmgrain.ScalingLUTSize]uint8
}

// FrameWorkFilmGrainPostFilterLumaGrain is the generated luma grain template
// stored in caller-owned scratch.
type FrameWorkFilmGrainPostFilterLumaGrain struct {
	Active bool

	Grain []int16

	Width  int
	Height int
	Stride int
}

// FrameWorkFilmGrainPostFilterChromaGrain is one generated chroma grain
// template stored in caller-owned scratch.
type FrameWorkFilmGrainPostFilterChromaGrain struct {
	Active bool

	Plane int
	Grain []int16

	Width  int
	Height int
	Stride int
}

// FrameWorkFilmGrainPostFilterLumaRow summarizes one applied luma grain row
// band.
type FrameWorkFilmGrainPostFilterLumaRow struct {
	Active bool

	Row int

	Width  int
	Height int
	Stride int
}

// FrameWorkFilmGrainPostFilterChromaRow summarizes one applied chroma grain row
// band.
type FrameWorkFilmGrainPostFilterChromaRow struct {
	Active bool

	Plane int
	Row   int

	Width      int
	Height     int
	Stride     int
	LumaStride int
}

// FilmGrainPostFilterPlan validates active film-grain state and reports the
// plane-local synthesis inputs. It does not mutate ctx.Output.
func (ctx FrameWorkPostFilterContext) FilmGrainPostFilterPlan() (FrameWorkFilmGrainPostFilterPlan, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		return FrameWorkFilmGrainPostFilterPlan{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkFilmGrainPostFilterPlan{}, frame.ErrInvalidSlot
	}

	params := ctx.Event.FilmGrain
	if err := frameWorkValidateFilmGrainParams(params, ctx.Output.Format); err != nil {
		return FrameWorkFilmGrainPostFilterPlan{}, err
	}

	bitDepth := params.BitDepth
	if bitDepth == 0 {
		bitDepth = ctx.Output.Format.BitDepth
	}
	numYPos := frameWorkFilmGrainLumaARCoeffCount(params.ARCoeffLag)
	numUVPos := frameWorkFilmGrainChromaARCoeffCount(params)
	plan := FrameWorkFilmGrainPostFilterPlan{
		Active:                true,
		Params:                params,
		Format:                ctx.Output.Format,
		Seed:                  params.Seed,
		BitDepth:              bitDepth,
		ChromaScalingFromLuma: params.ChromaScalingFromLuma,
		Overlap:               params.Overlap,
		ClipToRestrictedRange: params.ClipToRestrictedRange,
		MatrixIdentity:        ctx.Event.SequenceHeader.ColorConfig.MatrixCoefficients == parser.MatrixCoefficientsIdentity,
	}
	if params.NumYPoints != 0 {
		plan.Planes[0] = FrameWorkFilmGrainPostFilterPlanePlan{
			Active:        true,
			ScalingPoints: int(params.NumYPoints),
			ARCoeffs:      numYPos,
			Width:         ctx.Output.Format.Width,
			Height:        ctx.Output.Format.Height,
			Stride:        frameWorkFilmGrainSampleStride(ctx.Output.Layout.YStride, ctx.Output.Layout.BytesPerSample),
		}
	}
	if !ctx.Output.Format.MonoChrome {
		if params.NumCbPoints != 0 || params.ChromaScalingFromLuma {
			plan.Planes[1] = FrameWorkFilmGrainPostFilterPlanePlan{
				Active:        true,
				ScalingPoints: int(params.NumCbPoints),
				ARCoeffs:      numUVPos,
				Width:         ctx.Output.Layout.ChromaWidth,
				Height:        ctx.Output.Layout.ChromaHeight,
				Stride:        frameWorkFilmGrainSampleStride(ctx.Output.Layout.UStride, ctx.Output.Layout.BytesPerSample),
			}
		}
		if params.NumCrPoints != 0 || params.ChromaScalingFromLuma {
			plan.Planes[2] = FrameWorkFilmGrainPostFilterPlanePlan{
				Active:        true,
				ScalingPoints: int(params.NumCrPoints),
				ARCoeffs:      numUVPos,
				Width:         ctx.Output.Layout.ChromaWidth,
				Height:        ctx.Output.Layout.ChromaHeight,
				Stride:        frameWorkFilmGrainSampleStride(ctx.Output.Layout.VStride, ctx.Output.Layout.BytesPerSample),
			}
		}
	}
	return plan, nil
}

// FilmGrainPostFilterScratchLen reports scratch lengths needed by the planned
// film-grain synthesis inputs.
func (ctx FrameWorkPostFilterContext) FilmGrainPostFilterScratchLen() (FrameWorkFilmGrainPostFilterScratchSize, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterScratchSize{}, err
	}
	if !plan.Active {
		return FrameWorkFilmGrainPostFilterScratchSize{}, nil
	}
	var size FrameWorkFilmGrainPostFilterScratchSize
	for plane := 0; plane < len(plan.Planes); plane++ {
		size.ScalingPoints[plane] = plan.Planes[plane].ScalingPoints
		size.ARCoeffs[plane] = plan.Planes[plane].ARCoeffs
	}
	if frameWorkFilmGrainAnyPlaneActive(plan) {
		lumaStride := frameWorkFilmGrainSampleStride(ctx.Output.Layout.YStride, ctx.Output.Layout.BytesPerSample)
		size.LumaSamples = lumaStride * ctx.Output.Format.Height
	}
	if plan.Planes[0].Active {
		size.LumaGrain = filmgrain.LumaGrainSamples
		size.LumaLine = filmgrain.LumaOverlapSamples * plan.Planes[0].Stride
		size.LumaColumn = filmgrain.LumaColumnScratchRows * filmgrain.LumaOverlapSamples
	}
	for plane := 1; plane <= 2; plane++ {
		if plan.Planes[plane].Active {
			size.ChromaGrain[plane-1] = filmgrain.ChromaGrainSamples
			size.ChromaSamples[plane-1] = plan.Planes[plane].Stride * plan.Planes[plane].Height
		}
	}
	return size, nil
}

// FilmGrainPostFilterScalingLUTs builds the per-plane film-grain scaling lookup
// tables from parsed frame parameters. It does not synthesize grain or mutate
// ctx.Output.
func (ctx FrameWorkPostFilterContext) FilmGrainPostFilterScalingLUTs() (FrameWorkFilmGrainPostFilterScalingLUTs, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterScalingLUTs{}, err
	}
	if !plan.Active {
		return FrameWorkFilmGrainPostFilterScalingLUTs{}, nil
	}

	var luts FrameWorkFilmGrainPostFilterScalingLUTs
	luts.Active = true
	params := plan.Params
	if err := frameWorkBuildFilmGrainYScalingLUT(luts.LUTs[0][:], params); err != nil {
		return FrameWorkFilmGrainPostFilterScalingLUTs{}, err
	}
	if !plan.Format.MonoChrome {
		if params.ChromaScalingFromLuma {
			copy(luts.LUTs[1][:], luts.LUTs[0][:])
			copy(luts.LUTs[2][:], luts.LUTs[0][:])
			return luts, nil
		}
		if err := frameWorkBuildFilmGrainUVScalingLUT(luts.LUTs[1][:], params.NumCbPoints, params.CbPoints); err != nil {
			return FrameWorkFilmGrainPostFilterScalingLUTs{}, err
		}
		if err := frameWorkBuildFilmGrainUVScalingLUT(luts.LUTs[2][:], params.NumCrPoints, params.CrPoints); err != nil {
			return FrameWorkFilmGrainPostFilterScalingLUTs{}, err
		}
	}
	return luts, nil
}

// GenerateFilmGrainLumaGrain fills caller-owned scratch with the luma grain
// template for the current frame. It does not place grain onto ctx.Output.
func (ctx FrameWorkPostFilterContext) GenerateFilmGrainLumaGrain(dst []int16) (FrameWorkFilmGrainPostFilterLumaGrain, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterLumaGrain{}, err
	}
	if !plan.Active || !plan.Planes[0].Active {
		return FrameWorkFilmGrainPostFilterLumaGrain{}, nil
	}
	if len(dst) < filmgrain.LumaGrainSamples {
		return FrameWorkFilmGrainPostFilterLumaGrain{}, frame.ErrShortBuffer
	}
	params := frameWorkFilmGrainLumaGrainParams(plan.Params, plan.BitDepth)
	if err := filmgrain.GenerateLumaGrain(dst, params); err != nil {
		return FrameWorkFilmGrainPostFilterLumaGrain{}, frame.ErrInvalidFormat
	}
	return FrameWorkFilmGrainPostFilterLumaGrain{
		Active: true,
		Grain:  dst[:filmgrain.LumaGrainSamples],
		Width:  filmgrain.LumaGrainWidth,
		Height: filmgrain.LumaGrainHeight,
		Stride: filmgrain.LumaGrainWidth,
	}, nil
}

// GenerateFilmGrainChromaGrain fills caller-owned scratch with one chroma grain
// template for the current frame. plane must be filmgrain.ChromaPlaneCb or
// filmgrain.ChromaPlaneCr.
func (ctx FrameWorkPostFilterContext) GenerateFilmGrainChromaGrain(dst []int16, luma []int16, plane int) (FrameWorkFilmGrainPostFilterChromaGrain, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterChromaGrain{}, err
	}
	if !plan.Active {
		return FrameWorkFilmGrainPostFilterChromaGrain{}, nil
	}
	if plane != filmgrain.ChromaPlaneCb && plane != filmgrain.ChromaPlaneCr {
		return FrameWorkFilmGrainPostFilterChromaGrain{}, frame.ErrInvalidFormat
	}
	if !plan.Planes[plane].Active {
		return FrameWorkFilmGrainPostFilterChromaGrain{}, nil
	}
	if len(dst) < filmgrain.ChromaGrainSamples ||
		(plan.Params.NumYPoints != 0 && len(luma) < filmgrain.LumaGrainSamples) {
		return FrameWorkFilmGrainPostFilterChromaGrain{}, frame.ErrShortBuffer
	}
	params := frameWorkFilmGrainChromaGrainParams(plan, plane)
	if err := filmgrain.GenerateChromaGrain(dst, luma, params); err != nil {
		return FrameWorkFilmGrainPostFilterChromaGrain{}, frame.ErrInvalidFormat
	}
	width, height := frameWorkFilmGrainChromaGrainDimensions(plan.Format)
	return FrameWorkFilmGrainPostFilterChromaGrain{
		Active: true,
		Plane:  plane,
		Grain:  dst[:filmgrain.ChromaGrainSamples],
		Width:  width,
		Height: height,
		Stride: filmgrain.ChromaGrainWidth,
	}, nil
}

// ApplyFilmGrainLumaRow applies luma film grain to one 32-line row band using
// caller-owned uint16 sample views. src and dst may alias.
func (ctx FrameWorkPostFilterContext) ApplyFilmGrainLumaRow(dst []uint16, src []uint16, grain []int16, scaling []uint8, row int) (FrameWorkFilmGrainPostFilterLumaRow, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterLumaRow{}, err
	}
	if !plan.Active || !plan.Planes[0].Active {
		return FrameWorkFilmGrainPostFilterLumaRow{}, nil
	}
	plane := plan.Planes[0]
	if row < 0 || row*filmgrain.LumaBlockSize >= plane.Height {
		return FrameWorkFilmGrainPostFilterLumaRow{}, frame.ErrInvalidFormat
	}
	height := filmgrain.LumaBlockSize
	if remaining := plane.Height - row*filmgrain.LumaBlockSize; remaining < height {
		height = remaining
	}
	need := (height-1)*plane.Stride + plane.Width
	if len(dst) < need || len(src) < need || len(grain) < filmgrain.LumaGrainSamples || len(scaling) < filmgrain.ScalingLUTSize {
		return FrameWorkFilmGrainPostFilterLumaRow{}, frame.ErrShortBuffer
	}
	params := frameWorkFilmGrainLumaRowParams(plan, row, height)
	if err := filmgrain.ApplyLumaRow(dst, src, grain, scaling, params); err != nil {
		return FrameWorkFilmGrainPostFilterLumaRow{}, frame.ErrInvalidFormat
	}
	return FrameWorkFilmGrainPostFilterLumaRow{
		Active: true,
		Row:    row,
		Width:  params.Width,
		Height: params.Height,
		Stride: params.Stride,
	}, nil
}

// ApplyFilmGrainChromaRow applies chroma film grain to one chroma row band
// using caller-owned uint16 sample views. src and dst may alias. luma must be a
// pre-grain luma sample view for the matching luma row band.
func (ctx FrameWorkPostFilterContext) ApplyFilmGrainChromaRow(dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, plane int, row int) (FrameWorkFilmGrainPostFilterChromaRow, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterChromaRow{}, err
	}
	if !plan.Active {
		return FrameWorkFilmGrainPostFilterChromaRow{}, nil
	}
	if plane != filmgrain.ChromaPlaneCb && plane != filmgrain.ChromaPlaneCr {
		return FrameWorkFilmGrainPostFilterChromaRow{}, frame.ErrInvalidFormat
	}
	if !plan.Planes[plane].Active {
		return FrameWorkFilmGrainPostFilterChromaRow{}, nil
	}
	planePlan := plan.Planes[plane]
	blockHeight := frameWorkFilmGrainChromaBlockHeight(plan.Format)
	if row < 0 || row*blockHeight >= planePlan.Height {
		return FrameWorkFilmGrainPostFilterChromaRow{}, frame.ErrInvalidFormat
	}
	height := blockHeight
	if remaining := planePlan.Height - row*blockHeight; remaining < height {
		height = remaining
	}
	shiftX := frameWorkFilmGrainSubsamplingShift(plan.Format.SubsamplingX)
	shiftY := frameWorkFilmGrainSubsamplingShift(plan.Format.SubsamplingY)
	lumaStride := frameWorkFilmGrainSampleStride(ctx.Output.Layout.YStride, ctx.Output.Layout.BytesPerSample)
	need := (height-1)*planePlan.Stride + planePlan.Width
	lumaRows := ((height - 1) << shiftY) + 1
	lumaNeed := (lumaRows-1)*lumaStride + ((planePlan.Width - 1) << shiftX) + 1 + shiftX
	if len(dst) < need || len(src) < need || len(luma) < lumaNeed ||
		len(grain) < filmgrain.ChromaGrainSamples || len(scaling) < filmgrain.ScalingLUTSize {
		return FrameWorkFilmGrainPostFilterChromaRow{}, frame.ErrShortBuffer
	}
	params := frameWorkFilmGrainChromaRowParams(plan, plane, row, height, lumaStride)
	if err := filmgrain.ApplyChromaRow(dst, src, luma, grain, scaling, params); err != nil {
		return FrameWorkFilmGrainPostFilterChromaRow{}, frame.ErrInvalidFormat
	}
	return FrameWorkFilmGrainPostFilterChromaRow{
		Active:     true,
		Plane:      plane,
		Row:        row,
		Width:      params.Width,
		Height:     params.Height,
		Stride:     params.Stride,
		LumaStride: params.LumaStride,
	}, nil
}

// ApplyFilmGrainPostFilter applies the currently supported film-grain subset.
func (ctx FrameWorkPostFilterContext) ApplyFilmGrainPostFilter(req FrameWorkFilmGrainPostFilterRequest) (FrameWorkFilmGrainPostFilterResult, error) {
	remaining := ctx.RemainingPostFilters()
	preFilmGrain := FrameWorkPostFilterLoopFilter |
		FrameWorkPostFilterCDEF |
		FrameWorkPostFilterSuperRes |
		FrameWorkPostFilterLoopRestoration
	if remaining&preFilmGrain != 0 {
		return FrameWorkFilmGrainPostFilterResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterFilmGrain) {
		return FrameWorkFilmGrainPostFilterResult{}, nil
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return FrameWorkFilmGrainPostFilterResult{}, err
	}
	if !frameWorkFilmGrainNoOp(plan.Params) {
		if !frameWorkFilmGrainAnyPlaneActive(plan) {
			return FrameWorkFilmGrainPostFilterResult{}, ErrUnsupportedPostFilter
		}
		if plan.Params.ScalingShift < 8 || plan.Params.ScalingShift > 11 {
			return FrameWorkFilmGrainPostFilterResult{}, frame.ErrInvalidFormat
		}
		luts, err := ctx.FilmGrainPostFilterScalingLUTs()
		if err != nil {
			return FrameWorkFilmGrainPostFilterResult{}, err
		}
		luma, err := frame.LoadSamplePlane(req.LumaSamples, ctx.Output.Y, ctx.Output.Layout.BytesPerSample)
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
			if !plan.Planes[plane].Active {
				continue
			}
			chromaGrain[plane-1], err = ctx.GenerateFilmGrainChromaGrain(req.ChromaGrain[plane-1], lumaGrain.Grain, plane)
			if err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, err
			}
		}

		result := FrameWorkFilmGrainPostFilterResult{Plan: plan}
		for plane := filmgrain.ChromaPlaneCb; plane <= filmgrain.ChromaPlaneCr; plane++ {
			if !plan.Planes[plane].Active {
				continue
			}
			chromaPlane := ctx.Output.U
			if plane == filmgrain.ChromaPlaneCr {
				chromaPlane = ctx.Output.V
			}
			chroma, err := frame.LoadSamplePlane(req.ChromaSamples[plane-1], chromaPlane, ctx.Output.Layout.BytesPerSample)
			if err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, err
			}
			blockHeight := frameWorkFilmGrainChromaBlockHeight(plan.Format)
			rows := (plan.Planes[plane].Height + blockHeight - 1) / blockHeight
			for row := 0; row < rows; row++ {
				rowStart := row * blockHeight * chroma.Stride
				lumaRowStart := row * filmgrain.LumaBlockSize * luma.Stride
				if _, err := ctx.ApplyFilmGrainChromaRow(chroma.Pix[rowStart:], chroma.Pix[rowStart:], luma.Pix[lumaRowStart:], chromaGrain[plane-1].Grain, luts.LUTs[plane][:], plane, row); err != nil {
					return FrameWorkFilmGrainPostFilterResult{}, err
				}
			}
			if err := frame.StoreSamplePlane(chromaPlane, ctx.Output.Layout.BytesPerSample, chroma); err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, err
			}
			result.ChromaRows[plane-1] = rows
		}

		if plan.Planes[0].Active {
			rows := (plan.Planes[0].Height + filmgrain.LumaBlockSize - 1) / filmgrain.LumaBlockSize
			for row := 0; row < rows; row++ {
				rowStart := row * filmgrain.LumaBlockSize * luma.Stride
				if _, err := ctx.ApplyFilmGrainLumaRow(luma.Pix[rowStart:], luma.Pix[rowStart:], lumaGrain.Grain, luts.LUTs[0][:], row); err != nil {
					return FrameWorkFilmGrainPostFilterResult{}, err
				}
			}
			if err := frame.StoreSamplePlane(ctx.Output.Y, ctx.Output.Layout.BytesPerSample, luma); err != nil {
				return FrameWorkFilmGrainPostFilterResult{}, err
			}
			result.LumaRows = rows
		}
		return result, nil
	}
	return FrameWorkFilmGrainPostFilterResult{Plan: plan, NoOp: true}, nil
}

func (ctx FrameWorkPostFilterContext) filmGrainPostFilterSupported() (bool, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return false, err
	}
	return plan.Active && (frameWorkFilmGrainNoOp(plan.Params) || frameWorkFilmGrainAnyPlaneActive(plan)), nil
}

func (ctx FrameWorkPostFilterContext) validateFilmGrainPostFilterRequest(req FrameWorkFilmGrainPostFilterRequest) error {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return err
	}
	if !plan.Active || frameWorkFilmGrainNoOp(plan.Params) {
		return nil
	}
	if !frameWorkFilmGrainAnyPlaneActive(plan) {
		return ErrUnsupportedPostFilter
	}
	if plan.Params.ScalingShift < 8 || plan.Params.ScalingShift > 11 {
		return frame.ErrInvalidFormat
	}
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		return err
	}
	if len(req.LumaGrain) < size.LumaGrain || len(req.LumaSamples) < size.LumaSamples {
		return frame.ErrShortBuffer
	}
	for plane := 0; plane < len(req.ChromaGrain); plane++ {
		if len(req.ChromaGrain[plane]) < size.ChromaGrain[plane] ||
			len(req.ChromaSamples[plane]) < size.ChromaSamples[plane] {
			return frame.ErrShortBuffer
		}
	}
	if _, err := ctx.FilmGrainPostFilterScalingLUTs(); err != nil {
		return err
	}
	return nil
}

func frameWorkValidateFilmGrainParams(params parser.FilmGrainParams, format frame.Format) error {
	if params.BitDepth != 0 && params.BitDepth != format.BitDepth {
		return frame.ErrInvalidFormat
	}
	if params.NumYPoints > parser.MaxFilmGrainYPoints ||
		params.NumCbPoints > parser.MaxFilmGrainUVPoints ||
		params.NumCrPoints > parser.MaxFilmGrainUVPoints ||
		params.ARCoeffLag > 3 {
		return frame.ErrInvalidFormat
	}
	if format.MonoChrome && (params.ChromaScalingFromLuma || params.NumCbPoints != 0 || params.NumCrPoints != 0) {
		return frame.ErrInvalidFormat
	}
	if format.SubsamplingX && format.SubsamplingY &&
		((params.NumCbPoints == 0) != (params.NumCrPoints == 0)) {
		return frame.ErrInvalidFormat
	}
	numYPos := frameWorkFilmGrainLumaARCoeffCount(params.ARCoeffLag)
	numUVPos := frameWorkFilmGrainChromaARCoeffCount(params)
	if numYPos > parser.MaxFilmGrainYCoeffs || numUVPos > parser.MaxFilmGrainUVCoeffs {
		return frame.ErrInvalidFormat
	}
	return nil
}

func frameWorkFilmGrainLumaGrainParams(params parser.FilmGrainParams, bitDepth uint8) filmgrain.LumaGrainParams {
	out := filmgrain.LumaGrainParams{
		Seed:            params.Seed,
		BitDepth:        bitDepth,
		NumYPoints:      params.NumYPoints,
		GrainScaleShift: params.GrainScaleShift,
		ARCoeffLag:      params.ARCoeffLag,
		ARCoeffShift:    params.ARCoeffShift,
	}
	copy(out.ARCoeffs[:], params.ARCoeffsY[:])
	return out
}

func frameWorkFilmGrainLumaRowParams(plan FrameWorkFilmGrainPostFilterPlan, row int, height int) filmgrain.LumaRowParams {
	return filmgrain.LumaRowParams{
		Seed:                  plan.Seed,
		Width:                 plan.Planes[0].Width,
		Height:                height,
		Stride:                plan.Planes[0].Stride,
		Row:                   row,
		BitDepth:              plan.BitDepth,
		ScalingShift:          plan.Params.ScalingShift,
		Overlap:               plan.Overlap,
		ClipToRestrictedRange: plan.ClipToRestrictedRange,
	}
}

func frameWorkFilmGrainChromaRowParams(plan FrameWorkFilmGrainPostFilterPlan, plane int, row int, height int, lumaStride int) filmgrain.ChromaRowParams {
	params := plan.Params
	out := filmgrain.ChromaRowParams{
		Seed:                  plan.Seed,
		Width:                 plan.Planes[plane].Width,
		Height:                height,
		Stride:                plan.Planes[plane].Stride,
		LumaStride:            lumaStride,
		Row:                   row,
		BitDepth:              plan.BitDepth,
		ScalingShift:          params.ScalingShift,
		SubsamplingX:          plan.Format.SubsamplingX,
		SubsamplingY:          plan.Format.SubsamplingY,
		Overlap:               plan.Overlap,
		ClipToRestrictedRange: plan.ClipToRestrictedRange,
		MatrixIdentity:        plan.MatrixIdentity,
		ChromaScalingFromLuma: params.ChromaScalingFromLuma,
	}
	if plane == filmgrain.ChromaPlaneCb {
		out.ChromaMult = params.CbMult
		out.ChromaLumaMult = params.CbLumaMult
		out.ChromaOffset = params.CbOffset
	} else {
		out.ChromaMult = params.CrMult
		out.ChromaLumaMult = params.CrLumaMult
		out.ChromaOffset = params.CrOffset
	}
	return out
}

func frameWorkFilmGrainChromaGrainParams(plan FrameWorkFilmGrainPostFilterPlan, plane int) filmgrain.ChromaGrainParams {
	params := plan.Params
	out := filmgrain.ChromaGrainParams{
		Seed:            params.Seed,
		Plane:           plane,
		BitDepth:        plan.BitDepth,
		NumYPoints:      params.NumYPoints,
		GrainScaleShift: params.GrainScaleShift,
		SubsamplingX:    plan.Format.SubsamplingX,
		SubsamplingY:    plan.Format.SubsamplingY,
		ARCoeffLag:      params.ARCoeffLag,
		ARCoeffShift:    params.ARCoeffShift,
	}
	if plane == filmgrain.ChromaPlaneCb {
		copy(out.ARCoeffs[:], params.ARCoeffsCb[:])
	} else {
		copy(out.ARCoeffs[:], params.ARCoeffsCr[:])
	}
	return out
}

func frameWorkFilmGrainChromaGrainDimensions(format frame.Format) (int, int) {
	width := filmgrain.ChromaGrainWidth
	if format.SubsamplingX {
		width = filmgrain.ChromaSubsampledGrainWidth
	}
	height := filmgrain.ChromaGrainHeight
	if format.SubsamplingY {
		height = filmgrain.ChromaSubsampledGrainHeight
	}
	return width, height
}

func frameWorkFilmGrainNoOp(params parser.FilmGrainParams) bool {
	return !params.ChromaScalingFromLuma &&
		params.NumYPoints == 0 &&
		params.NumCbPoints == 0 &&
		params.NumCrPoints == 0
}

func frameWorkFilmGrainAnyPlaneActive(plan FrameWorkFilmGrainPostFilterPlan) bool {
	return plan.Planes[0].Active || plan.Planes[1].Active || plan.Planes[2].Active
}

func frameWorkFilmGrainChromaBlockHeight(format frame.Format) int {
	height := filmgrain.LumaBlockSize
	if format.SubsamplingY {
		height >>= 1
	}
	return height
}

func frameWorkFilmGrainSubsamplingShift(subsampled bool) int {
	if subsampled {
		return 1
	}
	return 0
}

func frameWorkFilmGrainSampleStride(strideBytes int, bytesPerSample int) int {
	if bytesPerSample <= 0 {
		return 0
	}
	return strideBytes / bytesPerSample
}

func frameWorkBuildFilmGrainYScalingLUT(dst []uint8, params parser.FilmGrainParams) error {
	var points [parser.MaxFilmGrainYPoints]filmgrain.ScalingPoint
	for i := 0; i < int(params.NumYPoints); i++ {
		points[i] = filmgrain.ScalingPoint{Value: params.YPoints[i][0], Scaling: params.YPoints[i][1]}
	}
	if err := filmgrain.BuildScalingLUT(dst, points[:params.NumYPoints]); err != nil {
		return frame.ErrInvalidFormat
	}
	return nil
}

func frameWorkBuildFilmGrainUVScalingLUT(dst []uint8, count uint8, input [parser.MaxFilmGrainUVPoints][2]uint8) error {
	var points [parser.MaxFilmGrainUVPoints]filmgrain.ScalingPoint
	for i := 0; i < int(count); i++ {
		points[i] = filmgrain.ScalingPoint{Value: input[i][0], Scaling: input[i][1]}
	}
	if err := filmgrain.BuildScalingLUT(dst, points[:count]); err != nil {
		return frame.ErrInvalidFormat
	}
	return nil
}

func frameWorkFilmGrainLumaARCoeffCount(lag uint8) int {
	l := int(lag)
	return 2 * l * (l + 1)
}

func frameWorkFilmGrainChromaARCoeffCount(params parser.FilmGrainParams) int {
	count := frameWorkFilmGrainLumaARCoeffCount(params.ARCoeffLag)
	if params.NumYPoints != 0 {
		count++
	}
	return count
}
