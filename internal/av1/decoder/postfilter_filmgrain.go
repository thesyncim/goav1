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
}

// FrameWorkFilmGrainPostFilterScratchSize reports caller-owned scratch needed
// to stage scaling points and autoregressive coefficients for synthesis.
type FrameWorkFilmGrainPostFilterScratchSize struct {
	ScalingPoints [3]int
	ARCoeffs      [3]int

	LumaGrain  int
	LumaLine   int
	LumaColumn int
}

// FrameWorkFilmGrainPostFilterRequest carries caller-owned scratch for
// ApplyFilmGrainPostFilter. The currently supported no-op subset needs none.
type FrameWorkFilmGrainPostFilterRequest struct{}

// FrameWorkFilmGrainPostFilterResult summarizes film-grain postfilter work.
type FrameWorkFilmGrainPostFilterResult struct {
	Plan FrameWorkFilmGrainPostFilterPlan

	NoOp bool
}

// FrameWorkFilmGrainPostFilterScalingLUTs contains the 8-bit-domain scaling
// lookup tables used by film-grain synthesis.
type FrameWorkFilmGrainPostFilterScalingLUTs struct {
	Active bool

	LUTs [3][filmgrain.ScalingLUTSize]uint8
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
	if plan.Planes[0].Active {
		size.LumaGrain = filmgrain.LumaGrainSamples
		size.LumaLine = filmgrain.LumaOverlapSamples * plan.Planes[0].Stride
		size.LumaColumn = filmgrain.LumaColumnScratchRows * filmgrain.LumaOverlapSamples
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

// ApplyFilmGrainPostFilter applies the currently supported film-grain subset.
// It only completes the stage when the signaled grain is a true no-op; active
// synthesis still rejects before mutating ctx.Output.
func (ctx FrameWorkPostFilterContext) ApplyFilmGrainPostFilter(req FrameWorkFilmGrainPostFilterRequest) (FrameWorkFilmGrainPostFilterResult, error) {
	_ = req
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
		return FrameWorkFilmGrainPostFilterResult{}, ErrUnsupportedPostFilter
	}
	return FrameWorkFilmGrainPostFilterResult{Plan: plan, NoOp: true}, nil
}

func (ctx FrameWorkPostFilterContext) filmGrainPostFilterSupported() (bool, error) {
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		return false, err
	}
	return plan.Active && frameWorkFilmGrainNoOp(plan.Params), nil
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

func frameWorkFilmGrainNoOp(params parser.FilmGrainParams) bool {
	return !params.ChromaScalingFromLuma &&
		params.NumYPoints == 0 &&
		params.NumCbPoints == 0 &&
		params.NumCrPoints == 0
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
