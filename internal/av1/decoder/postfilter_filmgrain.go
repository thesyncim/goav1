package decoder

import (
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
			Stride:        ctx.Output.Layout.YStride,
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
				Stride:        ctx.Output.Layout.UStride,
			}
		}
		if params.NumCrPoints != 0 || params.ChromaScalingFromLuma {
			plan.Planes[2] = FrameWorkFilmGrainPostFilterPlanePlan{
				Active:        true,
				ScalingPoints: int(params.NumCrPoints),
				ARCoeffs:      numUVPos,
				Width:         ctx.Output.Layout.ChromaWidth,
				Height:        ctx.Output.Layout.ChromaHeight,
				Stride:        ctx.Output.Layout.VStride,
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
	return size, nil
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
	numYPos := frameWorkFilmGrainLumaARCoeffCount(params.ARCoeffLag)
	numUVPos := frameWorkFilmGrainChromaARCoeffCount(params)
	if numYPos > parser.MaxFilmGrainYCoeffs || numUVPos > parser.MaxFilmGrainUVCoeffs {
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
