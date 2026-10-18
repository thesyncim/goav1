package decoder

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFilmGrainDirectPublicationMatchesStaged(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		sx, sy, mono, fromLuma bool
		active                 uint8
	}{
		{"444-all", false, false, false, false, 7},
		{"444-from-luma", false, false, false, true, 7},
		{"422-all", true, false, false, false, 7},
		{"420-from-luma", true, true, false, true, 7},
		{"mono", true, true, true, false, 1},
		{"444-luma-only", false, false, false, false, 1},
		{"444-cb-only", false, false, false, false, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := testFilmGrainDirectContext(t, frame.Format{Width: 98, Height: 65, BitDepth: 8, SubsamplingX: tc.sx, SubsamplingY: tc.sy, MonoChrome: tc.mono, Align: 64}, tc.active, tc.fromLuma)
			ctx.Output.ExtendBorders()
			before := [3][]byte{bytes.Clone(ctx.Output.Y.Pix), bytes.Clone(ctx.Output.U.Pix), bytes.Clone(ctx.Output.V.Pix)}
			size, err := ctx.FilmGrainPostFilterScratchLen()
			if err != nil {
				t.Fatal(err)
			}
			legacyReq := testFrameWorkFilmGrainPostFilterRequest(size)
			var legacyView frame.Frame
			legacyReq.OutputView = &legacyView
			legacy, err := ctx.BindFilmGrainPostFilterOutput(legacyReq)
			if err != nil {
				t.Fatal(err)
			}
			legacyCtx := ctx
			legacyCtx.Output = legacy
			legacyCtx.detachedPostFilterOutput = true
			wantResult, err := legacyCtx.ApplyFilmGrainPostFilter(legacyReq)
			if err != nil {
				t.Fatal(err)
			}

			req := testFrameWorkFilmGrainPostFilterRequest(size)
			var view frame.Frame
			req.OutputView = &view
			for _, samples := range [][]uint16{req.LumaSamples, req.ChromaSamples[0], req.ChromaSamples[1]} {
				for i := range samples {
					samples[i] = 0xabad
				}
			}
			next, got, result, err := ctx.ApplySupportedPostFiltersForPublication(FrameWorkPostFilterRequest{FilmGrain: req})
			if err != nil {
				t.Fatal(err)
			}
			if next.Output != ctx.Output || got != &view || result.FilmGrain.Output != got || next.RemainingPostFilters() != 0 {
				t.Fatal("publication ownership/result changed")
			}
			if result.FilmGrain.Plan != wantResult.Plan || result.FilmGrain.LumaRows != wantResult.LumaRows || result.FilmGrain.ChromaRows != wantResult.ChromaRows || result.FilmGrain.OutputSize != wantResult.OutputSize {
				t.Fatal("result geometry changed")
			}
			for i, p := range []frame.Plane{got.Y, got.U, got.V} {
				want := []frame.Plane{legacy.Y, legacy.U, legacy.V}[i]
				if !bytes.Equal(p.Pix, want.Pix) {
					t.Fatalf("plane %d full bytes/padding differ", i)
				}
				original := []frame.Plane{ctx.Output.Y, ctx.Output.U, ctx.Output.V}[i]
				if !bytes.Equal(original.Pix, before[i]) {
					t.Fatalf("reference plane %d modified", i)
				}
			}
			if frameWorkFilmGrainDirectU8Available {
				for _, samples := range [][]uint16{req.LumaSamples, req.ChromaSamples[0], req.ChromaSamples[1]} {
					for _, v := range samples {
						if v != 0xabad {
							t.Fatal("direct path used staging scratch")
						}
					}
				}
			}
			if tc.name == "444-all" {
				if allocs := testing.AllocsPerRun(20, func() {
					_, _, _, err := ctx.ApplySupportedPostFiltersForPublication(FrameWorkPostFilterRequest{FilmGrain: req})
					if err != nil {
						panic(err)
					}
				}); allocs != 0 {
					t.Fatalf("steady publication allocs=%v", allocs)
				}
			}
		})
	}
}

func TestFilmGrainDirectPublicationFallbacks(t *testing.T) {
	for _, name := range []string{"alias", "odd-subsampled-width", "allocated-view", "high-bit-depth", "noncanonical-plane"} {
		t.Run(name, func(t *testing.T) {
			format := frame.Format{Width: 34, Height: 33, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
			if name == "odd-subsampled-width" {
				format.Width = 33
			}
			if name == "high-bit-depth" {
				format.BitDepth = 10
			}
			ctx := testFilmGrainDirectContext(t, format, 7, true)
			if name == "noncanonical-plane" {
				ctx.Output.Y.Width--
			}
			size, err := ctx.FilmGrainPostFilterScratchLen()
			if err != nil {
				t.Fatal(err)
			}
			req := testFrameWorkFilmGrainPostFilterRequest(size)
			var view frame.Frame
			req.OutputView = &view
			if name == "allocated-view" {
				req.OutputView = nil
			}
			if name == "alias" {
				req.OutputFrame = ctx.Output.Y.Pix[:ctx.Output.Layout.Size]
			}
			for i := range req.LumaSamples {
				req.LumaSamples[i] = 0xabad
			}
			_, _, err = frameWorkApplyFilmGrainForPublication(&ctx, req)
			if err != nil {
				t.Fatal(err)
			}
			if req.LumaSamples[0] == 0xabad {
				t.Fatal("legacy fallback did not run")
			}
		})
	}
}

func TestFilmGrainDirectPublicationErrorOrdering(t *testing.T) {
	ctx := testFilmGrainDirectContext(t, frame.Format{Width: 34, Height: 33, BitDepth: 8, Align: 32}, 7, false)
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size)
	var view frame.Frame
	req.OutputView = &view
	req.LumaSamples = req.LumaSamples[:len(req.LumaSamples)-1]
	before := bytes.Clone(ctx.Output.Y.Pix)
	_, _, _, err = ctx.ApplySupportedPostFiltersForPublication(FrameWorkPostFilterRequest{FilmGrain: req})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("err=%v", err)
	}
	if !bytes.Equal(before, ctx.Output.Y.Pix) {
		t.Fatal("short scratch mutated reference")
	}
}

func testFilmGrainDirectContext(t *testing.T, format frame.Format, active uint8, fromLuma bool) FrameWorkPostFilterContext {
	t.Helper()
	output := testFrameWorkCDEFFrame(t, format)
	for plane, p := range []frame.Plane{output.Y, output.U, output.V} {
		for i := range p.Pix {
			p.Pix[i] = byte(i*37 + plane*19)
		}
		if format.BitDepth > 8 {
			for y := 0; y < p.Height; y++ {
				for x := 0; x < p.Width; x++ {
					setTestFrameWorkFilmGrainPlaneSample(p, output.Layout.BytesPerSample, x, y, uint16((x*19+y*31+plane*7)&1023))
				}
			}
		}
	}
	fg := parser.FilmGrainParams{ParamsPresent: true, Apply: true, Seed: 0x4362, BitDepth: format.BitDepth, ScalingShift: 8, ARCoeffShift: 6, Overlap: true, ClipToRestrictedRange: true,
		ChromaScalingFromLuma: fromLuma, CbMult: 140, CrMult: 118, CbLumaMult: 146, CrLumaMult: 126, CbOffset: 260, CrOffset: 250}
	if active&1 != 0 {
		fg.NumYPoints = 2
		fg.YPoints[0] = [2]uint8{0, 255}
		fg.YPoints[1] = [2]uint8{255, 32}
	}
	if active&2 != 0 && !fromLuma {
		fg.NumCbPoints = 2
		fg.CbPoints[0] = [2]uint8{0, 64}
		fg.CbPoints[1] = [2]uint8{255, 250}
	}
	if active&4 != 0 && !fromLuma {
		fg.NumCrPoints = 2
		fg.CrPoints[0] = [2]uint8{0, 250}
		fg.CrPoints[1] = [2]uint8{255, 64}
	}
	ctx := FrameWorkPostFilterContext{Event: Event{FilmGrain: fg}, Output: output}
	if _, err := ctx.FilmGrainPostFilterPlan(); err != nil {
		t.Fatal(fmt.Errorf("fixture: %w", err))
	}
	return ctx
}
