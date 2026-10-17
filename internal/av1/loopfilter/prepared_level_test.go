package loopfilter

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPreparedLevelResolverMatchesScalarResolver(t *testing.T) {
	baseCases := []struct {
		params         parser.LoopFilterParams
		segmentationOn bool
	}{
		{params: parser.LoopFilterParams{LevelY: [2]uint8{31, 32}, LevelU: 31, LevelV: 32}},
		{segmentationOn: true, params: parser.LoopFilterParams{LevelY: [2]uint8{255, 1}, LevelU: 200, LevelV: 64}},
		{params: parser.LoopFilterParams{LevelY: [2]uint8{1, 63}, LevelU: 0, LevelV: 63, ModeRefDeltaEnabled: true,
			Deltas: parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{-128, 127, -127, 126, -1, 1, 0, 0}, Mode: [2]int8{127, -128}}}},
		{segmentationOn: true, params: parser.LoopFilterParams{LevelY: [2]uint8{255, 1}, LevelU: 200, LevelV: 64, ModeRefDeltaEnabled: true,
			Deltas: parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{127, -128, 127, -128, 63, -64, 1, -1}, Mode: [2]int8{-128, 127}}}},
	}
	segmentValues := []int16{-32768, -275, -1, 0, 1, 275, 32767}
	deltaValues := []int8{-128, -63, -1, 0, 1, 63, 127}
	for caseIndex, tc := range baseCases {
		params := tc.params
		for _, multiDelta := range []bool{false, true} {
			for _, deltaPresent := range []bool{false, true} {
				seg := parser.SegmentationParams{Enabled: tc.segmentationOn}
				for segment := range parser.MaxSegments {
					s := segmentValues[(segment+caseIndex)%len(segmentValues)]
					seg.Data.Segments[segment] = parser.SegmentData{
						DeltaLFYV: s,
						DeltaLFYH: segmentValues[(segment+1+caseIndex)%len(segmentValues)],
						DeltaLFU:  segmentValues[(segment+2+caseIndex)%len(segmentValues)],
						DeltaLFV:  segmentValues[(segment+3+caseIndex)%len(segmentValues)],
					}
				}
				delta := parser.DeltaParams{DeltaLFPresent: deltaPresent, DeltaLFMulti: multiDelta}
				prepared := PrepareLevelResolver(params, seg, delta, false)
				for segmentID := uint8(0); segmentID < parser.MaxSegments; segmentID++ {
					for ref := uint8(0); ref < parser.RefFrames; ref++ {
						for _, mode := range []ModeDeltaClass{ModeDeltaClassZero, ModeDeltaClassMotion} {
							for deltaIndex, deltaFromBase := range deltaValues {
								multi := [DeltaCount]int8{
									deltaValues[(deltaIndex+0)%len(deltaValues)],
									deltaValues[(deltaIndex+1)%len(deltaValues)],
									deltaValues[(deltaIndex+2)%len(deltaValues)],
									deltaValues[(deltaIndex+3)%len(deltaValues)],
								}
								got, err := prepared.ResolveBlockLevels(segmentID, ref, mode, deltaFromBase, multi)
								if err != nil {
									t.Fatalf("prepared case=%d seg=%d ref=%d mode=%d delta=%d multi=%v: %v", caseIndex, segmentID, ref, mode, deltaFromBase, multi, err)
								}
								for plane := PlaneY; plane <= PlaneV; plane++ {
									for edge := EdgeVertical; edge <= EdgeHorizontal; edge++ {
										blockDelta := int8(0)
										if deltaPresent {
											if multiDelta {
												index, _ := DeltaIndex(plane, edge)
												blockDelta = multi[index]
											} else {
												blockDelta = deltaFromBase
											}
										}
										want, err := ResolveLevel(params, seg, LevelRequest{
											Plane: plane, Edge: edge, SegmentID: segmentID,
											RefFrame: ref, Mode: mode, DeltaLF: blockDelta,
										})
										if err != nil {
											t.Fatalf("scalar case=%d seg=%d ref=%d mode=%d plane=%d edge=%d: %v", caseIndex, segmentID, ref, mode, plane, edge, err)
										}
										if got[plane][edge] != want {
											t.Fatalf("case=%d seg=%d ref=%d mode=%d multi=%v delta=%d plane=%d edge=%d: prepared=%d scalar=%d", caseIndex, segmentID, ref, mode, multiDelta, deltaFromBase, plane, edge, got[plane][edge], want)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestPreparedLevelResolverModeRefThresholdAndFlagCombinations(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{31, 32},
		ModeRefDeltaEnabled: true,
		Deltas:              parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{0, 1}},
	}
	prepared := PrepareLevelResolver(params, parser.SegmentationParams{}, parser.DeltaParams{}, false)
	got, err := prepared.ResolveBlockLevels(0, 1, ModeDeltaClassZero, 0, [DeltaCount]int8{})
	if err != nil {
		t.Fatalf("mode/ref-only resolve: %v", err)
	}
	if got[PlaneY][EdgeVertical] != 32 || got[PlaneY][EdgeHorizontal] != 34 {
		t.Fatalf("mode/ref threshold levels=%v want Y=(32,34)", got[PlaneY])
	}

	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[1].DeltaLFYV = 5
	segOnly := PrepareLevelResolver(parser.LoopFilterParams{LevelY: [2]uint8{20, 21}}, seg, parser.DeltaParams{}, false)
	segGot, err := segOnly.ResolveBlockLevels(1, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
	if err != nil || segGot[PlaneY][EdgeVertical] != 25 || segGot[PlaneY][EdgeHorizontal] != 21 {
		t.Fatalf("segmentation-only levels=%v err=%v want Y=(25,21)", segGot[PlaneY], err)
	}
}

func TestPreparedLevelResolverMonochromeAndRawBaseDomain(t *testing.T) {
	for raw := 0; raw <= 255; raw++ {
		params := parser.LoopFilterParams{
			LevelY: [2]uint8{uint8(raw), uint8(255 - raw)},
			LevelU: uint8(raw), LevelV: uint8(255 - raw),
		}
		prepared := PrepareLevelResolver(params, parser.SegmentationParams{}, parser.DeltaParams{}, true)
		got, err := prepared.ResolveBlockLevels(0, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
		if err != nil {
			t.Fatalf("raw base=%d resolve: %v", raw, err)
		}
		for plane := PlaneY; plane <= PlaneV; plane++ {
			for edge := EdgeVertical; edge <= EdgeHorizontal; edge++ {
				want, err := ResolveLevel(params, parser.SegmentationParams{}, LevelRequest{
					Plane: plane, Edge: edge, RefFrame: 0, Mode: ModeDeltaClassZero,
				})
				if err != nil {
					t.Fatalf("raw base=%d plane=%d edge=%d scalar: %v", raw, plane, edge, err)
				}
				if got[plane][edge] != want {
					t.Fatalf("raw base=%d plane=%d edge=%d prepared=%d scalar=%d", raw, plane, edge, got[plane][edge], want)
				}
			}
		}
	}

	base := parser.LoopFilterParams{LevelY: [2]uint8{20, 24}, LevelU: 31, LevelV: 47}
	noFlags := PrepareLevelResolver(base, parser.SegmentationParams{}, parser.DeltaParams{}, true)
	got, err := noFlags.ResolveBlockLevels(0, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
	if err != nil || got[PlaneU] != [2]uint8{31, 31} || got[PlaneV] != [2]uint8{47, 47} {
		t.Fatalf("monochrome no-flags legacy chroma fast return levels=%v err=%v", got, err)
	}

	cases := []struct {
		name  string
		seg   parser.SegmentationParams
		delta parser.DeltaParams
		mode  parser.LoopFilterParams
	}{
		{name: "delta", delta: parser.DeltaParams{DeltaLFPresent: true}},
		{name: "segmentation", seg: parser.SegmentationParams{Enabled: true}},
		{name: "mode-ref", mode: parser.LoopFilterParams{ModeRefDeltaEnabled: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := base
			params.ModeRefDeltaEnabled = tc.mode.ModeRefDeltaEnabled
			params.Deltas = tc.mode.Deltas
			seg := tc.seg
			if seg.Enabled {
				seg.Data.Segments[0].DeltaLFU = 7
				seg.Data.Segments[0].DeltaLFV = -9
			}
			prepared := PrepareLevelResolver(params, seg, tc.delta, true)
			levels, err := prepared.ResolveBlockLevels(0, 0, ModeDeltaClassZero, 1, [DeltaCount]int8{1, 2, 3, 4})
			if err != nil || levels[PlaneU] != [2]uint8{} || levels[PlaneV] != [2]uint8{} {
				t.Fatalf("monochrome %s gated chroma levels=%v err=%v", tc.name, levels, err)
			}
		})
	}
}

func TestPreparedLevelResolverLumaZeroFastReturnKeepsLegacyChromaBehavior(t *testing.T) {
	params := parser.LoopFilterParams{LevelU: 40, LevelV: 50}
	for _, mono := range []bool{false, true} {
		prepared := PrepareLevelResolver(params, parser.SegmentationParams{}, parser.DeltaParams{}, mono)
		got, err := prepared.ResolveBlockLevels(parser.MaxSegments, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
		if err != nil || got != ([3][2]uint8{}) {
			t.Fatalf("mono=%v levels=%v err=%v want luma-zero fast return", mono, got, err)
		}
	}
}

func TestPreparedLevelResolverFastReturnValidationOrder(t *testing.T) {
	params := parser.LoopFilterParams{}
	seg := parser.SegmentationParams{Enabled: true}
	prepared := PrepareLevelResolver(params, seg, parser.DeltaParams{}, true)
	if _, err := prepared.ResolveBlockLevels(0, parser.RefFrames, ModeDeltaClassZero, 0, [DeltaCount]int8{}); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid reference error=%v want %v", err, ErrInvalidFilter)
	}
	if _, err := prepared.ResolveBlockLevels(0, 0, ModeDeltaClass(2), 0, [DeltaCount]int8{}); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid mode error=%v want %v", err, ErrInvalidFilter)
	}
	if got, err := prepared.ResolveBlockLevels(parser.MaxSegments, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{}); err != nil || got != ([3][2]uint8{}) {
		t.Fatalf("luma-zero invalid-segment fast return levels=%v err=%v", got, err)
	}
}
