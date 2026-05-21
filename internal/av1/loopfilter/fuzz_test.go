package loopfilter

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func FuzzResolveLevel(f *testing.F) {
	f.Add(uint8(40), uint8(20), uint8(17), int8(-3), int16(-5), int8(-2), int8(3), uint8(0), uint8(4), uint8(1))
	f.Fuzz(func(t *testing.T, levelYV uint8, levelYH uint8, levelU uint8, deltaLF int8, segDelta int16, refDelta int8, modeDelta int8, edgeRaw uint8, refRaw uint8, modeRaw uint8) {
		edge := Edge(edgeRaw & 1)
		ref := int(refRaw % parser.RefFrames)
		mode := ModeDeltaClass(modeRaw & 1)

		params := parser.LoopFilterParams{
			LevelY:              [2]uint8{levelYV % (MaxLevel + 1), levelYH % (MaxLevel + 1)},
			LevelU:              levelU % (MaxLevel + 1),
			ModeRefDeltaEnabled: true,
		}
		params.Deltas.Ref[ref] = refDelta
		params.Deltas.Mode[mode] = modeDelta

		seg := parser.SegmentationParams{Enabled: true}
		seg.Data.Segments[0].DeltaLFYV = segDelta
		seg.Data.Segments[0].DeltaLFYH = segDelta

		got, err := ResolveLevel(params, seg, LevelRequest{
			Plane:     PlaneY,
			Edge:      edge,
			SegmentID: 0,
			RefFrame:  ref,
			Mode:      mode,
			DeltaLF:   deltaLF,
		})
		if err != nil {
			t.Fatal(err)
		}
		if got > MaxLevel {
			t.Fatalf("level=%d", got)
		}
	})
}

func FuzzThresholdsForLevel(f *testing.F) {
	f.Add(uint8(16), uint8(0))
	f.Add(uint8(63), uint8(7))
	f.Fuzz(func(t *testing.T, level uint8, sharpness uint8) {
		thresholds, err := ThresholdsForLevel(level, sharpness)
		if level > MaxLevel || sharpness > MaxSharpness {
			if err == nil {
				t.Fatalf("ThresholdsForLevel(%d,%d) succeeded for invalid input", level, sharpness)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if thresholds.Limit == 0 || thresholds.BlockLimit < thresholds.Limit || thresholds.HighEdgeVariance != level>>4 {
			t.Fatalf("thresholds=%+v level=%d sharpness=%d", thresholds, level, sharpness)
		}
	})
}
