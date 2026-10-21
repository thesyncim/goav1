package lfmask

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestCalcLFValuesMatchesResolver proves the dav1d level table reproduces the
// per-block resolver for every (segment, ref, mode, delta_lf) combination over
// randomized headers: segmentation on/off, mode/ref deltas on/off, levels on
// both sides of the base>=32 scale switch, zero chroma bases, zero luma, and
// delta_lf single vs multi.
func TestCalcLFValuesMatchesResolver(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	level := func() uint8 {
		switch rng.Intn(4) {
		case 0:
			return 0
		case 1:
			return uint8(28 + rng.Intn(8)) // around the base>=32 shift
		default:
			return uint8(rng.Intn(64))
		}
	}
	for iter := 0; iter < 20000; iter++ {
		var lf parser.LoopFilterParams
		lf.LevelY = [2]uint8{level(), level()}
		if rng.Intn(8) == 0 {
			lf.LevelY = [2]uint8{}
		}
		lf.LevelU, lf.LevelV = level(), level()
		lf.ModeRefDeltaEnabled = rng.Intn(2) == 0
		for i := range lf.Deltas.Ref {
			lf.Deltas.Ref[i] = int8(rng.Intn(127) - 63)
		}
		for i := range lf.Deltas.Mode {
			lf.Deltas.Mode[i] = int8(rng.Intn(127) - 63)
		}
		var seg parser.SegmentationParams
		seg.Enabled = rng.Intn(2) == 0
		for s := range seg.Data.Segments {
			d := &seg.Data.Segments[s]
			if rng.Intn(3) == 0 {
				continue
			}
			d.DeltaLFYV = int16(rng.Intn(127) - 63)
			d.DeltaLFYH = int16(rng.Intn(127) - 63)
			d.DeltaLFU = int16(rng.Intn(127) - 63)
			d.DeltaLFV = int16(rng.Intn(127) - 63)
		}
		var delta parser.DeltaParams
		delta.DeltaLFPresent = rng.Intn(2) == 0
		delta.DeltaLFMulti = rng.Intn(2) == 0
		var fromBase int8
		var multi [loopfilter.DeltaCount]int8
		if delta.DeltaLFPresent {
			fromBase = int8(rng.Intn(127) - 63)
			for i := range multi {
				multi[i] = int8(rng.Intn(127) - 63)
			}
		}
		res := loopfilter.PrepareLevelResolver(lf, seg, delta, false)
		p := NewLFValueParams(lf, seg, delta)
		var lfDelta [4]int8
		if delta.DeltaLFPresent {
			if delta.DeltaLFMulti {
				lfDelta = multi
			} else {
				lfDelta[0] = fromBase
			}
		}
		var values LFValues
		CalcLFValues(&values, &p, lfDelta)
		for s := uint8(0); s < parser.MaxSegments; s++ {
			ts := s
			if !seg.Enabled {
				ts = 0
			}
			for ref := uint8(0); ref < parser.RefFrames; ref++ {
				for mode := 0; mode < 2; mode++ {
					want, err := res.ResolveBlockLevels(s, ref, loopfilter.ModeDeltaClass(mode), fromBase, multi)
					if err != nil {
						t.Fatal(err)
					}
					got := values[ts]
					m := mode
					if ref == 0 {
						m = 0 // intra and ref slot 0 carry no mode delta
					}
					g := [3][2]uint8{
						{got[0][ref][m], got[1][ref][m]},
						{got[2][ref][m], got[2][ref][m]},
						{got[3][ref][m], got[3][ref][m]},
					}
					if g != want {
						t.Fatalf("iter %d seg %d ref %d mode %d: table %v resolver %v (lf %+v seg %v delta %+v)", iter, s, ref, mode, g, want, lf, seg.Enabled, delta)
					}
				}
			}
		}
	}
}
