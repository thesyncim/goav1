package encoder

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestMDS0RetainedPredictionMatchesWinner(t *testing.T) {
	zero := motion.Vector{}
	onePixel := motion.Vector{Col: 8}
	farMV := motion.Vector{Col: 64}
	tests := []struct {
		name     string
		sourceMV motion.Vector
		meMV     motion.Vector
		nearest  motion.Vector
		near     motion.Vector
		rdMult   int64
		rates    tile.InterModeRateTables
		wantMode tile.InterMode
		wantMV   motion.Vector
	}{
		{
			name:     "restore-overwritten-me-winner",
			sourceMV: farMV, meMV: farMV, nearest: zero, near: onePixel,
			rdMult: 512, wantMode: tile.InterModeNewMV, wantMV: farMV,
		},
		{
			name:     "copy-new-near-winner",
			sourceMV: onePixel, meMV: farMV, nearest: zero, near: onePixel,
			rdMult: 512, wantMode: tile.InterModeNearMV, wantMV: onePixel,
		},
		{
			name:     "restore-duplicate-global-winner",
			sourceMV: onePixel, meMV: farMV, nearest: zero, near: onePixel,
			rdMult: 1 << 30,
			rates: tile.InterModeRateTables{
				RefMV: [tile.RefMVModeContexts][2]uint32{{20_000, 10_000}},
			},
			wantMode: tile.InterModeGlobalMV, wantMV: zero,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			const width, height, block, px, py = 64, 64, 8, 24, 24
			rng := rand.New(rand.NewSource(91))
			refPlane := make([]byte, width*height)
			for i := range refPlane {
				refPlane[i] = byte(rng.Intn(256))
			}
			var sourcePrediction [block * block]byte
			var sourceScratch motion.ScaledConvolveScratch
			if err := predictInto(sourcePrediction[:], refPlane, width, width, height,
				px, py, block, block, tc.sourceMV, false, false, &sourceScratch); err != nil {
				t.Fatalf("source prediction: %v", err)
			}
			srcY := make([]byte, width*height)
			for row := range block {
				copy(srcY[(py+row)*width+px:(py+row)*width+px+block], sourcePrediction[row*block:(row+1)*block])
			}
			src := SourceFrame420{Y: srcY, YStride: width, Width: width, Height: height}
			stack := &tile.ReferenceMVStackResult{ModeContext: 0}
			stack.Stack.Count = 2
			stack.Stack.Candidates[0].This = tc.nearest
			stack.Stack.Candidates[1].This = tc.near
			st := &lossyEncodeState{rdMult: tc.rdMult, mds0Rates: tc.rates}

			winner, ok, retained := st.mds0PickInterMode(src, refPlane, width, stack, px, py, block, block, tc.meMV)
			if !ok {
				t.Fatal("MDS0 did not return a winner")
			}
			if winner.mode != tc.wantMode || winner.mv != tc.wantMV {
				t.Fatalf("winner=(%v,%v), want (%v,%v)", winner.mode, winner.mv, tc.wantMode, tc.wantMV)
			}
			if !retained {
				t.Fatal("winner prediction was not retained")
			}

			var want [block * block]byte
			var wantScratch motion.ScaledConvolveScratch
			if err := predictInto(want[:], refPlane, width, width, height,
				px, py, block, block, winner.mv, false, false, &wantScratch); err != nil {
				t.Fatalf("winner prediction: %v", err)
			}
			if !bytes.Equal(st.predY[:block*block], want[:]) {
				t.Fatal("retained predY does not match the selected winner")
			}
		})
	}
}

func TestMDS0PredictorMatchesCodedFilterGate(t *testing.T) {
	fullPel := motion.Vector{Row: 8, Col: 16}
	subPel := motion.Vector{Row: 1, Col: 9}
	regular := motion.RegularFilters
	smooth := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpEightTapRegular}
	dual := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	tests := []struct {
		name     string
		retained bool
		mv       motion.Vector
		filters  motion.InterpFilters
		want     bool
	}{
		{"invalid", false, subPel, regular, false},
		{"full-pel ignores filter", true, fullPel, smooth, true},
		{"sub-pel regular", true, subPel, regular, true},
		{"sub-pel smooth", true, subPel, smooth, false},
		{"sub-pel dual", true, subPel, dual, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := mds0PredictorMatchesCodedFilters(tc.retained, tc.mv, tc.filters); got != tc.want {
				t.Fatalf("reusable=%t, want %t", got, tc.want)
			}
		})
	}
}
