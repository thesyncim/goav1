package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// sceneCutToSmoothFrames is dense noise followed by a slowly moving smooth
// gradient: the cut makes the light-PD1 DC_PRED arm code whole 16x16/32x32
// blocks as intra inside the inter frames.
func sceneCutToSmoothFrames(n, w, h int) []SourceFrame420 {
	rng := rand.New(rand.NewSource(5))
	cw, ch := w/2, h/2
	frames := make([]SourceFrame420, n)
	for k := range frames {
		f := SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 130
		}
		for y := range h {
			for x := range w {
				if k == 0 {
					f.Y[y*w+x] = uint8(40 + rng.Intn(170))
				} else {
					f.Y[y*w+x] = uint8(60 + (x+y+2*k)/4)
				}
			}
		}
		frames[k] = f
	}
	return frames
}

func TestPFrameIntraDCSplitMatchesFused(t *testing.T) {
	const w, h = 256, 192
	frames := sceneCutToSmoothFrames(6, w, h)
	// The oracle below proves byte identity; this fused run proves the DC
	// arm actually fires in the oracle's configuration.
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 600_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	enc.SetDecisionStatsEnabled(true)
	for i, f := range frames {
		if _, _, err := enc.Encode(f, i == 0); err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
	}
	stats := enc.DecisionStats()
	if stats.IntraBlockSizes[tile.BlockSize16x16]+stats.IntraBlockSizes[tile.BlockSize32x32] == 0 {
		t.Fatal("oracle configuration coded no 16x16/32x32 DC intra blocks")
	}
	assertSplitMatchesFused(t, frames, w, h, 30, 600_000, []int{1, 2})
	assertWavefrontMatchesSerial(t, frames, w, h, 30, 600_000, []int{2, 3})
}
