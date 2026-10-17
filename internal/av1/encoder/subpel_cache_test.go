package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestSubpelProbeCacheMatchesPredictor checks every candidate delta the
// libaom tree can reach, in arbitrary order (cold and warm intermediates),
// against the motion predictor plus SAD.
func TestSubpelProbeCacheMatchesPredictor(t *testing.T) {
	if pred8Impl == nil {
		t.Skip("no vector predictor in this build")
	}
	const w, h = 192, 160
	rng := rand.New(rand.NewSource(23))
	pix := make([]byte, w*h)
	src := make([]byte, w*h)
	var c subpelProbeCache
	var scratch motion.ConvolveScratch
	want := make([]byte, 64*64)
	for trial := range 60 {
		rng.Read(pix)
		if trial%3 == 0 {
			for i := range pix {
				pix[i] = []byte{0, 255}[rng.Intn(2)]
			}
		}
		rng.Read(src)
		ref := frame.Plane{Pix: pix, Stride: w, Width: w, Height: h}
		n := []int{16, 32, 64}[trial%3]
		px := 8 + rng.Intn(w-n-40)
		py := 8 + rng.Intn(h-n-16)
		start := motion.Vector{Row: int16(rng.Intn(5)-2) * 8, Col: int16(rng.Intn(5)-2) * 8}
		c.init(ref, src, w, px, py, n, start)
		if !c.ok {
			t.Fatalf("trial %d: cache declined interior block", trial)
		}
		deltas := make([]motion.Vector, 0, 169)
		for dy := -12; dy <= 12; dy += 2 {
			for dx := -12; dx <= 12; dx += 2 {
				deltas = append(deltas, motion.Vector{Row: int16(dy), Col: int16(dx)})
			}
		}
		rng.Shuffle(len(deltas), func(i, j int) { deltas[i], deltas[j] = deltas[j], deltas[i] })
		for _, d := range deltas {
			cand := motion.Vector{Row: start.Row + d.Row, Col: start.Col + d.Col}
			refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, cand, false, false)
			if err != nil {
				t.Fatal(err)
			}
			dst := frame.Plane{Pix: want, Stride: n, Width: n, Height: n}
			if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst, ref, 1, 8, 0, 0, refX, refY, n, n, n, n, subX, subY, motion.RegularFilters, &scratch); err != nil {
				t.Fatal(err)
			}
			wantSAD := sadRectDualBlock(src[py*w+px:], w, want, n, n, n)
			got, ok := c.sad(d)
			if !ok || got != wantSAD {
				t.Fatalf("trial %d n=%d start %+v delta %+v: sad %d ok=%v want %d", trial, n, start, d, got, ok, wantSAD)
			}
		}
	}
}
