package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestRefPadPlaneMatchesClampedReads: every sample of the padded copy is the
// source sample at the clamped position, the decoder's reference edge rule.
func TestRefPadPlaneMatchesClampedReads(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const w, h, stride, border = 37, 23, 41, 9
	src := make([]byte, stride*h)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	var p refPadPlane
	p.build(src, stride, w, h, border)
	for y := -border; y < h+border; y++ {
		for x := -border; x < w+border; x++ {
			want := src[min(max(y, 0), h-1)*stride+min(max(x, 0), w-1)]
			if got := p.buf[(y+border)*p.stride+x+border]; got != want {
				t.Fatalf("(%d,%d) = %d, want %d", x, y, got, want)
			}
		}
	}
}

// TestPredictPlanePaddedMatchesClamped: edge and outside-picture predictions
// through the padded copy equal the clamped predictor's.
func TestPredictPlanePaddedMatchesClamped(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const w, h = 128, 96
	ref := make([]byte, w*h)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	var plain, padded lossyEncodeState
	var cache refPadCache
	frame := SourceFrame420{Y: ref, U: ref[:64*48], V: ref[64*48 : 2*64*48], YStride: w, ChromaStride: 64, Width: w, Height: h}
	padded.refPads = cache.padsFor(frame, nil, 64, 48)
	got := make([]byte, 64*64)
	want := make([]byte, 64*64)
	for range 400 {
		bw := 8 << rng.Intn(4)
		bh := bw
		px := rng.Intn(w/8) * 8
		py := rng.Intn(h/8) * 8
		mv := motion.Vector{Row: int16(rng.Intn(161) - 80), Col: int16(rng.Intn(161) - 80)}
		if err := plain.predictPlane(want[:bw*bh], ref, w, w, h, px, py, bw, bh, mv, false, false, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
		if err := padded.predictPlane(got[:bw*bh], ref, w, w, h, px, py, bw, bh, mv, false, false, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
		for i := range bw * bh {
			if got[i] != want[i] {
				t.Fatalf("block %dx%d at (%d,%d) mv %+v: sample %d = %d, want %d", bw, bh, px, py, mv, i, got[i], want[i])
			}
		}
	}
}
