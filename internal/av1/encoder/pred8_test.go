package encoder

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

func BenchmarkPred8(b *testing.B) {
	if pred8Impl == nil {
		b.Skip("no vector predictor in this build")
	}
	const w, h = 256, 128
	rng := rand.New(rand.NewSource(1))
	pix := make([]byte, w*h)
	rng.Read(pix)
	src := make([]byte, w*h)
	rng.Read(src)
	ref := frame.Plane{Pix: pix, Stride: w, Width: w, Height: h}
	var im [pred8IMLen]int16
	var scratch motion.ConvolveScratch
	dst := make([]byte, 64*64)
	for _, tc := range []struct {
		name       string
		subX, subY int
	}{{"H", 8, 0}, {"V", 0, 8}, {"2D", 8, 8}, {"FP", 0, 0}} {
		b.Run("fast16/"+tc.name, func(b *testing.B) {
			for range b.N {
				pred8Predict(dst, 16, &ref, 64, 32, 16, 16, tc.subX, tc.subY, motion.RegularFilters, &im)
			}
		})
		b.Run("fastSAD16/"+tc.name, func(b *testing.B) {
			for range b.N {
				pred8SAD(src[32*w+64:], w, &ref, 64, 32, 16, 16, tc.subX, tc.subY, &im)
			}
		})
		b.Run("motion16/"+tc.name, func(b *testing.B) {
			d := frame.Plane{Pix: dst, Stride: 16, Width: 16, Height: 16}
			for range b.N {
				_ = motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(d, ref, 1, 8, 0, 0, 64, 32, 16, 16, 16, 16, tc.subX, tc.subY, motion.RegularFilters, &scratch)
			}
		})
	}
}

// TestPred8MatchesMotionPredictor checks the fast luma predictor (store and
// SAD modes) against the motion package predictor for every Q4 phase, filter
// pair and block shape, over random, extreme and alternating planes.
func TestPred8MatchesMotionPredictor(t *testing.T) {
	if pred8Impl == nil {
		t.Skip("no vector predictor in this build")
	}
	const w, h = 160, 120
	rng := rand.New(rand.NewSource(17))
	pix := make([]byte, w*h)
	src := make([]byte, w*h)
	ref := frame.Plane{Pix: pix, Stride: w, Width: w, Height: h}
	var scratch motion.ConvolveScratch
	var im [pred8IMLen]int16
	want := make([]byte, 64*64)
	got := make([]byte, 64*64)
	for pattern := range 4 {
		for i := range pix {
			switch pattern {
			case 0:
				pix[i] = byte(rng.Intn(256))
			case 1:
				pix[i] = 255 * byte((i/w+i%w)&1)
			case 2:
				pix[i] = []byte{0, 255}[rng.Intn(2)]
			default:
				pix[i] = byte(128 + rng.Intn(9) - 4)
			}
		}
		rng.Read(src)
		for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}, {16, 8}, {8, 16}, {32, 16}, {16, 32}, {64, 16}, {16, 64}, {64, 32}, {32, 64}} {
			bw, bh := sz[0], sz[1]
			for trial := range 40 {
				refX := 3 + rng.Intn(w-bw-20)
				refY := 3 + rng.Intn(h-bh-8)
				subX := rng.Intn(16)
				subY := rng.Intn(16)
				if trial < 4 {
					subX, subY = []int{0, 8, 0, 4}[trial], []int{0, 0, 8, 12}[trial]
				}
				filters := motion.InterpFilters{X: motion.InterpFilter(rng.Intn(3)), Y: motion.InterpFilter(rng.Intn(3))}
				dst := frame.Plane{Pix: want, Stride: bw, Width: bw, Height: bh}
				if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst, ref, 1, 8, 0, 0, refX, refY, bw, bh, bw, bh, subX, subY, filters, &scratch); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !pred8Predict(got, bw, &ref, refX, refY, bw, bh, subX, subY, filters, &im) {
					t.Fatalf("%dx%d at (%d,%d): fast path declined", bw, bh, refX, refY)
				}
				if !bytes.Equal(got[:bw*bh], want[:bw*bh]) {
					t.Fatalf("pattern %d %dx%d at (%d,%d) sub (%d,%d) filters %+v: prediction differs", pattern, bw, bh, refX, refY, subX, subY, filters)
				}
				if bw%16 != 0 || filters != motion.RegularFilters {
					continue
				}
				so := rng.Intn(w*(h-bh-1) - bw)
				wantSAD := sadRectDualBlock(src[so:], w, want, bw, bw, bh)
				gotSAD, ok := pred8SAD(src[so:], w, &ref, refX, refY, bw, bh, subX, subY, &im)
				if !ok || gotSAD != wantSAD {
					t.Fatalf("pattern %d %dx%d sub (%d,%d): SAD %d ok=%v want %d", pattern, bw, bh, subX, subY, gotSAD, ok, wantSAD)
				}
			}
		}
	}
}
