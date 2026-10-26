//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// BenchmarkAMD64Pred8 compares the encoder's AVX2 full-pel predictor
// (impl=avx2) with the motion package predictor it replaces on amd64
// (impl=generic). Sub-pel blocks keep the motion package predictor on amd64,
// so only FP has an avx2 case.
func BenchmarkAMD64Pred8(b *testing.B) {
	const w, h = 256, 160
	rng := rand.New(rand.NewSource(1))
	pix := make([]byte, w*h)
	rng.Read(pix)
	ref := frame.Plane{Pix: pix, Stride: w, Width: w, Height: h}
	var im [pred8IMLen]int16
	var scratch motion.ConvolveScratch
	dst := make([]byte, 64*64)
	for _, n := range []int{8, 16, 32} {
		for _, tc := range []struct {
			name       string
			subX, subY int
		}{{"H", 8, 0}, {"V", 0, 8}, {"2D", 8, 8}, {"FP", 0, 0}} {
			b.Run(fmt.Sprintf("%dx%d/%s/impl=generic", n, n, tc.name), func(b *testing.B) {
				d := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
				for b.Loop() {
					_ = motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(d, ref, 1, 8, 0, 0, 64, 32, n, n, n, n, tc.subX, tc.subY, motion.RegularFilters, &scratch)
				}
			})
			b.Run(fmt.Sprintf("%dx%d/%s/impl=avx2", n, n, tc.name), func(b *testing.B) {
				if pred8Impl == nil || pred8FullPelOnly && tc.subX|tc.subY != 0 {
					b.Skip("motion package predictor")
				}
				for b.Loop() {
					pred8Predict(dst, n, &ref, 64, 32, n, n, tc.subX, tc.subY, motion.RegularFilters, &im)
				}
			})
		}
	}
}
