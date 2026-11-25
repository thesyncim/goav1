//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"math/rand"
	"testing"
)

// The BenchmarkAMD64* benchmarks time each encoder AVX2 kernel against its
// generic reference on the same inputs; the amd64-bench workflow compares the
// impl=generic and impl=avx2 sub-benchmarks with benchstat.

func benchAMD64Impls(b *testing.B, generic, avx2 func()) {
	b.Run("impl=generic", func(b *testing.B) {
		for b.Loop() {
			generic()
		}
	})
	b.Run("impl=avx2", func(b *testing.B) {
		if !gosimdKernelsSupported() {
			b.Skip("no AVX2")
		}
		for b.Loop() {
			avx2()
		}
	})
}

func benchAMD64Plane(n int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	p := make([]byte, n)
	for i := range p {
		p[i] = byte(rng.Intn(256))
	}
	return p
}

func BenchmarkAMD64SubpelVariance(b *testing.B) {
	const stride = 160
	ref := benchAMD64Plane(stride*80, 1)
	src := benchAMD64Plane(stride*80, 2)
	var scratch svtSubpelScratch
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		b.Run(sizeName(w, h), func(b *testing.B) {
			var sink uint32
			benchAMD64Impls(b, func() {
				for k := 1; k < 8; k += 2 {
					svtBilinearPredict(&scratch, ref, stride, w, h, k, 8-k)
					_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, w, stride, 0, 0, w, h)
					sink += v
				}
			}, func() {
				for k := 1; k < 8; k += 2 {
					sink += svtSubpelVarianceArch(&scratch, ref, stride, src, stride, w, h, k, 8-k)
				}
			})
			_ = sink
		})
	}
}

func BenchmarkAMD64Downsample2D(b *testing.B) {
	src := benchAMD64Plane(640*64, 3)
	dst := make([]byte, 320*32)
	benchAMD64Impls(b, func() { svtDownsample2D(dst, 320, src, 640, 320, 32) },
		func() { svtDownsample2DArch(dst, 320, src, 640, 320, 32) })
}

func BenchmarkAMD64BoxInterpolateRow8(b *testing.B) {
	s0 := benchAMD64Plane(640, 4)
	s1 := benchAMD64Plane(640, 5)
	dst := make([]byte, 640)
	benchAMD64Impls(b, func() { boxInterpolateRow(dst, s0, s1, 77, 128) },
		func() { boxInterpolateRow8(dst, s0, s1, 77) })
}

func BenchmarkAMD64CDEFDist8(b *testing.B) {
	const stride = 64
	src := benchAMD64Plane(stride*64, 6)
	flt := benchAMD64Plane(stride*64, 7)
	var dlist []cdef.BlockPosition
	for by := range 8 {
		for bx := range 8 {
			dlist = append(dlist, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	var sink uint64
	benchAMD64Impls(b, func() { sink += computeCDEFDist8(src, stride, flt, stride, dlist, 3, 3, 1) },
		func() { sink += cdefDist8(src, stride, flt, stride, dlist, 3, 3, 1) })
	_ = sink
}

func sizeName(w, h int) string {
	return "size=" + itoa(w) + "x" + itoa(h)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [8]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

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
