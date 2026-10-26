//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
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

func BenchmarkAMD64Avg8x8Quad(b *testing.B) {
	src := benchAMD64Plane(64*16, 8)
	var sink int
	benchAMD64Impls(b, func() { a, c, d, e := realtimeAvg8x8QuadWide(src, 64); sink += a + c + d + e },
		func() { a, c, d, e := realtimeAvg8x8QuadAVX2(src, 64); sink += a + c + d + e })
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
