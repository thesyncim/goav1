//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"
	"unsafe"
)

// Per-call benchmarks of the arm64 variance, ME sweep and HME kernels at the
// shapes the realtime encoder calls them with (stride 192, the reference
// one sample off the source), matching the SVT-AV1 NEON harness's shapes.

func sadVarBenchPlanes() (src, ref []byte) {
	rng := rand.New(rand.NewSource(1))
	src = make([]byte, 192*200)
	ref = make([]byte, 192*200)
	for i := range src {
		src[i] = byte(rng.Intn(256))
		ref[i] = byte(int(src[i]) + rng.Intn(41) - 20)
	}
	return src, ref
}

func BenchmarkPixelStatsRows16Arch(b *testing.B) {
	src, ref := sadVarBenchPlanes()
	for _, sh := range []struct{ w, h int }{{16, 16}, {32, 32}, {64, 32}} {
		b.Run(map[int]string{16: "16x16", 32: "32x32", 64: "64x32"}[sh.w], func(b *testing.B) {
			var sink uint32
			i := 0
			for b.Loop() {
				o := (i & 7) * 3
				i++
				sse, sum := pixelStatsRows16Arch(src[o:], 192, ref[1+o:], 192, sh.w, sh.h)
				sink += sse + uint32(sum)
			}
			_ = sink
		})
	}
}

func BenchmarkSVTHMEColSumRow(b *testing.B) {
	src, ref := sadVarBenchPlanes()
	for _, sh := range []struct{ bw, rows, w int }{{16, 8, 16}, {32, 16, 8}} {
		b.Run(map[int]string{16: "L0_16x16w16", 32: "L1_32x32w8"}[sh.bw], func(b *testing.B) {
			var cw [svtHMEMaxCols]uint16
			svtColTotalsArch(&cw, ref, 2*192, sh.rows, 64)
			srcSum := svtBlockSumSubArch(src, 2*192, sh.bw, sh.rows)
			var sink uint32
			for b.Loop() {
				sink += svtHMEColSumRow(nil, unsafe.Pointer(&src[0]), unsafe.Pointer(&ref[0]), unsafe.Pointer(&cw[0]), 2*192, sh.bw, sh.rows, sh.w, srcSum)
			}
			_ = sink
		})
	}
}
