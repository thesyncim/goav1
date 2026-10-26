//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"math/rand"
	"testing"
)

// TestAMD64CFL16MatchesPureGo compares the AVX 16bpc CfL subsampling and
// average subtraction with the scalar kernels, including out-of-range
// samples (both must report the error).
func TestAMD64CFL16MatchesPureGo(t *testing.T) {
	assertDispatchTarget(t, "subsampleLuma16Impl", subsampleLuma16Impl, "subsampleLuma16AVX")
	assertDispatchTarget(t, "subtractCFLAverageImpl", subtractCFLAverageImpl, "subtractCFLAverageAVX")
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 3000; trial++ {
		max := uint16(1023)
		if trial&1 != 0 {
			max = 4095
		}
		subX, subY := trial%3 != 0, trial%3 == 2
		w := 4 << rng.Intn(4)
		h := 4 << rng.Intn(4)
		outW, outH := w, h
		if subX {
			outW = w / 2
		}
		if subY {
			outH = h / 2
		}
		stride := w + 8
		input := make([]uint16, stride*h+16)
		for i := range input {
			input[i] = uint16(rng.Intn(int(max) + 1))
		}
		if trial%50 == 0 {
			input[rng.Intn(stride*h)] = max + 1
		}
		got := make([]uint16, CFLBufSquare)
		want := make([]uint16, CFLBufSquare)
		errG := subsampleLuma16AVX(got, input, stride, w, h, outW, outH, subX, subY, max)
		errW := subsampleLuma16PureGo(want, input, stride, w, h, outW, outH, subX, subY, max)
		if (errG == nil) != (errW == nil) {
			t.Fatalf("trial %d: error mismatch %v vs %v", trial, errG, errW)
		}
		if errW != nil {
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d %dx%d sub=%v,%v: idx %d got %d want %d", trial, w, h, subX, subY, i, got[i], want[i])
			}
		}
		log2, _ := log2PowerOfTwoInt(outW * outH)
		g, wv := make([]int16, CFLBufSquare), make([]int16, CFLBufSquare)
		subtractCFLAverageAVX(got, g, outW, outH, log2)
		subtractCFLAveragePureGo(want, wv, outW, outH, log2)
		for i := range g {
			if g[i] != wv[i] {
				t.Fatalf("trial %d: subtract idx %d got %d want %d", trial, i, g[i], wv[i])
			}
		}
	}
}
