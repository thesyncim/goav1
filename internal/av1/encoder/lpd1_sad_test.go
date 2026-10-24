package encoder

import (
	"math/rand"
	"testing"
)

// TestLPD1SampledSADMatchesPureGo checks the dispatched sampled SAD against
// its reference over the detector's widths, row counts and steps.
func TestLPD1SampledSADMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const stride = 80
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 4000; trial++ {
		for i := range src {
			src[i], ref[i] = byte(rng.Intn(256)), byte(rng.Intn(256))
		}
		w := []int{4, 8, 16, 32}[rng.Intn(4)]
		rows := 1 + rng.Intn(8)
		step := 1 << rng.Intn(3)
		maxOff := stride*40 - ((rows-1)*step*stride + w)
		so, ro := rng.Intn(maxOff), rng.Intn(maxOff)
		got := lpd1SampledSAD(src, ref, so, ro, stride, stride, w, rows, step)
		want := lpd1SampledSADPureGo(src, ref, so, ro, stride, stride, w, rows, step)
		if got != want {
			t.Fatalf("w=%d rows=%d step=%d: got %d want %d", w, rows, step, got, want)
		}
	}
}
