package encoder

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestReconCopyKernelsMatchPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const stride = 80
	for _, sz := range [][2]int{{4, 4}, {8, 8}, {8, 16}, {16, 8}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 16}, {64, 64}} {
		w, h := sz[0], sz[1]
		pred := make([]byte, w*h+3)
		rng.Read(pred)
		for trial := range 40 {
			px, py := rng.Intn(stride-w+1), rng.Intn(4)
			want := make([]byte, stride*(py+h+1))
			rng.Read(want)
			got := bytes.Clone(want)
			copyPredToReconBlockPureGo(want, pred, w, stride, px, py, w, h)
			copyPredToReconBlock(got, pred, w, stride, px, py, w, h)
			if !bytes.Equal(got, want) {
				t.Fatalf("copy %dx%d trial %d differs", w, h, trial)
			}
			for _, r := range []int16{int16(rng.Intn(801) - 400), 0, 1, -1, 255, -255, 256, -256, 32767, -32768} {
				rng.Read(want)
				copy(got, want)
				addConstantResidualToPredReconBlockPureGo(want, pred, w, stride, px, py, w, h, r)
				addConstantResidualToPredReconBlock(got, pred, w, stride, px, py, w, h, r)
				if !bytes.Equal(got, want) {
					t.Fatalf("add-constant %dx%d r=%d trial %d differs", w, h, r, trial)
				}
			}
		}
	}
}
