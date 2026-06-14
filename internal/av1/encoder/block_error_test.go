package encoder

import (
	"math/rand"
	"testing"
)

func TestBlockErrorImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(6201))
	for _, count := range []int{0, 1, 7, 8, 16, 32, 64, 128, 256, 1024} {
		coeff := make([]int32, max(count, 1))
		dqcoeff := make([]int32, max(count, 1))
		for trial := 0; trial < 200; trial++ {
			for i := range count {
				// Include values outside int16 range to pin SVT's
				// load_tran_low_to_s16q narrowing behavior.
				coeff[i] = int32(rng.Intn(1<<22) - 1<<21)
				dqcoeff[i] = int32(rng.Intn(1<<22) - 1<<21)
			}
			wantErr, wantSSZ := blockErrorPureGo(coeff, dqcoeff, count)
			gotErr, gotSSZ := blockError(coeff, dqcoeff, count)
			if gotErr != wantErr || gotSSZ != wantSSZ {
				t.Fatalf("count=%d trial=%d: got err=%d ssz=%d want err=%d ssz=%d",
					count, trial, gotErr, gotSSZ, wantErr, wantSSZ)
			}
		}
	}
}
