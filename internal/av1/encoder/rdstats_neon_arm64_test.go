//go:build arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

func TestResidualBlockNEONParity(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	shapes := []struct{ w, h int }{
		{4, 4}, {8, 4}, {4, 8}, {8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 32},
	}
	for _, sh := range shapes {
		stride := sh.w + 13
		predStride := sh.w + 5
		src := make([]byte, stride*(sh.h+2)+7)
		pred := make([]byte, predStride*sh.h)
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		off := stride + 3
		want := make([]int16, sh.w*sh.h)
		got := make([]int16, sh.w*sh.h)
		residualBlockPureGo(want, src, off, stride, pred, predStride, sh.w, sh.h)
		residualBlockNEON(got, src, off, stride, pred, predStride, sh.w, sh.h)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("%dx%d residual[%d]: got %d want %d", sh.w, sh.h, i, got[i], want[i])
			}
		}
	}
}

func TestRDStatsBlockNEONParity(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	counts := []int{16, 32, 64, 128, 256, 1024}
	steps := []int32{4, 8, 21, 1365, 1828}
	for _, count := range counts {
		for _, step := range steps {
			for ts := uint8(0); ts <= 2; ts++ {
				tran := make([]int32, count)
				qcoeff := make([]int16, count)
				for i := range tran {
					tran[i] = int32(rng.Intn(1<<20) - 1<<19)
					switch rng.Intn(3) {
					case 0:
						qcoeff[i] = 0
					case 1:
						qcoeff[i] = int16(rng.Intn(64) - 32)
					default:
						qcoeff[i] = int16(rng.Intn(1<<16) - 1<<15)
					}
				}
				wantSkip, wantCode, wantRate, wantZero := rdStatsBlockPureGo(tran, qcoeff, count, step, ts)
				gotSkip, gotCode, gotRate, gotZero := rdStatsBlockNEON(tran, qcoeff, count, step, ts)
				if wantSkip != gotSkip || wantCode != gotCode || wantRate != gotRate || wantZero != gotZero {
					t.Fatalf("count=%d step=%d ts=%d: got (%d,%d,%d,%v) want (%d,%d,%d,%v)",
						count, step, ts, gotSkip, gotCode, gotRate, gotZero, wantSkip, wantCode, wantRate, wantZero)
				}
			}
		}
	}
	// All-zero coefficients report the skip flag.
	tran := make([]int32, 64)
	qcoeff := make([]int16, 64)
	for i := range tran {
		tran[i] = int32(rng.Intn(2048) - 1024)
	}
	_, _, _, zero := rdStatsBlockNEON(tran, qcoeff, 64, 21, 1)
	if !zero {
		t.Fatal("all-zero qcoeff must report allZero")
	}
}
