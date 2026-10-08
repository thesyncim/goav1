//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// TestRDStatsGosimdBinding confirms the goexperiment.simd init bound the
// residual, RD-statistics and block-error dispatch vars to the Go SIMD kernels
// wherever the CPU supports them.
func TestRDStatsGosimdBinding(t *testing.T) {
	if !gosimdKernelsSupported() {
		t.Skip("Go SIMD kernels unsupported on this CPU")
	}
	cases := []struct {
		name string
		fn   any
		want string
	}{
		{"residualBlockImpl", residualBlockImpl, "residualBlockSIMD"},
		{"rdStatsBlockImpl", rdStatsBlockImpl, "rdStatsBlockSIMD"},
		{"blockErrorImpl", blockErrorImpl, "blockErrorSIMD"},
	}
	for _, tc := range cases {
		name := runtime.FuncForPC(reflect.ValueOf(tc.fn).Pointer()).Name()
		if !strings.Contains(name, tc.want) {
			t.Errorf("%s bound to %q, want %q", tc.name, name, tc.want)
		}
	}
}

func TestResidualBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5D01))
	shapes := []struct{ w, h int }{{4, 4}, {8, 8}, {8, 16}, {12, 4}, {16, 16}, {16, 8}, {24, 3}, {32, 32}, {64, 16}, {40, 2}}
	for _, sh := range shapes {
		for trial := range 200 {
			stride := sh.w + rng.Intn(9)
			predStride := sh.w + rng.Intn(9)
			src := make([]byte, stride*sh.h+8)
			pred := make([]byte, predStride*sh.h+8)
			fillRDStatsBytes(rng, src)
			fillRDStatsBytes(rng, pred)
			srcOff := rng.Intn(4)
			got := make([]int16, sh.w*sh.h)
			want := make([]int16, sh.w*sh.h)
			residualBlockSIMD(got, src, srcOff, stride, pred, predStride, sh.w, sh.h)
			residualBlockPureGo(want, src, srcOff, stride, pred, predStride, sh.w, sh.h)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%dx%d trial %d [%d]: got %d want %d", sh.w, sh.h, trial, i, got[i], want[i])
				}
			}
		}
	}
}

func TestRDStatsBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5D02))
	for _, count := range []int{0, 1, 3, 7, 8, 9, 15, 16, 31, 64, 256, 1024} {
		for trial := range 300 {
			tran := make([]int32, count)
			qcoeff := make([]int16, count)
			fillRDStatsCoeffs(rng, tran, qcoeff, trial)
			step := int32(rng.Intn(29247) + 1)
			ts := uint8(rng.Intn(4))
			if trial%3 == 0 {
				step = 1 << 15
			}
			wd, wc, wr, wz := rdStatsBlockPureGo(tran, qcoeff, count, step, ts)
			gd, gc, gr, gz := rdStatsBlockSIMD(tran, qcoeff, count, step, ts)
			if gd != wd || gc != wc || gr != wr || gz != wz {
				t.Fatalf("count %d trial %d step %d ts %d: got (%d,%d,%d,%v) want (%d,%d,%d,%v)",
					count, trial, step, ts, gd, gc, gr, gz, wd, wc, wr, wz)
			}
		}
	}
}

func TestBlockErrorSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5D03))
	for _, count := range []int{0, 1, 3, 4, 5, 7, 8, 16, 63, 64, 256, 1024} {
		for trial := range 300 {
			coeff := make([]int32, count)
			dq := make([]int32, count)
			for i := range coeff {
				coeff[i] = int32(rng.Uint32())
				dq[i] = int32(rng.Uint32())
				if trial%2 == 0 {
					coeff[i] = int32(rng.Intn(65536) - 32768)
					dq[i] = int32(rng.Intn(65536) - 32768)
				}
			}
			we, ws := blockErrorPureGo(coeff, dq, count)
			ge, gs := blockErrorSIMD(coeff, dq, count)
			if ge != we || gs != ws {
				t.Fatalf("count %d trial %d: got (%d,%d) want (%d,%d)", count, trial, ge, gs, we, ws)
			}
		}
	}
}

func TestRDStatsBlockSIMDNoAlloc(t *testing.T) {
	tran := make([]int32, 256)
	qcoeff := make([]int16, 256)
	for i := range tran {
		tran[i] = int32(i*37 - 4000)
		qcoeff[i] = int16(i%7 - 3)
	}
	if a := testing.AllocsPerRun(100, func() { _, _, _, _ = rdStatsBlockSIMD(tran, qcoeff, 256, 21, 1) }); a != 0 {
		t.Errorf("rdStatsBlockSIMD: %.1f allocs/op, want 0", a)
	}
}

// fillRDStatsBytes fills b with random bytes, with saturated runs so the
// widening edges are reached.
func fillRDStatsBytes(rng *rand.Rand, b []byte) {
	for i := range b {
		switch rng.Intn(4) {
		case 0:
			b[i] = 255
		case 1:
			b[i] = 0
		default:
			b[i] = byte(rng.Intn(256))
		}
	}
}

// fillRDStatsCoeffs fills transform and quantized coefficients. Some trials
// hold all-zero levels, and some hold the int16 extremes.
func fillRDStatsCoeffs(rng *rand.Rand, tran []int32, qcoeff []int16, trial int) {
	for i := range tran {
		switch {
		case trial%5 == 0:
			tran[i] = int32(rng.Intn(1<<20) - 1<<19)
		default:
			tran[i] = int32(rng.Intn(8192) - 4096)
		}
		switch {
		case trial%4 == 0:
			qcoeff[i] = 0
		case trial%4 == 1 && rng.Intn(2) == 0:
			qcoeff[i] = int16(rng.Intn(65536) - 32768)
		default:
			qcoeff[i] = int16(rng.Intn(41) - 20)
		}
	}
}
