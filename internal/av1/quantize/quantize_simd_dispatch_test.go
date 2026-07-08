package quantize

import (
	"math/rand"
	"testing"
)

func TestQuantizeBlockImplMatchesScalar(t *testing.T) {
	if quantizeBlockImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(67))
	edges := []int32{
		-(1 << 30), -1 << 20, -32768, -32767, -1, 0, 1, 32767, 32768, 1 << 20, 1 << 30,
	}
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1, 2} {
			for trial := 0; trial < 200; trial++ {
				q := Quantizer{
					DC: int32(1 + rng.Intn(8000)),
					AC: int32(1 + rng.Intn(8000)),
				}
				coeff := make([]int32, n*n)
				for i := range coeff {
					if i < len(edges) {
						coeff[i] = edges[i]
					} else {
						coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
					}
				}
				want := make([]int16, n*n)
				got := make([]int16, n*n)
				for i := range coeff {
					scale := q.AC
					if i == 0 {
						scale = q.DC
					}
					want[i] = quantizeScalar(coeff[i], scale, ts)
				}
				if !quantizeBlockImpl(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d trial=%d q=%+v coeff0=%d coeff1=%d: kernel refused", n, ts, trial, q, coeff[0], coeff[1])
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("n=%d ts=%d trial=%d q[%d] impl %d want %d coeff=%d dc=%v q=%+v",
							n, ts, trial, i, got[i], want[i], coeff[i], i == 0, q)
					}
				}
			}
		}
	}
}

func TestQuantizeFPNoQMatrixImplMatchesScalar(t *testing.T) {
	if quantizeFPNoQMatrixImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(71))
	edges := []int32{
		minInt32, -(1 << 30), -32768, -32767, -1, 0, 1, 32767, 32768, 1 << 30,
	}
	for _, count := range []int{1, 4, 16, 64} {
		for _, logScale := range []uint8{0, 1, 3, 15} {
			for trial := 0; trial < 100; trial++ {
				q := FPQuantizer{
					Quant:    [2]int16{int16(1 + rng.Intn(2048)), int16(1 + rng.Intn(2048))},
					Dequant:  [2]int16{int16(1 + rng.Intn(2048)), int16(1 + rng.Intn(2048))},
					Round:    [2]int16{int16(rng.Intn(2048)), int16(rng.Intn(2048))},
					LogScale: logScale,
				}
				coeff := make([]int32, count)
				for i := range coeff {
					if i < len(edges) {
						coeff[i] = edges[i]
					} else {
						coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
					}
				}
				scan := make([]int16, count)
				for i := range scan {
					scan[i] = int16(i)
				}
				wantQ := make([]int32, count)
				wantDQ := make([]int32, count)
				old := quantizeFPNoQMatrixImpl
				quantizeFPNoQMatrixImpl = nil
				wantEOB, err := QuantizeFPNoQMatrix(wantQ, wantDQ, coeff, scan, q)
				quantizeFPNoQMatrixImpl = old
				if err != nil {
					t.Fatal(err)
				}

				gotQ := make([]int32, count)
				gotDQ := make([]int32, count)
				gotEOB, ok := quantizeFPNoQMatrixImpl(gotQ, gotDQ, coeff, scan, q)
				if !ok {
					t.Fatalf("count=%d logScale=%d: kernel refused identity scan", count, logScale)
				}
				if gotEOB != wantEOB {
					t.Fatalf("count=%d logScale=%d trial=%d eob=%d want %d", count, logScale, trial, gotEOB, wantEOB)
				}
				for i := range wantQ {
					if gotQ[i] != wantQ[i] || gotDQ[i] != wantDQ[i] {
						t.Fatalf("count=%d logScale=%d trial=%d coeff[%d]=%d q/dq=%d/%d want %d/%d q=%+v",
							count, logScale, trial, i, coeff[i], gotQ[i], gotDQ[i], wantQ[i], wantDQ[i], q)
					}
				}
			}
		}
	}
}

func TestQuantizeFPNoQMatrixImplRejectsNonIdentityScan(t *testing.T) {
	if quantizeFPNoQMatrixImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	q := FPQuantizer{
		Quant:   [2]int16{840, 704},
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
	}
	qcoeff := make([]int32, 4)
	dqcoeff := make([]int32, 4)
	coeff := []int32{-449, 624, -14, 24}
	scan := []int16{0, 2, 1, 3}
	if _, ok := quantizeFPNoQMatrixImpl(qcoeff, dqcoeff, coeff, scan, q); ok {
		t.Fatal("kernel accepted non-identity scan")
	}
}

func TestQuantizeVectorImplsAllocs(t *testing.T) {
	if quantizeBlockImpl == nil || quantizeFPBlockImpl == nil || quantizeFPNoQMatrixImpl == nil {
		t.Skip("missing vector kernel on this architecture")
	}
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, 16*16)
	for i := range coeff {
		coeff[i] = int32(i*37 - 4096)
	}
	qcoeff16 := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeBlockImpl(qcoeff16, coeff, 16, q, 1) {
			t.Fatal("quantizeBlockImpl refused")
		}
		if !quantizeFPBlockImpl(qcoeff16, coeff, 16, q, 1) {
			t.Fatal("quantizeFPBlockImpl refused")
		}
	})
	if allocs != 0 {
		t.Fatalf("block vector quantizers allocated %f objects/run, want 0", allocs)
	}

	fpq := FPQuantizer{
		Quant:    [2]int16{840, 704},
		Dequant:  [2]int16{78, 93},
		Round:    [2]int16{39, 46},
		LogScale: 1,
	}
	qcoeff32 := make([]int32, 16*16)
	dqcoeff32 := make([]int32, 16*16)
	scan := make([]int16, 16*16)
	for i := range scan {
		scan[i] = int16(i)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		if _, ok := quantizeFPNoQMatrixImpl(qcoeff32, dqcoeff32, coeff, scan, fpq); !ok {
			t.Fatal("quantizeFPNoQMatrixImpl refused")
		}
	})
	if allocs != 0 {
		t.Fatalf("fp no-qmatrix vector quantizer allocated %f objects/run, want 0", allocs)
	}
}
