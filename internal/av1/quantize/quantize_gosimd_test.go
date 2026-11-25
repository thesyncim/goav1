//go:build goexperiment.simd && (arm64 || amd64) && !purego

package quantize

import (
	"math"
	"math/bits"
	"math/rand"
	"testing"
)

// dequantSIMDCoeffDomain covers every int16 coefficient value once, followed
// by a repeat of the first seven values so every n%8 tail length is exercised
// after a full-width body.
func dequantSIMDCoeffDomain() []int16 {
	coeff := make([]int16, 65536+8)
	for i := range 65536 {
		coeff[i] = int16(i)
	}
	for i := range 8 {
		coeff[65536+i] = int16(i)
	}
	return coeff
}

// TestDequantColumnSIMDMatchesPureGoExhaustive proves the SIMD dequant kernel
// bit-exact with dequantColumnPureGo over the full int16 coefficient domain,
// every tx scale and bit-depth clamp, and scales spanning the table range plus
// values whose products wrap int32 (only the low 24 bits matter).
func TestDequantColumnSIMDMatchesPureGoExhaustive(t *testing.T) {
	if !dequantSIMDSupported() {
		t.Skip("SIMD dequant kernel not supported on this CPU")
	}
	coeff := dequantSIMDCoeffDomain()
	scales := []int32{1, 4, 9, 38, 44, 74, 88, 1828, 7312, 29247, 65535, 1<<20 + 3}
	bounds := [][2]int32{
		{-1 << 15, 1<<15 - 1},
		{-1 << 17, 1<<17 - 1},
		{-1 << 19, 1<<19 - 1},
		{math.MinInt32, math.MaxInt32},
	}
	for _, n := range []int{65536, 65536 + 1, 65536 + 3, 65536 + 7, 65536 + 8} {
		c := coeff[:n]
		for _, scale := range scales {
			for txScale := uint8(0); txScale <= 2; txScale++ {
				for _, b := range bounds {
					want := make([]int32, n)
					got := make([]int32, n)
					dequantColumnPureGo(want, c, scale, txScale, b[0], b[1])
					dequantColumnSIMD(got, c, scale, txScale, b[0], b[1])
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("n=%d scale=%d txScale=%d bounds=%v coeff[%d]=%d: got %d want %d",
								n, scale, txScale, b, i, c[i], got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestDequantColumnSIMDZeroAlloc(t *testing.T) {
	if !dequantSIMDSupported() {
		t.Skip("SIMD dequant kernel not supported on this CPU")
	}
	coeff := dequantSIMDCoeffDomain()[:1024]
	dst := make([]int32, 1024)
	allocs := testing.AllocsPerRun(200, func() {
		dequantColumnSIMD(dst, coeff, 88, 1, -32768, 32767)
	})
	if allocs != 0 {
		t.Fatalf("dequantColumnSIMD allocs = %v, want 0", allocs)
	}
}

// bSIMDQuantBParams builds the scalar quantize_b parameters for one step, the
// same way the scalar block path does, as the oracle for the vector kernel.
// zbinFactor comes from the DC step for both coefficient classes, as in
// QuantizeBlockScaledB.
func bSIMDQuantBParams(step int32, zbinFactor int32, txScale uint8) quantBParams {
	quant, shift := invertQuant(step)
	return quantBParams{
		zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*step, 7), txScale),
		round: roundPowerOfTwo((48*step)>>7, txScale),
		quant: quant,
		shift: shift,
	}
}

// bSIMDCoeffs builds a block covering the quantize_b branch points: zero, the
// zbin deadzone edge, the 32767 saturation edge, the int32 extremes (INT32_MIN
// included), and log-uniform random magnitudes of both signs.
func bSIMDCoeffs(rng *rand.Rand, count int, dc, ac quantBParams) []int32 {
	coeff := make([]int32, count)
	for i := range coeff {
		mag := rng.Int63n(int64(1) << uint(rng.Intn(32)))
		v := int32(mag)
		if rng.Intn(2) == 0 {
			v = -v
		}
		coeff[i] = v
	}
	satEdge := 32767 - ac.round
	extremes := []int32{
		0, 1, -1,
		ac.zbin - 1, ac.zbin, -ac.zbin, -ac.zbin + 1, ac.zbin + 1,
		dc.zbin - 1, dc.zbin, -dc.zbin, dc.zbin + 1,
		satEdge - 1, satEdge, satEdge + 1, -satEdge,
		32766, 32767, 32768, -32767, -32768, 65535, 65536,
		1 << 20, -(1 << 20), 1<<30 - 1, -(1 << 30),
		math.MaxInt32, math.MaxInt32 - 1, math.MinInt32, math.MinInt32 + 1,
	}
	// Scatter the edges through the block so every lane position sees them.
	for range count / 2 {
		coeff[rng.Intn(count)] = extremes[rng.Intn(len(extremes))]
	}
	return coeff
}

// TestQuantizeBBlockSIMDMatchesScalar proves the SIMD quantize_b kernel
// bit-exact with quantizeScalarB over every square block size, every tx scale,
// quantizer steps across the table range, and coefficients at the zbin,
// saturation, and int32 edges. A refusal is only accepted when the step's
// l = log2(step) is below txScale, the one case the kernel defers to scalar.
func TestQuantizeBBlockSIMDMatchesScalar(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	rng := rand.New(rand.NewSource(0xb51dfa))
	for _, n := range []int{4, 8, 16, 32} {
		for ts := uint8(0); ts <= 2; ts++ {
			for trial := 0; trial < 300; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(29244)),
					AC: int32(4 + rng.Intn(29244)),
				}
				if trial == 0 {
					q = Quantizer{DC: 4, AC: 4}
				}
				zbinFactor := int32(80)
				if q.DC < 148 {
					zbinFactor = 84
				}
				l := 31 - bits.LeadingZeros32(uint32(q.AC))
				dc := bSIMDQuantBParams(q.DC, zbinFactor, ts)
				ac := bSIMDQuantBParams(q.AC, zbinFactor, ts)
				coeff := bSIMDCoeffs(rng, n*n, dc, ac)
				got := make([]int16, n*n)
				if l < int(ts) {
					if quantizeBBlockSIMD(got, coeff, n, q, ts) {
						t.Fatalf("n=%d ts=%d q=%+v: accepted l=%d < txScale", n, ts, q, l)
					}
					continue
				}
				if !quantizeBBlockSIMD(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d q=%+v trial %d: kernel refused in-range input", n, ts, q, trial)
				}
				for i := range coeff {
					p := &ac
					if i == 0 {
						p = &dc
					}
					if want := quantizeScalarB(coeff[i], p, ts); got[i] != want {
						t.Fatalf("n=%d ts=%d q=%+v trial %d: qcoeff[%d] simd %d want %d (coeff %d)",
							n, ts, q, trial, i, got[i], want, coeff[i])
					}
				}
			}
		}
	}
}

// TestQuantizeBBlockSIMDZeroBinGroups covers the kernel's zero-bin test on
// sparse blocks, where most eight-coefficient groups lie wholly below the AC
// zero bin (magnitudes up to zbin-1 of both signs) and the others carry one
// lane at the bin's edge or an int32 extreme. Levels and the end of block
// must match quantizeScalarB, with and without iscan.
func TestQuantizeBBlockSIMDZeroBinGroups(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	rng := rand.New(rand.NewSource(0x2b1))
	for _, n := range []int{4, 8, 16, 32} {
		for ts := uint8(0); ts <= 2; ts++ {
			for trial := range 200 {
				q := Quantizer{DC: int32(4 + rng.Intn(2000)), AC: int32(4 + rng.Intn(2000))}
				zbinFactor := int32(80)
				if q.DC < 148 {
					zbinFactor = 84
				}
				if 31-bits.LeadingZeros32(uint32(q.AC)) < int(ts) {
					continue
				}
				dc := bSIMDQuantBParams(q.DC, zbinFactor, ts)
				ac := bSIMDQuantBParams(q.AC, zbinFactor, ts)
				coeff := make([]int32, n*n)
				for i := range coeff {
					if rng.Intn(3) == 0 {
						coeff[i] = rng.Int31n(2*ac.zbin-1) - (ac.zbin - 1)
					}
				}
				edges := []int32{ac.zbin, -ac.zbin, ac.zbin - 1, -(ac.zbin - 1), dc.zbin, -dc.zbin,
					math.MinInt32, math.MaxInt32, 32767, -32768}
				for range trial % 4 {
					coeff[rng.Intn(len(coeff))] = edges[rng.Intn(len(edges))]
				}
				want := make([]int16, n*n)
				for i := range coeff {
					p := &ac
					if i == 0 {
						p = &dc
					}
					want[i] = quantizeScalarB(coeff[i], p, ts)
				}
				iscan := make([]int16, n*n)
				for i, v := range rng.Perm(n*n - 1) {
					iscan[i+1] = int16(v + 1)
				}
				wantEOB := 0
				for i, v := range want {
					if v != 0 {
						wantEOB = max(wantEOB, int(iscan[i])+1)
					}
				}
				for _, sc := range [][]int16{nil, iscan} {
					got := make([]int16, n*n)
					for i := range got {
						got[i] = 0x5a5a
					}
					eob, ok := quantizeBBlockEOBSIMD(got, coeff, n, q, ts, sc)
					if !ok {
						t.Fatalf("n=%d ts=%d q=%+v: kernel refused", n, ts, q)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("n=%d ts=%d q=%+v iscan=%v: qcoeff[%d] simd %d want %d (coeff %d)",
								n, ts, q, sc != nil, i, got[i], want[i], coeff[i])
						}
					}
					if sc != nil && eob != wantEOB {
						t.Fatalf("n=%d ts=%d q=%+v: eob %d want %d", n, ts, q, eob, wantEOB)
					}
				}
			}
		}
	}
}

// TestQuantizeBBlockSIMDRejectsUnsupportedShapes checks the guards that keep
// the vector lane math exact: a block whose count is not a multiple of 16 and a
// step above quantizeVectorMaxStep both defer to the scalar rule.
func TestQuantizeBBlockSIMDRejectsUnsupportedShapes(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	coeff := make([]int32, 16)
	got := make([]int16, 16)
	if quantizeBBlockSIMD(got, coeff, 3, Quantizer{DC: 8, AC: 8}, 0) {
		t.Fatal("accepted a block whose count is not a multiple of 16")
	}
	if quantizeBBlockSIMD(got, coeff, 4, Quantizer{DC: 8, AC: quantizeVectorMaxStep + 1}, 0) {
		t.Fatal("accepted an AC step above quantizeVectorMaxStep")
	}
}

// TestQuantizeBlockSIMDZeroAlloc asserts the fp and quantize-b vector block
// kernels, which both arches dispatch, do not allocate on the hot path.
func TestQuantizeBlockSIMDZeroAlloc(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, 16*16)
	for i := range coeff {
		coeff[i] = int32(i*37 - 4096)
	}
	qcoeff := make([]int16, 16*16)
	if allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeFPBlockSIMD(qcoeff, coeff, 16, q, 1) {
			t.Fatal("quantizeFPBlockSIMD refused")
		}
	}); allocs != 0 {
		t.Fatalf("quantizeFPBlockSIMD allocated %f objects/run, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeBBlockSIMD(qcoeff, coeff, 16, q, 1) {
			t.Fatal("quantizeBBlockSIMD refused")
		}
	}); allocs != 0 {
		t.Fatalf("quantizeBBlockSIMD allocated %f objects/run, want 0", allocs)
	}
}

// fpSIMDCoeffs builds a block of coefficients that covers the boundaries the
// fp rule branches on: zero, the skip threshold (|c| << (1+ts) against the
// dequant step), the 32767 clamp, the int32 extremes, and log-uniform random
// magnitudes of both signs.
func fpSIMDCoeffs(rng *rand.Rand, count int, dequant int32, ts uint8) []int32 {
	coeff := make([]int32, count)
	for i := range coeff {
		mag := rng.Int63n(int64(1) << uint(rng.Intn(32)))
		v := int32(mag)
		if rng.Intn(2) == 0 {
			v = -v
		}
		coeff[i] = v
	}
	thresh := dequant >> (1 + ts)
	extremes := []int32{
		0, 1, -1, thresh - 1, thresh, -thresh, -thresh + 1, thresh + 1,
		32766, 32767, 32768, -32767, -32768, -32769,
		65535, 65536, 1 << 20, -(1 << 20), 1<<30 - 1, -(1 << 30),
		math.MaxInt32, math.MinInt32, math.MinInt32 + 1, math.MaxInt32 - 1,
		// Magnitudes whose rounding add wraps int32: the keep test must drop them.
		math.MaxInt32 - 1<<14, math.MaxInt32 - 1<<13 - 7, -(math.MaxInt32 - 1<<14),
	}
	// Scatter the boundary values through the block so every vector lane
	// position sees them, including the DC slot.
	for range count / 2 {
		coeff[rng.Intn(count)] = extremes[rng.Intn(len(extremes))]
	}
	return coeff
}

// TestQuantizeFPBlockSIMDMatchesScalar proves the SIMD fp kernel bit-exact
// with quantizeScalarFP over every square block size, every tx scale, quantizer
// steps across the table range, and coefficients at the extremes of int32.
func TestQuantizeFPBlockSIMDMatchesScalar(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	rng := rand.New(rand.NewSource(0x51dfa))
	for _, n := range []int{4, 8, 16, 32} {
		for ts := uint8(0); ts <= 2; ts++ {
			for trial := 0; trial < 300; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(29244)),
					AC: int32(4 + rng.Intn(29244)),
				}
				if trial == 0 {
					q = Quantizer{DC: 4, AC: 4}
				}
				coeff := fpSIMDCoeffs(rng, n*n, q.AC, ts)
				got := make([]int16, n*n)
				if !quantizeFPBlockSIMD(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d q=%+v trial %d: kernel refused in-range input", n, ts, q, trial)
				}
				quantDC := int64(1<<16) / int64(q.DC)
				roundDC := roundPowerOfTwo((64*q.DC)>>7, ts)
				quantAC := int64(1<<16) / int64(q.AC)
				roundAC := roundPowerOfTwo((64*q.AC)>>7, ts)
				for i := range coeff {
					want := quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, ts)
					if i == 0 {
						want = quantizeScalarFP(coeff[i], q.DC, quantDC, roundDC, ts)
					}
					if got[i] != want {
						t.Fatalf("n=%d ts=%d q=%+v trial %d: qcoeff[%d] simd %d want %d (coeff %d)",
							n, ts, q, trial, i, got[i], want, coeff[i])
					}
				}
			}
		}
	}
}

// TestQuantizeFPBlockSIMDRejectsOutOfRangeSteps keeps the guard that protects
// the int32 lane math: steps below 4 (quant above 1<<14) are refused so the
// scalar rule handles them.
func TestQuantizeFPBlockSIMDRejectsOutOfRangeSteps(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	coeff := make([]int32, 16)
	got := make([]int16, 16)
	if quantizeFPBlockSIMD(got, coeff, 4, Quantizer{DC: 3, AC: 3}, 0) {
		t.Fatal("accepted AC/DC step 3 (quant above 1<<14)")
	}
	if quantizeFPBlockSIMD(got, coeff, 3, Quantizer{DC: 8, AC: 8}, 0) {
		t.Fatal("accepted a block whose count is not a multiple of 16")
	}
}
