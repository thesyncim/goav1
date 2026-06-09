package transform

import (
	"math/rand"
	"testing"
)

// The forward WHT's parity proof is an exact round-trip against the decoder's
// InverseWHT4x4Block (a byte-exact libaom inverse): for any int16 residual,
// ForwardWHT4x4 then InverseWHT4x4Block reproduces the residual bit-for-bit.

func roundTripWHT(t *testing.T, residual *[16]int16) {
	t.Helper()
	var coeff [16]int32
	if err := ForwardWHT4x4(coeff[:], 4, residual[:], 4); err != nil {
		t.Fatalf("forward: %v", err)
	}
	var dst [16]int16
	if err := InverseWHT4x4Block(dst[:], 4, coeff[:], 4, 16); err != nil {
		t.Fatalf("inverse: %v", err)
	}
	if dst != *residual {
		t.Fatalf("round-trip mismatch:\n in    %v\n out   %v\n coeff %v", *residual, dst, coeff)
	}
}

func TestForwardWHT4x4RoundTripRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for range 5000 {
		var residual [16]int16
		for i := range residual {
			// 12-bit residual span ([-4095,4095]) covers every supported depth.
			residual[i] = int16(rng.Intn(2*4095+1) - 4095)
		}
		roundTripWHT(t, &residual)
	}
}

func TestForwardWHT4x4RoundTripExtremes(t *testing.T) {
	cases := [][16]int16{
		{},               // all zero
		{37},             // DC only
		{0: -37},         // negative DC
		fill16(32767),    // max int16, proves no forward/inverse overflow
		fill16(-32768),   // min int16
		altSign16(32767), // alternating extremes
		{1, -1, 1, -1, 2, -2, 2, -2, 3, -3, 3, -3, 4, -4, 4, -4},
	}
	for i := range cases {
		c := cases[i]
		roundTripWHT(t, &c)
	}
}

// TestForwardWHT4x4Strided exercises the strided scatter/gather mapping with
// coeffStride and residualStride larger than the 4x4 block width.
func TestForwardWHT4x4Strided(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const rs, cs = 6, 5
	residual := make([]int16, 4*rs)
	for i := range residual {
		residual[i] = int16(rng.Intn(1024) - 512)
	}
	coeff := make([]int32, 4*cs)
	if err := ForwardWHT4x4(coeff, cs, residual, rs); err != nil {
		t.Fatalf("forward: %v", err)
	}
	dst := make([]int16, 4*rs)
	if err := InverseWHT4x4Block(dst, rs, coeff, cs, 16); err != nil {
		t.Fatalf("inverse: %v", err)
	}
	for r := range 4 {
		for c := range 4 {
			if dst[r*rs+c] != residual[r*rs+c] {
				t.Fatalf("strided mismatch at (%d,%d): got %d want %d", r, c, dst[r*rs+c], residual[r*rs+c])
			}
		}
	}
}

func TestForwardWHT4x4BoundsErrors(t *testing.T) {
	coeff := make([]int32, 16)
	residual := make([]int16, 16)
	if err := ForwardWHT4x4(coeff, 3, residual, 4); err == nil {
		t.Fatal("expected error for coeffStride < 4")
	}
	if err := ForwardWHT4x4(coeff, 4, residual, 3); err == nil {
		t.Fatal("expected error for residualStride < 4")
	}
	if err := ForwardWHT4x4(coeff[:8], 4, residual, 4); err == nil {
		t.Fatal("expected error for short coeff")
	}
	if err := ForwardWHT4x4(coeff, 4, residual[:8], 4); err == nil {
		t.Fatal("expected error for short residual")
	}
}

func TestForwardWHT4x4ZeroAlloc(t *testing.T) {
	residual := make([]int16, 16)
	for i := range residual {
		residual[i] = int16(i*7 - 50)
	}
	coeff := make([]int32, 16)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ForwardWHT4x4(coeff, 4, residual, 4)
	})
	if allocs != 0 {
		t.Fatalf("ForwardWHT4x4 allocated %v objects/run, want 0", allocs)
	}
}

func fill16(v int16) [16]int16 {
	var a [16]int16
	for i := range a {
		a[i] = v
	}
	return a
}

func altSign16(v int16) [16]int16 {
	var a [16]int16
	for i := range a {
		if i&1 == 0 {
			a[i] = v
		} else {
			a[i] = -v
		}
	}
	return a
}
