package transform

import (
	"math/rand"
	"testing"
)

// The forward DCT's parity proof at this stage is the combined-pipeline check:
// ForwardDCT then the decoder's byte-exact InverseDCTBlock must reproduce the
// residual within the AV1 fixed-point pipeline's known rounding (the same
// fwd+inv error bound libaom's own unit tests assert, |err| <= 2 at 8-bit
// magnitudes). The bit-exact end-to-end gate lands with the non-lossless
// encoder: there the decoder reconstructs from the same coefficients the
// encoder used, so encoder/decoder recon match exactly regardless of forward
// rounding.

func TestForwardDCT4x4InverseRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	scratch := make([]int32, 16)
	for range 2000 {
		var residual [16]int16
		for i := range residual {
			residual[i] = int16(rng.Intn(2*255+1) - 255) // 8-bit residual span
		}
		var coeff [16]int32
		if err := ForwardDCT4x4(coeff[:], 4, residual[:], 4); err != nil {
			t.Fatalf("forward: %v", err)
		}
		var dst [16]int16
		if err := InverseDCTBlock(dst[:], 4, coeff[:], 4, scratch, Size{Width: 4, Height: 4}); err != nil {
			t.Fatalf("inverse: %v", err)
		}
		for i := range dst {
			diff := int(dst[i]) - int(residual[i])
			if diff < -2 || diff > 2 {
				t.Fatalf("4x4 round-trip error %d at %d:\n in  %v\n out %v\n co  %v", diff, i, residual, dst, coeff)
			}
		}
	}
}

func TestForwardDCT8x8InverseRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	scratch := make([]int32, 64)
	for range 1000 {
		var residual [64]int16
		for i := range residual {
			residual[i] = int16(rng.Intn(2*255+1) - 255)
		}
		var coeff [64]int32
		if err := ForwardDCT8x8(coeff[:], 8, residual[:], 8); err != nil {
			t.Fatalf("forward: %v", err)
		}
		var dst [64]int16
		if err := InverseDCTBlock(dst[:], 8, coeff[:], 8, scratch, Size{Width: 8, Height: 8}); err != nil {
			t.Fatalf("inverse: %v", err)
		}
		for i := range dst {
			diff := int(dst[i]) - int(residual[i])
			if diff < -2 || diff > 2 {
				t.Fatalf("8x8 round-trip error %d at %d", diff, i)
			}
		}
	}
}

// TestForwardDCTDCScale pins the DC gain as a regression guard: a constant
// block transforms to a single DC coefficient with zero AC. The DC magnitudes
// are the AV1 fixed-point pipeline's outputs (per-pass gain 2*cos(pi/4) in Q13
// plus the stage shifts), pinned from the implementation whose correctness the
// inverse round-trip tests establish.
func TestForwardDCTDCScale(t *testing.T) {
	var residual [16]int16
	for i := range residual {
		residual[i] = 100
	}
	var coeff [16]int32
	if err := ForwardDCT4x4(coeff[:], 4, residual[:], 4); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < 16; i++ {
		if coeff[i] != 0 {
			t.Fatalf("AC coeff[%d] = %d, want 0 for constant input", i, coeff[i])
		}
	}
	// 4x4: 100<<2 per sample, two passes of gain 2*cos(pi/4).
	if coeff[0] != 3199 {
		t.Fatalf("4x4 DC = %d, want 3199", coeff[0])
	}

	var residual8 [64]int16
	for i := range residual8 {
		residual8[i] = 100
	}
	var coeff8 [64]int32
	if err := ForwardDCT8x8(coeff8[:], 8, residual8[:], 8); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < 64; i++ {
		if coeff8[i] != 0 {
			t.Fatalf("8x8 AC coeff[%d] = %d, want 0", i, coeff8[i])
		}
	}
	if coeff8[0] != 6404 {
		t.Fatalf("8x8 DC = %d, want 6404", coeff8[0])
	}
}

func TestForwardDCTZeroAlloc(t *testing.T) {
	residual := make([]int16, 64)
	for i := range residual {
		residual[i] = int16(i*3 - 90)
	}
	coeff := make([]int32, 64)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ForwardDCT4x4(coeff[:16], 4, residual[:16], 4)
		_ = ForwardDCT8x8(coeff, 8, residual, 8)
	})
	if allocs != 0 {
		t.Fatalf("forward DCT allocated %v objects/run, want 0", allocs)
	}
}

// TestForwardDCTRectInverseRoundTrip proves the rectangular forward DCTs
// against the decoder's byte-exact inverse within the same fixed-point
// rounding bound the square sizes assert. The rectangular pipeline adds the
// sqrt(2) row scaling, so this also pins the NewSqrt2 rounding.
func TestForwardDCTRectInverseRoundTrip(t *testing.T) {
	type rect struct {
		name string
		w, h int
		fwd  func([]int32, int, []int16, int) error
	}
	rects := []rect{
		{"16x8", 16, 8, ForwardDCT16x8},
		{"8x16", 8, 16, ForwardDCT8x16},
		{"8x4", 8, 4, ForwardDCT8x4},
		{"4x8", 4, 8, ForwardDCT4x8},
	}
	for _, rc := range rects {
		t.Run(rc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(3))
			n := rc.w * rc.h
			scratch := make([]int32, n)
			residual := make([]int16, n)
			coeff := make([]int32, n)
			dst := make([]int16, n)
			for range 1000 {
				for i := range residual {
					residual[i] = int16(rng.Intn(2*255+1) - 255)
				}
				if err := rc.fwd(coeff, rc.h, residual, rc.w); err != nil {
					t.Fatalf("forward: %v", err)
				}
				if err := InverseDCTBlock(dst, rc.w, coeff, rc.h, scratch, Size{Width: uint8(rc.w), Height: uint8(rc.h)}); err != nil {
					t.Fatalf("inverse: %v", err)
				}
				for i := range dst {
					diff := int(dst[i]) - int(residual[i])
					if diff < -2 || diff > 2 {
						t.Fatalf("%s round-trip error %d at %d", rc.name, diff, i)
					}
				}
			}
		})
	}
}

// TestForwardDCTRectDCScale pins the rectangular DC gains (per-pass 2*cos(pi/4)
// gains, the stage shifts, and the sqrt(2) row scaling) as regression guards.
func TestForwardDCTRectDCScale(t *testing.T) {
	check := func(name string, w, h int, fwd func([]int32, int, []int16, int) error, wantDC int32) {
		n := w * h
		residual := make([]int16, n)
		for i := range residual {
			residual[i] = 100
		}
		coeff := make([]int32, n)
		if err := fwd(coeff, h, residual, w); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := 1; i < n; i++ {
			if coeff[i] != 0 {
				t.Fatalf("%s AC coeff[%d] = %d, want 0", name, i, coeff[i])
			}
		}
		if coeff[0] != wantDC {
			t.Fatalf("%s DC = %d, want %d", name, coeff[0], wantDC)
		}
	}
	check("16x8", 16, 8, ForwardDCT16x8, 9057)
	check("8x16", 8, 16, ForwardDCT8x16, 9057)
	check("8x4", 8, 4, ForwardDCT8x4, 4529)
	check("4x8", 4, 8, ForwardDCT4x8, 4529)
}
