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
		tol  int
	}
	// The 32-point rectangles run the deeper {2,-4,0} shift pipeline, which
	// discards two more fixed-point bits mid-pass than the small rectangles;
	// one extra unit of round-trip noise is the corresponding bound.
	rects := []rect{
		{"16x8", 16, 8, ForwardDCT16x8, 2},
		{"8x16", 8, 16, ForwardDCT8x16, 2},
		{"8x4", 8, 4, ForwardDCT8x4, 2},
		{"4x8", 4, 8, ForwardDCT4x8, 2},
		{"32x16", 32, 16, ForwardDCT32x16, 3},
		{"16x32", 16, 32, ForwardDCT16x32, 3},
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
					if diff < -rc.tol || diff > rc.tol {
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
	check("32x16", 32, 16, ForwardDCT32x16, 9057)
	check("16x32", 16, 32, ForwardDCT16x32, 9057)
	check("8x16", 8, 16, ForwardDCT8x16, 9057)
	check("8x4", 8, 4, ForwardDCT8x4, 4529)
	check("4x8", 4, 8, ForwardDCT4x8, 4529)
}

func TestForwardDCT64ExtentDCScale(t *testing.T) {
	check := func(name string, w, h int, fwd func([]int32, int, []int16, int) error, wantDC int32) {
		coeffSize := adjustedScanSize(Size{Width: uint8(w), Height: uint8(h)})
		coeffW := int(coeffSize.Width)
		coeffH := int(coeffSize.Height)
		residual := make([]int16, w*h)
		for i := range residual {
			residual[i] = 100
		}
		coeff := make([]int32, coeffW*coeffH)
		if err := fwd(coeff, coeffH, residual, w); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := 1; i < len(coeff); i++ {
			if coeff[i] != 0 {
				t.Fatalf("%s AC coeff[%d]=%d want 0", name, i, coeff[i])
			}
		}
		if coeff[0] != wantDC {
			t.Fatalf("%s DC=%d want %d", name, coeff[0], wantDC)
		}
	}
	check("64x64", 64, 64, ForwardDCT64x64, 12806)
	check("32x64", 32, 64, ForwardDCT32x64, 9056)
	check("64x32", 64, 32, ForwardDCT64x32, 9056)
}

func TestForwardDCT64ExtentDispatchMatchesForwardBlock(t *testing.T) {
	cases := []struct {
		name string
		size Size
		fwd  func([]int32, int, []int16, int) error
	}{
		{name: "64x64", size: Size{Width: 64, Height: 64}, fwd: ForwardDCT64x64},
		{name: "32x64", size: Size{Width: 32, Height: 64}, fwd: ForwardDCT32x64},
		{name: "64x32", size: Size{Width: 64, Height: 32}, fwd: ForwardDCT64x32},
	}
	rng := rand.New(rand.NewSource(6400))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			width := int(tc.size.Width)
			height := int(tc.size.Height)
			coeffSize := adjustedScanSize(tc.size)
			coeffW := int(coeffSize.Width)
			coeffH := int(coeffSize.Height)
			residual := make([]int16, width*height)
			for i := range residual {
				residual[i] = int16(rng.Intn(511) - 255)
			}
			got := make([]int32, coeffW*coeffH)
			want := make([]int32, coeffW*coeffH)
			scratch := make([]int32, width*height)
			if err := ForwardBlock(got, coeffH, residual, width, scratch, tc.size, TypeDCTDCT); err != nil {
				t.Fatalf("ForwardBlock: %v", err)
			}
			if err := tc.fwd(want, coeffH, residual, width); err != nil {
				t.Fatalf("specialized: %v", err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestForwardDCT64ExtentZeroAlloc(t *testing.T) {
	var residual [64 * 64]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [32 * 32]int32
	allocs := testing.AllocsPerRun(20, func() {
		_ = ForwardDCT64x64(coeff[:], 32, residual[:], 64)
		_ = ForwardDCT32x64(coeff[:], 32, residual[:32*64], 32)
		_ = ForwardDCT64x32(coeff[:], 32, residual[:64*32], 64)
	})
	if allocs != 0 {
		t.Fatalf("forward DCT64 extent allocated %v objects/run, want 0", allocs)
	}
}
