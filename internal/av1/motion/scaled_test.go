// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import "testing"

func TestNewScaleFactorsIdentity(t *testing.T) {
	sf, err := NewScaleFactors(1280, 720, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	if !sf.Identity {
		t.Fatalf("expected identity scaling, got Identity=false")
	}
	if got, want := sf.XScaleFP, int32(1<<RefScaleShift); got != want {
		t.Errorf("XScaleFP=%d want %d", got, want)
	}
	if got, want := sf.YScaleFP, int32(1<<RefScaleShift); got != want {
		t.Errorf("YScaleFP=%d want %d", got, want)
	}
	if got, want := sf.XStepQN, int32(1<<ScaleSubpelBits); got != want {
		t.Errorf("XStepQN=%d want %d", got, want)
	}
	if got, want := sf.YStepQN, int32(1<<ScaleSubpelBits); got != want {
		t.Errorf("YStepQN=%d want %d", got, want)
	}
}

func TestNewScaleFactorsSVCEnhancementUpscale(t *testing.T) {
	// libaom SVC L2T1: S=1 enhancement layer (1280x720) references the S=0
	// base layer (640x360). For the enhancement-layer block consuming a
	// half-size base reference, x_scale_fp == 1/2 in Q14.
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	if sf.Identity {
		t.Fatalf("expected non-identity scaling for SVC enhancement layer")
	}
	wantScale := int32(((640 << RefScaleShift) + 1280/2) / 1280)
	if sf.XScaleFP != wantScale {
		t.Errorf("XScaleFP=%d want %d", sf.XScaleFP, wantScale)
	}
	if sf.YScaleFP != wantScale {
		t.Errorf("YScaleFP=%d want %d", sf.YScaleFP, wantScale)
	}
	wantStep := int32(((640 << ScaleSubpelBits) + 1280/2) / 1280)
	if sf.XStepQN != wantStep {
		t.Errorf("XStepQN=%d want %d", sf.XStepQN, wantStep)
	}
	if sf.YStepQN != wantStep {
		t.Errorf("YStepQN=%d want %d", sf.YStepQN, wantStep)
	}
	// For an exact 2:1 ratio the step is exactly 0.5 in Q10.
	if sf.XStepQN != ScaleSubpelScale/2 {
		t.Errorf("XStepQN=%d, expected exact 0.5 Q10 step=%d", sf.XStepQN, ScaleSubpelScale/2)
	}
}

func TestNewScaleFactorsRejectsOutOfRange(t *testing.T) {
	tests := []struct {
		name                                       string
		refW, refH, curW, curH int
	}{
		{"zero width", 0, 360, 640, 360},
		{"zero height", 640, 0, 640, 360},
		{"zero cur width", 640, 360, 0, 360},
		{"zero cur height", 640, 360, 640, 0},
		// Upscale > 16x (ref tiny, cur huge): 16 * 32 < 1280  => 512 < 1280 → reject.
		{"upscale beyond 16x width", 32, 360, 1280, 720},
		{"upscale beyond 16x height", 640, 32, 1280, 1280},
		// Downscale > 2x (ref huge, cur small): cur*2 < ref.
		{"downscale beyond 2x width", 2000, 360, 640, 360},
		{"downscale beyond 2x height", 640, 1600, 640, 720},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewScaleFactors(tc.refW, tc.refH, tc.curW, tc.curH); err == nil {
				t.Fatalf("NewScaleFactors(%d,%d,%d,%d): expected error, got nil",
					tc.refW, tc.refH, tc.curW, tc.curH)
			}
		})
	}
}

func TestScaledBlockOriginIdentityMatchesSameSize(t *testing.T) {
	// For identity scaling the scaled origin must agree with the existing
	// same-size ReferenceOriginSubsampled path: same integer-pel reference
	// origin and same Q4 sub-pel fractional offsets.
	sf, err := NewScaleFactors(1280, 720, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	cases := []struct {
		name string
		mv   Vector
	}{
		{"zero", Vector{}},
		{"integer +1 col", Vector{Col: 1 << SubpelBits}},
		{"integer -2 row", Vector{Row: -2 << SubpelBits}},
		{"half-pel col", Vector{Col: 4}},          // 4/8 sample
		{"quarter-pel row", Vector{Row: 2}},       // 2/8 sample
		{"eighth-pel both", Vector{Row: 1, Col: 3}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(40, 24, tc.mv, false, false)
			if err != nil {
				t.Fatalf("ScaledBlockOrigin: %v", err)
			}
			if xStep != ScaleSubpelScale {
				t.Errorf("xStep=%d want %d", xStep, ScaleSubpelScale)
			}
			if yStep != ScaleSubpelScale {
				t.Errorf("yStep=%d want %d", yStep, ScaleSubpelScale)
			}
			refX, refY, subX, subY, err := ReferenceOriginSubsampled(40, 24, tc.mv, false, false)
			if err != nil {
				t.Fatalf("ReferenceOriginSubsampled: %v", err)
			}
			gotIntX, gotSubQ4X := SplitScaledPosition(startX)
			gotIntY, gotSubQ4Y := SplitScaledPosition(startY)
			if int(gotIntX) != refX {
				t.Errorf("intX=%d want %d", gotIntX, refX)
			}
			if int(gotIntY) != refY {
				t.Errorf("intY=%d want %d", gotIntY, refY)
			}
			if gotSubQ4X != subX {
				t.Errorf("subX(Q4)=%d want %d", gotSubQ4X, subX)
			}
			if gotSubQ4Y != subY {
				t.Errorf("subY(Q4)=%d want %d", gotSubQ4Y, subY)
			}
		})
	}
}

func TestScaledBlockOriginUpscaledRef(t *testing.T) {
	// Upscaled current frame (1280x720) referencing half-size base (640x360):
	// each output pel maps to 0.5 source pel, so the stepper is 0.5 Q10 per
	// output pel. The destination block origin (16, 8) plus libaom's
	// av1_scaled_x bias (off = (x_scale_fp - REF_NO_SCALE) * SUBPEL_OFF; for
	// 2:1 that bias is half a source pel) lands the first tap at source
	// integer column 7 with Q4 phase 12, not at "src 8.0 / phase 0". This
	// libaom behaviour is what SVC L2T1 spatial=1 enhancement layers expect.
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(16, 8, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	if xStep != ScaleSubpelScale/2 {
		t.Errorf("xStep=%d want %d", xStep, ScaleSubpelScale/2)
	}
	if yStep != ScaleSubpelScale/2 {
		t.Errorf("yStep=%d want %d", yStep, ScaleSubpelScale/2)
	}
	intX, fracQ4X := SplitScaledPosition(startX)
	intY, fracQ4Y := SplitScaledPosition(startY)
	if intX != 7 || fracQ4X != 12 {
		t.Errorf("startX split=(%d, %d) want (7, 12)", intX, fracQ4X)
	}
	if intY != 3 || fracQ4Y != 12 {
		t.Errorf("startY split=(%d, %d) want (3, 12)", intY, fracQ4Y)
	}
}

func TestScaledBlockOriginCarriesNegativeMV(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	// MV = (-8, 0) in Q3: 1 luma pel to the left. With 2:1 downscale that's
	// 0.5 source pel less. Combined with libaom's av1_scaled_x bias for the
	// 2:1 ratio, the start lands at source integer column 7 with Q4 phase 4
	// — a half-pel offset to the LEFT of phase 12 (the zero-MV case at
	// dstX=16 — see TestScaledBlockOriginUpscaledRef).
	startX, _, _, _, err := sf.ScaledBlockOrigin(16, 8, Vector{Col: -8}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	intX, fracQ4X := SplitScaledPosition(startX)
	if intX != 7 || fracQ4X != 4 {
		t.Errorf("start=(%d, %d) want (7, 4)", intX, fracQ4X)
	}
}

func TestScaledBlockOriginRejectsZeroFactors(t *testing.T) {
	var sf ScaleFactors
	if _, _, _, _, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false); err == nil {
		t.Fatal("expected error for zero ScaleFactors")
	}
}

// TestScaledBlockOriginLibaomBiasFrameTopLeft pins the bias-aware Q10 origin
// for the very first block of an SVC L2T1 enhancement layer (1280x720
// referencing the 640x360 base): a 128x128 mi-(0,0) NEARESTMV block with
// zero MV. Before the libaom bias fix, this case returned (0,0) at xStep=512
// with no SCALE_EXTRA_OFF, which drove the convolver to filter phase 0 at
// each output integer column. libaom's av1_scaled_x bias instead yields a
// negative integer-pel origin with Q4 phase 12 — the half-pel offset the
// convolver expects for non-identity ratios.
func TestScaledBlockOriginLibaomBiasFrameTopLeft(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	if xStep != ScaleSubpelScale/2 || yStep != ScaleSubpelScale/2 {
		t.Fatalf("xStep=%d yStep=%d want %d", xStep, yStep, ScaleSubpelScale/2)
	}
	// libaom for x=0, mv=0, 2:1 downscale:
	//   off = (8192-16384)*8 = -65536
	//   tval = 0 + (-65536) = -65536
	//   pos = ROUND_POWER_OF_TWO_SIGNED_64(-65536, 8) = -256
	//   startX = -256 + SCALE_EXTRA_OFF(=32) = -224
	if want := int64(-224); startX != want {
		t.Errorf("startX=%d want %d", startX, want)
	}
	if want := int64(-224); startY != want {
		t.Errorf("startY=%d want %d", startY, want)
	}
}

// TestScaledBlockOriginLibaomBiasStepsCleanly walks the first eight output
// integer columns of the same SVC L2T1 frame-top-left block and pins each
// Q10 stepper position, integer-pel index, and Q4 sub-pel filter index.
// Phase alternation (12/4/12/4...) and the negative integer-pel origin at
// the left edge are the libaom-faithful behaviour; the AV1 spec convolver
// (av1_convolve_2d_scale_c) requires these for bit-exact output.
func TestScaledBlockOriginLibaomBiasStepsCleanly(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, _, xStep, _, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	cases := []struct {
		x       int
		wantPos int64
		wantInt int64
		wantQ4  int
	}{
		// startX=-224 + n*512, then >>10 / phase4 derivation:
		// x=0: -224 (intX=-1, phase=12)
		// x=1: 288  (intX=0,  phase=4)
		// x=2: 800  (intX=0,  phase=12)
		// x=3: 1312 (intX=1,  phase=4)
		// x=4: 1824 (intX=1,  phase=12)
		// x=5: 2336 (intX=2,  phase=4)
		// x=6: 2848 (intX=2,  phase=12)
		// x=7: 3360 (intX=3,  phase=4)
		{0, -224, -1, 12},
		{1, 288, 0, 4},
		{2, 800, 0, 12},
		{3, 1312, 1, 4},
		{4, 1824, 1, 12},
		{5, 2336, 2, 4},
		{6, 2848, 2, 12},
		{7, 3360, 3, 4},
	}
	for _, tc := range cases {
		pos := startX + int64(tc.x)*xStep
		if pos != tc.wantPos {
			t.Errorf("x=%d pos=%d want %d", tc.x, pos, tc.wantPos)
		}
		intPart, subQ4 := SplitScaledPosition(pos)
		if intPart != tc.wantInt || subQ4 != tc.wantQ4 {
			t.Errorf("x=%d split=(%d, %d) want (%d, %d)", tc.x, intPart, subQ4, tc.wantInt, tc.wantQ4)
		}
	}
}

// TestScaledBlockOriginLibaomBiasChromaSubsampling pins the chroma path:
// the av1_scaled_x bias applies independently of subsampling, but the MV
// component scale (×2 for luma, ×1 for subsampled chroma) survives.
func TestScaledBlockOriginLibaomBiasChromaSubsampling(t *testing.T) {
	sf, err := NewScaleFactors(320, 180, 640, 360)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	// Chroma block at (0,0) with non-zero Q3 MV. Subsampled chroma uses the
	// MV components directly (×1), not ×2 like luma.
	startX, _, _, _, err := sf.ScaledBlockOrigin(0, 0, Vector{Col: 4}, true, true)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	// val_Q4 = 0*16 + 4*1 = 4.
	// off    = (8192-16384)*8 = -65536.
	// tval   = 4*8192 - 65536 = 32768 - 65536 = -32768.
	// pos    = ROUND_POWER_OF_TWO_SIGNED_64(-32768, 8) = -((32768+128)>>8) = -128.
	// startX = -128 + 32 = -96.
	if want := int64(-96); startX != want {
		t.Errorf("startX=%d want %d", startX, want)
	}
}

func TestSplitScaledPositionZero(t *testing.T) {
	intPart, sub := SplitScaledPosition(0)
	if intPart != 0 || sub != 0 {
		t.Fatalf("split(0)=(%d, %d) want (0, 0)", intPart, sub)
	}
}

func TestSplitScaledPositionAtOneSample(t *testing.T) {
	// Q10 position of exactly 1 sample: int=1, frac=0.
	intPart, sub := SplitScaledPosition(ScaleSubpelScale)
	if intPart != 1 || sub != 0 {
		t.Fatalf("split(1.0)=(%d, %d) want (1, 0)", intPart, sub)
	}
}

func TestSplitScaledPositionHalfSample(t *testing.T) {
	// Q10 position of 0.5 sample: int=0, frac=8/16 (Q4).
	intPart, sub := SplitScaledPosition(ScaleSubpelScale / 2)
	if intPart != 0 || sub != 8 {
		t.Fatalf("split(0.5)=(%d, %d) want (0, 8)", intPart, sub)
	}
}
