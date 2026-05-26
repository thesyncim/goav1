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
	// each output pel maps to 0.5 source pel, so a block whose dst origin is
	// (16, 8) with zero MV must start at source position (8, 4) and step by
	// 0.5 source pels per output pel.
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
	if intX != 8 || fracQ4X != 0 {
		t.Errorf("startX split=(%d, %d) want (8, 0)", intX, fracQ4X)
	}
	if intY != 4 || fracQ4Y != 0 {
		t.Errorf("startY split=(%d, %d) want (4, 0)", intY, fracQ4Y)
	}
}

func TestScaledBlockOriginCarriesNegativeMV(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	// MV = (-8, 0) in Q3: 1 luma pel to the left. With 2:1 downscale that's
	// 0.5 source pel.
	startX, _, _, _, err := sf.ScaledBlockOrigin(16, 8, Vector{Col: -8}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	// dst_x=16 maps to src 8.0; MV -1 luma pel maps to -0.5 source pel.
	// Expected start = (8 - 0.5) = 7.5 in source coords = 7 integer + 0.5
	// frac. In Q10 that is 7*1024 + 512 = 7680. Filter index = (512>>6)=8.
	intX, fracQ4X := SplitScaledPosition(startX)
	if intX != 7 || fracQ4X != 8 {
		t.Errorf("start=(%d, %d) want (7, 8)", intX, fracQ4X)
	}
}

func TestScaledBlockOriginRejectsZeroFactors(t *testing.T) {
	var sf ScaleFactors
	if _, _, _, _, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false); err == nil {
		t.Fatal("expected error for zero ScaleFactors")
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
