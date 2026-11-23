package filmgrain

import (
	"testing"
)

func TestApplyLumaSample(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 64}}); err != nil {
		t.Fatal(err)
	}
	got := applyLumaSample(128, 32, lut[:], 8, 8, false)
	if got != 136 {
		t.Fatalf("sample=%d want 136", got)
	}
	got = applyLumaSample(512, 32, lut[:], 10, 8, false)
	if got != 520 {
		t.Fatalf("hbd sample=%d want 520", got)
	}
}

func TestApplyLumaSampleClips(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	got := applyLumaSample(10, -128, lut[:], 8, 8, true)
	if got != LumaLegalMin {
		t.Fatalf("restricted low sample=%d want %d", got, LumaLegalMin)
	}
	got = applyLumaSample(250, 128, lut[:], 8, 8, true)
	if got != LumaLegalMax {
		t.Fatalf("restricted high sample=%d want %d", got, LumaLegalMax)
	}
}

func TestBlendLumaOverlap(t *testing.T) {
	got := blendLumaOverlap(100, 20, 0, 8)
	if got != 95 {
		t.Fatalf("first overlap=%d want 95", got)
	}
	got = blendLumaOverlap(100, 20, 1, 8)
	if got != 70 {
		t.Fatalf("second overlap=%d want 70", got)
	}
}

func TestBlendLumaOverlapClips(t *testing.T) {
	got := blendLumaOverlap(300, 300, 0, 8)
	if got != 127 {
		t.Fatalf("high overlap=%d want 127", got)
	}
	got = blendLumaOverlap(-300, -300, 0, 8)
	if got != -128 {
		t.Fatalf("low overlap=%d want -128", got)
	}
}
