package filmgrain

import (
	"errors"
	"testing"
)

func TestApplyLumaSample(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 64}}); err != nil {
		t.Fatal(err)
	}
	got, err := ApplyLumaSample(128, 32, lut[:], 8, 8, false)
	if err != nil {
		t.Fatal(err)
	}
	if got != 136 {
		t.Fatalf("sample=%d want 136", got)
	}
	got, err = ApplyLumaSample(512, 32, lut[:], 10, 8, false)
	if err != nil {
		t.Fatal(err)
	}
	if got != 520 {
		t.Fatalf("hbd sample=%d want 520", got)
	}
}

func TestApplyLumaSampleClips(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	got, err := ApplyLumaSample(10, -128, lut[:], 8, 8, true)
	if err != nil {
		t.Fatal(err)
	}
	if got != LumaLegalMin {
		t.Fatalf("restricted low sample=%d want %d", got, LumaLegalMin)
	}
	got, err = ApplyLumaSample(250, 128, lut[:], 8, 8, true)
	if err != nil {
		t.Fatal(err)
	}
	if got != LumaLegalMax {
		t.Fatalf("restricted high sample=%d want %d", got, LumaLegalMax)
	}
}

func TestApplyLumaSampleRejectsInvalidInputs(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if _, err := ApplyLumaSample(0, 0, lut[:], 8, 7, false); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("low scaling shift err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := ApplyLumaSample(0, 0, lut[:ScalingLUTSize-1], 8, 8, false); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("short lut err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := ApplyLumaSample(1<<12, 0, lut[:], 12, 8, false); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("large sample err=%v want %v", err, ErrInvalidParams)
	}
}

func TestApplyLumaSampleAllocs(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 64}}); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyLumaSample(128, 32, lut[:], 8, 8, false); err != nil {
			t.Fatal(err)
		}
		if _, err := ApplyLumaSample(512, -32, lut[:], 10, 8, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLumaSample allocated: %f", allocs)
	}
}

func BenchmarkApplyLumaSample(b *testing.B) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 64}}); err != nil {
		b.Fatal(err)
	}
	var sample uint16
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var err error
		sample, err = ApplyLumaSample(uint16(i&255), 32, lut[:], 8, 8, false)
		if err != nil {
			b.Fatal(err)
		}
	}
	if sample == 0 {
		b.Fatal("unexpected zero sample")
	}
}
