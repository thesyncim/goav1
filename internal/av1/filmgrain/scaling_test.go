package filmgrain

import (
	"errors"
	"testing"
)

func TestBuildScalingLUT(t *testing.T) {
	tests := []struct {
		name    string
		points  []ScalingPoint
		samples map[int]uint8
	}{
		{
			name:    "empty",
			samples: map[int]uint8{0: 0, 1: 0, 255: 0},
		},
		{
			name:    "single point",
			points:  []ScalingPoint{{Value: 10, Scaling: 20}},
			samples: map[int]uint8{0: 20, 9: 20, 10: 20, 255: 20},
		},
		{
			name:    "rising segment",
			points:  []ScalingPoint{{Value: 10, Scaling: 20}, {Value: 20, Scaling: 40}},
			samples: map[int]uint8{0: 20, 9: 20, 10: 20, 11: 22, 14: 28, 15: 30, 19: 38, 20: 40, 255: 40},
		},
		{
			name:    "falling segment",
			points:  []ScalingPoint{{Value: 0, Scaling: 100}, {Value: 10, Scaling: 0}},
			samples: map[int]uint8{0: 100, 1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40, 7: 30, 8: 20, 9: 10, 10: 0, 255: 0},
		},
		{
			name:    "uneven segments",
			points:  []ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}},
			samples: map[int]uint8{0: 7, 2: 7, 3: 7, 4: 8, 5: 9, 6: 9, 7: 10, 8: 11, 9: 10, 19: 3, 20: 2, 255: 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var lut [ScalingLUTSize]uint8
			if err := BuildScalingLUT(lut[:], tt.points); err != nil {
				t.Fatal(err)
			}
			for x, want := range tt.samples {
				if got := lut[x]; got != want {
					t.Fatalf("lut[%d]=%d want %d", x, got, want)
				}
			}
		})
	}
}

func TestNormativeDimensions(t *testing.T) {
	if GaussianSequenceLen != 2048 || GaussianBits != 11 {
		t.Fatalf("gaussian bits=%d len=%d", GaussianBits, GaussianSequenceLen)
	}
	if LumaGrainWidth != 82 || LumaGrainHeight != 73 || LumaGrainSamples != 5986 {
		t.Fatalf("luma grain dimensions=%dx%d samples=%d", LumaGrainWidth, LumaGrainHeight, LumaGrainSamples)
	}
	if LumaBlockSize != 32 || NoiseStripeHeight != 34 || LumaOverlapSamples != 2 {
		t.Fatalf("luma block=%d stripe=%d overlap=%d", LumaBlockSize, NoiseStripeHeight, LumaOverlapSamples)
	}
}

func TestScaleLUT(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	tests := map[int]uint8{
		0:    0,
		1:    0,
		2:    1,
		3:    1,
		4:    1,
		510:  128,
		511:  128,
		512:  128,
		1022: 255,
		1023: 255,
	}
	for index, want := range tests {
		got, err := ScaleLUT(lut[:], index, 10)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("ScaleLUT(%d, 10)=%d want %d", index, got, want)
		}
	}
	got, err := ScaleLUT(lut[:], 4095, 12)
	if err != nil {
		t.Fatal(err)
	}
	if got != 255 {
		t.Fatalf("ScaleLUT(4095, 12)=%d want 255", got)
	}
}

func TestScalingRejectsInvalidInputs(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:ScalingLUTSize-1], nil); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("short lut err=%v want %v", err, ErrInvalidParams)
	}
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 10}, {Value: 10}}); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("duplicate point err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := ScaleLUT(lut[:ScalingLUTSize-1], 0, 8); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("short scale err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := ScaleLUT(lut[:], 1<<12, 12); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := ScaleLUT(lut[:], 0, 7); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("low bit depth err=%v want %v", err, ErrInvalidParams)
	}
}

// TestScaleLUTClampsOutOfRangeIndex exercises the defense-in-depth clamp in
// the internal scaleLUT helper. Internal apply callers feed raw uint16 sample
// values that, on a malformed bitstream, can exceed (1<<bitDepth)-1 and would
// otherwise drive lut[x] out of bounds.
func TestScaleLUTClampsOutOfRangeIndex(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name     string
		index    int
		bitDepth uint8
		want     uint8
	}{
		{"8bit above max", 1 << 12, 8, lut[ScalingLUTSize-1]},
		{"10bit above max", 1 << 14, 10, lut[ScalingLUTSize-1]},
		{"12bit above max", 1 << 16, 12, lut[ScalingLUTSize-1]},
		{"negative index", -1, 10, lut[0]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := scaleLUT(lut[:], tc.index, tc.bitDepth)
			if got != tc.want {
				t.Fatalf("scaleLUT(%d, %d)=%d want %d", tc.index, tc.bitDepth, got, tc.want)
			}
		})
	}
}

// TestApplyLumaSampleClampsOutOfRangeOrig confirms ApplyLumaSample no longer
// panics when the original sample is outside the legal bitdepth range. The
// public entrypoint rejects such samples with ErrInvalidParams, and the
// internal helper clips defensively so concurrent apply paths cannot OOB.
func TestApplyLumaSampleClampsOutOfRangeOrig(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	// applyLumaSample must not panic for any uint16 input under any
	// supported bitdepth even when the sample exceeds the bitdepth range.
	for _, bd := range []uint8{8, 10, 12} {
		for _, orig := range []uint16{0, 1, 1<<bd - 1, 1 << bd, (1 << 16) - 1} {
			_ = applyLumaSample(orig, 0, lut[:], bd, 8, false)
			_ = applyLumaSample(orig, 0, lut[:], bd, 11, true)
		}
	}
}

func TestScalingAllocs(t *testing.T) {
	points := [...]ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}}
	var lut [ScalingLUTSize]uint8
	allocs := testing.AllocsPerRun(1000, func() {
		if err := BuildScalingLUT(lut[:], points[:]); err != nil {
			t.Fatal(err)
		}
		for _, index := range [...]int{0, 17, 255, 510, 1023, 2047, 4095} {
			if _, err := ScaleLUT(lut[:], index, 12); err != nil {
				t.Fatal(err)
			}
		}
	})
	if allocs != 0 {
		t.Fatalf("filmgrain scaling allocated: %f", allocs)
	}
}

func BenchmarkBuildScalingLUT(b *testing.B) {
	points := [...]ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}}
	var lut [ScalingLUTSize]uint8
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BuildScalingLUT(lut[:], points[:])
	}
}

func BenchmarkScaleLUT(b *testing.B) {
	points := [...]ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], points[:]); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ScaleLUT(lut[:], i&4095, 12)
	}
}
