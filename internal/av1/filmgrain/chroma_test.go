package filmgrain

import (
	"errors"
	"testing"
)

func TestGenerateChromaGrainWhiteNoise420(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
	}
	if err := GenerateChromaGrain(grain, nil, params); err != nil {
		t.Fatal(err)
	}
	wantCb := [...]int16{12, -37, 4, -18, -13, 6, -5, -43, -8, 11, -29, 48, 67, 57, 20, 13}
	for i, want := range wantCb {
		if grain[i] != want {
			t.Fatalf("cb grain[%d]=%d want %d", i, grain[i], want)
		}
	}
	if got := grain[3*ChromaGrainWidth+3]; got != 14 {
		t.Fatalf("cb grain[3][3]=%d want 14", got)
	}
	if grain[44] != 0 || grain[38*ChromaGrainWidth] != 0 {
		t.Fatalf("subsampled inactive samples=%d %d want zero", grain[44], grain[38*ChromaGrainWidth])
	}

	params.Plane = ChromaPlaneCr
	if err := GenerateChromaGrain(grain, nil, params); err != nil {
		t.Fatal(err)
	}
	wantCr := [...]int16{5, -38, 7, 23, 0, -33, 34, 15, -5, -25, -11, 48, 29, -11, 8, 3}
	for i, want := range wantCr {
		if grain[i] != want {
			t.Fatalf("cr grain[%d]=%d want %d", i, grain[i], want)
		}
	}
}

func TestGenerateChromaGrainAppliesLumaAR(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	for i := range luma {
		luma[i] = 32
	}
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		NumYPoints:      1,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
		ARCoeffShift:    6,
	}
	params.ARCoeffs[0] = 64
	if err := GenerateChromaGrain(grain, luma, params); err != nil {
		t.Fatal(err)
	}
	if got := grain[3*ChromaGrainWidth+3]; got != 46 {
		t.Fatalf("luma AR grain[3][3]=%d want 46", got)
	}
}

func TestGenerateChromaGrainAppliesChromaAR(t *testing.T) {
	baseline := make([]int16, ChromaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
	}
	if err := GenerateChromaGrain(baseline, nil, params); err != nil {
		t.Fatal(err)
	}

	grain := make([]int16, ChromaGrainSamples)
	params.ARCoeffLag = 1
	params.ARCoeffShift = 6
	params.ARCoeffs[3] = 64
	if err := GenerateChromaGrain(grain, nil, params); err != nil {
		t.Fatal(err)
	}
	want := baseline[3*ChromaGrainWidth+3] + baseline[3*ChromaGrainWidth+2]
	if got := grain[3*ChromaGrainWidth+3]; got != want {
		t.Fatalf("chroma AR grain[3][3]=%d want %d", got, want)
	}
}

func TestGenerateChromaGrainRejectsInvalidInputs(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	tests := []struct {
		name   string
		dst    []int16
		luma   []int16
		params ChromaGrainParams
	}{
		{name: "short", dst: grain[:ChromaGrainSamples-1], params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8}},
		{name: "plane", dst: grain, params: ChromaGrainParams{Plane: 0, BitDepth: 8}},
		{name: "bit-depth", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 9}},
		{name: "points", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, NumYPoints: MaxLumaScalingPoints + 1}},
		{name: "grain-shift", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, GrainScaleShift: 4}},
		{name: "lag", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, ARCoeffLag: 4}},
		{name: "coeff-shift", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, ARCoeffLag: 1, ARCoeffShift: 5}},
		{name: "short-luma", dst: grain, luma: luma[:LumaGrainSamples-1], params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, NumYPoints: 1, ARCoeffShift: 6}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := GenerateChromaGrain(tt.dst, tt.luma, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("GenerateChromaGrain err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func TestGenerateChromaGrainAllocs(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		NumYPoints:      1,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
	}
	params.ARCoeffs[3] = 64
	params.ARCoeffs[4] = 32
	allocs := testing.AllocsPerRun(1000, func() {
		if err := GenerateChromaGrain(grain, luma, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateChromaGrain allocated: %f", allocs)
	}
}

func BenchmarkGenerateChromaGrain(b *testing.B) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		NumYPoints:      1,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
	}
	params.ARCoeffs[3] = 64
	params.ARCoeffs[4] = 32
	b.ReportAllocs()
	for b.Loop() {
		if err := GenerateChromaGrain(grain, luma, params); err != nil {
			b.Fatal(err)
		}
	}
}
