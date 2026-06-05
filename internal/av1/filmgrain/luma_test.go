package filmgrain

import (
	"errors"
	"testing"
)

func TestGenerateLumaGrainZeroPoints(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	for i := range grain {
		grain[i] = 99
	}
	if err := GenerateLumaGrain(grain, LumaGrainParams{BitDepth: 8}); err != nil {
		t.Fatal(err)
	}
	for i, sample := range grain {
		if sample != 0 {
			t.Fatalf("grain[%d]=%d want 0", i, sample)
		}
	}
}

func TestGenerateLumaGrainWhiteNoise(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	if err := GenerateLumaGrain(grain, params); err != nil {
		t.Fatal(err)
	}
	wantFirst := [...]int16{-29, -1, -31, -20, 49, 8, -56, 54, 18, -7, -35, -47, -51, -3, 60, 26}
	for i, want := range wantFirst {
		if grain[i] != want {
			t.Fatalf("grain[%d]=%d want %d", i, grain[i], want)
		}
	}
	if got := grain[3*LumaGrainWidth+3]; got != -14 {
		t.Fatalf("grain[3][3]=%d want -14", got)
	}
}

func TestGenerateLumaGrainAppliesAR(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	params.ARCoeffs[3] = 64
	if err := GenerateLumaGrain(grain, params); err != nil {
		t.Fatal(err)
	}
	wantRow := [...]int16{-24, 42, -26, -40, 24, 42, 13, -66, -67, -70, -74}
	row := grain[3*LumaGrainWidth:]
	for i, want := range wantRow {
		if row[i] != want {
			t.Fatalf("row[3][%d]=%d want %d", i, row[i], want)
		}
	}
}

func TestGenerateLumaGrainARLagSpecializationsMatchReference(t *testing.T) {
	for _, lag := range []uint8{1, 2, 3} {
		t.Run("lag"+string(rune('0'+lag)), func(t *testing.T) {
			params := LumaGrainParams{
				Seed:            0x4567,
				BitDepth:        10,
				NumYPoints:      1,
				GrainScaleShift: 1,
				ARCoeffLag:      lag,
				ARCoeffShift:    7,
			}
			for i := range params.ARCoeffs {
				params.ARCoeffs[i] = int8((i%9)-4) * 3
			}
			want := referenceLumaGrain(t, params)
			got := make([]int16, LumaGrainSamples)
			if err := GenerateLumaGrain(got, params); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("grain[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestGenerateLumaGrainRejectsInvalidInputs(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	tests := []struct {
		name   string
		dst    []int16
		params LumaGrainParams
	}{
		{name: "short", dst: grain[:LumaGrainSamples-1], params: LumaGrainParams{BitDepth: 8}},
		{name: "bit-depth", dst: grain, params: LumaGrainParams{BitDepth: 9}},
		{name: "points", dst: grain, params: LumaGrainParams{BitDepth: 8, NumYPoints: MaxLumaScalingPoints + 1}},
		{name: "grain-shift", dst: grain, params: LumaGrainParams{BitDepth: 8, GrainScaleShift: 4}},
		{name: "lag", dst: grain, params: LumaGrainParams{BitDepth: 8, ARCoeffLag: 4}},
		{name: "coeff-shift", dst: grain, params: LumaGrainParams{BitDepth: 8, NumYPoints: 1, ARCoeffLag: 1, ARCoeffShift: 5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := GenerateLumaGrain(tt.dst, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("GenerateLumaGrain err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func referenceLumaGrain(t *testing.T, params LumaGrainParams) []int16 {
	t.Helper()
	grain := make([]int16, LumaGrainSamples)
	rng := NewRandom(params.Seed)
	gaussianShift := int(12 - params.BitDepth + params.GrainScaleShift)
	for y := range LumaGrainHeight {
		for x := range LumaGrainWidth {
			index, err := rng.Number(GaussianBits)
			if err != nil {
				t.Fatal(err)
			}
			grain[y*LumaGrainWidth+x] = int16(roundPowerOfTwo(int(Gaussian(index)), gaussianShift))
		}
	}

	lag := int(params.ARCoeffLag)
	coeffShift := int(params.ARCoeffShift)
	roundingOffset := 1 << (coeffShift - 1)
	grainMin := -(1 << (params.BitDepth - 1))
	grainMax := (1 << (params.BitDepth - 1)) - 1
	for y := 3; y < LumaGrainHeight; y++ {
		for x := 3; x < LumaGrainWidth-3; x++ {
			sum := 0
			pos := 0
			for deltaRow := -lag; deltaRow <= 0; deltaRow++ {
				for deltaCol := -lag; deltaCol <= lag; deltaCol++ {
					if deltaRow == 0 && deltaCol == 0 {
						break
					}
					sample := int(grain[(y+deltaRow)*LumaGrainWidth+x+deltaCol])
					sum += int(params.ARCoeffs[pos]) * sample
					pos++
				}
			}
			v := int(grain[y*LumaGrainWidth+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[y*LumaGrainWidth+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
	return grain
}

func TestGenerateLumaGrainAllocs(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	params.ARCoeffs[3] = 64
	allocs := testing.AllocsPerRun(1000, func() {
		if err := GenerateLumaGrain(grain, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateLumaGrain allocated: %f", allocs)
	}
}

func BenchmarkGenerateLumaGrain(b *testing.B) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	params.ARCoeffs[3] = 64
	b.ReportAllocs()
	for b.Loop() {
		if err := GenerateLumaGrain(grain, params); err != nil {
			b.Fatal(err)
		}
	}
}
