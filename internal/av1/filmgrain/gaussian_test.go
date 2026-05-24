package filmgrain

import "testing"

func TestGaussianSequence(t *testing.T) {
	if len(gaussianSequence) != GaussianSequenceLen {
		t.Fatalf("len=%d want %d", len(gaussianSequence), GaussianSequenceLen)
	}
	wantFirst := [...]int16{56, 568, -180, 172, 124, -84, 172, -64}
	for i, want := range wantFirst {
		if got := Gaussian(uint16(i)); got != want {
			t.Fatalf("Gaussian(%d)=%d want %d", i, got, want)
		}
	}
	wantMid := [...]int16{320, 1032, 216, 320, -8, -64, 156, -1016}
	for i, want := range wantMid {
		index := 1020 + i
		if got := Gaussian(uint16(index)); got != want {
			t.Fatalf("Gaussian(%d)=%d want %d", index, got, want)
		}
	}
	wantLast := [...]int16{-80, 32, -16, 280, 288, 944, 428, -484}
	for i, want := range wantLast {
		index := GaussianSequenceLen - len(wantLast) + i
		if got := Gaussian(uint16(index)); got != want {
			t.Fatalf("Gaussian(%d)=%d want %d", index, got, want)
		}
	}
	if Gaussian(GaussianSequenceLen) != Gaussian(0) {
		t.Fatalf("masked index sample=%d first=%d", Gaussian(GaussianSequenceLen), Gaussian(0))
	}
}

func TestGaussianSequenceRange(t *testing.T) {
	minValue := int16(2047)
	maxValue := int16(-2048)
	for i, sample := range gaussianSequence {
		if sample < -2048 || sample > 2047 || sample%4 != 0 {
			t.Fatalf("gaussian[%d]=%d outside normative range", i, sample)
		}
		if sample < minValue {
			minValue = sample
		}
		if sample > maxValue {
			maxValue = sample
		}
	}
	if minValue != -1752 || maxValue != 1688 {
		t.Fatalf("range=%d..%d", minValue, maxValue)
	}
}
