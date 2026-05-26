package filmgrain

import "testing"

func TestGaussianSequence(t *testing.T) {
	if len(gaussianSequence) != GaussianSequenceLen {
		t.Fatalf("len=%d want %d", len(gaussianSequence), GaussianSequenceLen)
	}
	var sum int64
	var abs int64
	var weighted int64
	for i, v := range gaussianSequence {
		value := int64(v)
		sum += value
		if value < 0 {
			abs -= value
		} else {
			abs += value
		}
		weighted += int64(i+1) * value
	}
	if sum != 1120 || abs != 837192 || weighted != -8011268 {
		t.Fatalf("gaussian checksum sum=%d abs=%d weighted=%d", sum, abs, weighted)
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

func TestGaussianAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		var sum int16
		for i := range uint16(256) {
			sum += Gaussian(i)
		}
		if sum == 0 {
			t.Fatal("unexpected zero gaussian sum")
		}
	})
	if allocs != 0 {
		t.Fatalf("Gaussian allocated: %f", allocs)
	}
}

func BenchmarkGaussian(b *testing.B) {
	var sum int16
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sum += Gaussian(uint16(i))
	}
	if sum == 0 {
		b.Fatal("unexpected zero gaussian sum")
	}
}
