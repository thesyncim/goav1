// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

import (
	"math/rand"
	"testing"
)

// fuzzScaleLUT returns a 256-entry scaling LUT. Beyond fully random tables it
// injects structured shapes (flat, ramps, saturated) so the interpolation and
// the uint8 truncation of start+round are exercised at their extremes.
func fuzzScaleLUT(rng *rand.Rand, shape int) []uint8 {
	lut := make([]uint8, ScalingLUTSize)
	switch shape {
	case 0:
		for i := range lut {
			lut[i] = uint8(rng.Intn(256))
		}
	case 1: // monotone ramp (typical film-grain shape)
		for i := range lut {
			lut[i] = uint8(i)
		}
	case 2: // descending ramp: forces negative diffs (arithmetic shift sign)
		for i := range lut {
			lut[i] = uint8(255 - i)
		}
	case 3: // flat
		v := uint8(rng.Intn(256))
		for i := range lut {
			lut[i] = v
		}
	default: // adjacent extremes: max |diff| interpolation
		for i := range lut {
			if i&1 == 0 {
				lut[i] = 0
			} else {
				lut[i] = 255
			}
		}
	}
	return lut
}

// fuzzScaleSrc returns a sample row for the given bit depth, saturated at both
// ends and salted with a few out-of-range samples to exercise the clamp.
func fuzzScaleSrc(rng *rand.Rand, n int, bitDepth uint8) []uint16 {
	maxSample := (1 << bitDepth) - 1
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(rng.Intn(maxSample + 1))
	}
	if n > 0 {
		src[0] = 0
		src[n-1] = uint16(maxSample)
	}
	// Out-of-range samples (defense-in-depth clamp path).
	for _, i := range []int{n / 3, n / 2, (2 * n) / 3} {
		if i >= 0 && i < n {
			src[i] = uint16(1024 + rng.Intn(64512))
		}
	}
	return src
}

func TestBuildScaleRow10MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5CA1E10))
	lengths := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100}
	for shape := 0; shape < 5; shape++ {
		for rep := 0; rep < 4; rep++ {
			lut := fuzzScaleLUT(rng, shape)
			for _, n := range lengths {
				src := fuzzScaleSrc(rng, n, 10)
				ref := make([]uint16, n)
				got := make([]uint16, n)
				for i := 0; i < n; i++ {
					ref[i] = uint16(scaleLUT10(lut, int(src[i])))
				}
				buildScaleRow10(got, src, lut)
				for i := 0; i < n; i++ {
					if got[i] != ref[i] {
						t.Fatalf("10-bit shape=%d n=%d i=%d src=%d: row=%d scalar=%d",
							shape, n, i, src[i], got[i], ref[i])
					}
				}
			}
		}
	}
}

// TestBuildScaleRow10ExhaustiveIndex checks every legal 10-bit sample index
// against the scalar, so the x==255 boundary (idx in [1020,1023]) and every
// interpolation weight are covered without relying on random draws.
func TestBuildScaleRow10ExhaustiveIndex(t *testing.T) {
	rng := rand.New(rand.NewSource(0xEECE))
	src := make([]uint16, 1024)
	for i := range src {
		src[i] = uint16(i)
	}
	for shape := 0; shape < 5; shape++ {
		lut := fuzzScaleLUT(rng, shape)
		ref := make([]uint16, len(src))
		got := make([]uint16, len(src))
		for i := range src {
			ref[i] = uint16(scaleLUT10(lut, int(src[i])))
		}
		buildScaleRow10(got, src, lut)
		for i := range src {
			if got[i] != ref[i] {
				t.Fatalf("10-bit exhaustive shape=%d idx=%d: row=%d scalar=%d",
					shape, i, got[i], ref[i])
			}
		}
	}
}

func TestBuildScaleRow10ZeroAlloc(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(1))
	lut := fuzzScaleLUT(rng, 0)
	src := fuzzScaleSrc(rng, n, 10)
	scale := make([]uint16, n)
	if a := testing.AllocsPerRun(1000, func() {
		buildScaleRow10(scale, src, lut)
	}); a != 0 {
		t.Fatalf("buildScaleRow10 allocated: %f", a)
	}
}

func BenchmarkBuildScaleRow10(b *testing.B) {
	const n = 32
	rng := rand.New(rand.NewSource(7))
	lut := fuzzScaleLUT(rng, 1)
	src := fuzzScaleSrc(rng, n, 10)
	scale := make([]uint16, n)
	b.ReportAllocs()
	for b.Loop() {
		buildScaleRow10(scale, src, lut)
	}
}
