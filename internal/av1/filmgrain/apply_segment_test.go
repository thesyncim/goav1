// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

import (
	"math/rand"
	"testing"
)

// fuzzSegmentInputs builds a deterministic but adversarial set of inputs for
// the apply kernel: full-range source samples, the full int16 grain range,
// and 0..255 scale values, so the differential test exercises the rounding,
// the arithmetic shift sign, and both clamp bounds.
func fuzzSegmentInputs(rng *rand.Rand, n int, bitDepth uint8) (dst, src, scale []uint16, grain []int16) {
	maxSample := (1 << bitDepth) - 1
	dst = make([]uint16, n)
	src = make([]uint16, n)
	scale = make([]uint16, n)
	grain = make([]int16, n)
	grainLimit := 1 << (bitDepth - 1)
	for i := 0; i < n; i++ {
		src[i] = uint16(rng.Intn(maxSample + 1))
		scale[i] = uint16(rng.Intn(256))
		grain[i] = int16(rng.Intn(2*grainLimit) - grainLimit)
	}
	// Sprinkle in saturating extremes.
	if n > 0 {
		src[0], scale[0], grain[0] = uint16(maxSample), 255, int16(grainLimit-1)
		src[n-1], scale[n-1], grain[n-1] = 0, 255, int16(-grainLimit)
	}
	return dst, src, scale, grain
}

func BenchmarkApplyGrainSegmentPureGo(b *testing.B) {
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(7)), n, 8)
	b.ReportAllocs()
	for b.Loop() {
		applyGrainSegmentPureGo(dst, src, scale, grain, 8, 0, 255)
	}
}
