// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

import "testing"

// TestFilterBlockDispatchMatchesPureGo is the bit-exactness guard for the
// dispatched CDEF block filter. It exercises the resolved dispatch slot
// (filterBlockImpl, which routes through the NEON asm on arm64) against the
// canonical pure-Go reference across the full matrix of directions,
// primary/secondary strengths, damping, bit depths, and the four CDEF border
// configurations. Every output sample must match the reference exactly.
func TestFilterBlockDispatchMatchesPureGo(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()

	for _, depth := range []int{8, 10, 12} {
		coeffShift := depth - 8
		for boundary := 0; boundary < 16; boundary++ {
			input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), depth, boundary, boundary+1)
			for dir := 0; dir <= 7; dir++ {
				for _, pri := range cdefPrimaryStrengthCorpus(coeffShift) {
					for _, sec := range cdefSecondaryStrengthCorpus(coeffShift) {
						for _, damping := range []int{3 + coeffShift, 5 + coeffShift, 6 + coeffShift} {
							params := BlockFilterParams{
								PrimaryStrength:   pri,
								SecondaryStrength: sec,
								Direction:         dir,
								PrimaryDamping:    damping,
								SecondaryDamping:  damping,
								CoeffShift:        coeffShift,
								Width:             8,
								Height:            8,
							}
							want := make([]uint16, dstStride*8)
							got := make([]uint16, dstStride*8)
							filterBlockPureGo(want, dstStride, 0, input, origin, params)
							filterBlockImpl(got, dstStride, 0, input, origin, params)
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("depth=%d boundary=%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
										depth, boundary, dir, pri, sec, damping, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestFilterBlockDispatchIsZeroAlloc protects the hot-path contract that the
// dispatched filter does not allocate per call.
func TestFilterBlockDispatchIsZeroAlloc(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 5, 3)
	dst := make([]uint16, dstStride*8)
	params := BlockFilterParams{
		PrimaryStrength:   12,
		SecondaryStrength: 4,
		Direction:         3,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		CoeffShift:        0,
		Width:             8,
		Height:            8,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		filterBlockImpl(dst, dstStride, 0, input, origin, params)
	})
	if allocs != 0 {
		t.Fatalf("dispatch allocated %f times per call", allocs)
	}
}
