// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import (
	"testing"
)

// TestFilterBlockAVX2MatchesPureGo is the bit-exactness guard for the AVX2 CDEF
// block kernel. It calls filterBlockAVX2 directly (not through the dispatch
// slot) so the AVX2 asm is exercised even when feature detection has not
// enabled it for the live decoder, then compares every output sample against
// the canonical pure-Go reference across the full matrix of directions,
// primary/secondary strengths, damping, bit depths, and the CDEF border
// configurations.
//
// The AVX2 (VEX-encoded) instructions execute under Rosetta 2 even though it
// does not advertise AVX2 in CPUID, so this test calls the kernel directly and
// validates byte-exactness regardless of the dispatch gate.
func TestFilterBlockAVX2MatchesPureGo(t *testing.T) {
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
								PrimaryStrength:   uint8(pri),
								SecondaryStrength: uint8(sec),
								Direction:         uint8(dir),
								PrimaryDamping:    uint8(damping),
								SecondaryDamping:  uint8(damping),
								CoeffShift:        uint8(coeffShift),
								Width:             8,
								Height:            8,
							}
							want := make([]uint16, dstStride*8)
							got := make([]uint16, dstStride*8)
							filterBlockPureGo(want, dstStride, 0, input, origin, params)
							filterBlockAVX2(got, dstStride, 0, input, origin, params)
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
