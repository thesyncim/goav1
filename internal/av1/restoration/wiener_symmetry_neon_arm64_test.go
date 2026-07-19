// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "testing"

// These tests are the byte-exactness gate for the Wiener symmetric-pair MAC
// reduction (see wiener_neon_arm64.s / wiener_u8_neon_arm64.s). They sweep
// randomized valid Wiener coefficient sets -- every filter NewWienerFilter can
// produce, i.e. every mirrored tap triple in [Min,Max] -- against randomized
// samples, and require the NEON kernels to match the pure-Go reference sample
// for sample. The pairing (f0==f6, f1==f5, f2==f4) is only bit-identical for
// symmetric filters, so this exercises exactly the property the rewrite relies
// on across the full coefficient space, not just the fixed dispatch corpus.

// randomWienerFilter draws a random valid Wiener filter: each of the three
// distinct taps is uniform over its [Min,Max] range and NewWienerFilter mirrors
// them and derives the center so the taps sum to zero (validWienerFilter holds).
func randomWienerFilter(rnd *restorationRandom) WienerFilter {
	t0 := int16(WienerTap0Min + rnd.pseudoUniform(WienerTap0Max-WienerTap0Min+1))
	t1 := int16(WienerTap1Min + rnd.pseudoUniform(WienerTap1Max-WienerTap1Min+1))
	t2 := int16(WienerTap2Min + rnd.pseudoUniform(WienerTap2Max-WienerTap2Min+1))
	return NewWienerFilter(t0, t1, t2)
}

var wienerSymmetrySizes = []struct{ width, height int }{
	{8, 8}, {16, 5}, {24, 9}, {40, 12}, {64, 64},
}

// TestWienerHorizontalNEONSymmetryRandomTaps checks the u16 horizontal
// symmetric-pair kernel against the reference over random valid taps and random
// samples, at every bit depth (the horizontal pair-sums stay in the int16 lane
// for all of 8/10/12-bit since source samples are <=4095).
func TestWienerHorizontalNEONSymmetryRandomTaps(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5117)
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, _ := wienerRounds(bitDepth)
		for _, sz := range wienerSymmetrySizes {
			for trial := 0; trial < 24; trial++ {
				filter := randomWienerFilter(rnd)
				stride := sz.width + 2*WienerHalfwin + 4
				origin := WienerHalfwin*stride + WienerHalfwin + 2
				src := make([]uint16, stride*(sz.height+2*WienerHalfwin))
				for i := range src {
					src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)
				want := make([]uint16, tempLen)
				got := make([]uint16, tempLen)
				wienerHorizontal(src, stride, origin, sz.width, sz.height, filter, bitDepth, round0, max, want)
				wienerHorizontalNEON(src, stride, origin, sz.width, sz.height, filter, bitDepth, round0, max, got)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bd=%d sz=%dx%d filter=%v temp[%d]=%d want %d", bitDepth, sz.width, sz.height, filter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestWienerVerticalNEONRandomTaps checks the general u16 vertical kernel (which
// deliberately keeps the plain 7-tap MAC) against the reference over random
// valid taps and random temp values spanning the full [0,maxClamp] intermediate
// range, at every bit depth.
func TestWienerVerticalNEONRandomTaps(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5118)
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(bitDepth)
		limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
		for _, sz := range wienerSymmetrySizes {
			for trial := 0; trial < 24; trial++ {
				filter := randomWienerFilter(rnd)
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)
				temp := make([]uint16, tempLen)
				for i := range temp {
					temp[i] = uint16(rnd.pseudoUniform(int(limit))) // [0, maxClamp]
				}
				dstStride := sz.width + 3
				want := make([]uint16, dstStride*sz.height)
				got := make([]uint16, dstStride*sz.height)
				wienerVertical(temp, sz.width, want, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)
				wienerVerticalNEON(temp, sz.width, got, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bd=%d sz=%dx%d filter=%v dst[%d]=%d want %d", bitDepth, sz.width, sz.height, filter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestWienerU8NEONSymmetryRandomTaps checks the 8-bit symmetric-pair kernels
// (both horizontal and vertical) against the pure-Go u8 references over random
// valid taps. The vertical pair-sums are safe because temp is clamped to
// [0,8191] at 8-bit, so a pair-sum (<=16382) stays in the int16 lane.
func TestWienerU8NEONSymmetryRandomTaps(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5119)
	round0, round1 := wienerRounds(8)
	for _, sz := range wienerSymmetrySizes {
		for trial := 0; trial < 24; trial++ {
			hFilter := randomWienerFilter(rnd)
			vFilter := randomWienerFilter(rnd)
			stride := sz.width + 2*WienerHalfwin + 3
			origin := WienerHalfwin*stride + WienerHalfwin + 1
			src := randomU8Plane(rnd, stride, sz.height+2*WienerHalfwin)
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)

			// Horizontal: NEON vs reference.
			wantTemp := make([]uint16, tempLen)
			gotTemp := make([]uint16, tempLen)
			wienerHorizontalU8(src, stride, origin, sz.width, sz.height, hFilter, round0, wantTemp)
			wienerHorizontalU8NEON(src, stride, origin, sz.width, sz.height, hFilter, round0, gotTemp)
			for i := range wantTemp {
				if gotTemp[i] != wantTemp[i] {
					t.Fatalf("horiz sz=%dx%d filter=%v temp[%d]=%d want %d", sz.width, sz.height, hFilter, i, gotTemp[i], wantTemp[i])
				}
			}

			// Vertical over the shared (reference) temp buffer.
			dstStride := sz.width + 2
			wantDst := make([]uint8, dstStride*sz.height)
			gotDst := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(wantTemp, sz.width, wantDst, dstStride, sz.width, sz.height, vFilter, round1)
			wienerVerticalU8NEON(wantTemp, sz.width, gotDst, dstStride, sz.width, sz.height, vFilter, round1)
			for i := range wantDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("vert sz=%dx%d filter=%v dst[%d]=%d want %d", sz.width, sz.height, vFilter, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}
