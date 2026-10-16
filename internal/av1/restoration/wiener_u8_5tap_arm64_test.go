// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "testing"

// TestWienerU8FiveTapMatchesReference checks the five-tap kernels against the
// scalar references on symmetric filters whose outer tap pair is zero, and
// that the dispatch really routes them to the five-tap kernels (a zeroed f[7]
// alone would leave the filter asymmetric and silently take the scalar path).
func TestWienerU8FiveTapMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5a5)
	round0, round1 := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {64, 64}, {62, 12}, {24, 7}, {48, 3}}, u8DiffSizes...)
	for _, sz := range sizes {
		for fi, info := range wienerDispatchFilters() {
			hf, vf := info.HFilter, info.VFilter
			hf[0], hf[WienerSubpelTaps-2] = 0, 0
			vf[0], vf[WienerSubpelTaps-2] = 0, 0
			if !wienerFilterFiveTap(hf) || !wienerFilterSymmetric(hf) || !wienerFilterFiveTap(vf) {
				t.Fatalf("filter %d is not a symmetric five-tap filter", fi)
			}
			stride := sz.width + 2*WienerHalfwin + 32
			origin := WienerHalfwin*stride + WienerHalfwin + 8
			src := make([]uint8, stride*(sz.height+2*WienerHalfwin+4))
			for i := range src {
				src[i] = uint8(rnd.pseudoUniform(256))
			}
			if sz.width >= 8 && sz.width%8 == 0 && !wienerHorizontalU8SIMDCanRun(len(src), stride, origin, sz.width, sz.height) {
				t.Fatalf("sz=%dx%d: SIMD horizontal cannot run; test geometry too tight", sz.width, sz.height)
			}
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			want := make([]uint16, tempLen)
			got := make([]uint16, tempLen)
			wienerHorizontalU8(src, stride, origin, sz.width, sz.height, hf, round0, want)
			wienerHorizontalU8SIMDTaps(src, stride, origin, sz.width, sz.height, hf, round0, got)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("horizontal sz=%dx%d f=%d temp[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
			dstStride := sz.width + 5
			wantDst := make([]uint8, dstStride*sz.height)
			gotDst := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(want, sz.width, wantDst, dstStride, sz.width, sz.height, vf, round1)
			wienerVerticalU8SIMDTaps(want, sz.width, gotDst, dstStride, sz.width, sz.height, vf, round1)
			for i := range wantDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("vertical sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}
