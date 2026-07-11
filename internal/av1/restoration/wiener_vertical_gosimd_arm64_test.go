// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "testing"

func TestWienerVerticalU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5c3)
	_, round1 := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {62, 12}, {17, 4}, {31, 7}, {7, 5}}, u8DiffSizes...)
	for _, sz := range sizes {
		for fi, info := range wienerDispatchFilters() {
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			temp := make([]uint16, tempLen)
			for i := range temp {
				temp[i] = uint16(rnd.pseudoUniform(8192))
			}
			dstStride := sz.width + 2
			want := make([]uint8, dstStride*sz.height)
			got := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(temp, sz.width, want, dstStride, sz.width, sz.height, info.VFilter, round1)
			wienerVerticalU8SIMD(temp, sz.width, got, dstStride, sz.width, sz.height, info.VFilter, round1)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkWienerVerticalU8SIMD(b *testing.B) {
	_, round1 := wienerRounds(8)
	const w, h = 64, 64
	temp := make([]uint16, w*(h+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16((i*97 + 31) % 8192)
	}
	dst := make([]uint8, w*h)
	info := wienerDispatchFilters()[0]
	b.SetBytes(int64(w * h))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wienerVerticalU8SIMD(temp, w, dst, w, w, h, info.VFilter, round1)
	}
}
