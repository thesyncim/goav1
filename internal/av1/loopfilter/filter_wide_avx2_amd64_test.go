// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// wideAVX2Kernel pairs an AVX2 8-bit wide kernel with its pure-Go reference and
// the tap headroom the differential buffer needs on each side of the edge.
type wideAVX2Kernel struct {
	name string
	ref  func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	avx2 func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	pad  int
}

func wideAVX2Kernels() []wideAVX2Kernel {
	return []wideAVX2Kernel{
		{"filter6", filter6EdgePureGo, filter6EdgeAVX2, 3},
		{"filter8", filter8EdgePureGo, filter8EdgeAVX2, 4},
		{"filter14", filter14EdgePureGo, filter14EdgeAVX2, 7},
	}
}

// TestWideFilterAVX2MatchesPureGo is the bit-exactness guard for the AVX2 8-bit
// six/eight/fourteen-sample kernels. It calls the AVX2 wrappers directly (not
// through the dispatch slot) so the VEX-encoded asm is exercised even when CPUID
// does not advertise AVX2 (Rosetta 2 executes it regardless), deliberately not
// skipping on cpu.Detected.AVX2 == false. Every output byte must match across the
// content regimes (random, near-flat, wide-flat, clamp edges), length matrix
// (including tails that route to the scalar remainder), and threshold spread.
func TestWideFilterAVX2MatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 47, 48, 63, 64}
	for _, k := range wideAVX2Kernels() {
		var seed int64 = 1
		for _, params := range wideFilterParamsCorpus() {
			for _, length := range lengths {
				for rep := 0; rep < 5; rep++ {
					runWideAVX2(t, k, seed, length, params)
					seed++
				}
			}
		}
	}
}

func runWideAVX2(t *testing.T, k wideAVX2Kernel, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step := stride
	outer := 1
	q0Base := 8 * stride // q0 row 8 leaves >=7 rows of headroom on either side

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	k.ref(want, q0Base, step, outer, length, 1, params)
	k.avx2(got, q0Base, step, outer, length, 1, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				k.name, seed, length, params, i, got[i], want[i])
		}
	}
}
