// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// wideFilterParamsCorpus mirrors filter4DispatchParamsCorpus but for the wide
// filters, exercising the needsFilter, flat, hev, and all-pass/all-reject
// branches at 8-bit (scale == 1).
func wideFilterParamsCorpus() []filter4Params {
	mk := func(limit, blimit, hev int) filter4Params {
		return filter4Params{
			limit:  limit,
			blimit: blimit,
			hev:    hev,
			min:    -128,
			max:    127,
			center: 128,
		}
	}
	return []filter4Params{
		mk(0, 0, 0),
		mk(1, 1, 0),
		mk(2, 5, 1),
		mk(8, 20, 4),
		mk(16, 40, 8),
		mk(32, 80, 16),
		mk(63, 128, 32),
		mk(255, 510, 0),
		mk(255, 510, 255),
	}
}

type wideKernel struct {
	name string
	ref  func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	impl func() func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	pad  int // tap headroom rows on each side
}

func wideKernels() []wideKernel {
	return []wideKernel{
		{name: "filter6", ref: filter6EdgePureGo, impl: func() func([]byte, int, int, int, int, int, filter4Params) { return filter6EdgeImpl }, pad: 3},
		{name: "filter8", ref: filter8EdgePureGo, impl: func() func([]byte, int, int, int, int, int, filter4Params) { return filter8EdgeImpl }, pad: 4},
		{name: "filter14", ref: filter14EdgePureGo, impl: func() func([]byte, int, int, int, int, int, filter4Params) { return filter14EdgeImpl }, pad: 7},
	}
}

func runWideKernel(t *testing.T, k wideKernel, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}
	step := stride
	outer := 1
	q0Base := 8 * stride // q0 row 8 leaves >=7 rows of headroom either side

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	k.ref(want, q0Base, step, outer, length, 1, params)
	k.impl()(got, q0Base, step, outer, length, 1, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				k.name, seed, length, params, i, got[i], want[i])
		}
	}
}

// TestWideFilterDispatchMatchesPureGo is the bit-exactness guard for the
// dispatched six/eight/fourteen-sample kernels. It drives the resolved dispatch
// slot (NEON asm on arm64) against the pure-Go reference over a spread of edge
// lengths (including non-multiples of eight to exercise the scalar tail) and
// threshold configurations. Every output byte must match.
func TestWideFilterDispatchMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	for _, k := range wideKernels() {
		var seed int64 = 1
		for _, params := range wideFilterParamsCorpus() {
			for _, length := range lengths {
				for rep := 0; rep < 4; rep++ {
					runWideKernel(t, k, seed, length, params)
					seed++
				}
			}
		}
	}
}

// TestWideFilterDispatchForcedPureGo confirms the differential holds when the
// dispatcher is forced onto the pure-Go branch, so the test still has meaning
// on a NEON host where the slot would otherwise always pick the asm.
func TestWideFilterDispatchForcedPureGo(t *testing.T) {
	restore := cpu.OverrideForTest(cpu.Features{})
	defer restore()
	prev6, prev8, prev14 := filter6EdgeImpl, filter8EdgeImpl, filter14EdgeImpl
	filter6EdgeImpl, filter8EdgeImpl, filter14EdgeImpl = filter6EdgePureGo, filter8EdgePureGo, filter14EdgePureGo
	defer func() { filter6EdgeImpl, filter8EdgeImpl, filter14EdgeImpl = prev6, prev8, prev14 }()

	for _, k := range wideKernels() {
		for i, params := range wideFilterParamsCorpus() {
			runWideKernel(t, k, int64(2000+i), 16, params)
		}
	}
}
