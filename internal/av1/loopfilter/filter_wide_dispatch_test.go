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
			limit:  int16(limit),
			blimit: int16(blimit),
			hev:    int16(hev),
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

// fillWideContent fills buf with one of several content regimes. Pure-random
// content almost never satisfies the flat8in/flat8out predicates, so the flat
// and near-flat modes are essential to exercise filter8's flat branch and
// filter14's fourteen-tap wide path (where a divergence once slipped past a
// random-only test and only surfaced on a real monochrome frame).
func fillWideContent(buf []byte, rng *rand.Rand, mode int) {
	for i := range buf {
		switch mode {
		case 0:
			buf[i] = byte(rng.Intn(256)) // random
		case 1:
			buf[i] = byte(120 + rng.Intn(16)) // near-flat
		case 2:
			buf[i] = byte(127 + rng.Intn(3)) // very flat (wide path)
		case 3:
			buf[i] = byte(rng.Intn(4)) // flat near 0 (clamp edge)
		default:
			buf[i] = byte(252 + rng.Intn(4)) // flat near 255 (clamp edge)
		}
	}
}

func runWideKernel(t *testing.T, k wideKernel, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
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

// runWideKernelVertical mirrors runWideKernel for a vertical edge: the taps are
// one byte apart (step == 1) and successive positions advance by the row stride
// (outer == stride). This exercises the transposing ld4/ld2 vertical NEON path.
func runWideKernelVertical(t *testing.T, k wideKernel, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 96
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step := 1
	outer := stride
	q0Base := 16 // column 16 leaves >=7 bytes of tap headroom on the left

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	k.ref(want, q0Base, step, outer, length, 1, params)
	k.impl()(got, q0Base, step, outer, length, 1, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s vertical seed=%d len=%d params=%+v idx=%d got=%d want=%d",
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

// TestWideFilterDispatchVerticalMatchesPureGo is the bit-exactness guard for the
// vertical-edge six/eight/fourteen-sample kernels (transposing ld4/ld2 NEON on
// arm64) against the pure-Go reference over the same length and threshold
// spread. Every output byte must match.
func TestWideFilterDispatchVerticalMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	for _, k := range wideKernels() {
		var seed int64 = 7000
		for _, params := range wideFilterParamsCorpus() {
			for _, length := range lengths {
				for rep := 0; rep < 4; rep++ {
					runWideKernelVertical(t, k, seed, length, params)
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
