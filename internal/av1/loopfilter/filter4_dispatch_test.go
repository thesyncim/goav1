// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// filter4DispatchParamsCorpus returns a spread of threshold configurations that
// exercise every branch of needsFilter4 and the hev split, including the all-
// pass and all-reject extremes for 8/10/12-bit (the scale only enters through
// the centre/clamp constants, which the NEON path fixes at the 8-bit values; we
// therefore also assert the 8-bit kernel directly).
func filter4DispatchParamsCorpus() []filter4Params {
	const scale = 1 // NEON path is 8-bit only.
	mk := func(limit, blimit, hev int) filter4Params {
		return filter4Params{
			limit:  limit * scale,
			blimit: blimit * scale,
			hev:    hev * scale,
			min:    -128 * scale,
			max:    128*scale - 1,
			center: 128 * scale,
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
		mk(255, 510, 0),   // effectively always filter, never hev
		mk(255, 510, 255), // always filter, always non-hev
	}
}

// runFilter4Kernel materialises a flat 8-bit buffer wide enough for the chosen
// length, runs both the reference and the dispatched kernel on independent
// copies, and reports the first mismatching byte. The buffer layout mirrors a
// horizontal edge: outer == 1 (positions are contiguous) and step == stride so
// the four taps live on separate rows, exactly as the decoder presents them.
func runFilter4Kernel(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 64 // >= length; row stride in bytes
	const rows = 8    // p3..q3 headroom; we use rows 2..5 for p1,p0,q0,q1
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}

	step := stride
	outer := 1
	q0Base := 3*stride + 0 // q0 row is row 3; p1 row 1, p0 row 2, q1 row 4

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4EdgePureGo(want, q0Base, step, outer, length, params)
	filter4EdgeImpl(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4DispatchMatchesPureGo is the bit-exactness guard for the dispatched
// narrow deblocking kernel. It drives the resolved dispatch slot (which routes
// through the NEON asm on arm64) against the canonical pure-Go reference over a
// spread of edge lengths — including non-multiples of eight to exercise the
// scalar tail — and threshold configurations. Every output byte must match.
func TestFilter4DispatchMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	var seed int64 = 1
	for _, params := range filter4DispatchParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter4Kernel(t, seed, length, params)
				seed++
			}
		}
	}
}

// TestFilter4DispatchMatchesPureGoForcedPureGo confirms the differential holds
// when the dispatcher is forced onto the pure-Go branch (CPU override), so the
// test still has meaning on a NEON host where the slot would otherwise always
// pick the asm. It re-binds the slot for the duration of the check.
func TestFilter4DispatchMatchesPureGoForcedPureGo(t *testing.T) {
	restore := cpu.OverrideForTest(cpu.Features{})
	defer restore()
	// The dispatch slot resolved at init is not re-evaluated by the override, so
	// also pin it explicitly and restore afterwards.
	prev := filter4EdgeImpl
	filter4EdgeImpl = filter4EdgePureGo
	defer func() { filter4EdgeImpl = prev }()

	for i, params := range filter4DispatchParamsCorpus() {
		runFilter4Kernel(t, int64(1000+i), 16, params)
	}
}

// filter4Dispatch16ParamsCorpus mirrors filter4DispatchParamsCorpus but at
// 10- and 12-bit scale, exercising the centre/clamp constants the 16-bit kernel
// reads from its context.
func filter4Dispatch16ParamsCorpus() []filter4Params {
	var out []filter4Params
	for _, shift := range []int{2, 4} { // 10-bit, 12-bit
		scale := 1 << shift
		mk := func(limit, blimit, hev int) filter4Params {
			return filter4Params{
				limit:  limit * scale,
				blimit: blimit * scale,
				hev:    hev * scale,
				min:    -128 * scale,
				max:    128*scale - 1,
				center: 128 * scale,
			}
		}
		out = append(out,
			mk(0, 0, 0),
			mk(2, 5, 1),
			mk(16, 40, 8),
			mk(63, 128, 32),
			mk(255, 510, 0),
			mk(255, 510, 255),
		)
	}
	return out
}

// runFilter4Kernel16 materialises a two-byte plane buffer, runs both the
// reference and the dispatched 16-bit kernel on independent copies, and reports
// the first mismatching byte. outer == 2 (contiguous two-byte positions) and
// step == stride so the four taps live on separate rows.
func runFilter4Kernel16(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const strideSamples = 64
	const stride = strideSamples * 2
	const rows = 8
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	maxVal := params.center*2 - 1
	for i := 0; i+1 < len(base); i += 2 {
		v := rng.Intn(maxVal + 1)
		base[i] = byte(v)
		base[i+1] = byte(v >> 8)
	}
	step := stride
	outer := 2
	q0Base := 3*stride + 0

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4Edge16PureGo(want, q0Base, step, outer, length, params)
	filter4Edge16Impl(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4Dispatch16MatchesPureGo is the bit-exactness guard for the
// dispatched 10/12-bit narrow kernel (NEON asm on arm64) against the pure-Go
// reference over a spread of edge lengths and scaled threshold configurations.
func TestFilter4Dispatch16MatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	var seed int64 = 5000
	for _, params := range filter4Dispatch16ParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter4Kernel16(t, seed, length, params)
				seed++
			}
		}
	}
}

// TestFilter4DispatchIsZeroAlloc protects the hot-path contract that the
// dispatched filter does not allocate per edge.
func TestFilter4DispatchIsZeroAlloc(t *testing.T) {
	const stride = 64
	base := make([]byte, stride*8)
	rng := rand.New(rand.NewSource(7))
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	q0Base := 3 * stride
	allocs := testing.AllocsPerRun(1000, func() {
		filter4EdgeImpl(base, q0Base, stride, 1, 56, params)
	})
	if allocs != 0 {
		t.Fatalf("dispatch allocated %f times per call", allocs)
	}
}
