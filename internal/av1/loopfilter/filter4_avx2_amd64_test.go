// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// TestFilter4EdgeAVX2MatchesPureGo is the bit-exactness guard for the AVX2
// narrow deblocking kernel. It calls filter4EdgeAVX2 directly (not through the
// dispatch slot) so the AVX2 asm is exercised even when feature detection has
// not enabled it. The VEX-encoded instructions execute under Rosetta 2 even
// though it does not advertise AVX2 in CPUID, so this test deliberately does
// NOT skip on cpu.Detected.AVX2 == false: that is exactly the configuration
// that previously surfaced a non-byte-exact divergence. Every output byte must
// match the pure-Go reference across the threshold and length matrix.
func TestFilter4EdgeAVX2MatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64}
	var seed int64 = 1
	for _, params := range filter4DispatchParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 8; rep++ {
				runFilter4AVX2(t, seed, length, params)
				seed++
			}
		}
	}
}

// runFilter4AVX2 materialises a flat 8-bit buffer for a horizontal edge
// (outer == 1, step == stride) and compares filter4EdgeAVX2 against the pure-Go
// reference on independent copies.
func runFilter4AVX2(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 8
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}
	step := stride
	outer := 1
	q0Base := 3 * stride // q0 row 3; p1 row 1, p0 row 2, q1 row 4

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4EdgePureGo(want, q0Base, step, outer, length, params)
	filter4EdgeAVX2(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4EdgeAVX2NearFlat stresses the content regime where most positions
// pass needsFilter4 and split across the hev branch (adjacent samples close
// together), the path the kernel's masks and blends must reproduce exactly.
func TestFilter4EdgeAVX2NearFlat(t *testing.T) {
	const stride = 128
	const rows = 8
	for seed := int64(0); seed < 64; seed++ {
		rng := rand.New(rand.NewSource(seed + 500))
		base := make([]byte, stride*rows)
		for i := range base {
			base[i] = byte(120 + rng.Intn(16))
		}
		params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
		q0Base := 3 * stride
		want := append([]byte(nil), base...)
		got := append([]byte(nil), base...)
		filter4EdgePureGo(want, q0Base, stride, 1, 64, params)
		filter4EdgeAVX2(got, q0Base, stride, 1, 64, params)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("near-flat seed=%d idx=%d got=%d want=%d", seed, i, got[i], want[i])
			}
		}
	}
}
