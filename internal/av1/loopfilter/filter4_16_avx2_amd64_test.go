// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// TestFilter4Edge16AVX2MatchesPureGo is the bit-exactness guard for the AVX2
// 10/12-bit narrow deblocking kernel. It calls filter4Edge16AVX2 directly (not
// through the dispatch slot) so the AVX2 asm is exercised even when feature
// detection has not enabled it: the VEX-encoded instructions execute under
// Rosetta 2 even though it does not advertise AVX2 in CPUID, so this test
// deliberately does NOT skip on cpu.Detected.AVX2 == false. Every output byte
// must match the pure-Go reference across the threshold and length matrix at
// both 10- and 12-bit scale.
func TestFilter4Edge16AVX2MatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64}
	var seed int64 = 1
	for _, params := range filter4Dispatch16ParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 8; rep++ {
				runFilter4Edge16AVX2(t, seed, length, params)
				seed++
			}
		}
	}
}

// runFilter4Edge16AVX2 materialises a two-byte plane buffer for a horizontal
// edge (outer == 2, step == stride) and compares filter4Edge16AVX2 against the
// pure-Go reference on independent copies.
func runFilter4Edge16AVX2(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const strideSamples = 128
	const stride = strideSamples * 2
	const rows = 8
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	maxVal := int(params.center)*2 - 1
	for i := 0; i+1 < len(base); i += 2 {
		v := rng.Intn(maxVal + 1)
		base[i] = byte(v)
		base[i+1] = byte(v >> 8)
	}
	step := stride
	outer := 2
	q0Base := 3 * stride

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4Edge16PureGo(want, q0Base, step, outer, length, params)
	filter4Edge16AVX2(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4Edge16AVX2NearFlat stresses the content regime where most positions
// pass needsFilter4 and split across the hev branch (adjacent samples close
// together), the path the kernel's masks and blends must reproduce exactly.
func TestFilter4Edge16AVX2NearFlat(t *testing.T) {
	const strideSamples = 128
	const stride = strideSamples * 2
	const rows = 8
	for _, shift := range []int{2, 4} {
		scale := 1 << shift
		center := 128 * scale
		params := filter4Params{
			limit:  int16(16 * scale),
			blimit: int16(40 * scale),
			hev:    int16(8 * scale),
			min:    int16(-128 * scale),
			max:    int16(128*scale - 1),
			center: int16(center),
		}
		for seed := int64(0); seed < 64; seed++ {
			rng := rand.New(rand.NewSource(seed + 900))
			base := make([]byte, stride*rows)
			for i := 0; i+1 < len(base); i += 2 {
				v := center - 8 + rng.Intn(16)
				base[i] = byte(v)
				base[i+1] = byte(v >> 8)
			}
			q0Base := 3 * stride
			want := append([]byte(nil), base...)
			got := append([]byte(nil), base...)
			filter4Edge16PureGo(want, q0Base, stride, 2, 64, params)
			filter4Edge16AVX2(got, q0Base, stride, 2, 64, params)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("near-flat scale=%d seed=%d idx=%d got=%d want=%d", scale, seed, i, got[i], want[i])
				}
			}
		}
	}
}
