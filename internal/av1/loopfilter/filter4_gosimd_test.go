// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// The Go-native SIMD narrow kernels are called directly, not through the
// dispatch slot, so they are exercised on every host, including amd64 hosts
// whose CPUID does not advertise AVX2 (Rosetta). Each case compares the kernel
// against the scalar reference on independent copies of the same buffer.

// narrowCase describes one narrow-filter edge: horizontal (taps one row apart,
// positions contiguous) or vertical (taps contiguous, positions one row apart).
type narrowCase struct {
	bytesPerSample int
	maxVal         int // largest valid sample at the case's bit depth
	vertical       bool
	length         int
	params         filter4Params
}

// runNarrowCase fills a buffer with seeded content, runs the SIMD kernel and the
// reference on copies, and fails on the first differing byte.
func runNarrowCase(t *testing.T, seed int64, c narrowCase, content func(*rand.Rand, int) int) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	sz := c.bytesPerSample
	var buf []byte
	var q0Base, step, outer int
	if c.vertical {
		// 8 columns of samples; q0 sits in column 4, so p1 (col 2) through q1
		// (col 5) are four contiguous samples on each row.
		const cols = 8
		rows := c.length + 2
		buf = make([]byte, rows*cols*sz)
		step = sz
		outer = cols * sz
		q0Base = 4 * sz
	} else {
		const stride = 64
		const rows = 8
		buf = make([]byte, stride*rows*sz)
		step = stride * sz
		outer = sz
		q0Base = 3 * stride * sz
	}
	for i := 0; i < len(buf); i += sz {
		v := content(rng, c.maxVal)
		buf[i] = byte(v)
		if sz == 2 {
			buf[i+1] = byte(v >> 8)
		}
	}
	want := append([]byte(nil), buf...)
	got := append([]byte(nil), buf...)
	if sz == 1 {
		filter4EdgePureGo(want, q0Base, step, outer, c.length, c.params)
		filter4EdgeSIMD(got, q0Base, step, outer, c.length, c.params)
	} else {
		filter4Edge16PureGo(want, q0Base, step, outer, c.length, c.params)
		filter4Edge16SIMD(got, q0Base, step, outer, c.length, c.params)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d sz=%d vertical=%v len=%d params=%+v idx=%d got=%d want=%d",
				seed, sz, c.vertical, c.length, c.params, i, got[i], want[i])
		}
	}
}

// uniformContent draws each sample uniformly over the sample range.
func uniformContent(rng *rand.Rand, max int) int { return rng.Intn(max + 1) }

// nearFlatContent keeps samples inside a narrow band around mid-range so most
// positions pass needsFilter4 and split across the hev branch.
func nearFlatContent(rng *rand.Rand, max int) int {
	mid := (max + 1) / 2
	return mid - 8 + rng.Intn(16)
}

// narrowParamsCorpus returns threshold sets scaled to a bit depth whose sample
// scale is scale (1 at 8-bit, 4 at 10-bit, 16 at 12-bit).
func narrowParamsCorpus(scale int) []filter4Params {
	mk := func(limit, blimit, hev int) filter4Params {
		return filter4Params{
			limit:  int16(limit * scale),
			blimit: int16(blimit * scale),
			hev:    int16(hev * scale),
			min:    int16(-128 * scale),
			max:    int16(128*scale - 1),
			center: int16(128 * scale),
		}
	}
	return []filter4Params{
		mk(0, 0, 0),
		mk(2, 5, 1),
		mk(8, 20, 4),
		mk(16, 40, 8),
		mk(63, 128, 32),
		mk(255, 510, 0),
		mk(255, 510, 255),
	}
}

// TestFilter4SIMDMatchesPureGo covers both edge orientations, every tail
// length, and both content regimes at 8-bit, 10-bit and 12-bit.
func TestFilter4SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	depths := []struct{ sz, scale, maxVal int }{
		{1, 1, 255},
		{2, 4, 1023},  // 10-bit
		{2, 16, 4095}, // 12-bit
	}
	var seed int64 = 1
	for _, d := range depths {
		for _, params := range narrowParamsCorpus(d.scale) {
			for _, vertical := range []bool{false, true} {
				for _, length := range lengths {
					for _, content := range []func(*rand.Rand, int) int{uniformContent, nearFlatContent} {
						for rep := 0; rep < 3; rep++ {
							c := narrowCase{bytesPerSample: d.sz, maxVal: d.maxVal, vertical: vertical, length: length, params: params}
							runNarrowCase(t, seed, c, content)
							seed++
						}
					}
				}
			}
		}
	}
}
