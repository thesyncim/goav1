// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// The Go-native SIMD wide kernels (six-, eight- and fourteen-tap) are called
// directly, not through the dispatch slot, so they are exercised on every host,
// including amd64 hosts whose CPUID does not advertise AVX2 (Rosetta). Each case
// compares the kernel against the scalar reference on independent copies of the
// same buffer.

// wideSIMDKernel pairs a Go SIMD wrapper with its scalar reference for one
// filter family and sample width. pad is the number of taps before q0 and taps
// is the window size.
type wideSIMDKernel struct {
	name string
	pad  int
	taps int
	bps  int
	ref  func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	simd func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
}

func wideSIMDKernels() []wideSIMDKernel {
	return []wideSIMDKernel{
		{"filter6", 3, 6, 1, filter6EdgePureGo, filter6EdgeSIMD},
		{"filter8", 4, 8, 1, filter8EdgePureGo, filter8EdgeSIMD},
		{"filter14", 7, 14, 1, filter14EdgePureGo, filter14EdgeSIMD},
		{"filter6-16", 3, 6, 2, filter6Edge16PureGo, filter6Edge16SIMD},
		{"filter8-16", 4, 8, 2, filter8Edge16PureGo, filter8Edge16SIMD},
		{"filter14-16", 7, 14, 2, filter14Edge16PureGo, filter14Edge16SIMD},
	}
}

// wideSIMDLayout returns the buffer, q0 offset, tap step and position stride for
// one orientation. The horizontal layout has q0 on row 8 of a 16-row buffer;
// the vertical layout has q0 in column 8 of a 16-column buffer, so every window
// lies inside the buffer for taps up to 14.
func wideSIMDLayout(bps int, length int, vertical bool) (buf []byte, q0Base, step, outer int) {
	if vertical {
		const cols = 16
		rows := length + 2
		buf = make([]byte, rows*cols*bps)
		return buf, 8 * bps, bps, cols * bps
	}
	const stride = 64
	const rows = 16
	buf = make([]byte, stride*rows*bps)
	return buf, 8 * stride * bps, stride * bps, bps
}

// wideSIMDContent fills one sample with the content regime selected by mode:
// 0 uniform, 1 near-flat around mid-range, 2 flat within the scale, 3 extremes.
func wideSIMDContent(rng *rand.Rand, mode, scale, maxVal int) int {
	mid := (maxVal + 1) / 2
	switch mode {
	case 0:
		return rng.Intn(maxVal + 1)
	case 1:
		return mid - 8*scale + rng.Intn(16*scale)
	case 2:
		return mid + rng.Intn(scale+1)
	}
	if rng.Intn(2) == 0 {
		return 0
	}
	return maxVal
}

// runWideSIMDCase runs the SIMD kernel and the reference on copies of one seeded
// buffer and fails on the first differing byte.
func runWideSIMDCase(t *testing.T, seed int64, k wideSIMDKernel, length int, vertical bool, mode, scale, maxVal int, params filter4Params) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	buf, q0Base, step, outer := wideSIMDLayout(k.bps, length, vertical)
	for i := 0; i < len(buf); i += k.bps {
		v := wideSIMDContent(rng, mode, scale, maxVal)
		buf[i] = byte(v)
		if k.bps == 2 {
			buf[i+1] = byte(v >> 8)
		}
	}
	want := append([]byte(nil), buf...)
	got := append([]byte(nil), buf...)
	k.ref(want, q0Base, step, outer, length, scale, params)
	k.simd(got, q0Base, step, outer, length, scale, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s seed=%d vertical=%v len=%d mode=%d params=%+v idx=%d got=%d want=%d",
				k.name, seed, vertical, length, mode, params, i, got[i], want[i])
		}
	}
}

// TestWideSIMDMatchesPureGo covers every wide kernel at 8, 10 and 12 bits, both
// edge orientations, every tail length, and all four content regimes. The flat
// and flat2 branches only fire on near-flat content, so the regimes matter.
func TestWideSIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	depths := []struct{ bps, scale, maxVal int }{
		{1, 1, 255},
		{2, 4, 1023},
		{2, 16, 4095},
	}
	var seed int64 = 9000
	for _, k := range wideSIMDKernels() {
		for _, d := range depths {
			if (k.bps == 1) != (d.bps == 1) {
				continue
			}
			for _, params := range narrowParamsCorpus(d.scale) {
				for _, vertical := range []bool{false, true} {
					for _, length := range lengths {
						for mode := 0; mode < 4; mode++ {
							for rep := 0; rep < 2; rep++ {
								runWideSIMDCase(t, seed, k, length, vertical, mode, d.scale, d.maxVal, params)
								seed++
							}
						}
					}
				}
			}
		}
	}
}

// TestWideSIMDZeroAlloc guards the hot-path contract that the Go SIMD edge
// kernels, including the vertical gather/scatter path, do not allocate.
func TestWideSIMDZeroAlloc(t *testing.T) {
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	for _, k := range wideSIMDKernels() {
		for _, vertical := range []bool{false, true} {
			buf, q0Base, step, outer := wideSIMDLayout(k.bps, 64, vertical)
			for i := range buf {
				buf[i] = byte(100 + i%7)
			}
			allocs := testing.AllocsPerRun(200, func() {
				k.simd(buf, q0Base, step, outer, 64, 1, params)
			})
			if allocs != 0 {
				t.Fatalf("%s vertical=%v allocated %.1f objects/run, want 0", k.name, vertical, allocs)
			}
		}
	}
}
