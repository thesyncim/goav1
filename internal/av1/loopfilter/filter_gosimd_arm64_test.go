// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"math/rand"
	"reflect"
	"runtime"
	"testing"
)

func TestFilterSIMDDispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	checks := []struct {
		name      string
		got, want interface{}
	}{
		{"filter4EdgeImpl", filter4EdgeImpl, filter4EdgeNEON},
		{"filter4Edge16Impl", filter4Edge16Impl, filter4Edge16NEON},
		{"filter6EdgeImpl", filter6EdgeImpl, filter6EdgeNEON},
		{"filter6Edge16Impl", filter6Edge16Impl, filter6Edge16SIMD},
		{"filter8EdgeImpl", filter8EdgeImpl, filter8EdgeNEON},
		{"filter8Edge16Impl", filter8Edge16Impl, filter8Edge16NEON},
		{"filter14EdgeImpl", filter14EdgeImpl, filter14EdgeNEON},
		{"filter14Edge16Impl", filter14Edge16Impl, filter14Edge16SIMD},
	}
	for _, c := range checks {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Errorf("%s = %s, want %s", c.name, got, want)
		}
	}
}

func runFilterWide16SIMD(t *testing.T, name string, vertical bool, seed int64, length int, c wide16Case,
	ref, simd, baseline func([]byte, int, int, int, int, int, filter4Params)) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	maxVal := (1 << c.bitDepth) - 1
	scale, params := wide16Params(c.bitDepth, c.limit, c.blimit, c.hev)
	var strideBytes, rows, step, outer, q0Base int
	if vertical {
		strideBytes, rows = 128, 96
		step, outer = 2, strideBytes
		q0Base = 16 * 2
	} else {
		strideBytes, rows = 256, 32
		step, outer = strideBytes, 2
		q0Base = 16*strideBytes + 16*2
	}
	base := make([]byte, strideBytes*rows)
	fillWide16Content(base, rng, int(seed)%5, maxVal)
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	baselinePixels := append([]byte(nil), base...)
	ref(want, q0Base, step, outer, length, scale, params)
	simd(got, q0Base, step, outer, length, scale, params)
	baseline(baselinePixels, q0Base, step, outer, length, scale, params)
	dir := "H"
	if vertical {
		dir = "V"
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s-16 %s %s seed=%d len=%d idx=%d got=%d want=%d", name, dir, c.name, seed, length, i, got[i], want[i])
		}
		if baselinePixels[i] != want[i] {
			t.Fatalf("%s-16 %s baseline %s seed=%d len=%d idx=%d got=%d want=%d", name, dir, c.name, seed, length, i, baselinePixels[i], want[i])
		}
	}
}

func TestFilter14Edge16SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 4000
	for _, c := range wide16Corpus() {
		for _, length := range lengths {
			for rep := 0; rep < 3; rep++ {
				runFilterWide16SIMD(t, "filter14", false, seed, length, c,
					filter14Edge16PureGo, filter14Edge16SIMD, filter14Edge16PureGoFallback)
				runFilterWide16SIMD(t, "filter14", true, seed+700000, length, c,
					filter14Edge16PureGo, filter14Edge16SIMD, filter14Edge16PureGoFallback)
				seed++
			}
		}
	}
}

func TestFilter6Edge16SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 5000
	for _, c := range wide16Corpus() {
		for _, length := range lengths {
			for rep := 0; rep < 3; rep++ {
				runFilterWide16SIMD(t, "filter6", false, seed, length, c,
					filter6Edge16PureGo, filter6Edge16SIMD, filter6Edge16PureGo)
				runFilterWide16SIMD(t, "filter6", true, seed+800000, length, c,
					filter6Edge16PureGo, filter6Edge16SIMD, filter6Edge16PureGo)
				seed++
			}
		}
	}
}

// TestFilterWide16SIMDExtremePixels drives min/max 10/12-bit samples with
// maximal thresholds: the all-max 12-bit pattern pushes the fourteen-tap sum
// to its 65528 ceiling, the exact case the -32768 accumulator offset exists
// for the 12-bit SIMD implementation.
func TestFilterWide16SIMDExtremePixels(t *testing.T) {
	const strideBytes = 256
	const rows = 32
	writeS := func(buf []byte, off, v int) {
		buf[off] = byte(v)
		buf[off+1] = byte(v >> 8)
	}
	for _, bd := range []uint8{10, 12} {
		maxVal := (1 << bd) - 1
		patterns := []func(row, col int) int{
			func(row, col int) int { return maxVal },
			func(row, col int) int { return 0 },
			func(row, col int) int { return (col % 2) * maxVal },
			func(row, col int) int { return (row % 2) * maxVal },
			func(row, col int) int { return ((col / 8) % 2) * maxVal },
		}
		for _, thr := range [][3]int{{255, 510, 255}, {255, 510, 0}, {0, 0, 0}} {
			scale, params := wide16Params(bd, thr[0], thr[1], thr[2])
			for pi, fill := range patterns {
				base := make([]byte, strideBytes*rows)
				for row := 0; row < rows; row++ {
					for col := 0; col < strideBytes/2; col++ {
						writeS(base, row*strideBytes+col*2, fill(row, col))
					}
				}
				q0Base := 16 * strideBytes
				for _, k := range []struct {
					name string
					ref  func([]byte, int, int, int, int, int, filter4Params)
					simd func([]byte, int, int, int, int, int, filter4Params)
				}{
					{"filter14", filter14Edge16PureGo, filter14Edge16SIMD},
					{"filter6", filter6Edge16PureGo, filter6Edge16SIMD},
				} {
					want := append([]byte(nil), base...)
					got := append([]byte(nil), base...)
					k.ref(want, q0Base, strideBytes, 2, 64, scale, params)
					k.simd(got, q0Base, strideBytes, 2, 64, scale, params)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("%s-16 extreme bd=%d thr=%v pattern=%d idx=%d got=%d want=%d", k.name, bd, thr, pi, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestFilterSIMDZeroAlloc checks the selected high-bit-depth SIMD hot paths.
func TestFilterSIMDZeroAlloc(t *testing.T) {
	const stride = 256
	const rows = 80
	pix := make([]byte, stride*rows)
	rng := rand.New(rand.NewSource(9))
	for i := 0; i+1 < len(pix); i += 2 {
		v := rng.Intn(1024)
		pix[i], pix[i+1] = byte(v), byte(v>>8)
	}
	params := filter4Params{limit: 64, blimit: 160, hev: 32, min: -512, max: 511, center: 512}
	cases := []struct {
		name string
		fn   func()
	}{
		{"filter6Edge16SIMD", func() { filter6Edge16SIMD(pix, 8*stride+32, stride, 2, 64, 4, params) }},
		{"filter6Edge16SIMD vertical", func() { filter6Edge16SIMD(pix, 32, 2, stride, 64, 4, params) }},
		{"filter14Edge16SIMD", func() { filter14Edge16SIMD(pix, 8*stride+32, stride, 2, 64, 4, params) }},
		{"filter14Edge16SIMD_V", func() { filter14Edge16SIMD(pix, 32, 2, stride, 8, 4, params) }},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %.1f objects/run, want 0", c.name, a)
		}
	}
}
