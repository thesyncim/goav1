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

	"simd/archsimd"
)

// TestFilterKernelSIMDCanonical exercises the outlined canonical kernel forms
// (filter4TapVecs, filter8KernelSIMD) that the hot edge loops paste inline, so a
// future edit to one and not the other is caught. It checks a single 8-position
// group against the scalar per-sample reference for a spread of thresholds and
// content.
func TestFilterKernelSIMDCanonical(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4b))
	load := func(v [8]int16) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&v) }
	for _, params := range wideFilterParamsCorpus() {
		cst4 := makeFilter4SIMDConst(params)
		cst8 := makeFilter8SIMDConst(1, params)
		wk4 := params.widen()
		for iter := 0; iter < 40; iter++ {
			var s [8][8]int16 // taps p3..q3 across 8 lanes
			for tap := 0; tap < 8; tap++ {
				for lane := 0; lane < 8; lane++ {
					switch iter % 3 {
					case 0:
						s[tap][lane] = int16(rng.Intn(256))
					case 1:
						s[tap][lane] = int16(120 + rng.Intn(16))
					default:
						s[tap][lane] = int16(127 + rng.Intn(3))
					}
				}
			}
			p3, p2, p1, p0 := load(s[0]), load(s[1]), load(s[2]), load(s[3])
			q0, q1, q2, q3 := load(s[4]), load(s[5]), load(s[6]), load(s[7])

			// filter4 canonical vs scalar.
			g4p1, g4p0, g4q0, g4q1 := filter4TapVecs(p1, p0, q0, q1, cst4)
			var o4p1, o4p0, o4q0, o4q1 [8]int16
			g4p1.StoreArray(&o4p1)
			g4p0.StoreArray(&o4p0)
			g4q0.StoreArray(&o4q0)
			g4q1.StoreArray(&o4q1)
			for lane := 0; lane < 8; lane++ {
				ep1, ep0, eq0, eq1 := int(s[2][lane]), int(s[3][lane]), int(s[4][lane]), int(s[5][lane])
				if needsFilter4(ep1, ep0, eq0, eq1, wk4) {
					ep1, ep0, eq0, eq1 = filter4Samples(ep1, ep0, eq0, eq1, wk4)
				}
				if int(o4p1[lane]) != ep1 || int(o4p0[lane]) != ep0 || int(o4q0[lane]) != eq0 || int(o4q1[lane]) != eq1 {
					t.Fatalf("filter4 kernel lane=%d got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)",
						lane, o4p1[lane], o4p0[lane], o4q0[lane], o4q1[lane], ep1, ep0, eq0, eq1)
				}
			}

			// filter8 canonical vs scalar.
			g8p2, g8p1, g8p0, g8q0, g8q1, g8q2 := filter8KernelSIMD(p3, p2, p1, p0, q0, q1, q2, q3, cst8)
			var o8 [6][8]int16
			g8p2.StoreArray(&o8[0])
			g8p1.StoreArray(&o8[1])
			g8p0.StoreArray(&o8[2])
			g8q0.StoreArray(&o8[3])
			g8q1.StoreArray(&o8[4])
			g8q2.StoreArray(&o8[5])
			for lane := 0; lane < 8; lane++ {
				ep3, ep2, ep1, ep0 := int(s[0][lane]), int(s[1][lane]), int(s[2][lane]), int(s[3][lane])
				eq0, eq1, eq2, eq3 := int(s[4][lane]), int(s[5][lane]), int(s[6][lane]), int(s[7][lane])
				wp2, wp1, wp0, wq0, wq1, wq2 := ep2, ep1, ep0, eq0, eq1, eq2
				if needsFilter8(ep3, ep2, ep1, ep0, eq0, eq1, eq2, eq3, wk4) {
					wp2, wp1, wp0, wq0, wq1, wq2 = filter8Samples(ep3, ep2, ep1, ep0, eq0, eq1, eq2, eq3, 1, wk4)
				}
				want := [6]int{wp2, wp1, wp0, wq0, wq1, wq2}
				for k := 0; k < 6; k++ {
					if int(o8[k][lane]) != want[k] {
						t.Fatalf("filter8 kernel lane=%d out=%d got=%d want=%d", lane, k, o8[k][lane], want[k])
					}
				}
			}
		}
	}
}

// TestFilterSIMDDispatchBound is the FuncForPC probe: under the goexperiment.simd
// build every narrow and wide dispatch slot (8-bit and 16-bit) must resolve to
// the Go-native SIMD kernels.
func TestFilterSIMDDispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	checks := []struct {
		name      string
		got, want interface{}
	}{
		{"filter4EdgeImpl", filter4EdgeImpl, filter4EdgeSIMD},
		{"filter4Edge16Impl", filter4Edge16Impl, filter4Edge16SIMD},
		{"filter8EdgeImpl", filter8EdgeImpl, filter8EdgeSIMD},
		{"filter8Edge16Impl", filter8Edge16Impl, filter8Edge16SIMD},
		{"filter6EdgeImpl", filter6EdgeImpl, filter6EdgeSIMD},
		{"filter6Edge16Impl", filter6Edge16Impl, filter6Edge16SIMD},
		{"filter14EdgeImpl", filter14EdgeImpl, filter14EdgeSIMD},
		{"filter14Edge16Impl", filter14Edge16Impl, filter14Edge16SIMD},
	}
	for _, c := range checks {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Errorf("%s = %s, want %s", c.name, got, want)
		}
	}
}

// --- narrow (filter4) SIMD differential -------------------------------------

// runFilter4SIMDHorizontal materialises an 8-bit horizontal-edge buffer and
// checks filter4EdgeSIMD byte-for-byte against filter4EdgePureGo.
func runFilter4SIMDHorizontal(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 8
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := stride, 1
	q0Base := 3 * stride
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	filter4EdgePureGo(want, q0Base, step, outer, length, params)
	filter4EdgeSIMD(got, q0Base, step, outer, length, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter4 H seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
	}
}

// runFilter4SIMDVertical checks the vertical orientation (strided taps): the
// SIMD wrapper must route to the NEON transpose path and stay byte-exact.
func runFilter4SIMDVertical(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 96
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := 1, stride
	q0Base := 16
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	filter4EdgePureGo(want, q0Base, step, outer, length, params)
	filter4EdgeSIMD(got, q0Base, step, outer, length, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter4 V seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
	}
}

func TestFilter4SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 1
	for _, params := range wideFilterParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter4SIMDHorizontal(t, seed, length, params)
				runFilter4SIMDVertical(t, seed+100000, length, params)
				seed++
			}
		}
	}
}

// runFilter4Edge16SIMD checks the 10/12-bit narrow SIMD kernel for one
// orientation.
func runFilter4Edge16SIMD(t *testing.T, vertical bool, seed int64, length int, c wide16Case) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	maxVal := (1 << c.bitDepth) - 1
	scale, params := wide16Params(c.bitDepth, c.limit, c.blimit, c.hev)
	_ = scale
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
	filter4Edge16PureGo(want, q0Base, step, outer, length, params)
	filter4Edge16SIMD(got, q0Base, step, outer, length, params)
	dir := "H"
	if vertical {
		dir = "V"
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter4-16 %s %s seed=%d len=%d idx=%d got=%d want=%d", dir, c.name, seed, length, i, got[i], want[i])
		}
	}
}

func TestFilter4Edge16SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 500
	for _, c := range wide16Corpus() {
		for _, length := range lengths {
			for rep := 0; rep < 3; rep++ {
				runFilter4Edge16SIMD(t, false, seed, length, c)
				runFilter4Edge16SIMD(t, true, seed+200000, length, c)
				seed++
			}
		}
	}
}

// --- wide (filter8) SIMD differential ---------------------------------------

func runFilter8SIMDHorizontal(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := stride, 1
	q0Base := 8 * stride
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	filter8EdgePureGo(want, q0Base, step, outer, length, 1, params)
	filter8EdgeSIMD(got, q0Base, step, outer, length, 1, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter8 H seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
	}
}

func runFilter8SIMDVertical(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 96
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := 1, stride
	q0Base := 16
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	filter8EdgePureGo(want, q0Base, step, outer, length, 1, params)
	filter8EdgeSIMD(got, q0Base, step, outer, length, 1, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter8 V seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
	}
}

func TestFilter8SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 1000
	for _, params := range wideFilterParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter8SIMDHorizontal(t, seed, length, params)
				runFilter8SIMDVertical(t, seed+300000, length, params)
				seed++
			}
		}
	}
}

func runFilter8Edge16SIMD(t *testing.T, vertical bool, seed int64, length int, c wide16Case) {
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
	filter8Edge16PureGo(want, q0Base, step, outer, length, scale, params)
	filter8Edge16SIMD(got, q0Base, step, outer, length, scale, params)
	dir := "H"
	if vertical {
		dir = "V"
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter8-16 %s %s seed=%d len=%d idx=%d got=%d want=%d", dir, c.name, seed, length, i, got[i], want[i])
		}
	}
}

func TestFilter8Edge16SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 1500
	for _, c := range wide16Corpus() {
		for _, length := range lengths {
			for rep := 0; rep < 3; rep++ {
				runFilter8Edge16SIMD(t, false, seed, length, c)
				runFilter8Edge16SIMD(t, true, seed+400000, length, c)
				seed++
			}
		}
	}
}

// --- wide (filter6) SIMD differential ----------------------------------------

func runFilter6SIMDHorizontal(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := stride, 1
	q0Base := 8 * stride
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	asm := append([]byte(nil), base...)
	filter6EdgePureGo(want, q0Base, step, outer, length, 1, params)
	filter6EdgeSIMD(got, q0Base, step, outer, length, 1, params)
	filter6EdgeNEON(asm, q0Base, step, outer, length, 1, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter6 H vs purego seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
		if asm[i] != want[i] {
			t.Fatalf("filter6 H asm vs purego seed=%d len=%d params=%+v idx=%d asm=%d want=%d", seed, length, params, i, asm[i], want[i])
		}
	}
}

func runFilter6SIMDVertical(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 96
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	filter6EdgePureGo(want, 16, 1, stride, length, 1, params)
	filter6EdgeSIMD(got, 16, 1, stride, length, 1, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter6 V seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
	}
}

func TestFilter6SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 3000
	for _, params := range wideFilterParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter6SIMDHorizontal(t, seed, length, params)
				runFilter6SIMDVertical(t, seed+600000, length, params)
				seed++
			}
		}
	}
}

// TestFilter6SIMDExtremePixels mirrors the filter14 extreme-value guard.
func TestFilter6SIMDExtremePixels(t *testing.T) {
	const stride = 128
	const rows = 16
	patterns := []func(row, col int) byte{
		func(row, col int) byte { return 255 },
		func(row, col int) byte { return 0 },
		func(row, col int) byte { return byte((col % 2) * 255) },
		func(row, col int) byte { return byte((row % 2) * 255) },
		func(row, col int) byte { return byte(255 * ((col / 8) % 2)) },
	}
	params := []filter4Params{
		{limit: 255, blimit: 510, hev: 255, min: -128, max: 127, center: 128},
		{limit: 255, blimit: 510, hev: 0, min: -128, max: 127, center: 128},
		{limit: 0, blimit: 0, hev: 0, min: -128, max: 127, center: 128},
	}
	for pi, fill := range patterns {
		for _, p := range params {
			base := make([]byte, stride*rows)
			for row := 0; row < rows; row++ {
				for col := 0; col < stride; col++ {
					base[row*stride+col] = fill(row, col)
				}
			}
			want := append([]byte(nil), base...)
			got := append([]byte(nil), base...)
			filter6EdgePureGo(want, 8*stride, stride, 1, 64, 1, p)
			filter6EdgeSIMD(got, 8*stride, stride, 1, 64, 1, p)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("filter6 extreme pattern=%d params=%+v idx=%d got=%d want=%d", pi, p, i, got[i], want[i])
				}
			}
		}
	}
}

// --- wide (filter14) SIMD differential ---------------------------------------

func runFilter14SIMDHorizontal(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := stride, 1
	q0Base := 8 * stride
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	asm := append([]byte(nil), base...)
	filter14EdgePureGo(want, q0Base, step, outer, length, 1, params)
	filter14EdgeSIMD(got, q0Base, step, outer, length, 1, params)
	// Differential against the NEON asm as well: all three must agree.
	filter14EdgeNEON(asm, q0Base, step, outer, length, 1, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter14 H vs purego seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
		if asm[i] != want[i] {
			t.Fatalf("filter14 H asm vs purego seed=%d len=%d params=%+v idx=%d asm=%d want=%d", seed, length, params, i, asm[i], want[i])
		}
	}
}

func runFilter14SIMDVertical(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 96
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step, outer := 1, stride
	q0Base := 16
	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	filter14EdgePureGo(want, q0Base, step, outer, length, 1, params)
	filter14EdgeSIMD(got, q0Base, step, outer, length, 1, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filter14 V seed=%d len=%d params=%+v idx=%d got=%d want=%d", seed, length, params, i, got[i], want[i])
		}
	}
}

func TestFilter14SIMDMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	var seed int64 = 2000
	for _, params := range wideFilterParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter14SIMDHorizontal(t, seed, length, params)
				runFilter14SIMDVertical(t, seed+500000, length, params)
				seed++
			}
		}
	}
}

// TestFilter14SIMDMixedGroups builds edges whose eight-position groups
// alternate between wide-flat, filter8-flat, narrow and rejected content, so a
// single kernel invocation walks every branch of the ladder, including groups
// where lanes disagree (mixed masks inside one vector).
func TestFilter14SIMDMixedGroups(t *testing.T) {
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(77))
	for _, params := range wideFilterParamsCorpus() {
		base := make([]byte, stride*rows)
		for col := 0; col < stride; col++ {
			regime := (col / 4) % 5 // 4-wide regimes split every vector group
			for row := 0; row < rows; row++ {
				var v byte
				switch regime {
				case 0:
					v = byte(rng.Intn(256)) // reject/narrow
				case 1:
					v = byte(127 + rng.Intn(3)) // wide flat
				case 2:
					v = byte(120 + rng.Intn(16)) // near flat (filter8-ish)
				case 3:
					v = byte(rng.Intn(4)) // flat at clamp-low
				default:
					v = byte(252 + rng.Intn(4)) // flat at clamp-high
				}
				base[row*stride+col] = v
			}
		}
		want := append([]byte(nil), base...)
		got := append([]byte(nil), base...)
		filter14EdgePureGo(want, 8*stride, stride, 1, 128, 1, params)
		filter14EdgeSIMD(got, 8*stride, stride, 1, 128, 1, params)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("filter14 mixed params=%+v idx=%d got=%d want=%d", params, i, got[i], want[i])
			}
		}
	}
}

// TestFilter14SIMDExtremePixels drives min/max samples with maximal thresholds
// so the widest sums and the clamp edges are exercised (per the skill: a guard
// removed may have been masking a divergence — there is no guard here, the
// arithmetic must hold at the extremes).
func TestFilter14SIMDExtremePixels(t *testing.T) {
	const stride = 128
	const rows = 16
	patterns := []func(row, col int) byte{
		func(row, col int) byte { return 255 },                       // all-max: largest wide sums
		func(row, col int) byte { return 0 },                         // all-min
		func(row, col int) byte { return byte((col % 2) * 255) },     // 0/255 columns: max lane diffs
		func(row, col int) byte { return byte((row % 2) * 255) },     // 0/255 rows: max tap diffs
		func(row, col int) byte { return byte(255 * ((col / 8) % 2)) }, // group-alternating extremes
	}
	params := []filter4Params{
		{limit: 255, blimit: 510, hev: 255, min: -128, max: 127, center: 128},
		{limit: 255, blimit: 510, hev: 0, min: -128, max: 127, center: 128},
		{limit: 0, blimit: 0, hev: 0, min: -128, max: 127, center: 128},
	}
	for pi, fill := range patterns {
		for _, p := range params {
			base := make([]byte, stride*rows)
			for row := 0; row < rows; row++ {
				for col := 0; col < stride; col++ {
					base[row*stride+col] = fill(row, col)
				}
			}
			want := append([]byte(nil), base...)
			got := append([]byte(nil), base...)
			filter14EdgePureGo(want, 8*stride, stride, 1, 64, 1, p)
			filter14EdgeSIMD(got, 8*stride, stride, 1, 64, 1, p)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("filter14 extreme pattern=%d params=%+v idx=%d got=%d want=%d", pi, p, i, got[i], want[i])
				}
			}
		}
	}
}

// --- wide 16-bit (filter6/filter14) SIMD differential -------------------------

func runFilterWide16SIMD(t *testing.T, name string, vertical bool, seed int64, length int, c wide16Case,
	ref, simd, neon func([]byte, int, int, int, int, int, filter4Params)) {
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
	asm := append([]byte(nil), base...)
	ref(want, q0Base, step, outer, length, scale, params)
	simd(got, q0Base, step, outer, length, scale, params)
	neon(asm, q0Base, step, outer, length, scale, params)
	dir := "H"
	if vertical {
		dir = "V"
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s-16 %s %s seed=%d len=%d idx=%d got=%d want=%d", name, dir, c.name, seed, length, i, got[i], want[i])
		}
		if asm[i] != want[i] {
			t.Fatalf("%s-16 %s asm %s seed=%d len=%d idx=%d asm=%d want=%d", name, dir, c.name, seed, length, i, asm[i], want[i])
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
					filter14Edge16PureGo, filter14Edge16SIMD, filter14Edge16NEON)
				runFilterWide16SIMD(t, "filter14", true, seed+700000, length, c,
					filter14Edge16PureGo, filter14Edge16SIMD, filter14Edge16NEON)
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
					filter6Edge16PureGo, filter6Edge16SIMD, filter6Edge16NEON)
				runFilterWide16SIMD(t, "filter6", true, seed+800000, length, c,
					filter6Edge16PureGo, filter6Edge16SIMD, filter6Edge16NEON)
				seed++
			}
		}
	}
}

// TestFilterWide16SIMDExtremePixels drives min/max 10/12-bit samples with
// maximal thresholds: the all-max 12-bit pattern pushes the fourteen-tap sum
// to its 65528 ceiling, the exact case the -32768 accumulator offset exists
// for (and the case the NEON asm refuses).
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

// --- zero-alloc guard -------------------------------------------------------

// TestFilterSIMDZeroAlloc confirms the SIMD kernels do not heap-allocate on the
// hot horizontal path (array-pointer loads/stores, stack-resident vectors).
func TestFilterSIMDZeroAlloc(t *testing.T) {
	const stride = 128
	const rows = 16
	pix := make([]byte, stride*rows)
	rng := rand.New(rand.NewSource(9))
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	params := filter4Params{limit: 32, blimit: 80, hev: 16, min: -128, max: 127, center: 128}
	pix16 := make([]byte, stride*rows*2)
	for i := range pix16 {
		pix16[i] = byte(rng.Intn(256))
	}
	params16 := filter4Params{limit: 32, blimit: 80, hev: 16, min: -512, max: 511, center: 512}

	cases := []struct {
		name string
		fn   func()
	}{
		{"filter4EdgeSIMD", func() { filter4EdgeSIMD(pix, 3*stride, stride, 1, 64, params) }},
		{"filter6EdgeSIMD", func() { filter6EdgeSIMD(pix, 8*stride, stride, 1, 64, 1, params) }},
		{"filter8EdgeSIMD", func() { filter8EdgeSIMD(pix, 8*stride, stride, 1, 64, 1, params) }},
		{"filter14EdgeSIMD", func() { filter14EdgeSIMD(pix, 8*stride, stride, 1, 64, 1, params) }},
		{"filter14EdgeSIMD_V", func() { filter14EdgeSIMD(pix, 16, 1, stride, 8, 1, params) }},
		{"filter4Edge16SIMD", func() { filter4Edge16SIMD(pix16, 8*stride*2+32, stride*2, 2, 64, params16) }},
		{"filter8Edge16SIMD", func() { filter8Edge16SIMD(pix16, 8*stride*2+32, stride*2, 2, 64, 4, params16) }},
		{"filter6Edge16SIMD", func() { filter6Edge16SIMD(pix16, 8*stride*2+32, stride*2, 2, 64, 4, params16) }},
		{"filter14Edge16SIMD", func() { filter14Edge16SIMD(pix16, 8*stride*2+32, stride*2, 2, 64, 4, params16) }},
		{"filter14Edge16SIMD_V", func() { filter14Edge16SIMD(pix16, 16*2, 2, stride*2, 8, 4, params16) }},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %.1f objects/run, want 0", c.name, a)
		}
	}
}
