//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

import (
	"bytes"
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

// TestFilterSIMDExactMinimumWindow covers the final load and store at the end
// of a slice with no padding. The vertical cases also hit one full tile plus
// an odd tail, and the fourteen-tap cases exercise overlapping tiles.
func TestFilterSIMDExactMinimumWindow(t *testing.T) {
	type edgeFn func([]byte, int, int, int, int, int, filter4Params)
	for _, tc := range []struct {
		name           string
		before, taps   int
		pure8, simd8   edgeFn
		pure16, simd16 edgeFn
	}{
		{"4", 2, 4,
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4EdgePureGo(p, q, s, o, n, f) },
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4EdgeSIMD(p, q, s, o, n, f) },
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4Edge16PureGo(p, q, s, o, n, f) },
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4Edge16SIMD(p, q, s, o, n, f) }},
		{"6", 3, 6, filter6EdgePureGo, filter6EdgeSIMD, filter6Edge16PureGo, filter6Edge16SIMD},
		{"8", 4, 8, filter8EdgePureGo, filter8EdgeSIMD, filter8Edge16PureGo, filter8Edge16SIMD},
		{"14", 7, 14, filter14EdgePureGo, filter14EdgeSIMD, filter14Edge16PureGo, filter14Edge16SIMD},
	} {
		for _, depth := range []uint8{8, 10, 12} {
			sz := 1
			if depth > 8 {
				sz = 2
			}
			maxVal := (1 << depth) - 1
			for _, vertical := range []bool{false, true} {
				for _, length := range []int{8, 9, 15, 32} {
					for _, thr := range []Thresholds{
						{Limit: 0, BlockLimit: 0, HighEdgeVariance: 0},
						{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10},
						{Limit: 255, BlockLimit: 255, HighEdgeVariance: 255},
					} {
						scale, params := filter4ParamsFor(depth, thr)
						step, outer := length*sz, sz
						q0 := tc.before * step
						if vertical {
							step, outer = sz, tc.taps*sz
							q0 = tc.before * sz
						}
						buf := make([]byte, length*tc.taps*sz)
						rng := rand.New(rand.NewSource(int64(1000 + int(depth)*100 + length*17 + tc.taps)))
						for pattern := 0; pattern < 4; pattern++ {
							for i := 0; i < len(buf); i += sz {
								v := 0
								switch pattern {
								case 1:
									v = maxVal
								case 2:
									if (i/sz)&1 != 0 {
										v = maxVal
									}
								case 3:
									v = rng.Intn(maxVal + 1)
								}
								buf[i] = byte(v)
								if sz == 2 {
									buf[i+1] = byte(v >> 8)
								}
							}
							want := bytes.Clone(buf)
							got := bytes.Clone(buf)
							pure, simd := tc.pure8, tc.simd8
							if sz == 2 {
								pure, simd = tc.pure16, tc.simd16
							}
							pure(want, q0, step, outer, length, scale, params)
							simd(got, q0, step, outer, length, scale, params)
							if !bytes.Equal(got, want) {
								t.Fatalf("filter%s depth=%d vertical=%v len=%d thr=%+v pattern=%d", tc.name, depth, vertical, length, thr, pattern)
							}
						}
					}
				}
			}
		}
	}
}

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
