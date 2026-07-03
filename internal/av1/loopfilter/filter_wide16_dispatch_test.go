// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"math/rand"
	"testing"
)

// wide16Params builds a two-byte (10/12-bit) parameter set the way
// filter4ParamsFor does for a given bit depth and raw thresholds, so the
// differential exercises the same centre/min/max/scale the decoder would use.
func wide16Params(bitDepth uint8, limit, blimit, hev int) (int, filter4Params) {
	scale := 1 << int(bitDepth-8)
	return scale, filter4Params{
		limit:  int16(limit * scale),
		blimit: int16(blimit * scale),
		hev:    int16(hev * scale),
		min:    int16(-128 * scale),
		max:    int16(128*scale - 1),
		center: int16(128 * scale),
	}
}

type wide16Case struct {
	name     string
	bitDepth uint8
	limit    int
	blimit   int
	hev      int
}

func wide16Corpus() []wide16Case {
	cases := []wide16Case{}
	for _, bd := range []uint8{10, 12} {
		for _, t := range [][3]int{
			{0, 0, 0}, {1, 1, 0}, {2, 5, 1}, {8, 20, 4}, {16, 40, 8},
			{32, 80, 16}, {63, 128, 32}, {255, 510, 0}, {255, 510, 255},
		} {
			cases = append(cases, wide16Case{
				name:     "bd" + itoa(int(bd)),
				bitDepth: bd,
				limit:    t[0], blimit: t[1], hev: t[2],
			})
		}
	}
	return cases
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [8]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// fillWide16Content fills a two-byte sample buffer with content regimes chosen
// to exercise the needsFilter / flat8in / flat8out branches, mirroring the
// 8-bit fillWideContent but scaled to the bit depth's sample range.
func fillWide16Content(buf []byte, rng *rand.Rand, mode int, maxVal int) {
	writeS := func(off, v int) {
		buf[off] = byte(v)
		buf[off+1] = byte(v >> 8)
	}
	mid := maxVal / 2
	for i := 0; i+1 < len(buf); i += 2 {
		var v int
		switch mode {
		case 0:
			v = rng.Intn(maxVal + 1)
		case 1:
			v = mid + rng.Intn(16)
		case 2:
			v = mid + rng.Intn(3)
		case 3:
			v = rng.Intn(4)
		default:
			v = maxVal - 3 + rng.Intn(4)
		}
		if v > maxVal {
			v = maxVal
		}
		writeS(i, v)
	}
}

type wide16Kernel struct {
	name string
	ref  func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	neon func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
}

func wide16Kernels() []wide16Kernel {
	return []wide16Kernel{
		{"filter6", filter6Edge16PureGo, filter6Edge16NEON},
		{"filter8", filter8Edge16PureGo, filter8Edge16NEON},
		{"filter14", filter14Edge16PureGo, filter14Edge16NEON},
	}
}

func runWide16Horizontal(t *testing.T, k wide16Kernel, seed int64, length int, c wide16Case) {
	t.Helper()
	const strideBytes = 256 // 128 samples per row
	const rows = 32
	rng := rand.New(rand.NewSource(seed))
	maxVal := (1 << c.bitDepth) - 1
	base := make([]byte, strideBytes*rows)
	fillWide16Content(base, rng, int(seed)%5, maxVal)
	step := strideBytes
	outer := 2
	q0Base := 16*strideBytes + 16*2 // row 16, sample col 16
	scale, params := wide16Params(c.bitDepth, c.limit, c.blimit, c.hev)

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	k.ref(want, q0Base, step, outer, length, scale, params)
	k.neon(got, q0Base, step, outer, length, scale, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s H %s seed=%d len=%d idx=%d got=%d want=%d", k.name, c.name, seed, length, i, got[i], want[i])
		}
	}
}

func runWide16Vertical(t *testing.T, k wide16Kernel, seed int64, length int, c wide16Case) {
	t.Helper()
	const strideBytes = 128 // 64 samples per row
	const rows = 96
	rng := rand.New(rand.NewSource(seed))
	maxVal := (1 << c.bitDepth) - 1
	base := make([]byte, strideBytes*rows)
	fillWide16Content(base, rng, int(seed)%5, maxVal)
	step := 2
	outer := strideBytes
	q0Base := 16 * 2 // sample col 16, row 0
	scale, params := wide16Params(c.bitDepth, c.limit, c.blimit, c.hev)

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)
	k.ref(want, q0Base, step, outer, length, scale, params)
	k.neon(got, q0Base, step, outer, length, scale, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s V %s seed=%d len=%d idx=%d got=%d want=%d", k.name, c.name, seed, length, i, got[i], want[i])
		}
	}
}

// TestWide16FilterNEONMatchesPureGo drives the 10/12-bit wide NEON kernels
// (executed directly) against the pure-Go reference across all widths, levels,
// thresholds, and both 10- and 12-bit sample ranges, for horizontal and
// vertical edges. Every output byte must match.
func TestWide16FilterNEONMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	for _, k := range wide16Kernels() {
		var seed int64 = 100000
		for _, c := range wide16Corpus() {
			for _, length := range lengths {
				for rep := 0; rep < 4; rep++ {
					runWide16Horizontal(t, k, seed, length, c)
					runWide16Vertical(t, k, seed+500000, length, c)
					seed++
				}
			}
		}
	}
}

// TestWide16FilterNEONZeroAlloc guards that the accelerated 10-bit paths
// (horizontal direct and vertical repack) allocate nothing per call.
func TestWide16FilterNEONZeroAlloc(t *testing.T) {
	const strideBytes = 256
	const rows = 96
	base := make([]byte, strideBytes*rows)
	rng := rand.New(rand.NewSource(1))
	fillWide16Content(base, rng, 2, 1023)
	scale, params := wide16Params(10, 16, 40, 8)
	for _, k := range wide16Kernels() {
		// horizontal
		hBuf := append([]byte(nil), base...)
		if n := testing.AllocsPerRun(50, func() {
			k.neon(hBuf, 16*strideBytes+16*2, strideBytes, 2, 64, scale, params)
		}); n != 0 {
			t.Fatalf("%s horizontal allocs=%v", k.name, n)
		}
		// vertical
		vBuf := append([]byte(nil), base...)
		if n := testing.AllocsPerRun(50, func() {
			k.neon(vBuf, 16*2, 2, strideBytes, 64, scale, params)
		}); n != 0 {
			t.Fatalf("%s vertical allocs=%v", k.name, n)
		}
	}
}
