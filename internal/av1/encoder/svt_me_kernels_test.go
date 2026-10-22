package encoder

import (
	"math/rand"
	"testing"
)

// fillMEPattern fills b with random, flat (tie-heavy) or extreme content.
func fillMEPattern(rng *rand.Rand, b []byte, mode int) {
	for i := range b {
		switch mode % 4 {
		case 0:
			b[i] = byte(rng.Intn(256))
		case 1:
			b[i] = 128 // flat: every position ties
		case 2:
			if rng.Intn(2) == 0 {
				b[i] = 0
			} else {
				b[i] = 255
			}
		default:
			b[i] = byte(100 + rng.Intn(4))
		}
	}
}

func TestSVTSAD16xhx8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const stride = 96
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 64; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		for rows := 1; rows <= 16; rows++ {
			so := rng.Intn(16)
			ro := rng.Intn(16)
			// Exact-length slices: the kernel's last read is the slice end.
			s := src[so : so+(rows-1)*2*stride+16]
			r := ref[ro : ro+(rows-1)*stride+23]
			var got [8]uint32
			svtSAD16xhx8(&got, s, 2*stride, r, stride, rows)
			want := svtSAD16xhx8Scalar(s, 2*stride, r, stride, rows)
			if got != want {
				t.Fatalf("trial %d rows %d: got %v want %v", trial, rows, got, want)
			}
		}
	}
}

func BenchmarkSVTSAD16xhx8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const stride = 416
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	fillMEPattern(rng, src, 0)
	fillMEPattern(rng, ref, 0)
	for _, rows := range []int{8, 16} {
		b.Run(map[int]string{8: "L0rows8", 16: "L1rows16"}[rows], func(b *testing.B) {
			var sink uint32
			for b.Loop() {
				var a [8]uint32
				svtSAD16xhx8(&a, src, 2*stride, ref, 2*stride, rows)
				sink += a[3]
			}
			_ = sink
		})
		b.Run(map[int]string{8: "old_x4x2_L0rows8", 16: "old_x4x2_L1rows16"}[rows], func(b *testing.B) {
			var sink int
			for b.Loop() {
				a0, _, _, _ := sad16xNx4(src, 2*stride, ref, 2*stride, rows)
				a4, _, _, _ := sad16xNx4(src, 2*stride, ref[4:], 2*stride, rows)
				sink += a0 + a4
			}
			_ = sink
		})
	}
}

func BenchmarkSVTHMESearch(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	var cur, ref svtPlane
	cur.size(320, 180, svtSixBorder)
	ref.size(320, 180, svtSixBorder)
	fillMEPattern(rng, cur.pix, 0)
	fillMEPattern(rng, ref.pix, 0)
	var half, halfRef svtPlane
	half.size(640, 360, svtHalfBorder)
	halfRef.size(640, 360, svtHalfBorder)
	fillMEPattern(rng, half.pix, 0)
	fillMEPattern(rng, halfRef.pix, 0)
	b.Run("L0", func(b *testing.B) {
		for b.Loop() {
			svtHMESearch(&cur, &ref, 160, 96, 16, 16, -16, -4, 16, 4)
		}
	})
	b.Run("L1", func(b *testing.B) {
		for b.Loop() {
			svtHMESearch(&half, &halfRef, 320, 192, 32, 32, -4, -1, svtHMEL1W, svtHMEL1H)
		}
	})
}

// TestSVTHMECorrectArea pins hme_level_0 / hme_level_1's search-area
// correction on the 1280x720 pyramid (borders 32 and 64).
func TestSVTHMECorrectArea(t *testing.T) {
	for _, c := range []struct{ orgX, orgY, ox, oy, w, h, picW, picH, pad, wx, wy, ww, wh int }{
		{160, 96, -16, -4, 16, 4, 320, 180, 31, -16, -4, 16, 4}, // L0 inside
		{608, 0, 26, -1, 8, 3, 640, 360, 63, 26, -1, 6, 3},      // L1 right edge: 608+26+8 > 640
		{608, 0, 40, -1, 8, 3, 640, 360, 63, 31, -1, 1, 3},      // origin past the right edge
		{0, 0, -70, -70, 8, 3, 640, 360, 63, -63, -63, 8, 3},    // left/top: origin moves, size kept
		{0, 352, 0, 7, 8, 3, 640, 360, 63, 0, 7, 8, 1},          // bottom shrink
		{304, 0, 0, 0, 16, 4, 320, 180, 31, 0, 0, 16, 4},        // L0 last column, exact fit
		{304, 176, 0, 2, 16, 4, 320, 180, 31, 0, 2, 16, 2},      // L0 bottom shrink
	} {
		ox, oy, w, h := svtHMECorrectArea(c.orgX, c.orgY, c.ox, c.oy, c.w, c.h, c.picW, c.picH, c.pad)
		if ox != c.wx || oy != c.wy || w != c.ww || h != c.wh {
			t.Errorf("%+v: got %d %d %d %d", c, ox, oy, w, h)
		}
	}
}

// TestSVTHMEL0SearchArea pins get_hme_l0_search_area's per-region size at
// preset 12 rtc for the low-delay reference distances.
func TestSVTHMEL0SearchArea(t *testing.T) {
	for _, c := range []struct{ dist, w, h int }{{1, 16, 4}, {2, 16, 8}, {4, 16, 12}, {8, 32, 20}, {16, 48, 40}, {32, 48, 48}} {
		var p svtPyramid
		p.setDistance(c.dist)
		if p.l0W != c.w || p.l0H != c.h {
			t.Errorf("dist %d: %dx%d, want %dx%d", c.dist, p.l0W, p.l0H, c.w, c.h)
		}
	}
}

func TestSVTSAD32xhx8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	const stride = 96
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 64; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		if trial%5 == 0 { // the largest SADs: every difference 255
			for i := range src {
				src[i], ref[i] = 0, 255
			}
		}
		for rows := 1; rows <= 16; rows++ {
			so := rng.Intn(16)
			ro := rng.Intn(16)
			s := src[so : so+(rows-1)*2*stride+32]
			r := ref[ro : ro+(rows-1)*stride+39]
			var got [8]uint32
			got[3] = 7 // accumulates
			svtSAD32xhx8(&got, s, 2*stride, r, stride, rows)
			a := svtSAD16xhx8Scalar(s, 2*stride, r, stride, rows)
			b := svtSAD16xhx8Scalar(s[16:], 2*stride, r[16:], stride, rows)
			for k := range a {
				want := a[k] + b[k]
				if k == 3 {
					want += 7
				}
				if got[k] != want {
					t.Fatalf("trial %d rows %d pos %d: got %d want %d", trial, rows, k, got[k], want)
				}
			}
		}
	}
}
