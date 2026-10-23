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
		ox, oy, w, h := svtSearchAreaCorrect(c.orgX, c.orgY, c.ox, c.oy, c.w, c.h, c.picW, c.picH, c.pad, true)
		if ox != c.wx || oy != c.wy || w != c.ww || h != c.wh {
			t.Errorf("%+v: got %d %d %d %d", c, ox, oy, w, h)
		}
	}
}

// TestSVTHMEL0SearchArea pins get_hme_l0_search_area's per-region size at
// preset 12 rtc (CBR sequence qp 35) for the low-delay reference distances,
// camera and screen content.
func TestSVTHMEL0SearchArea(t *testing.T) {
	for _, c := range []struct {
		screen     bool
		dist, w, h int
	}{
		{false, 1, 16, 4}, {false, 2, 16, 8}, {false, 4, 16, 12}, {false, 8, 32, 20}, {false, 16, 48, 40}, {false, 32, 48, 48},
		{true, 1, 16, 9}, {true, 2, 32, 18}, {true, 4, 32, 27},
	} {
		ctrl := svtMEControls(c.screen, 1280, 720, 35)
		var p svtPyramid
		p.setDistance(c.dist, ctrl.l0Min, ctrl.l0Max)
		if p.l0W != c.w || p.l0H != c.h {
			t.Errorf("screen %v dist %d: %dx%d, want %dx%d", c.screen, c.dist, p.l0W, p.l0H, c.w, c.h)
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

// TestSVTMEControls pins the qp-scaled preset-12 rtc ME/HME areas
// (set_hme_search_params, set_me_search_params with
// svt_aom_get_qp_based_th_scaling_factors at the CBR sequence qp 35) and the
// integer-search area per reference distance.
func TestSVTMEControls(t *testing.T) {
	cam := svtMEControls(false, 1280, 720, 35)
	if cam.l0Min != (svtSearchArea{8, 8}) || cam.l0Max != (svtSearchArea{96, 96}) ||
		cam.meMin != (svtSearchArea{8, 3}) || cam.meMax != (svtSearchArea{8, 3}) || cam.hmeL2 || cam.prehme {
		t.Fatalf("camera controls %+v", cam)
	}
	sc := svtMEControls(true, 1280, 720, 35)
	if sc.l0Min != (svtSearchArea{18, 18}) || sc.l0Max != (svtSearchArea{107, 107}) ||
		sc.meMin != (svtSearchArea{13, 9}) || sc.meMax != (svtSearchArea{18, 9}) || !sc.hmeL2 || !sc.prehme {
		t.Fatalf("screen controls %+v", sc)
	}
	for _, c := range []struct {
		ctrl         svtMECtrl
		f            int
		sad, zz      uint32
		wantW, wantH int
	}{
		{cam, 1, 1 << 20, 1 << 20, 8, 3},
		{cam, 3, 0, 1 << 20, 8, 3},
		{cam, 1, 0, 100, 1, 1},
		{sc, 1, 1 << 20, 1 << 20, 16, 9}, // 13x9 rounded up to 16
		{sc, 3, 1 << 20, 1 << 20, 24, 9}, // capped at 18x9, rounded up
		{sc, 3, 1000, 1 << 20, 8, 3},     // low hme_sad: divisor 8
		{sc, 3, 1 << 20, 600, 1, 1},      // below me_early_exit_th / 6
	} {
		if w, h := c.ctrl.meArea(c.f, 64, 64, c.sad, c.zz); w != c.wantW || h != c.wantH {
			t.Errorf("screen %v f %d sad %d zz %d: %dx%d, want %dx%d", c.ctrl.screen, c.f, c.sad, c.zz, w, h, c.wantW, c.wantH)
		}
	}
}
