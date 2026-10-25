package encoder

import (
	"math/rand"
	"testing"
	"unsafe"
)

// randomSVTPlane is a padded w x h plane of random samples with a smooth
// component, so the searches see both ties and distinct minima.
func randomSVTPlane(rng *rand.Rand, w, h, border int, smooth bool) *svtPlane {
	p := &svtPlane{}
	p.size(w, h, border)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := rng.Intn(256)
			if smooth {
				v = (x*3+y*5)&255 ^ rng.Intn(4)
			}
			p.pix[p.org+y*p.stride+x] = byte(v)
		}
	}
	p.pad()
	return p
}

func TestSVTHMEEightColSumMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 400; iter++ {
		smooth := iter%3 == 0
		ref := randomSVTPlane(rng, 160, 90, 32, smooth)
		cur := randomSVTPlane(rng, 160, 90, 32, !smooth)
		if iter%7 == 0 {
			for i := range ref.pix {
				ref.pix[i] = 255
			}
			for i := range cur.pix {
				cur.pix[i] = byte(255 * (iter & 1))
			}
		}
		ref.buildColSums()
		bw := []int{16, 32}[iter&1]
		rows := 1 + rng.Intn(16)
		s2 := 2 * ref.stride
		r0 := rng.Intn(ref.h + 2*ref.border - 2*rows + 1)
		c0 := rng.Intn(ref.stride - bw - 8)
		var got, want [8]uint32
		s := cur.pix[rng.Intn(4)*cur.stride+rng.Intn(40):]
		refPix := ref.pix[r0*ref.stride+c0:]
		srcSum := svtBlockSumSub(s, s2, bw, rows)
		top := ref.colSum[r0*ref.stride+c0:]
		bot := ref.colSum[(r0+2*rows)*ref.stride+c0:]
		svtHMEEightColSum(&got, unsafe.Pointer(&s[0]), unsafe.Pointer(&refPix[0]), unsafe.Pointer(&top[0]), unsafe.Pointer(&bot[0]), s2, bw, rows, srcSum)
		svtHMEEightColSumScalar(&want, s, s2, refPix, top, bot, bw, rows, srcSum)
		// The direct SAD definition.
		var direct [8]uint32
		for k := range direct {
			for r := 0; r < rows; r++ {
				for c := 0; c < bw; c++ {
					a, b := int(s[r*s2+c]), int(refPix[r*s2+k+c])
					direct[k] += uint32(max(a-b, b-a))
				}
			}
		}
		if got != want || got != direct {
			t.Fatalf("iter %d bw %d rows %d: got %v scalar %v direct %v", iter, bw, rows, got, want, direct)
		}
	}
}

func TestSVTHMESearchColSumMatchesSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for iter := 0; iter < 300; iter++ {
		smooth := iter%2 == 0
		w, h := 160, 96
		border := 32
		cur := randomSVTPlane(rng, w, h, border, smooth)
		ref := randomSVTPlane(rng, w, h, border, smooth)
		bw := []int{16, 32}[iter%2]
		bh := bw
		bx := rng.Intn(w/bw) * bw
		by := rng.Intn(h/bh) * bh
		sw := 8 * (1 + rng.Intn(6))
		sh := 1 + rng.Intn(12)
		ox, oy, sw, sh := svtSearchAreaCorrect(bx, by, -sw/2, -sh/2, sw, sh, w, h, border-1, true)
		if sw%8 != 0 {
			continue
		}
		ref.colSumOK = false
		wantSAD, wantX, wantY := svtHMESearch(cur, ref, bx, by, bw, bh, ox, oy, sw, sh)
		ref.buildColSums()
		gotSAD, gotX, gotY := svtHMESearchColSum(cur, ref, bx, by, bw, bh, ox, oy, sw, sh)
		if gotSAD != wantSAD || gotX != wantX || gotY != wantY {
			t.Fatalf("iter %d: colsum (%d,%d,%d) != search (%d,%d,%d)", iter, gotSAD, gotX, gotY, wantSAD, wantX, wantY)
		}
	}
}

func TestSVTPlaneColSumsFollowCopies(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	p := randomSVTPlane(rng, 64, 48, 32, false)
	p.buildColSums()
	var q svtPlane
	q.copyFrom(p)
	if !q.colSumOK {
		t.Fatal("copy of a plane with column sums has none")
	}
	want := make([]uint16, len(p.colSum))
	clear(want[:2*p.stride])
	svtColSumRows(want, p.pix, p.stride, p.h+2*p.border)
	n := p.stride * (p.h + 2*p.border + 2)
	for i := 0; i < n; i++ {
		if q.colSum[i] != want[i] {
			t.Fatalf("colSum[%d] = %d, want %d", i, q.colSum[i], want[i])
		}
	}
	q.size(64, 48, 32)
	if q.colSumOK {
		t.Fatal("resized plane keeps stale column sums")
	}
}

func benchmarkSVTHMESearch(b *testing.B, bw, w, h int, colSum bool) {
	rng := rand.New(rand.NewSource(4))
	cur := randomSVTPlane(rng, 320, 180, 64, false)
	ref := randomSVTPlane(rng, 320, 180, 64, false)
	if colSum {
		ref.buildColSums()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		svtHMESearch(cur, ref, 64, 64, bw, bw, -w/2, -h/2, w, h)
	}
}

func BenchmarkSVTHMESearchL0(b *testing.B) {
	b.Run("maxmin", func(b *testing.B) { benchmarkSVTHMESearch(b, 16, 16, 8, false) })
	b.Run("colsum", func(b *testing.B) { benchmarkSVTHMESearch(b, 16, 16, 8, true) })
}

func BenchmarkSVTHMESearchL1(b *testing.B) {
	b.Run("maxmin", func(b *testing.B) { benchmarkSVTHMESearch(b, 32, 8, 3, false) })
	b.Run("colsum", func(b *testing.B) { benchmarkSVTHMESearch(b, 32, 8, 3, true) })
}

func BenchmarkSVTPlaneBuildColSums(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	half := randomSVTPlane(rng, 640, 360, svtHalfBorder, false)
	six := randomSVTPlane(rng, 320, 180, svtSixBorder, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		half.buildColSums()
		six.buildColSums()
	}
}
