package encoder

import (
	"math/rand"
	"testing"
)

func TestSweepSearchPoint8MatchesPoint4(t *testing.T) {
	const stride = 160
	rng := rand.New(rand.NewSource(5))
	src := make([]byte, stride*100)
	ref := make([]byte, stride*100)
	for trial := 0; trial < 20; trial++ {
		for i := range src {
			src[i] = byte(rng.Intn(256))
			ref[i] = byte(rng.Intn(256))
		}
		if trial%2 == 1 { // structured: ref is src shifted, so argmins move
			for i := range ref {
				ref[i] = src[(i+3)%len(src)] + byte(rng.Intn(5))
			}
		}
		var a, b realtimeSBMultiSizeME
		a.reset()
		b.reset()
		srcBase, refBase := 10*stride+20, 9*stride+14
		for y := 0; y < 3; y++ {
			rb := refBase + y*stride
			if !a.sweepSearchPoint8(src, ref, stride, srcBase, rb, -6, y-1) {
				t.Skip("no batch kernel in this build")
			}
			b.sweepSearchPoint4(src, ref, stride, srcBase, rb, -6, y-1)
			b.sweepSearchPoint4(src, ref, stride, srcBase, rb+4, -2, y-1)
		}
		if a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: batch differs: %v %v vs %v %v", trial, a.bestSAD[:6], a.bestMV[:6], b.bestSAD[:6], b.bestMV[:6])
		}
	}
}

func TestSVTHMEKernelsMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	const w, h, stride = 100, 40, 208
	src := make([]byte, stride*(2*h+2))
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	got := make([]byte, w*h)
	want := make([]byte, w*h)
	svtDownsample2DArch(got, w, src, stride, w, h)
	svtDownsample2D(want, w, src, stride, w, h)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("downsample sample %d = %d, want %d", i, got[i], want[i])
		}
	}
	ref := make([]byte, stride*40)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	for rows := 1; rows <= 16; rows++ {
		a := [4]int{}
		a[0], a[1], a[2], a[3] = sad16xNx4(src[3:], 2*stride, ref[5:], stride, rows)
		for k := range a {
			want := 0
			for r := 0; r < rows; r++ {
				for c := 0; c < 16; c++ {
					d := int(src[3+r*2*stride+c]) - int(ref[5+r*stride+k+c])
					if d < 0 {
						d = -d
					}
					want += d
				}
			}
			if a[k] != want {
				t.Fatalf("rows %d pos %d: sad %d, want %d", rows, k, a[k], want)
			}
		}
	}
}
