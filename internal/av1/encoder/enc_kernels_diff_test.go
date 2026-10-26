package encoder

import (
	"math/rand"
	"testing"
)

// TestEncoderSmallKernelsMatchScalar checks the architecture forms of the
// small SAD and downsampling kernels against their scalar references on
// random and extreme planes.
func TestEncoderSmallKernelsMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const stride = 96
	src := make([]byte, stride*80)
	ref := make([]byte, stride*80)
	absSAD := func(a, b []byte, as, bs, w, h int) int {
		s := 0
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				s += max(int(a[r*as+c])-int(b[r*bs+c]), int(b[r*bs+c])-int(a[r*as+c]))
			}
		}
		return s
	}
	for pattern := range 3 {
		for i := range src {
			switch pattern {
			case 0:
				src[i], ref[i] = byte(rng.Intn(256)), byte(rng.Intn(256))
			case 1:
				src[i], ref[i] = 255, 0
			default:
				src[i], ref[i] = []byte{0, 255}[rng.Intn(2)], []byte{0, 255}[rng.Intn(2)]
			}
		}
		for trial := range 50 {
			so := rng.Intn(10*stride) + rng.Intn(16)
			ro := rng.Intn(10*stride) + rng.Intn(16)
			got := [4]int{}
			got[0], got[1], got[2], got[3] = sad8x4x4(src[so:], ref[ro:], ref[ro+1:], ref[ro+5:], ref[ro+stride:], stride)
			for k, o := range []int{ro, ro + 1, ro + 5, ro + stride} {
				if want := sad8x4PureGo(src[so:], ref[o:], stride); got[k] != want {
					t.Fatalf("pattern %d trial %d: sad8x4x4[%d] = %d, want %d", pattern, trial, k, got[k], want)
				}
			}
			for _, rows := range []int{4, 8, 16} {
				got[0], got[1], got[2], got[3] = sad16xNx4(src[so:], 2*stride, ref[ro:], 2*stride, rows)
				for k := range 4 {
					if want := absSAD(src[so:], ref[ro+k:], 2*stride, 2*stride, 16, rows); got[k] != want {
						t.Fatalf("pattern %d rows %d: sad16xNx4[%d] = %d, want %d", pattern, rows, k, got[k], want)
					}
				}
			}
		}
		for _, sz := range [][2]int{{16, 8}, {24, 5}, {40, 30}, {7, 3}} {
			w, h := sz[0], sz[1]
			want := make([]byte, 48*h)
			got := make([]byte, 48*h)
			svtDownsample2D(want, 48, src, stride, w, h)
			svtDownsample2DArch(got, 48, src, stride, w, h)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("pattern %d %dx%d: downsample byte %d = %d, want %d", pattern, w, h, i, got[i], want[i])
				}
			}
		}
	}
}
