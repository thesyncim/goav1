package encoder

import (
	"math/rand"
	"testing"
)

// bilinearSubpelSADLibaom is aom_dsp/variance.c's two-pass bilinear
// prediction (full (n+1)-row first pass into a buffer, then the vertical
// pass), scored by SAD: the reference the fused kernels must equal.
func bilinearSubpelSADLibaom(src []byte, srcStride int, ref []byte, refStride, n, fx, fy int) int {
	tmp := make([]uint16, (n+1)*n)
	for r := 0; r <= n; r++ {
		for c := 0; c < n; c++ {
			a, b := int(ref[r*refStride+c]), int(ref[r*refStride+c+1])
			tmp[r*n+c] = uint16((a*int(bilinearFilters2T[fx][0]) + b*int(bilinearFilters2T[fx][1]) + 64) >> 7)
		}
	}
	sad := 0
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			a, b := int(tmp[r*n+c]), int(tmp[(r+1)*n+c])
			p := (a*int(bilinearFilters2T[fy][0]) + b*int(bilinearFilters2T[fy][1]) + 64) >> 7
			d := int(src[r*srcStride+c]) - p
			if d < 0 {
				d = -d
			}
			sad += d
		}
	}
	return sad
}

func TestBilinearSubpelSADMatchesLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const w, h = 160, 160
	ref := make([]byte, w*h)
	src := make([]byte, w*h)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
		src[i] = byte(rng.Intn(256))
	}
	for _, n := range []int{8, 16, 32, 64} {
		for fx := 0; fx < 8; fx++ {
			for fy := 0; fy < 8; fy++ {
				for _, at := range [][2]int{{3, 5}, {w - n - 1, h - n - 1}, {0, 0}} {
					x, y := at[0], at[1]
					want := bilinearSubpelSADLibaom(src[y*w+x:], w, ref[y*w+x:], w, n, fx, fy)
					got, ok := bilinearSubpelSAD(src[y*w+x:], w, ref, w, w, h, x, y, n, fx, fy)
					if !ok || got != want {
						t.Fatalf("n=%d f=(%d,%d) at=%v: got %d ok=%v, want %d", n, fx, fy, at, got, ok, want)
					}
					if s := bilinearSubpelSADScalar(src[y*w+x:], w, ref[y*w+x:], w, n, fx, fy); s != want {
						t.Fatalf("scalar n=%d f=(%d,%d): got %d want %d", n, fx, fy, s, want)
					}
				}
			}
		}
	}
}
