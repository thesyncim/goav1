//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// itxFusedRectOracle is the staged path the fused rectangular kernels
// replace: the raw column output, Round2(., 4), added to pred and saturated.
func itxFusedRectOracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, size Size, activeRows, activeCols int) {
	t.Helper()
	w, h := int(size.Width), int(size.Height)
	raw := make([]int32, w*h)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, size, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < h; r++ {
		for c := 0; c < w; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*w+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCTRectMatchesStagedPath: every (rows, cols) shape of the
// 16x32 and 32x16 blocks, every network form pairing, byte-exact against the
// staged path; declined blocks untouched; no declines at +-64 / +-1024;
// magnitudes up to 2^29 exercise the guards and the rect2 pre-clamp.
func TestInverseAddDCTRectMatchesStagedPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{"16x32", 16, 32, InverseAddDCT16x32},
		{"32x16", 32, 16, InverseAddDCT32x16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(61 + tc.w)))
			size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
			stride := tc.w + 8
			coeff := make([]int32, tc.w*tc.h)
			pred := make([]byte, tc.h*stride)
			got := make([]byte, tc.h*stride)
			want := make([]byte, tc.h*stride)
			accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
			for trial := 0; trial < 40000; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(tc.h), 1+rng.Intn(tc.w)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
				case 1:
					activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
				}
				mode := trial % 6
				mag := []int{64, 1024, 4096, 32767, 70000, 1 << 29}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*tc.h+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := range pred {
					pred[i] = byte(rng.Intn(256))
				}
				copy(want, pred)
				itxFusedRectOracle(t, want, stride, pred, stride, coeff, tc.h, size, activeRows, activeCols)
				copy(got, pred)
				ok := tc.fn(got, stride, got, stride, coeff, tc.h, activeRows, activeCols)
				if mode <= 1 {
					realistic++
				}
				if !ok {
					declined++
					if mode <= 1 {
						realisticDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
							trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
					}
				}
				for i := range got {
					got[i] = byte(0xa5)
				}
				if !tc.fn(got, stride, pred, stride, coeff, tc.h, activeRows, activeCols) {
					t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
				}
				for r := 0; r < tc.h; r++ {
					for c := 0; c < tc.w; c++ {
						if got[r*stride+c] != want[r*stride+c] {
							t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
						}
					}
					for c := tc.w; c < stride; c++ {
						if got[r*stride+c] != 0xa5 {
							t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
						}
					}
				}
			}
			t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
			if realisticDeclined != 0 {
				t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
			}
		})
	}
}

func BenchmarkInverseAddDCTRect(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	for _, tc := range []struct {
		name string
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{"16x32", 16, 32, InverseAddDCT16x32},
		{"32x16", 32, 16, InverseAddDCT32x16},
	} {
		size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
		for _, shape := range []struct {
			name       string
			rows, cols int
		}{{"low4", 4, 4}, {"low8", 8, 8}, {"full", tc.h, tc.w}} {
			coeff := make([]int32, tc.w*tc.h)
			for c := 0; c < shape.cols; c++ {
				for r := 0; r < shape.rows; r++ {
					coeff[c*tc.h+r] = int32(rng.Intn(2049) - 1024)
				}
			}
			dst := make([]byte, tc.w*tc.h)
			for i := range dst {
				dst[i] = byte(rng.Intn(256))
			}
			raw := make([]int32, tc.w*tc.h)
			b.Run(tc.name+"/"+shape.name+"/fused", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if !tc.fn(dst, tc.w, dst, tc.w, coeff, tc.h, shape.rows, shape.cols) {
						b.Fatal("declined")
					}
				}
			})
			b.Run(tc.name+"/"+shape.name+"/staged-transform-only", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := InverseBlockBitDepthRawBounded(raw, coeff, tc.h, size, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// TestInverseAddDCTRectScaleEdges: a lone coefficient across the rect2 clamp
// boundary (scale(46341) is the first value past the int16 row clamp) and the
// pre-clamp boundary, at several positions, against the staged path.
func TestInverseAddDCTRectScaleEdges(t *testing.T) {
	for _, tc := range []struct {
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{16, 32, InverseAddDCT16x32},
		{32, 16, InverseAddDCT32x16},
	} {
		size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
		pred := make([]byte, tc.w*tc.h)
		for i := range pred {
			pred[i] = byte(i * 7)
		}
		for _, v := range []int32{0, 1, 2, 100, 32767, 46339, 46340, 46341, 46342, 65535, 1 << 20, 1<<22 - 1, 1 << 22, 1<<22 + 1, 1 << 30, 1<<31 - 1} {
			for _, sign := range []int32{1, -1} {
				for _, pos := range [][2]int{{0, 0}, {1, 0}, {0, 1}, {3, 5}} {
					coeff := make([]int32, tc.w*tc.h)
					coeff[pos[1]*tc.h+pos[0]] = sign * v
					want := append([]byte(nil), pred...)
					itxFusedRectOracle(t, want, tc.w, pred, tc.w, coeff, tc.h, size, pos[0]+1, pos[1]+1)
					got := append([]byte(nil), pred...)
					if !tc.fn(got, tc.w, got, tc.w, coeff, tc.h, pos[0]+1, pos[1]+1) {
						continue // declined by the guard: the staged path serves it
					}
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("%dx%d coeff %d at row %d col %d: mismatch at %d: fused %d staged %d", tc.w, tc.h, sign*v, pos[0], pos[1], i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}
