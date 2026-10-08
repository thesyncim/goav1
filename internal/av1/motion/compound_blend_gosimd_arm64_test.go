//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestBlendCompoundAvg8GoSIMDMatchesPureGo checks the Go-native SIMD compound
// average / distance-weighted blend is byte-identical to the pure-Go reference
// over the full uint16 CONV_BUF range (which drives the unsigned-multiply and
// [0,255] saturation edges), every weight pair, and a range of block shapes.
func TestBlendCompoundAvg8GoSIMDMatchesPureGo(t *testing.T) {
	const roundOffset, roundBits = 6144, 4 // the 8-bit compound constants
	sizes := []struct{ w, h int }{{8, 8}, {16, 16}, {24, 32}, {32, 32}, {64, 64}, {40, 8}, {8, 1}, {4, 4}}
	weights := [][2]int{{8, 8}, {4, 12}, {12, 4}, {2, 14}, {14, 2}, {0, 16}, {16, 0}}
	rng := rand.New(rand.NewSource(7))
	for _, sz := range sizes {
		for _, w := range weights {
			src0 := make([]uint16, sz.w*sz.h)
			src1 := make([]uint16, sz.w*sz.h)
			for i := range src0 {
				src0[i] = uint16(rng.Intn(1 << 16))
				src1[i] = uint16(rng.Intn(1 << 16))
			}
			want, _ := testPlane(sz.w, sz.h, 1, sz.w)
			got, _ := testPlane(sz.w, sz.h, 1, sz.w)
			blendCompoundAvg8PureGo(want, src0, src1, 0, 0, sz.w, sz.h, w[0], w[1], roundOffset, roundBits)
			blendCompoundAvg8GoSIMD(got, src0, src1, 0, 0, sz.w, sz.h, w[0], w[1], roundOffset, roundBits)
			for i := range want.Pix {
				if want.Pix[i] != got.Pix[i] {
					t.Fatalf("size %dx%d w=%v: mismatch at %d: got %d want %d", sz.w, sz.h, w, i, got.Pix[i], want.Pix[i])
				}
			}
		}
	}
}

func benchBlendCompoundAvg8GoSIMDAndNEON(b *testing.B, fn func(dst frame.Plane, src0, src1 []uint16, dstX, dstY, w, h, fwd, bck, roundOffset, roundBits int)) {
	const roundOffset, roundBits = 6144, 4
	const w, h = 32, 32
	src0 := make([]uint16, w*h)
	src1 := make([]uint16, w*h)
	for i := range src0 {
		src0[i] = uint16(3000 + i%5000)
		src1[i] = uint16(4000 + (i*7)%5000)
	}
	dst, _ := testPlane(w, h, 1, w)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src0, src1, 0, 0, w, h, 8, 8, roundOffset, roundBits)
	}
}

func BenchmarkBlendCompoundAvg8GoSIMD_32(b *testing.B) {
	benchBlendCompoundAvg8GoSIMDAndNEON(b, blendCompoundAvg8GoSIMD)
}

func BenchmarkBlendCompoundAvg8NEON_32(b *testing.B) {
	benchBlendCompoundAvg8GoSIMDAndNEON(b, blendCompoundAvg8NEON)
}
