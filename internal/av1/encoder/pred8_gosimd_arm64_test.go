// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

var pred8VIMBenchSink byte

func pred8VIMScalarOracle(dst []byte, src []byte, srcStride int, im []int16, imStride, w, h int, k *[8]int16) int {
	sad := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sum := int64(-261120)
			for tap := 0; tap < 8; tap++ {
				sum += int64(im[(y+tap)*imStride+x]) * int64(k[tap])
			}
			v := sum >> 11
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			pred := byte(v)
			if dst != nil {
				dst[y*w+x] = pred
			}
			if src != nil {
				delta := int(src[y*srcStride+x]) - int(pred)
				if delta < 0 {
					delta = -delta
				}
				sad += delta
			}
		}
	}
	return sad
}

// TestPred8VIMZeroOuterTapDispatch covers each filter phase and supported
// block shape against an independent scalar eight-tap oracle. The gate mirrors
// the six-tap dispatch in the pinned libaom av1_convolve_2d_sr_neon; the
// coefficients are defined by av1/common/filter.h. Exact-size intermediate
// and output allocations expose row-offset errors to checkptr=2, and SAD cases
// exercise each 16-row accumulator flush.
func TestPred8VIMZeroOuterTapDispatch(t *testing.T) {
	var rng = rand.New(rand.NewSource(0x6a09e667))
	for _, size := range [][2]int{
		{8, 8}, {8, 16}, {16, 8}, {16, 16}, {32, 16}, {16, 32},
		{32, 32}, {64, 16}, {16, 64}, {64, 32}, {32, 64}, {64, 64},
	} {
		w, h := size[0], size[1]
		const srcPadding = 7
		srcStride := w + srcPadding
		src := make([]byte, (h-1)*srcStride+w)
		rng.Read(src)
		for pattern := 0; pattern < 3; pattern++ {
			imStride := w
			im := make([]int16, (h+7)*imStride)
			for i := range im {
				switch pattern {
				case 0: // values produced by the rounded horizontal first pass
					im[i] = int16(263 + rng.Intn(7913-263+1))
				case 1: // signed extrema and rounding boundaries
					switch i % 5 {
					case 0:
						im[i] = -32768
					case 1:
						im[i] = 32767
					case 2:
						im[i] = 0
					case 3:
						im[i] = 1
					default:
						im[i] = -1
					}
				default: // unrestricted int16 intermediates
					im[i] = int16(rng.Intn(65536) - 32768)
				}
			}
			for filter := motion.InterpEightTapRegular; filter <= motion.InterpMultiTapSharp; filter++ {
				for phase := range 16 {
					k := &pred8Kernels[filter][phase]
					zeroOuter := k[0] == 0 && k[7] == 0
					if filter != motion.InterpMultiTapSharp && !zeroOuter {
						t.Fatalf("filter=%d phase=%d has nonzero outer taps: %v", filter, phase, *k)
					}
					if filter == motion.InterpMultiTapSharp && phase != 0 && zeroOuter {
						t.Fatalf("sharp phase=%d unexpectedly has zero outer taps: %v", phase, *k)
					}

					want := make([]byte, w*h)
					wantSAD := pred8VIMScalarOracle(want, src, srcStride, im, w, w, h, k)
					got := make([]byte, w*h)
					gotSAD := pred8VIM(unsafe.Pointer(&got[0]), w, nil, 0,
						unsafe.Pointer(&im[0]), w*2, w, h, k)
					if gotSAD != 0 {
						t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d: store SAD=%d", filter, phase, w, h, pattern, gotSAD)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d byte=%d: got %d want %d", filter, phase, w, h, pattern, i, got[i], want[i])
						}
					}

					if w >= 16 {
						gotSAD = pred8VIM(nil, 0, unsafe.Pointer(&src[0]), srcStride,
							unsafe.Pointer(&im[0]), w*2, w, h, k)
						if gotSAD != wantSAD {
							t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d: SAD=%d want %d", filter, phase, w, h, pattern, gotSAD, wantSAD)
						}
					}
				}
			}
		}
	}
}

// BenchmarkPred8VIMTapPaths measures regular six-tap dispatch and sharp fallback
// against the same public entry point used by the encoder.
func BenchmarkPred8VIMTapPaths(b *testing.B) {
	const w, h = 64, 64
	im := make([]int16, (h+7)*w)
	for i := range im {
		im[i] = int16((i*97)%7651 + 263)
	}
	dst := make([]byte, w*h)
	regular := &pred8Kernels[motion.InterpEightTapRegular][8]
	sharp := &pred8Kernels[motion.InterpMultiTapSharp][8]
	bench := func(b *testing.B, k *[8]int16) {
		b.ReportAllocs()
		b.SetBytes(int64(w * h))
		for b.Loop() {
			pred8VIM(unsafe.Pointer(&dst[0]), w, nil, 0, unsafe.Pointer(&im[0]), w*2, w, h, k)
		}
		pred8VIMBenchSink = dst[len(dst)-1]
	}
	b.Run("regular", func(b *testing.B) { bench(b, regular) })
	b.Run("sharp", func(b *testing.B) { bench(b, sharp) })
}
