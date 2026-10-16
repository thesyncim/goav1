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
var pred8VBenchSink byte

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

// pred8VScalarOracle evaluates the full eight-tap convolve_y_sr expression
// using the original 7-bit coefficient scale, independently of the SIMD
// implementation's halved taps and 6-bit shift.
func pred8VScalarOracle(dst, src []byte, srcStride int, ref []byte, refOffset, refStride, w, h int, k *[8]int16) int {
	sad := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sum := int64(64)
			for tap := 0; tap < 8; tap++ {
				sum += int64(ref[refOffset+(y+tap-3)*refStride+x]) * int64(k[tap])
			}
			v := sum >> 7
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

// TestPred8VZeroOuterTapDispatch covers every filter phase and representative
// 8-, 16-, and 64-pixel strips against the independent full-eight-tap oracle.
// The source has explicit row and horizontal padding because the 8-pixel tail
// intentionally loads 16 bytes, and its odd row stride tests unaligned loads.
// The primary destination windows are exact-sized for checkptr; a final case
// checks padded source and destination strides with row-padding canaries.
func TestPred8VZeroOuterTapDispatch(t *testing.T) {
	var rng = rand.New(rand.NewSource(0xbb67ae85))
	for _, size := range [][2]int{
		{8, 8}, {8, 16}, {16, 8}, {16, 16}, {16, 24}, {16, 64},
		{24, 8}, {40, 16}, {64, 8}, {64, 16}, {64, 32}, {64, 64},
	} {
		w, h := size[0], size[1]
		refStride := w + 25
		ref := make([]byte, (h+7)*refStride)
		refOffset := 3*refStride + 8
		refPtr := unsafe.Pointer(&ref[refOffset])
		for pattern := 0; pattern < 4; pattern++ {
			switch pattern {
			case 0:
				_, _ = rng.Read(ref)
			case 1:
				clear(ref)
			case 2:
				for i := range ref {
					ref[i] = 255
				}
			default:
				for i := range ref {
					y, x := i/refStride, i%refStride
					ref[i] = byte((x+y)&1) * 255
				}
			}
			src := make([]byte, w*h)
			_, _ = rng.Read(src)
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
					wantSAD := pred8VScalarOracle(want, src, w, ref, refOffset, refStride, w, h, k)
					got := make([]byte, w*h)
					gotSAD := pred8V(unsafe.Pointer(&got[0]), w, nil, 0, refPtr, refStride, w, h, k)
					if gotSAD != 0 {
						t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d: store SAD=%d", filter, phase, w, h, pattern, gotSAD)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d byte=%d: got %d want %d", filter, phase, w, h, pattern, i, got[i], want[i])
						}
					}

					if w%16 == 0 {
						gotSAD = pred8V(nil, 0, unsafe.Pointer(&src[0]), w, refPtr, refStride, w, h, k)
						if gotSAD != wantSAD {
							t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d: SAD=%d want %d", filter, phase, w, h, pattern, gotSAD, wantSAD)
						}
					}
				}
			}
		}
	}

	const w, h = 64, 24
	const srcPad, dstPad, srcCanary, dstCanary = 7, 11, 0xa7, 0xcd
	refStride := w + 25
	ref := make([]byte, (h+7)*refStride)
	_, _ = rng.Read(ref)
	refOffset := 3*refStride + 8
	refPtr := unsafe.Pointer(&ref[refOffset])
	srcStride := w + srcPad
	src := make([]byte, h*srcStride)
	for i := range src {
		src[i] = srcCanary
	}
	for y := range h {
		_, _ = rng.Read(src[y*srcStride : y*srcStride+w])
	}
	dstStride := w + dstPad
	dst := make([]byte, h*dstStride)
	for filter := motion.InterpEightTapRegular; filter <= motion.InterpMultiTapSharp; filter++ {
		for phase := range 16 {
			k := &pred8Kernels[filter][phase]
			want := make([]byte, w*h)
			wantSAD := pred8VScalarOracle(want, src, srcStride, ref, refOffset, refStride, w, h, k)
			for i := range dst {
				dst[i] = dstCanary
			}
			gotSAD := pred8V(unsafe.Pointer(&dst[0]), dstStride, nil, 0, refPtr, refStride, w, h, k)
			if gotSAD != 0 {
				t.Fatalf("padded filter=%d phase=%d: store SAD=%d", filter, phase, gotSAD)
			}
			for y := range h {
				for x := range w {
					if got, expected := dst[y*dstStride+x], want[y*w+x]; got != expected {
						t.Fatalf("padded filter=%d phase=%d row=%d col=%d: got %d want %d", filter, phase, y, x, got, expected)
					}
				}
				for x := w; x < dstStride; x++ {
					if got := dst[y*dstStride+x]; got != dstCanary {
						t.Fatalf("padded filter=%d phase=%d wrote destination row padding at row=%d col=%d: got %#x", filter, phase, y, x, got)
					}
				}
			}
			gotSAD = pred8V(nil, 0, unsafe.Pointer(&src[0]), srcStride, refPtr, refStride, w, h, k)
			if gotSAD != wantSAD {
				t.Fatalf("padded filter=%d phase=%d: SAD=%d want %d", filter, phase, gotSAD, wantSAD)
			}
			for y := range h {
				for x := w; x < srcStride; x++ {
					if got := src[y*srcStride+x]; got != srcCanary {
						t.Fatalf("padded filter=%d phase=%d changed source row padding at row=%d col=%d: got %#x", filter, phase, y, x, got)
					}
				}
			}
		}
	}
}

// BenchmarkPred8VTapPaths captures small-block dispatch overhead as well as
// the steady-state large-block gain, with sharp fallback through the same
// public entry point.
func BenchmarkPred8VTapPaths(b *testing.B) {
	for _, shape := range []struct {
		name string
		w, h int
	}{
		{name: "8x8", w: 8, h: 8},
		{name: "16x8", w: 16, h: 8},
		{name: "64x64", w: 64, h: 64},
	} {
		shape := shape
		refStride := shape.w + 25
		ref := make([]byte, (shape.h+7)*refStride)
		for i := range ref {
			ref[i] = byte((i*37 + i/refStride*17) % 256)
		}
		refPtr := unsafe.Pointer(&ref[3*refStride+8])
		dst := make([]byte, shape.w*shape.h)
		for _, kernel := range []struct {
			name   string
			filter motion.InterpFilter
		}{
			{name: "regular", filter: motion.InterpEightTapRegular},
			{name: "sharp", filter: motion.InterpMultiTapSharp},
		} {
			kernel := kernel
			k := &pred8Kernels[kernel.filter][8]
			b.Run(shape.name+"/"+kernel.name, func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(shape.w * shape.h))
				for b.Loop() {
					pred8V(unsafe.Pointer(&dst[0]), shape.w, nil, 0, refPtr, refStride, shape.w, shape.h, k)
				}
				pred8VBenchSink = dst[len(dst)-1]
			})
		}
	}
}
