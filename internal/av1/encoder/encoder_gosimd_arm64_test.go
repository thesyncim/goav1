//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"bytes"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"math/rand"
	"simd/archsimd"
	"testing"
)

var pred8VIMBenchSink byte
var pred8VBenchSink byte
var pred8IMBenchSink byte

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
					gotSAD := pred8VIM(got, w, nil, 0, im, w, w, h, k)
					if gotSAD != 0 {
						t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d: store SAD=%d", filter, phase, w, h, pattern, gotSAD)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d byte=%d: got %d want %d", filter, phase, w, h, pattern, i, got[i], want[i])
						}
					}

					if w >= 16 {
						gotSAD = pred8VIM(nil, 0, src, srcStride, im, w, w, h, k)
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
			pred8VIM(dst, w, nil, 0, im, w, w, h, k)
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
					gotSAD := pred8V(got, w, nil, 0, ref, refOffset, refStride, w, h, k)
					if gotSAD != 0 {
						t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d: store SAD=%d", filter, phase, w, h, pattern, gotSAD)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("filter=%d phase=%d size=%dx%d pattern=%d byte=%d: got %d want %d", filter, phase, w, h, pattern, i, got[i], want[i])
						}
					}

					if w%16 == 0 {
						gotSAD = pred8V(nil, 0, src, w, ref, refOffset, refStride, w, h, k)
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
			gotSAD := pred8V(dst, dstStride, nil, 0, ref, refOffset, refStride, w, h, k)
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
			gotSAD = pred8V(nil, 0, src, srcStride, ref, refOffset, refStride, w, h, k)
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
		refOff := 3*refStride + 8
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
					pred8V(dst, shape.w, nil, 0, ref, refOff, refStride, shape.w, shape.h, k)
				}
				pred8VBenchSink = dst[len(dst)-1]
			})
		}
	}
}

// pred8IMScalarOracle evaluates the original eight-tap expression at the
// original coefficient scale, independently of the SIMD tap reduction.
func pred8IMScalarOracle(dst []int16, dstStride int, ref []byte, refOffset, refStride, w, rows int, k *[8]int16) {
	for y := 0; y < rows; y++ {
		for x := 0; x < w; x++ {
			sum := int64(16384)
			for tap := 0; tap < 8; tap++ {
				sum += int64(ref[refOffset+y*refStride+x+tap-3]) * int64(k[tap])
			}
			dst[y*dstStride+x] = int16((sum + 4) >> 3)
		}
	}
}

// TestPred8IMZeroOuterTapDispatch compares every filter phase against the
// original eight-tap expression. The pinned libaom
// av1/common/arm/av1_convolve_scale_neon.c (convolve6_8_h) omits coefficients 0
// and 7; regular and smooth filters have zero outer taps, while fractional
// sharp filters retain the full eight-tap path.
func TestPred8IMZeroOuterTapDispatch(t *testing.T) {
	var rng = rand.New(rand.NewSource(0x3c6ef372))
	shapes := []struct {
		name              string
		w, rows, imStride int
	}{
		{name: "8x15-tight", w: 8, rows: 15, imStride: 8},
		{name: "16x23-tight", w: 16, rows: 23, imStride: 16},
		{name: "32x39-tight", w: 32, rows: 39, imStride: 32},
		{name: "64x71-tight", w: 64, rows: 71, imStride: 64},
		{name: "32x39-padded", w: 32, rows: 39, imStride: 39},
		{name: "cache-24x26-s80", w: 24, rows: 26, imStride: 80},
		{name: "cache-40x42-s80", w: 40, rows: 42, imStride: 80},
		{name: "cache-72x74-s80", w: 72, rows: 74, imStride: 80},
	}
	for _, shape := range shapes {
		refStride := shape.w + 33 // odd pitch plus horizontal read padding
		// pred8IM advances p after the final row; pred8Fits and
		// subpelProbeCache.init provide this post-row padding at the call sites.
		ref := make([]byte, (shape.rows+1)*refStride)
		refOffset := 8
		got := make([]int16, (shape.rows-1)*shape.imStride+shape.w) // exact final span
		want := make([]int16, len(got))
		for pattern := 0; pattern < 5; pattern++ {
			switch pattern {
			case 0: // random bytes
				_, _ = rng.Read(ref)
			case 1: // all zero
				clear(ref)
			case 2: // all 255
				for i := range ref {
					ref[i] = 255
				}
			case 3: // checkerboard extremes
				for i := range ref {
					y, x := i/refStride, i%refStride
					ref[i] = byte((x+y)&1) * 255
				}
			default: // randomized binary extremes
				for i := range ref {
					if rng.Intn(2) == 0 {
						ref[i] = 0
					} else {
						ref[i] = 255
					}
				}
			}
			for filter := motion.InterpEightTapRegular; filter <= motion.InterpMultiTapSharp; filter++ {
				for phase := range 16 {
					k := &pred8Kernels[filter][phase]
					if filter != motion.InterpMultiTapSharp && (k[0] != 0 || k[7] != 0) {
						t.Fatalf("filter=%d phase=%d has nonzero outer taps: %v", filter, phase, *k)
					}
					if filter == motion.InterpMultiTapSharp && phase != 0 && k[0] == 0 && k[7] == 0 {
						t.Fatalf("sharp phase=%d unexpectedly has zero outer taps: %v", phase, *k)
					}

					for i := range got {
						got[i] = -1234
						want[i] = -1234
					}
					pred8IMScalarOracle(want, shape.imStride, ref, refOffset, refStride, shape.w, shape.rows, k)
					pred8IM(got, shape.imStride, ref, refOffset, refStride, shape.w, shape.rows, k)
					for y := 0; y < shape.rows; y++ {
						for x := 0; x < shape.w; x++ {
							if got[y*shape.imStride+x] != want[y*shape.imStride+x] {
								t.Fatalf("filter=%d phase=%d shape=%s pattern=%d row=%d col=%d: got %d want %d", filter, phase, shape.name, pattern, y, x, got[y*shape.imStride+x], want[y*shape.imStride+x])
							}
						}
						for x := shape.w; x < shape.imStride && y < shape.rows-1; x++ {
							if got[y*shape.imStride+x] != -1234 {
								t.Fatalf("filter=%d phase=%d shape=%s wrote intermediate row padding at row=%d col=%d", filter, phase, shape.name, y, x)
							}
						}
					}
				}
			}
		}
	}
}

// BenchmarkPred8IMTapPaths measures the horizontal first pass on cache-sized
// blocks through the same public entry point for regular and sharp filters.
func BenchmarkPred8IMTapPaths(b *testing.B) {
	for _, shape := range []struct {
		name              string
		w, rows, imStride int
	}{
		{name: "cache-24x26-s80", w: 24, rows: 26, imStride: 80},
		{name: "cache-40x42-s80", w: 40, rows: 42, imStride: 80},
		{name: "cache-72x74-s80", w: 72, rows: 74, imStride: 80},
	} {
		shape := shape
		refStride := shape.w + 33
		ref := make([]byte, (shape.rows+1)*refStride)
		for i := range ref {
			ref[i] = byte((i*37 + i/refStride*17) % 256)
		}
		im := make([]int16, (shape.rows-1)*shape.imStride+shape.w)
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
				b.SetBytes(int64(shape.w * shape.rows))
				for b.Loop() {
					pred8IM(im, shape.imStride, ref, 8, refStride, shape.w, shape.rows, k)
				}
				pred8IMBenchSink = byte(im[len(im)-1])
			})
		}
	}
}

func makeSADPlane(seed int64, n int) ([]byte, []byte) {
	rng := rand.New(rand.NewSource(seed))
	src := make([]byte, n)
	ref := make([]byte, n)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	return src, ref
}

// TestSAD8x8x4SIMDByteExact covers random data, odd strides, and unaligned
// block origins against the scalar oracle.
func TestSAD8x8x4SIMDByteExact(t *testing.T) {
	for _, stride := range []int{24, 40, 79, 96} {
		src, ref := makeSADPlane(int64(stride)*19+9, stride*128)
		rng := rand.New(rand.NewSource(int64(stride) + 400))
		for range 2000 {
			row := rng.Intn(96) + 8
			col := rng.Intn(stride-16) + 8
			off := row*stride + col
			r0, r1, r2, r3 := off+2, off-2, off+2*stride, off-2*stride
			w0, w1, w2, w3 := sad8x8x4PureGo(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			g0, g1, g2, g3 := sad8x8x4SIMD(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("stride %d off %d: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
					stride, off, g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

// TestSAD8x8x4SIMDExactMinimumSlice checks the smallest legal block windows
// with both aligned and offset slice starts. In particular, it ensures the
// raw-pointer loop does not advance past the allocation after the final row
// pair; run this test with checkptr=2 as well as ordinary SIMD tests.
func TestSAD8x8x4SIMDExactMinimumSlice(t *testing.T) {
	for _, stride := range []int{17, 32, 79} {
		const rowCount = 8
		windowLen := (rowCount-1)*stride + 8
		for _, offset := range []int{0, 1, 7} {
			makeWindow := func(seed int64) []byte {
				rng := rand.New(rand.NewSource(seed))
				backing := make([]byte, windowLen+offset)
				for i := range backing {
					backing[i] = uint8(rng.Intn(256))
				}
				return backing[offset:]
			}

			src := makeWindow(int64(stride*100 + offset))
			refs := [4][]byte{
				makeWindow(int64(stride*200 + offset)),
				makeWindow(int64(stride*300 + offset)),
				makeWindow(int64(stride*400 + offset)),
				makeWindow(int64(stride*500 + offset)),
			}
			w0, w1, w2, w3 := sad8x8x4PureGo(src, refs[0], refs[1], refs[2], refs[3], stride)
			g0, g1, g2, g3 := sad8x8x4SIMD(src, refs[0], refs[1], refs[2], refs[3], stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("stride=%d offset=%d: SIMD (%d,%d,%d,%d), want (%d,%d,%d,%d)",
					stride, offset, g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

// TestSAD8x8SIMDPackedRowsExactWindows covers every entry point that packs two
// 8-byte rows through a local array. The allocation ends at the last legal
// byte, including for overlapping rows and misaligned window starts.
func TestSAD8x8SIMDPackedRowsExactWindows(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5ad8))
	for _, srcStride := range []int{1, 2, 7, 8, 17, 79} {
		for _, refStride := range []int{1, 8, 19} {
			for _, offset := range []int{0, 1, 7} {
				src := sadWindow(rng, 7*srcStride+8, offset, sadFillRandom)
				ref0 := sadWindow(rng, 7*refStride+8, offset, sadFillBinary)
				ref1 := sadWindow(rng, 7*refStride+8, offset, sadFillMax)
				if got, want := sad8x8SIMD(src, srcStride, ref0, refStride),
					sad8x8DualPureGo(src, srcStride, ref0, refStride); got != want {
					t.Fatalf("dual strides %d/%d offset %d: got %d want %d", srcStride, refStride, offset, got, want)
				}
				if got, want := sad8x8CompoundAvgSIMD(src, srcStride, ref0, refStride, ref1, refStride),
					sad8x8CompoundAvgBlockPureGo(src, srcStride, ref0, refStride, ref1, refStride); got != want {
					t.Fatalf("compound strides %d/%d offset %d: got %d want %d", srcStride, refStride, offset, got, want)
				}
			}
		}
	}
	for _, stride := range []int{1, 2, 7, 8, 17, 79} {
		for _, offset := range []int{0, 1, 7} {
			src := sadWindow(rng, 7*stride+8, offset, sadFillRandom)
			refs := [4][]byte{
				sadWindow(rng, 7*stride+8, offset, sadFillRandom),
				sadWindow(rng, 7*stride+8, offset, sadFillZero),
				sadWindow(rng, 7*stride+8, offset, sadFillMax),
				sadWindow(rng, 7*stride+8, offset, sadFillBinary),
			}
			g0, g1, g2, g3 := sad8x8x4SIMD(src, refs[0], refs[1], refs[2], refs[3], stride)
			w0, w1, w2, w3 := sad8x8x4PureGo(src, refs[0], refs[1], refs[2], refs[3], stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("x4 stride %d offset %d: got (%d,%d,%d,%d) want (%d,%d,%d,%d)", stride, offset,
					g0, g1, g2, g3, w0, w1, w2, w3)
			}
			ref := sadWindow(rng, 7*stride+20, offset, sadFillRandom)
			g0, g1, g2, g3 = sad8x8x4Step4SIMD(src, ref, stride)
			w0, w1, w2, w3 = sad8x8x4Step4PureGo(src, ref, stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("step4 stride %d offset %d: got (%d,%d,%d,%d) want (%d,%d,%d,%d)", stride, offset,
					g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

func TestSAD8x8x4SIMDExtremes(t *testing.T) {
	const stride = 32
	src := make([]byte, stride*32)
	ref := make([]byte, stride*32)
	for i := range src {
		src[i] = 255
	}
	g0, g1, g2, g3 := sad8x8x4SIMD(src, ref, ref, ref, ref, stride)
	w0, w1, w2, w3 := sad8x8x4PureGo(src, ref, ref, ref, ref, stride)
	if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
		t.Fatalf("all-255 vs all-0: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
			g0, g1, g2, g3, w0, w1, w2, w3)
	}
}

func TestSADSIMDAbsDiffUnsignedEdges(t *testing.T) {
	var a, b, got [16]uint8
	for i := range a {
		a[i] = uint8(i * 17)
		b[i] = 255 - a[i]
	}
	absd := absDiffU8x16(archsimd.LoadUint8x16Array(&a), archsimd.LoadUint8x16Array(&b))
	absd.StoreArray(&got)
	for i := range got {
		want := int(a[i]) - int(b[i])
		if want < 0 {
			want = -want
		}
		if int(got[i]) != want {
			t.Fatalf("lane %d: absdiff(%d,%d)=%d, want %d", i, a[i], b[i], got[i], want)
		}
	}
}

var sadSIMDBenchSink int

func benchPlane() ([]byte, []byte) {
	src := make([]byte, 128*128)
	ref := make([]byte, 128*128)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	return src, ref
}

func BenchmarkSAD8x8x4_Scalar(b *testing.B) {
	src, ref := benchPlane()
	r0, r1, r2, r3 := ref, ref[4:], ref[8:], ref[12:]
	var sums [4]int
	b.ReportAllocs()
	for b.Loop() {
		s0, s1, s2, s3 := sad8x8x4PureGo(src, r0, r1, r2, r3, 128)
		sums[0] += s0
		sums[1] += s1
		sums[2] += s2
		sums[3] += s3
	}
	sadSIMDBenchSink = sums[0] + sums[1] + sums[2] + sums[3]
	if sadSIMDBenchSink == 0 {
		b.Fatal("unexpected zero SAD")
	}
}

func BenchmarkSAD8x8x4_SIMD(b *testing.B) {
	src, ref := benchPlane()
	r0, r1, r2, r3 := ref, ref[4:], ref[8:], ref[12:]
	var sums [4]int
	b.ReportAllocs()
	for b.Loop() {
		s0, s1, s2, s3 := sad8x8x4SIMD(src, r0, r1, r2, r3, 128)
		sums[0] += s0
		sums[1] += s1
		sums[2] += s2
		sums[3] += s3
	}
	sadSIMDBenchSink = sums[0] + sums[1] + sums[2] + sums[3]
	if sadSIMDBenchSink == 0 {
		b.Fatal("unexpected zero SAD")
	}
}

// boxColsRef is ScaleFilterCols_C with the ARM BLENDER, tap for tap.
func boxColsRef(dst, s []uint8, x, dx int) {
	for i := range dst {
		xi := x >> 16
		a, b := int(s[xi]), int(s[min(xi+1, len(s)-1)])
		dst[i] = uint8(a + (((x&0xffff)*(b-a) + 0x8000) >> 16))
		x += dx
	}
}

func boxColsRandomRow(rng *rand.Rand, n int) []uint8 {
	s := make([]uint8, n)
	switch rng.Intn(3) {
	case 0:
		rng.Read(s)
	case 1: // extremes: the largest |b-a|
		for i := range s {
			s[i] = []uint8{0, 255}[rng.Intn(2)]
		}
	default:
		for i := range s {
			s[i] = uint8(i * 7)
		}
	}
	return s
}

// The table kernel gives boxFilterCols' samples for any width, phase and
// step in [1, 2], including a last chunk pulled back over the previous one
// and a right tap clamped to the row.
func TestBoxColsPlanMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ran := 0
	var cache boxColsCache
	for iter := 0; iter < 4000; iter++ {
		sw := 32 + rng.Intn(300)
		if iter%7 == 0 {
			sw = 32 + rng.Intn(8)
		}
		dx := 1<<16 + rng.Intn(1<<16+1)
		switch iter % 11 {
		case 0:
			dx = 1 << 16
		case 1:
			dx = 2 << 16
		case 2:
			dx = 0x18000
		}
		// The output count that fits: the last tap stays inside the row.
		x := 0
		if iter%3 == 0 {
			x = dx>>1 - 32768
		} else {
			x = rng.Intn(1 << 16)
		}
		dwMax := (((sw-1)<<16)-x)/dx + 1
		if dwMax < 16 {
			continue
		}
		dw := 16 + rng.Intn(dwMax-15)
		if iter%5 == 0 {
			dw = dwMax
		}
		plan := cache.plan(sw, dw, x, dx, 1<<10)
		if plan == nil {
			t.Fatalf("sw=%d dw=%d x=%#x dx=%#x: no plan", sw, dw, x, dx)
		}
		s := boxColsRandomRow(rng, sw)
		got, want := make([]uint8, dw+3), make([]uint8, dw+3)
		got[dw], got[dw+1], got[dw+2] = 0xa5, 0xa5, 0xa5
		want[dw], want[dw+1], want[dw+2] = 0xa5, 0xa5, 0xa5
		ran++
		plan.filter(got[:dw], s)
		boxColsRef(want[:dw], s, x, dx)
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("sw=%d dw=%d x=%#x dx=%#x: sample %d = %d, want %d", sw, dw, x, dx, i, got[i], want[i])
				}
			}
		}
	}
	if ran < 3000 {
		t.Fatalf("only %d of 4000 random rows ran", ran)
	}
}

// Every (a, b) pair at the weights where the byte split of the blend is
// most likely to slip.
func TestBoxColsBlendByteSplit(t *testing.T) {
	fs := []int{0, 1, 2, 127, 128, 129, 254, 255, 256, 257, 0x7fff, 0x8000, 0x8001, 0xfeff, 0xff00, 0xff01, 0xfffe, 0xffff}
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 64; i++ {
		fs = append(fs, rng.Intn(1<<16))
	}
	var cache boxColsCache
	const sw = 64
	for _, f := range fs {
		// A step of exactly 1.0 from phase f: output i blends s[i], s[i+1] at f.
		dw := 40
		plan := cache.plan(sw, dw, f, 1<<16, 1<<10)
		if plan == nil {
			t.Fatalf("f=%#x: no plan", f)
		}
		for a := 0; a < 256; a++ {
			for b := 0; b < 256; b += 1 {
				s := make([]uint8, sw)
				for i := range s {
					s[i] = uint8(a)
					if i&1 == 1 {
						s[i] = uint8(b)
					}
				}
				got, want := make([]uint8, dw), make([]uint8, dw)
				plan.filter(got, s)
				boxColsRef(want, s, f, 1<<16)
				if !bytes.Equal(got, want) {
					t.Fatalf("f=%#x a=%d b=%d: %v want %v", f, a, b, got, want)
				}
			}
		}
	}
}

// A full plane through the bilinear-down path, SIMD columns against the
// scalar row and column reference, on the layer geometries and odd sizes.
func TestScalePlaneBilinearDownColsMatchesScalar(t *testing.T) {
	var scratch boxScaleScratch
	steps := append([]struct{ sw, sh, dw, dh int }{
		{1280, 720, 853, 480}, {853, 480, 568, 320}, {640, 360, 427, 240}, {427, 240, 284, 160},
		{135, 135, 68, 68}, {270, 135, 135, 68}, {101, 77, 67, 51}, {64, 64, 40, 41}, {1920, 1080, 1280, 720},
	}, boxTestSteps...)
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 40; i++ {
		sw, sh := 32+rng.Intn(200), 8+rng.Intn(60)
		steps = append(steps, struct{ sw, sh, dw, dh int }{sw, sh, max(sw*(2+rng.Intn(3))/4, 1), max(sh*(2+rng.Intn(3))/4, 1)})
	}
	for _, st := range steps {
		const pad = 5
		src := boxTestPlane(st.sw, st.sh, st.sw+pad)
		got, want := make([]uint8, st.dw*st.dh), make([]uint8, st.dw*st.dh)
		scalePlaneBox8(got, st.dw, st.dw, st.dh, src, st.sw+pad, st.sw, st.sh, &scratch)
		boxScalePlaneRef(want, st.dw, st.dh, src, st.sw+pad, st.sw, st.sh)
		if !bytes.Equal(got, want) {
			t.Fatalf("%+v: plane differs from the scalar scaler", st)
		}
		if st.sw >= 32 && st.dw >= 16 && st.sh >= 8 && st == steps[0] && scratch.cols.plans[0].dw != st.dw {
			t.Fatalf("%+v: the column table was not used", st)
		}
	}
}

// boxScalePlaneRef is scalePlaneBox8 with scalar columns only.
func boxScalePlaneRef(dst []uint8, dw, dh int, src []uint8, srcStride, sw, sh int) {
	filter := boxScaleFilter(sw, sh, dw, dh)
	x, y, dx, dy := boxScaleSlope(sw, sh, dw, dh, filter)
	switch {
	case dw == sw && dh == sh:
		for j := 0; j < dh; j++ {
			copy(dst[j*dw:j*dw+dw], src[j*srcStride:j*srcStride+sw])
		}
		return
	case 2*dw == sw && 2*dh == sh, filter == boxFilterBox && dh*2 < sh, filter == boxFilterNone:
		var scratch boxScaleScratch
		scalePlaneBox8(dst, dw, dw, dh, src, srcStride, sw, sh, &scratch)
		return
	}
	row := make([]uint8, sw)
	maxY := (sh - 1) << 16
	y = min(y, maxY)
	for j := 0; j < dh; j++ {
		yi := y >> 16
		s := src[yi*srcStride : yi*srcStride+sw]
		if filter == boxFilterBilinear {
			if yf := (y >> 8) & 255; yf != 0 {
				boxInterpolateRow(row, s, src[(yi+1)*srcStride:(yi+1)*srcStride+sw], yf, 128)
				s = row
			}
		}
		boxColsRef(dst[j*dw:j*dw+dw], s, x, dx)
		y = min(y+dy, maxY)
	}
}

func BenchmarkBoxColsPlan8(b *testing.B) {
	for _, tc := range []struct {
		name   string
		sw, dw int
		x, dx  int
	}{
		{"2/3 672", 1008, 672, 0x4000, 0x18000},
		{"640->427", 640, 427, 0, 0},
		{"853->568", 853, 568, 0, 0},
	} {
		x, dx := tc.x, tc.dx
		if dx == 0 {
			dx = boxFixedDiv(tc.sw, tc.dw)
			x = dx>>1 - 32768
		}
		s := boxTestPlane(tc.sw, 1, tc.sw)
		d := make([]byte, tc.dw)
		var cache boxColsCache
		plan := cache.plan(tc.sw, tc.dw, x, dx, 1<<10)
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(tc.dw))
			for b.Loop() {
				plan.filter(d, s)
			}
		})
	}
}
