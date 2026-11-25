//go:build goexperiment.simd && (arm64 || amd64) && !purego

package motion

import (
	"bytes"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"math/rand"
	"reflect"
	"testing"
)

// These differential tests assert the GoSIMD 8-bit compound kernels are
// bit-identical to the pure-Go references for every width/height and every
// subpel phase of each filter type. They call the GoSIMD wrappers directly rather
// than through the dispatch slots, so they validate the asm even on hosts whose
// CPUID does not advertise GoSIMD (e.g. amd64 under Rosetta 2, which translates
// GoSIMD anyway).

func compoundGoSIMDSizes() []struct{ w, h int } {
	return []struct{ w, h int }{
		{8, 1}, {8, 8}, {16, 7}, {16, 16}, {24, 3}, {32, 11}, {48, 5}, {64, 16},
	}
}

func compoundRoundParams8() (round0, offsetBits, roundOffset, roundBits int) {
	round0 = compoundRound0(8)
	offsetBits = 8 + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits = 2*filterBits - round0 - compoundRound1Bits
	return
}

func TestBlendCompoundAvg8GoSIMDSweepMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xB1E0D))
	_, _, roundOffset, roundBits := compoundRoundParams8()
	weights := [][2]int{{8, 8}, {9, 7}, {13, 3}, {4, 12}, {16, 0}, {0, 16}}
	for _, sz := range compoundGoSIMDSizes() {
		src0 := make([]uint16, sz.w*sz.h)
		src1 := make([]uint16, sz.w*sz.h)
		for i := range src0 {
			src0[i] = uint16(rng.Intn(1 << 12))
			src1[i] = uint16(rng.Intn(1 << 12))
		}
		for _, w := range weights {
			const stride = 96
			got := frame.Plane{Pix: make([]byte, stride*(sz.h+2)), Stride: stride, Width: sz.w + 4, Height: sz.h + 2}
			want := frame.Plane{Pix: make([]byte, stride*(sz.h+2)), Stride: stride, Width: sz.w + 4, Height: sz.h + 2}
			blendCompoundAvg8GoSIMD(got, src0, src1, 2, 1, sz.w, sz.h, w[0], w[1], roundOffset, roundBits)
			blendCompoundAvg8PureGo(want, src0, src1, 2, 1, sz.w, sz.h, w[0], w[1], roundOffset, roundBits)
			for i := range got.Pix {
				if got.Pix[i] != want.Pix[i] {
					t.Fatalf("blend %dx%d w=%v byte %d: GoSIMD=%d PureGo=%d", sz.w, sz.h, w, i, got.Pix[i], want.Pix[i])
				}
			}
		}
	}
}

func TestCompound8CopyGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0FFEE))
	round0, _, roundOffset, _ := compoundRoundParams8()
	const pad = filterTaps
	for _, sz := range compoundGoSIMDSizes() {
		side := sz.w
		if sz.h > side {
			side = sz.h
		}
		ref := randPlane(rng, side+2*pad, 1)
		got := make([]uint16, sz.w*sz.h)
		want := make([]uint16, sz.w*sz.h)
		predictInterCompoundRef8ToConvBufCopyGoSIMD(got, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
		predictInterCompoundRef8ToConvBufCopyPureGo(want, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("copy %dx%d sample %d: GoSIMD=%d PureGo=%d", sz.w, sz.h, i, got[i], want[i])
			}
		}
	}
}

func TestCompound8XGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA11CE))
	_, _, roundOffset, _ := compoundRoundParams8()
	const pad = filterTaps
	for _, tbl := range avx2FilterTables() {
		for ph := 1; ph < 16; ph++ {
			k := tbl[ph]
			for _, sz := range compoundGoSIMDSizes() {
				side := sz.w
				if sz.h > side {
					side = sz.h
				}
				ref := randPlane(rng, side+2*pad, 1)
				got := make([]uint16, sz.w*sz.h)
				want := make([]uint16, sz.w*sz.h)
				predictInterCompoundRef8ToConvBufXGoSIMD(got, ref, pad, pad, sz.w, sz.h, k, roundOffset)
				predictInterCompoundRef8ToConvBufXPureGo(want, ref, pad, pad, sz.w, sz.h, k, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("X %dx%d ph=%d sample %d: GoSIMD=%d PureGo=%d", sz.w, sz.h, ph, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompound8YGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xB0B))
	round0, _, roundOffset, _ := compoundRoundParams8()
	const pad = filterTaps
	for _, tbl := range avx2FilterTables() {
		for ph := 1; ph < 16; ph++ {
			k := tbl[ph]
			for _, sz := range compoundGoSIMDSizes() {
				side := sz.w
				if sz.h > side {
					side = sz.h
				}
				ref := randPlane(rng, side+2*pad, 1)
				got := make([]uint16, sz.w*sz.h)
				want := make([]uint16, sz.w*sz.h)
				predictInterCompoundRef8ToConvBufYGoSIMD(got, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
				predictInterCompoundRef8ToConvBufYPureGo(want, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("Y %dx%d ph=%d sample %d: GoSIMD=%d PureGo=%d", sz.w, sz.h, ph, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompound82DGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2D2D))
	_, offsetBits, _, _ := compoundRoundParams8()
	const pad = filterTaps
	tables := avx2FilterTables()
	for xi, xtbl := range tables {
		for xph := 1; xph < 16; xph++ {
			xk := xtbl[xph]
			ytbl := tables[(xi+1)%len(tables)]
			for _, yph := range []int{1, 5, 9, 15} {
				yk := ytbl[yph]
				for _, sz := range compoundGoSIMDSizes() {
					side := sz.w
					if sz.h > side {
						side = sz.h
					}
					ref := randPlane(rng, side+2*pad, 1)
					got := make([]uint16, sz.w*sz.h)
					want := make([]uint16, sz.w*sz.h)
					predictInterCompoundRef8ToConvBuf2DGoSIMD(got, ref, pad, pad, sz.w, sz.h, xk, yk, offsetBits, nil)
					predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, pad, pad, sz.w, sz.h, xk, yk, offsetBits, nil)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("2D %dx%d xph=%d yph=%d sample %d: GoSIMD=%d PureGo=%d", sz.w, sz.h, xph, yph, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// The clamped (edge-overhanging) 2D path materializes a halo via emu_edge and
// reruns the resident kernel; assert it stays bit-identical to the per-tap
// clamping pure-Go reference at every block edge.
func TestCompound82DGoSIMDEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xED9E))
	_, offsetBits, _, _ := compoundRoundParams8()
	tables := avx2FilterTables()
	xk := tables[2][11]
	yk := tables[0][7]
	const refW, refH, stride = 40, 24, 64
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	var scGot, scWant CompoundConvolveScratch
	for _, sz := range []struct{ w, h int }{{8, 8}, {16, 8}, {8, 16}, {16, 16}} {
		for _, org := range [][2]int{{-3, -2}, {refW - sz.w + 2, 1}, {1, refH - sz.h + 3}, {refW - 4, refH - 4}} {
			got := make([]uint16, sz.w*sz.h)
			want := make([]uint16, sz.w*sz.h)
			predictInterCompoundRef8ToConvBuf2DGoSIMD(got, ref, org[0], org[1], sz.w, sz.h, xk, yk, offsetBits, &scGot)
			predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, org[0], org[1], sz.w, sz.h, xk, yk, offsetBits, &scWant)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("2D edge %dx%d org=%v sample %d: GoSIMD=%d PureGo=%d", sz.w, sz.h, org, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompound8GoSIMDZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA110C))
	round0, offsetBits, roundOffset, roundBits := compoundRoundParams8()
	const pad = filterTaps
	const w, h = 32, 32
	ref := randPlane(rng, w+2*pad, 1)
	out := make([]uint16, w*h)
	src0 := make([]uint16, w*h)
	src1 := make([]uint16, w*h)
	dst := frame.Plane{Pix: make([]byte, 96*(h+2)), Stride: 96, Width: w + 4, Height: h + 2}
	k := avx2FilterTables()[0][7]
	var scratch CompoundConvolveScratch
	if a := testing.AllocsPerRun(20, func() {
		blendCompoundAvg8GoSIMD(dst, src0, src1, 0, 0, w, h, 9, 7, roundOffset, roundBits)
		predictInterCompoundRef8ToConvBufCopyGoSIMD(out, ref, pad, pad, w, h, round0, roundOffset)
		predictInterCompoundRef8ToConvBufXGoSIMD(out, ref, pad, pad, w, h, k, roundOffset)
		predictInterCompoundRef8ToConvBufYGoSIMD(out, ref, pad, pad, w, h, k, round0, roundOffset)
		predictInterCompoundRef8ToConvBuf2DGoSIMD(out, ref, pad, pad, w, h, k, k, offsetBits, &scratch)
	}); a != 0 {
		t.Fatalf("8-bit compound GoSIMD allocates: %v allocs/run", a)
	}
}

func TestBlendCompoundAvg8GoSIMDMatchesPureGo(t *testing.T) {
	const (
		stride = 160
		planeH = 140
	)
	rng := rand.New(rand.NewSource(0x8a17ee))
	roundOffset := (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	const roundBits = 2*filterBits - compoundRound0Bits - compoundRound1Bits
	weights := [...][2]int{{8, 8}, {9, 7}, {7, 9}, {11, 5}, {5, 11}, {12, 4}, {4, 12}, {13, 3}, {3, 13}}
	sizes := [...]struct {
		width  int
		height int
	}{
		{2, 2}, {2, 8}, {4, 4}, {4, 7}, {4, 16}, {8, 2}, {8, 8}, {8, 32},
		{12, 4}, {12, 9}, {16, 5}, {16, 16}, {20, 6}, {32, 8}, {32, 32}, {64, 16}, {64, 64}, {128, 128},
	}
	for _, w := range weights {
		for _, sz := range sizes {
			for _, dstOff := range [...][2]int{{0, 0}, {3, 5}} {
				src0 := make([]uint16, sz.width*sz.height)
				src1 := make([]uint16, sz.width*sz.height)
				for i := range src0 {
					switch i % 7 {
					case 0:
						src0[i] = 0
						src1[i] = 65535
					case 1:
						src0[i] = 65535
						src1[i] = 0
					default:
						src0[i] = uint16(rng.Intn(1 << 16))
						src1[i] = uint16(rng.Intn(1 << 16))
					}
				}
				got := frame.Plane{Pix: make([]byte, stride*planeH), Stride: stride, Width: stride, Height: planeH}
				want := frame.Plane{Pix: make([]byte, stride*planeH), Stride: stride, Width: stride, Height: planeH}
				for i := range got.Pix {
					got.Pix[i] = byte(i * 31)
					want.Pix[i] = byte(i * 31)
				}
				blendCompoundAvg8GoSIMD(got, src0, src1, dstOff[0], dstOff[1], sz.width, sz.height, w[0], w[1], roundOffset, roundBits)
				blendCompoundAvg8PureGo(want, src0, src1, dstOff[0], dstOff[1], sz.width, sz.height, w[0], w[1], roundOffset, roundBits)
				for i := range got.Pix {
					if got.Pix[i] != want.Pix[i] {
						t.Fatalf("w=%v %dx%d dst=(%d,%d) byte %d: GoSIMD=%d PureGo=%d",
							w, sz.width, sz.height, dstOff[0], dstOff[1], i, got.Pix[i], want.Pix[i])
					}
				}
			}
		}
	}
}

func TestBlendCompoundAvg8GoSIMDZeroAlloc(t *testing.T) {
	const width, height = 32, 32
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	for i := range src0 {
		src0[i] = uint16(i * 37)
		src1[i] = uint16(i * 91)
	}
	dst := frame.Plane{Pix: make([]byte, 64*64), Stride: 64, Width: 64, Height: 64}
	roundOffset := (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	const roundBits = 2*filterBits - compoundRound0Bits - compoundRound1Bits
	allocs := testing.AllocsPerRun(100, func() {
		blendCompoundAvg8GoSIMD(dst, src0, src1, 0, 0, width, height, 9, 7, roundOffset, roundBits)
	})
	if allocs != 0 {
		t.Fatalf("blendCompoundAvg8GoSIMD allocates: %v allocs/run", allocs)
	}
}

func benchBlendCompoundAvg8(b *testing.B, width, height int, fn func(dst frame.Plane, src0, src1 []uint16, dstX, dstY, w, h, fwd, bck, roundOffset, roundBits int)) {
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	for i := range src0 {
		src0[i] = uint16(i * 37)
		src1[i] = uint16(i * 91)
	}
	dst := frame.Plane{Pix: make([]byte, 192*192), Stride: 192, Width: 192, Height: 192}
	roundOffset := (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	const roundBits = 2*filterBits - compoundRound0Bits - compoundRound1Bits
	runConvolveBench(b, width, height, func() {
		fn(dst, src0, src1, 0, 0, width, height, 9, 7, roundOffset, roundBits)
	})
}

func BenchmarkBlendCompoundAvg8GoSIMDDirect_32(b *testing.B) {
	benchBlendCompoundAvg8(b, 32, 32, blendCompoundAvg8GoSIMD)
}

func BenchmarkBlendCompoundAvg8PureGoDirect_32(b *testing.B) {
	benchBlendCompoundAvg8(b, 32, 32, blendCompoundAvg8PureGo)
}

func BenchmarkBlendCompoundAvg8GoSIMDDirect_8(b *testing.B) {
	benchBlendCompoundAvg8(b, 8, 8, blendCompoundAvg8GoSIMD)
}

func BenchmarkBlendCompoundAvg8PureGoDirect_8(b *testing.B) {
	benchBlendCompoundAvg8(b, 8, 8, blendCompoundAvg8PureGo)
}

func BenchmarkBlendCompoundAvg8GoSIMDDirect_4x8(b *testing.B) {
	benchBlendCompoundAvg8(b, 4, 8, blendCompoundAvg8GoSIMD)
}

func BenchmarkBlendCompoundAvg8PureGoDirect_4x8(b *testing.B) {
	benchBlendCompoundAvg8(b, 4, 8, blendCompoundAvg8PureGo)
}

func benchBlendCompoundAvg8Fair(b *testing.B, fn func(dst frame.Plane, src0, src1 []uint16, dstX, dstY, w, h, fwd, bck, roundOffset, roundBits int)) {
	const roundOffset, roundBits = 6144, 4
	const width, height = 32, 32
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	for i := range src0 {
		src0[i] = uint16(3000 + i%5000)
		src1[i] = uint16(4000 + (i*7)%5000)
	}
	dst, _ := testPlane(width, height, 1, width)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src0, src1, 0, 0, width, height, 8, 8, roundOffset, roundBits)
	}
}

func BenchmarkBlendCompoundAvg8GoSIMD_32(b *testing.B) {
	benchBlendCompoundAvg8Fair(b, blendCompoundAvg8GoSIMD)
}

func BenchmarkBlendCompoundAvg8Scalar_32(b *testing.B) {
	benchBlendCompoundAvg8Fair(b, blendCompoundAvg8PureGo)
}

// BenchmarkU8GoSIMDDirect measures the general 8-bit Go SIMD convolve and
// CONV_BUF kernels (the resident halved-tap forms) called directly.
func BenchmarkU8GoSIMDDirect(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	ref := frame.Plane{Pix: make([]byte, 192*192), Stride: 192, Width: 192, Height: 192}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	dst := frame.Plane{Pix: make([]byte, 192*192), Stride: 192, Width: 192, Height: 192}
	out := make([]uint16, 128*128)
	var scratch ConvolveScratch
	cscratch := new(CompoundConvolveScratch)
	xk, yk := subpelFilters8[5], subpelFilters8[11]
	const roundOffset = 6144
	for _, w := range []int{8, 16, 32} {
		h := w
		b.Run(fmt.Sprintf("x_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				convolveX8GoSIMD(dst, ref, 0, 0, 40, 40, w, h, xk)
			}
		})
		b.Run(fmt.Sprintf("y_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				convolveY8GoSIMD(dst, ref, 0, 0, 40, 40, w, h, yk)
			}
		})
		b.Run(fmt.Sprintf("2d_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				convolve2D8GoSIMDWithScratch(dst, ref, 0, 0, 40, 40, w, h, xk, yk, &scratch)
			}
		})
		b.Run(fmt.Sprintf("cx_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBufXGoSIMD(out, ref, 40, 40, w, h, xk, roundOffset)
			}
		})
		b.Run(fmt.Sprintf("cy_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBufYGoSIMD(out, ref, 40, 40, w, h, yk, compoundRound0Bits, roundOffset)
			}
		})
		b.Run(fmt.Sprintf("c2d_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBuf2DGoSIMD(out, ref, 40, 40, w, h, xk, yk, 19, cscratch)
			}
		})
	}
}

func TestCompound8HalfExtremesAndExactWindows(t *testing.T) {
	round0, offsetBits, roundOffset, _ := compoundRoundParams8()
	for _, tbl := range avx2FilterTables() {
		for phase := 0; phase < 16; phase++ {
			k := tbl[phase]
			for _, shape := range [][2]int{{4, 4}, {8, 16}, {32, 32}, {128, 128}} {
				w, h := shape[0], shape[1]
				for pattern := 0; pattern < 3; pattern++ {
					stride := w + 9
					ref := frame.Plane{Pix: make([]byte, (h+6)*stride+w+8), Stride: stride, Width: w + 8, Height: h + 7}
					for y := 0; y < ref.Height; y++ {
						for x := 0; x < ref.Width; x++ {
							if pattern == 1 || pattern == 2 && (x+y)&1 != 0 {
								ref.Pix[y*stride+x] = 255
							}
						}
					}
					var scratch CompoundConvolveScratch
					for axis := 0; axis < 4; axis++ {
						got := make([]uint16, w*h)
						want := make([]uint16, w*h)
						switch axis {
						case 0:
							predictInterCompoundRef8ToConvBufCopyGoSIMD(got, ref, 3, 3, w, h, round0, roundOffset)
							predictInterCompoundRef8ToConvBufCopyPureGo(want, ref, 3, 3, w, h, round0, roundOffset)
						case 1:
							predictInterCompoundRef8ToConvBufXGoSIMD(got, ref, 3, 3, w, h, k, roundOffset)
							predictInterCompoundRef8ToConvBufXPureGo(want, ref, 3, 3, w, h, k, roundOffset)
						case 2:
							predictInterCompoundRef8ToConvBufYGoSIMD(got, ref, 3, 3, w, h, k, round0, roundOffset)
							predictInterCompoundRef8ToConvBufYPureGo(want, ref, 3, 3, w, h, k, round0, roundOffset)
						default:
							predictInterCompoundRef8ToConvBuf2DGoSIMD(got, ref, 3, 3, w, h, k, k, offsetBits, &scratch)
							predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, 3, 3, w, h, k, k, offsetBits, &scratch)
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("axis=%d phase=%d shape=%dx%d pattern=%d sample=%d: got=%d want=%d", axis, phase, w, h, pattern, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// blendHighBDRoundParams returns the CONV_BUF blend rounding constants for a
// bit depth, mirroring BlendCompoundAvg.
func blendHighBDRoundParams(bitDepth uint8) (roundOffset int, roundBits int) {
	bd := int(bitDepth)
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits = 2*filterBits - round0 - compoundRound1Bits
	return roundOffset, roundBits
}

// TestBlendCompoundAvgHighBDGoSIMDMatchesPureGo asserts the HBD GoSIMD compound
// average / dist-wtd blend is bit-identical to the pure-Go reference for every
// AV1 block width (including the width-4 two-rows-per-iteration variant and
// the odd-height fallback), every dist-wtd weight pair, at bit depths 10 and
// 12, over full-range uint16 CONV_BUF inputs that exercise both clip bounds.
func TestBlendCompoundAvgHighBDGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xb1e4d))
	widths := []int{4, 8, 16, 24, 32, 64, 128}
	heights := []int{1, 2, 3, 4, 5, 8, 16, 32, 64}
	weights := [][2]int{{8, 8}, {9, 7}, {7, 9}, {11, 5}, {5, 11}, {13, 3}, {3, 13}}
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		roundOffset, roundBits := blendHighBDRoundParams(bd)
		for _, w := range widths {
			for _, h := range heights {
				src0 := make([]uint16, w*h)
				src1 := make([]uint16, w*h)
				for i := range src0 {
					// Full uint16 range stresses both the >= 0 and <= max clamps.
					src0[i] = uint16(rng.Intn(1 << 16))
					src1[i] = uint16(rng.Intn(1 << 16))
				}
				for _, wt := range weights {
					for _, org := range [][2]int{{0, 0}, {2, 3}} {
						dstX, dstY := org[0], org[1]
						planeW := dstX + w
						planeH := dstY + h
						got, _ := testPlane(planeW, planeH, 2, planeW*2)
						want, _ := testPlane(planeW, planeH, 2, planeW*2)
						blendCompoundAvgHighBDGoSIMD(got, src0, src1, max, dstX, dstY, w, h, wt[0], wt[1], roundOffset, roundBits)
						blendCompoundAvgHighBD(want, src0, src1, max, dstX, dstY, w, h, wt[0], wt[1], roundOffset, roundBits)
						for y := 0; y < h; y++ {
							for x := 0; x < w; x++ {
								g := getSample(got, 2, dstX+x, dstY+y)
								e := getSample(want, 2, dstX+x, dstY+y)
								if g != e {
									t.Fatalf("bd=%d %dx%d wt=%v org=%v (%d,%d): GoSIMD=%d PureGo=%d",
										bd, w, h, wt, org, x, y, g, e)
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestBlendCompoundAvgHighBDGoSIMDZeroAlloc asserts the HBD blend GoSIMD wrapper
// allocates nothing on either asm path.
func TestBlendCompoundAvgHighBDGoSIMDZeroAlloc(t *testing.T) {
	max, _ := highBDMax(10)
	roundOffset, roundBits := blendHighBDRoundParams(10)
	src0 := make([]uint16, 32*32)
	src1 := make([]uint16, 32*32)
	for i := range src0 {
		src0[i] = uint16(i * 37)
		src1[i] = uint16(i * 91)
	}
	dst, _ := testPlane(32, 32, 2, 64)
	cases := []struct {
		name string
		fn   func()
	}{
		{"W8", func() {
			blendCompoundAvgHighBDGoSIMD(dst, src0[:32*8], src1[:32*8], max, 0, 0, 32, 8, 9, 7, roundOffset, roundBits)
		}},
		{"W4", func() {
			blendCompoundAvgHighBDGoSIMD(dst, src0[:4*8], src1[:4*8], max, 0, 0, 4, 8, 8, 8, roundOffset, roundBits)
		}},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(20, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
}

// TestConvolve2DHighBDGoSIMDWithScratchMatchesPureGo asserts the scratch-carrying
// HBD 2D GoSIMD convolve (resident and edge-clamped emu_edge shapes) stays
// bit-identical to the pure-Go references with a deliberately poisoned scratch,
// proving every intermediate sample read is written first.
func TestConvolve2DHighBDGoSIMDWithScratchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5c4a7c))
	scratch := &ConvolveScratch{}
	poison := func() {
		for i := range scratch.imHBD {
			scratch.imHBD[i] = int32(0x7bad0000) | int32(i&0xffff)
		}
		for i := range scratch.edge16 {
			scratch.edge16[i] = byte(0xa5)
		}
	}
	sizes := []int{4, 8, 16, 32, 64, 128}
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		xk := subpelFilters8[6]
		yk := subpelFilters8[9]
		for _, w := range sizes {
			for _, h := range sizes {
				side := w
				if h > side {
					side = h
				}
				ref := makeHighBDRef(side, pad, max, true, rng)
				// Resident window.
				got, _ := testPlane(w, h, 2, w*2)
				want, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDGoSIMDWithScratch(got, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(want, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				eqHighBDBlock(t, got, want, w, h, "2DHBDscratch", bd, w, h)
				// Overhanging window (emu_edge path through scratch.edge16).
				gotC, _ := testPlane(w, h, 2, w*2)
				wantC, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedGoSIMDWithScratch(gotC, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedPureGo(wantC, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				eqHighBDBlock(t, gotC, wantC, w, h, "2DHBDscratch-clamped", bd, w, h)
			}
		}
	}
}

// TestConvolve2DHighBDGoSIMDWithScratchZeroAlloc asserts the scratch path (both
// resident and emu_edge clamped) allocates nothing.
func TestConvolve2DHighBDGoSIMDWithScratchZeroAlloc(t *testing.T) {
	max, _ := highBDMax(10)
	rng := rand.New(rand.NewSource(11))
	const pad = filterTaps
	ref := makeHighBDRef(64, pad, max, true, rng)
	dst, _ := testPlane(64, 64, 2, 128)
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	scratch := &ConvolveScratch{}
	cases := []struct {
		name string
		fn   func()
	}{
		{"resident", func() {
			convolve2DHighBDGoSIMDWithScratch(dst, ref, 10, max, 0, 0, pad, pad, 64, 64, xk, yk, scratch)
		}},
		{"clamped", func() {
			convolve2DHighBDClampedGoSIMDWithScratch(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, xk, yk, scratch)
		}},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(20, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
}

// Differential tests for the GoSIMD high-bit-depth compound kernels. They call
// the GoSIMD wrappers directly so the asm executes (not skips) under GOARCH=amd64
// on a Rosetta host.

func compoundHBDParams(bitDepth uint8) (round0, offsetBits, roundOffset int) {
	round0 = compoundRound0(bitDepth)
	offsetBits = int(bitDepth) + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	return
}

func compoundHBDSizes() []struct{ w, h int } {
	return []struct{ w, h int }{
		{4, 4}, {4, 13}, {8, 1}, {8, 8}, {12, 3}, {16, 7}, {32, 11}, {64, 16},
	}
}

func TestCompoundHBDCopyGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE01))
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		for _, sz := range compoundHBDSizes() {
			side := sz.w
			if sz.h > side {
				side = sz.h
			}
			ref := randPlaneHBD(rng, side+2*pad, max)
			got := make([]uint16, sz.w*sz.h)
			want := make([]uint16, sz.w*sz.h)
			predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD(got, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(want, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("copy bd=%d %dx%d sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompoundHBDXGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE02))
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		for _, tbl := range avx2FilterTables() {
			for ph := 1; ph < 16; ph++ {
				k := tbl[ph]
				for _, sz := range compoundHBDSizes() {
					side := sz.w
					if sz.h > side {
						side = sz.h
					}
					ref := randPlaneHBD(rng, side+2*pad, max)
					got := make([]uint16, sz.w*sz.h)
					want := make([]uint16, sz.w*sz.h)
					predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(got, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufXResident(want, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("X bd=%d %dx%d ph=%d sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, ph, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHBDYGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE03))
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		for _, tbl := range avx2FilterTables() {
			for ph := 1; ph < 16; ph++ {
				k := tbl[ph]
				for _, sz := range compoundHBDSizes() {
					side := sz.w
					if sz.h > side {
						side = sz.h
					}
					ref := randPlaneHBD(rng, side+2*pad, max)
					got := make([]uint16, sz.w*sz.h)
					want := make([]uint16, sz.w*sz.h)
					predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(got, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufYResident(want, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("Y bd=%d %dx%d ph=%d sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, ph, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHBD2DGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE04))
	const pad = filterTaps
	tables := avx2FilterTables()
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, offsetBits, _ := compoundHBDParams(bd)
		for xi, xtbl := range tables {
			for xph := 1; xph < 16; xph++ {
				xk := xtbl[xph]
				ytbl := tables[(xi+1)%len(tables)]
				for _, yph := range []int{1, 5, 9, 15} {
					yk := ytbl[yph]
					for _, sz := range compoundHBDSizes() {
						side := sz.w
						if sz.h > side {
							side = sz.h
						}
						ref := randPlaneHBD(rng, side+2*pad, max)
						got := make([]uint16, sz.w*sz.h)
						want := make([]uint16, sz.w*sz.h)
						var imG, imW compoundIM
						predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got, ref, pad, pad, sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imG)
						predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, pad, pad, sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imW)
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("2D bd=%d %dx%d xph=%d yph=%d sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, xph, yph, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

func TestCompoundHBD2DClampedGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE05))
	tables := avx2FilterTables()
	xk := tables[2][11]
	yk := tables[0][7]
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, offsetBits, _ := compoundHBDParams(bd)
		const refW, refH = 40, 24
		stride := refW * 2
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		for y := range refH {
			for x := range refW {
				storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
			}
		}
		for _, sz := range []struct{ w, h int }{{4, 4}, {8, 8}, {16, 8}, {8, 16}} {
			for _, org := range [][2]int{{-3, -2}, {refW - sz.w + 2, 1}, {1, refH - sz.h + 3}, {refW - 4, refH - 4}} {
				got := make([]uint16, sz.w*sz.h)
				want := make([]uint16, sz.w*sz.h)
				var imG, imW compoundIM
				var edge emuEdge16Buf
				for i := range edge {
					edge[i] = 0xa5
				}
				gotEdge := make([]uint16, sz.w*sz.h)
				predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(got, ref, org[0], org[1], sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imG, nil)
				predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(gotEdge, ref, org[0], org[1], sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imG, &edge)
				predictInterCompoundRefHighBDToConvBuf2DClamped(want, ref, org[0], org[1], sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imW)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("2D clamped bd=%d %dx%d org=%v sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, org, i, got[i], want[i])
					}
					if gotEdge[i] != want[i] {
						t.Fatalf("2D clamped (edge) bd=%d %dx%d org=%v sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, org, i, gotEdge[i], want[i])
					}
				}
			}
		}
	}
}

// TestCompoundHBDXYClampedEmuEdgeGoSIMDMatchesPureGo runs the GoSIMD X/Y resident
// kernels over emu_edge halo windows of edge-overhanging blocks (the compound
// X/Y clamped fast path) and asserts bit-identity with the pure-Go per-tap
// clamping references.
func TestCompoundHBDXYClampedEmuEdgeGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE07))
	tables := avx2FilterTables()
	xk := tables[2][11]
	yk := tables[0][7]
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		const refW, refH = 40, 24
		stride := refW * 2
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		for y := range refH {
			for x := range refW {
				storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
			}
		}
		for _, sz := range []struct{ w, h int }{{4, 4}, {8, 8}, {16, 8}, {8, 16}, {12, 4}, {64, 32}} {
			for _, org := range [][2]int{{-3, -2}, {refW - sz.w + 2, 1}, {1, refH - sz.h + 3}, {refW - 4, refH - 4}, {-sz.w, -sz.h}} {
				gotX := make([]uint16, sz.w*sz.h)
				gotY := make([]uint16, sz.w*sz.h)
				wantX := make([]uint16, sz.w*sz.h)
				wantY := make([]uint16, sz.w*sz.h)
				var edge emuEdge16Buf
				for i := range edge {
					edge[i] = 0xa5
				}
				emu, emuX := emuEdgeWindow16X(ref, org[0], org[1], sz.w, sz.h, &edge)
				predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(gotX, emu, emuX, 0, sz.w, sz.h, xk, round0, roundOffset)
				predictInterCompoundRefHighBDToConvBufXClamped(wantX, ref, org[0], org[1], sz.w, sz.h, xk, round0, roundOffset)
				for i := range edge {
					edge[i] = 0x5a
				}
				emu, emuY := emuEdgeWindow16Y(ref, org[0], org[1], sz.w, sz.h, &edge)
				predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(gotY, emu, 0, emuY, sz.w, sz.h, yk, round0, roundOffset)
				predictInterCompoundRefHighBDToConvBufYClamped(wantY, ref, org[0], org[1], sz.w, sz.h, yk, round0, roundOffset)
				for i := range wantX {
					if gotX[i] != wantX[i] {
						t.Fatalf("X emu bd=%d %dx%d org=%v sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, org, i, gotX[i], wantX[i])
					}
					if gotY[i] != wantY[i] {
						t.Fatalf("Y emu bd=%d %dx%d org=%v sample %d: GoSIMD=%d PureGo=%d", bd, sz.w, sz.h, org, i, gotY[i], wantY[i])
					}
				}
			}
		}
	}
}

func TestCompoundHBDGoSIMDZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE06))
	const pad = filterTaps
	const w, h = 32, 32
	round0, offsetBits, roundOffset := compoundHBDParams(12)
	ref := randPlaneHBD(rng, w+2*pad, (1<<12)-1)
	out := make([]uint16, w*h)
	k := avx2FilterTables()[0][7]
	var im compoundIM
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD(out, ref, pad, pad, w, h, round0, roundOffset)
		predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(out, ref, pad, pad, w, h, k, round0, roundOffset)
		predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(out, ref, pad, pad, w, h, k, round0, roundOffset)
		predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, pad, pad, w, h, k, k, round0, offsetBits, 12, &im)
	}); a != 0 {
		t.Fatalf("HBD compound GoSIMD allocates: %v allocs/run", a)
	}
}

func TestCompoundHighBD4TapWidth4ExactWindow(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		for _, height := range []int{1, 3, 13} {
			for _, stride := range []int{14, 15, 16, 31} {
				t.Run(fmt.Sprintf("%d/%d/%d", bitDepth, height, stride), func(t *testing.T) {
					ref := make([]byte, (height-1)*stride+14)
					maxSample := (1 << bitDepth) - 1
					for row := range height {
						for sample := range 7 {
							value := uint16((row*733 + sample*599) & maxSample)
							ref[row*stride+sample*2] = byte(value)
							ref[row*stride+sample*2+1] = byte(value >> 8)
						}
					}
					storage := make([]uint16, height*4+2)
					storage[0], storage[len(storage)-1] = 0xbeef, 0xcafe
					out := storage[1 : len(storage)-1]
					round0 := compoundRound0(bitDepth)
					offsetBits := int(bitDepth) + 2*filterBits - round0
					offset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
					kernel := [filterTaps]int16{-8, 80, 64, -8}
					ctx := compoundFilterGoSIMDCtx{out: out, ref: ref, kernel: kernel, refStr: stride,
						width: 4, height: height, round0: round0, roundOffset: int32(offset), taps: 4}
					compoundXHighBDKernel(&ctx)
					for row := range height {
						for column := range 4 {
							var sum int32
							for tap := range 4 {
								index := row*stride + (column+tap)*2
								sample := uint16(ref[index]) | uint16(ref[index+1])<<8
								sum += int32(sample) * int32(kernel[tap])
							}
							want := uint16(((sum + (1 << (round0 - 1))) >> round0) + int32(offset))
							if got := out[row*4+column]; got != want {
								t.Fatalf("row=%d column=%d got=%d want=%d", row, column, got, want)
							}
						}
					}
					if storage[0] != 0xbeef || storage[len(storage)-1] != 0xcafe {
						t.Fatal("output guard overwritten")
					}
					if allocs := testing.AllocsPerRun(100, func() { compoundXHighBDKernel(&ctx) }); allocs != 0 {
						t.Fatalf("allocs=%g, want zero", allocs)
					}
				})
			}
		}
	}
}

func TestCompoundHighBDCopyGoSIMDMatchesPureGo(t *testing.T) {
	const (
		refW   = 48
		refH   = 19
		stride = 112
		refX   = 5
		refY   = 4
	)
	for _, bitDepth := range []uint8{10, 12} {
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		mask := (1 << int(bitDepth)) - 1
		for y := range refH {
			for x := range refW {
				storeHighBDSample(ref, x, y, uint16((x*97+y*53+x*y*11+int(bitDepth)*17)&mask))
			}
		}
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		for _, tc := range [...]struct {
			width  int
			height int
		}{
			{width: 8, height: 1},
			{width: 16, height: 7},
			{width: 32, height: 11},
		} {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD(got, ref, refX, refY, tc.width, tc.height, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(want, ref, refX, refY, tc.width, tc.height, round0, roundOffset)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bd=%d %dx%d sample %d: GoSIMD=%d PureGo=%d", bitDepth, tc.width, tc.height, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompoundHighBDXGoSIMDMatchesPureGo(t *testing.T) {
	const (
		pad = filterTaps
	)
	rng := rand.New(rand.NewSource(0x4b1d0a))
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{4, 4},
		{4, 13},
		{8, 1},
		{16, 7},
		{32, 11},
		{64, 16},
	}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		for _, size := range sizes {
			refW := size.width + 2*pad
			refH := size.height + 2*pad
			stride := refW * 2
			ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
			for y := range refH {
				for x := range refW {
					storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
				}
			}
			for _, filter := range filters {
				for subX := 1; subX <= subpelQ4Mask; subX++ {
					kernel, err := interpKernel(filter, size.width, subX)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]uint16, size.width*size.height)
					want := make([]uint16, size.width*size.height)
					predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(got, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufXResident(want, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d filter=%d size=%dx%d subX=%d sample=%d GoSIMD=%d PureGo=%d",
								bitDepth, filter, size.width, size.height, subX, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHighBDXGoSIMDFallbackMatchesPureGo(t *testing.T) {
	const (
		refW   = 48
		refH   = 24
		stride = refW * 2
		refX   = filterTaps
		refY   = filterTaps
	)
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[5]
	for _, tc := range []struct {
		name          string
		width, height int
		kernel        [filterTaps]int16
	}{
		{name: "odd_width", width: 12, height: 8, kernel: kernel},
		{name: "width4_non_4tap", width: 4, height: 8, kernel: kernel},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(got, ref, refX, refY, tc.width, tc.height, tc.kernel, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufXResident(want, ref, refX, refY, tc.width, tc.height, tc.kernel, round0, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d GoSIMD wrapper=%d PureGo=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundHighBDYGoSIMDMatchesPureGo(t *testing.T) {
	const pad = filterTaps
	rng := rand.New(rand.NewSource(0x7c0f21))
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{4, 4},
		{4, 13},
		{8, 1},
		{16, 7},
		{32, 11},
		{64, 16},
	}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		for _, size := range sizes {
			refW := size.width + 2*pad
			refH := size.height + 2*pad
			stride := refW * 2
			ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
			for y := range refH {
				for x := range refW {
					storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
				}
			}
			for _, filter := range filters {
				for subY := 1; subY <= subpelQ4Mask; subY++ {
					kernel, err := interpKernel(filter, size.height, subY)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]uint16, size.width*size.height)
					want := make([]uint16, size.width*size.height)
					predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(got, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufYResident(want, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d filter=%d size=%dx%d subY=%d sample=%d GoSIMD=%d PureGo=%d",
								bitDepth, filter, size.width, size.height, subY, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHighBDYGoSIMDFallbackMatchesPureGo(t *testing.T) {
	const (
		refW   = 48
		refH   = 24
		stride = refW * 2
		refX   = filterTaps
		refY   = filterTaps
	)
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[5]
	got := make([]uint16, 12*8)
	want := make([]uint16, 12*8)
	predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(got, ref, refX, refY, 12, 8, kernel, round0, roundOffset)
	predictInterCompoundRefHighBDToConvBufYResident(want, ref, refX, refY, 12, 8, kernel, round0, roundOffset)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("odd_width sample=%d GoSIMD wrapper=%d PureGo=%d", i, got[i], want[i])
		}
	}
}

func TestCompoundHighBD2DGoSIMDMatchesPureGo(t *testing.T) {
	const pad = filterTaps
	rng := rand.New(rand.NewSource(0x2d4c91))
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{4, 4},
		{4, 13},
		{8, 1},
		{16, 7},
		{32, 11},
		{64, 16},
	}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		for _, size := range sizes {
			refW := size.width + 2*pad
			refH := size.height + 2*pad
			stride := refW * 2
			ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
			for y := range refH {
				for x := range refW {
					storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
				}
			}
			for _, xFilter := range filters {
				for _, yFilter := range filters {
					for subX := 1; subX <= subpelQ4Mask; subX++ {
						xKernel, err := interpKernel(xFilter, size.width, subX)
						if err != nil {
							t.Fatal(err)
						}
						for subY := 1; subY <= subpelQ4Mask; subY++ {
							yKernel, err := interpKernel(yFilter, size.height, subY)
							if err != nil {
								t.Fatal(err)
							}
							got := make([]uint16, size.width*size.height)
							want := make([]uint16, size.width*size.height)
							var gotIM compoundIM
							var wantIM compoundIM
							predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got, ref, pad, pad, size.width, size.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM)
							predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, pad, pad, size.width, size.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &wantIM)
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("bd=%d filters=%d/%d size=%dx%d sub=%d/%d sample=%d GoSIMD=%d PureGo=%d",
										bitDepth, xFilter, yFilter, size.width, size.height, subX, subY, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestCompoundHighBD2DGoSIMDFallbackMatchesPureGo(t *testing.T) {
	const (
		width  = 12
		height = 8
		refX   = filterTaps
		refY   = filterTaps
	)
	fo := filterTaps/2 - 1
	refW := refX - fo + width + filterTaps - 1
	refH := refY - fo + height + filterTaps - 1
	stride := refW * 2
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	got := make([]uint16, width*height)
	want := make([]uint16, width*height)
	var gotIM compoundIM
	var wantIM compoundIM
	predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, 10, &gotIM)
	predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, 10, &wantIM)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("right_edge_fallback sample=%d GoSIMD wrapper=%d PureGo=%d", i, got[i], want[i])
		}
	}
}

// TestCompoundHighBD2DClampedEmuEdgeMatchesPureGo drives the HBD compound 2D
// clamped GoSIMD path over tap windows that overhang the reference plane on every
// side, forcing the emu_edge halo materialization, and asserts bit-identity
// with the pure-Go per-tap-clamping reference at bit depths 10 and 12.
func TestCompoundHighBD2DClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d16bced))
	const refW, refH = 24, 20
	sizes := []int{4, 8, 12, 16, 32, 64}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
		for y := 0; y < refH; y++ {
			for x := 0; x < refW; x++ {
				v := uint16(rng.Intn(int(max) + 1))
				if (x == 0 || x == refW-1) && (y == 0 || y == refH-1) {
					if (x+y)&1 == 0 {
						v = 0
					} else {
						v = max
					}
				}
				storeHighBDSample(ref, x, y, v)
			}
		}
		xKernel := subpelFilters8[6]
		yKernel := subpelFilters8[9]
		for _, w := range sizes {
			for _, h := range sizes {
				offs := [][2]int{
					{-w, -h}, {-3, -3}, {2, -5}, {refW - 2, -1},
					{refW - 1, 5}, {refW, refH}, {5, refH - 2},
					{-4, refH - 3}, {-6, 4}, {4, 4},
				}
				for _, o := range offs {
					got := make([]uint16, w*h)
					gotEdge := make([]uint16, w*h)
					want := make([]uint16, w*h)
					var gotIM, wantIM compoundIM
					var edge emuEdge16Buf
					// Poison the caller-owned edge window to prove every sample
					// read by the resident kernel was materialized first.
					for i := range edge {
						edge[i] = 0xa5
					}
					predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(got, ref, o[0], o[1], w, h, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM, nil)
					predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(gotEdge, ref, o[0], o[1], w, h, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM, &edge)
					predictInterCompoundRefHighBDToConvBuf2DClamped(want, ref, o[0], o[1], w, h, xKernel, yKernel, round0, offsetBits, int(bitDepth), &wantIM)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d %dx%d off=%v sample=%d GoSIMD=%d PureGo=%d",
								bitDepth, w, h, o, i, got[i], want[i])
						}
						if gotEdge[i] != want[i] {
							t.Fatalf("bd=%d %dx%d off=%v sample=%d GoSIMD(edge)=%d PureGo=%d",
								bitDepth, w, h, o, i, gotEdge[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestCompoundHighBD2DClampedEmuEdgeZeroAlloc proves the emu_edge halo scratch
// stays on the stack through the func-ptr dispatch slot.
func TestCompoundHighBD2DClampedEmuEdgeZeroAlloc(t *testing.T) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	xKernel := subpelFilters8[6]
	yKernel := subpelFilters8[9]
	out := make([]uint16, 64*64)
	var im compoundIM
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out, ref, -3, -3, 64, 64, xKernel, yKernel, round0, offsetBits, 10, &im, nil)
	}); a != 0 {
		t.Fatalf("emu_edge compound 2D clamped allocated %v times, want 0", a)
	}
	var edge emuEdge16Buf
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out, ref, -3, -3, 64, 64, xKernel, yKernel, round0, offsetBits, 10, &im, &edge)
	}); a != 0 {
		t.Fatalf("emu_edge compound 2D clamped (caller edge) allocated %v times, want 0", a)
	}
}

func BenchmarkCompoundHighBD2DClampedEmuEdge(b *testing.B) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	xKernel := subpelFilters8[6]
	yKernel := subpelFilters8[9]
	for _, w := range []int{8, 16, 32, 64} {
		out := make([]uint16, w*w)
		var im compoundIM
		b.Run("neon_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out, ref, -3, -3, w, w, xKernel, yKernel, round0, offsetBits, 10, &im, nil)
			}
		})
		b.Run("purego_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, -3, -3, w, w, xKernel, yKernel, round0, offsetBits, 10, &im)
			}
		})
	}
}

func BenchmarkCompoundConvBufXHighBDGoSIMDDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[3]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBufXHighBDPureGoDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[3]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBufYHighBDGoSIMDDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBufYHighBDPureGoDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBuf2DHighBDGoSIMDDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	var im compoundIM
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, round0, offsetBits, 10, &im)
	})
}

func BenchmarkCompoundConvBuf2DHighBDPureGoDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	var im compoundIM
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBuf2DResident(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, round0, offsetBits, 10, &im)
	})
}

// convolve2DKernelPhases returns every 8-tap subpel kernel phase across the four
// filter types so the differential test exercises all 16 x/y phases per type.

// TestConvolve2D8GoSIMDMatchesPureGo asserts the GoSIMD 2D convolve is bit-identical
// to convolve2D8PureGo for every width/height in 4..64, all 16 x and 16 y subpel
// phases of each filter type, over both deterministic and random reference
// pixels. It also asserts the GoSIMD path allocates nothing.
func TestConvolve2D8GoSIMDMatchesPureGo(t *testing.T) {
	tables := convolve2DKernelTables()
	rng := rand.New(rand.NewSource(0x2d20feed))

	const pad = filterTaps
	// Pick a representative subpel phase per filter for the full width/height
	// sweep, then sweep all 16x16 phases on a couple of fixed shapes.
	sweepSizes := []int{4, 8, 12, 16, 24, 32, 48, 64}

	makeRef := func(side int, randomize bool) frame.Plane {
		refSide := side + 2*pad
		r, _ := testPlane(refSide, refSide, 1, refSide)
		if randomize {
			for i := range r.Pix {
				r.Pix[i] = byte(rng.Intn(256))
			}
		} else {
			fillMotionTestPlane(r)
		}
		return r
	}

	run := func(t *testing.T, w, h int, xk, yk [filterTaps]int16, randomize bool) {
		t.Helper()
		side := w
		if h > side {
			side = h
		}
		ref := makeRef(side, randomize)
		got, _ := testPlane(w, h, 1, w)
		gotScratch, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		var scratch ConvolveScratch
		convolve2D8GoSIMD(got, ref, 0, 0, pad, pad, w, h, xk, yk)
		convolve2D8GoSIMDWithScratch(gotScratch, ref, 0, 0, pad, pad, w, h, xk, yk, &scratch)
		convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, xk, yk)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				g := got.Pix[y*got.Stride+x]
				gs := gotScratch.Pix[y*gotScratch.Stride+x]
				e := want.Pix[y*want.Stride+x]
				if g != e || gs != e {
					t.Fatalf("w=%d h=%d (%d,%d): GoSIMD=%d GoSIMD-scratch=%d PureGo=%d xk=%v yk=%v", w, h, x, y, g, gs, e, xk, yk)
				}
			}
		}
	}

	// Full width/height sweep with a non-trivial phase per filter type.
	for ti, tbl := range tables {
		xk := tbl[3]
		yk := tbl[5]
		for _, w := range sweepSizes {
			for _, h := range sweepSizes {
				for _, randomize := range []bool{false, true} {
					t.Run("", func(t *testing.T) { _ = ti; run(t, w, h, xk, yk, randomize) })
				}
			}
		}
	}

	// All 16x16 phase combinations per filter type on fixed shapes.
	for _, tbl := range tables {
		for sx := 0; sx < 16; sx++ {
			for sy := 0; sy < 16; sy++ {
				run(t, 16, 16, tbl[sx], tbl[sy], false)
				run(t, 8, 24, tbl[sx], tbl[sy], true)
			}
		}
	}

	// Zero-alloc assertion on the GoSIMD path (width%8==0, 8-tap -> asm).
	ref := makeRef(32, true)
	dst, _ := testPlane(32, 32, 1, 32)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8GoSIMDWithScratch(dst, ref, 0, 0, pad, pad, 32, 32, xk, yk, &scratch)
	})
	if allocs != 0 {
		t.Fatalf("convolve2D8GoSIMDWithScratch allocated %v times, want 0", allocs)
	}
}

// makeHighBDRef builds a (side+2*pad)-square high-bit-depth reference plane
// whose samples are valid for the given bit depth.
func TestConvolveClampedGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc1a))
	sizes := []int{4, 8, 12, 16, 32}

	// 8-bit clamped: place the block at the plane origin so the negative-offset
	// halo clamps, plus an interior offset that stays resident.
	for _, w := range sizes {
		for _, h := range sizes {
			side := w
			if h > side {
				side = h
			}
			ref, _ := testPlane(side, side, 1, side)
			for i := range ref.Pix {
				ref.Pix[i] = byte(rng.Intn(256))
			}
			xk := subpelFilters8[6]
			yk := subpelFilters8[9]
			for _, org := range []int{0, 1} {
				gx, _ := testPlane(w, h, 1, w)
				ex, _ := testPlane(w, h, 1, w)
				convolveX8ClampedGoSIMD(gx, ref, 0, 0, org, org, w, h, xk)
				convolveX8ClampedPureGo(ex, ref, 0, 0, org, org, w, h, xk)
				assertBytesEqual(t, gx, ex, w, h, "X8clamped", w, h, org)

				gy, _ := testPlane(w, h, 1, w)
				ey, _ := testPlane(w, h, 1, w)
				convolveY8ClampedGoSIMD(gy, ref, 0, 0, org, org, w, h, yk)
				convolveY8ClampedPureGo(ey, ref, 0, 0, org, org, w, h, yk)
				assertBytesEqual(t, gy, ey, w, h, "Y8clamped", w, h, org)

				g2, _ := testPlane(w, h, 1, w)
				e2, _ := testPlane(w, h, 1, w)
				convolve2D8ClampedGoSIMD(g2, ref, 0, 0, org, org, w, h, xk, yk)
				convolve2D8ClampedPureGo(e2, ref, 0, 0, org, org, w, h, xk, yk)
				assertBytesEqual(t, g2, e2, w, h, "2D8clamped", w, h, org)
			}
		}
	}

}

func TestConvolve1D8ClampedEdgeGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1d8c1a))
	xWidths := []int{4, 8, 12, 15, 16, 24, 32}
	yWidths := []int{4, 8, 16, 24, 32}
	heights := []int{4, 8, 12, 15, 16, 24, 32}
	kernels := [][filterTaps]int16{
		subpelFilters8[3],
		subpelFilters8Smooth[6],
		subpelFilters8Sharp[9],
		bilinearFilters[7],
	}

	for _, w := range xWidths {
		for _, h := range heights {
			for _, k := range kernels {
				for _, edge := range []string{"left", "right"} {
					const refW = 96
					refH := h + 2*filterTaps
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := 1
					if edge == "right" {
						refX = refW - w - 3
					}
					refY := filterTaps
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveX8ClampedGoSIMD(got, ref, 0, 0, refX, refY, w, h, k)
					convolveX8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, k)
					assertBytesEqual(t, got, want, w, h, "X8horizontal-edge", w, h, edge)
				}
			}
		}
	}

	for _, w := range yWidths {
		for _, h := range heights {
			for _, k := range kernels {
				for _, edge := range []string{"top", "bottom"} {
					refW := w + 2*filterTaps
					const refH = 96
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := filterTaps
					refY := 1
					if edge == "bottom" {
						refY = refH - h - 3
					}
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveY8ClampedGoSIMD(got, ref, 0, 0, refX, refY, w, h, k)
					convolveY8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, k)
					assertBytesEqual(t, got, want, w, h, "Y8vertical-edge", w, h, edge)
				}
			}
		}
	}
}

func TestConvolve2D8ClampedHorizontalEdgeGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d8c1a))
	widths := []int{8, 12, 15, 16, 24, 32}
	heights := []int{4, 8, 16, 32}
	phasePairs := [][2][filterTaps]int16{
		{subpelFilters8[3], subpelFilters8[5]},
		{subpelFilters8Smooth[6], subpelFilters8Smooth[11]},
		{subpelFilters8Sharp[9], subpelFilters8Sharp[13]},
		{bilinearFilters[7], bilinearFilters[2]},
	}

	for _, w := range widths {
		for _, h := range heights {
			for _, kernels := range phasePairs {
				for _, edge := range []string{"left", "right"} {
					const refW = 96
					refH := h + 2*filterTaps
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := 1
					if edge == "right" {
						refX = refW - w - 3
					}
					refY := filterTaps
					got, _ := testPlane(w, h, 1, w)
					gotScratch, _ := testPlane(w, h, 1, w)
					gotSplit, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					var scratch ConvolveScratch
					var splitScratch ConvolveScratch
					convolve2D8ClampedGoSIMD(got, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					convolve2D8ClampedGoSIMDWithScratch(gotScratch, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratch)
					if !convolve2D8ClampedEdgeSplitGoSIMDWithScratch(gotSplit, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &splitScratch) {
						t.Fatalf("2D8horizontal-edge split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolve2D8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					assertBytesEqual(t, got, want, w, h, "2D8horizontal-edge", w, h, edge)
					assertBytesEqual(t, gotScratch, want, w, h, "2D8horizontal-edge-scratch", w, h, edge)
					assertBytesEqual(t, gotSplit, want, w, h, "2D8horizontal-edge-split", w, h, edge)
				}
			}
		}
	}

	const refW = 64
	const w = 16
	const h = 16
	ref, _ := testPlane(refW, h+2*filterTaps, 1, refW)
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	dst, _ := testPlane(w, h, 1, w)
	var scratch ConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8ClampedGoSIMDWithScratch(dst, ref, 0, 0, refW-w-3, filterTaps, w, h, subpelFilters8[3], subpelFilters8[5], &scratch)
	})
	if allocs != 0 {
		t.Fatalf("convolve2D8ClampedGoSIMDWithScratch horizontal edge allocated %v times, want 0", allocs)
	}
}

func assertBytesEqual(t *testing.T, got, want frame.Plane, w, h int, tag string, ctx ...any) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.Pix[y*got.Stride+x]
			e := want.Pix[y*want.Stride+x]
			if g != e {
				t.Fatalf("%s (%d,%d): GoSIMD=%d PureGo=%d ctx=%v", tag, x, y, g, e, ctx)
			}
		}
	}
}

// These differential tests assert the GoSIMD convolve kernels are bit-identical to
// the pure-Go references for every width/height and every subpel phase of each
// filter type. They call the GoSIMD wrappers directly rather than through the
// dispatch slots, so they validate the asm even on hosts whose CPUID does not
// advertise GoSIMD (e.g. amd64 under Rosetta 2, which translates GoSIMD anyway). On
// a true non-GoSIMD amd64 host these would fault; the harness that runs them is
// expected to be GoSIMD-capable (real CI) or GoSIMD-translating (Rosetta).

func TestConvolveX8GoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA1A2B3C4))
	const pad = filterTaps
	sizes := []int{8, 16, 24, 32, 48, 64}
	for _, tbl := range avx2FilterTables() {
		for ph := 0; ph < 16; ph++ {
			k := tbl[ph]
			for _, w := range sizes {
				for _, h := range []int{1, 4, 8, 17, 32} {
					side := w
					if h > side {
						side = h
					}
					ref := randPlane(rng, side+2*pad, 1)
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveX8GoSIMD(got, ref, 0, 0, pad, pad, w, h, k)
					convolveX8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
					diffPlanes8(t, got, want, w, h, "X", k, k)
				}
			}
		}
	}
}

func TestConvolveY8GoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xB1B2C3D4))
	const pad = filterTaps
	sizes := []int{8, 16, 24, 32, 48, 64}
	for _, tbl := range avx2FilterTables() {
		for ph := 0; ph < 16; ph++ {
			k := tbl[ph]
			for _, w := range sizes {
				for _, h := range []int{1, 4, 8, 17, 32} {
					side := w
					if h > side {
						side = h
					}
					ref := randPlane(rng, side+2*pad, 1)
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveY8GoSIMD(got, ref, 0, 0, pad, pad, w, h, k)
					convolveY8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
					diffPlanes8(t, got, want, w, h, "Y", k, k)
				}
			}
		}
	}
}

func TestConvolve1D8ClampedEdgeSplitGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1d8a11e))
	widths := []int{16, 24, 32}
	heights := []int{4, 8, 16, 32}
	kernels := [][filterTaps]int16{
		subpelFilters8Sharp[2],
		subpelFilters8Sharp[9],
	}
	for _, k := range kernels {
		if isFourTap(k) {
			t.Fatalf("test kernel unexpectedly uses GoSIMD four-tap fallback: %v", k)
		}
	}
	if !isFourTap(bilinearFilters[7]) {
		t.Fatalf("bilinear fallback guard no longer exercises a four-tap kernel: %v", bilinearFilters[7])
	}

	for _, w := range widths {
		for _, h := range heights {
			for _, k := range kernels {
				for _, edge := range []string{"left", "right"} {
					const refW = 96
					refH := h + 2*filterTaps
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := 1
					if edge == "right" {
						refX = refW - w - 3
					}
					refY := filterTaps
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					if !convolveX8HorizontalEdgeGoSIMD(got, ref, 0, 0, refX, refY, w, h, k) {
						t.Fatalf("X8horizontal-edge GoSIMD split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolveX8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, k)
					diffPlanes8(t, got, want, w, h, "Xclamped-edge", k, k)
				}
				for _, edge := range []string{"top", "bottom"} {
					const refH = 96
					refW := w + 2*filterTaps
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := filterTaps
					refY := 1
					if edge == "bottom" {
						refY = refH - h - 3
					}
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					if !convolveY8VerticalEdgeGoSIMD(got, ref, 0, 0, refX, refY, w, h, k) {
						t.Fatalf("Y8vertical-edge GoSIMD split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolveY8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, k)
					diffPlanes8(t, got, want, w, h, "Yclamped-edge", k, k)
				}
			}
		}
	}

	// The Go SIMD edge paths accept four-tap kernels (the GoSIMD asm they replace
	// rejected them). Whenever a path reports that it ran, it must match the
	// per-tap clamped reference.
	ref, _ := testPlane(64, 64, 1, 64)
	got, _ := testPlane(16, 16, 1, 16)
	want, _ := testPlane(16, 16, 1, 16)
	if convolveX8HorizontalEdgeGoSIMD(got, ref, 0, 0, 1, filterTaps, 16, 16, bilinearFilters[7]) {
		convolveX8ClampedPureGo(want, ref, 0, 0, 1, filterTaps, 16, 16, bilinearFilters[7])
		diffPlanes8(t, got, want, 16, 16, "X8horizontal-edge four-tap", bilinearFilters[7], bilinearFilters[7])
	}
	if convolveY8VerticalEdgeGoSIMD(got, ref, 0, 0, filterTaps, 1, 16, 16, bilinearFilters[7]) {
		convolveY8ClampedPureGo(want, ref, 0, 0, filterTaps, 1, 16, 16, bilinearFilters[7])
		diffPlanes8(t, got, want, 16, 16, "Y8vertical-edge four-tap", bilinearFilters[7], bilinearFilters[7])
	}
}

func TestConvolve2D8GoSIMDSweepMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d20feed))
	const pad = filterTaps
	sizes := []int{8, 16, 24, 32, 48, 64}
	tables := avx2FilterTables()

	// Full size sweep with one representative phase per filter type.
	for _, tbl := range tables {
		xk := tbl[3]
		yk := tbl[5]
		for _, w := range sizes {
			for _, h := range []int{4, 8, 16, 32} {
				side := w
				if h > side {
					side = h
				}
				ref := randPlane(rng, side+2*pad, 1)
				got, _ := testPlane(w, h, 1, w)
				want, _ := testPlane(w, h, 1, w)
				convolve2D8GoSIMD(got, ref, 0, 0, pad, pad, w, h, xk, yk)
				convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, xk, yk)
				diffPlanes8(t, got, want, w, h, "2D", xk, yk)
			}
		}
	}

	// All 16x16 phase combinations on fixed shapes.
	for _, tbl := range tables {
		for sx := 0; sx < 16; sx++ {
			for sy := 0; sy < 16; sy++ {
				ref := randPlane(rng, 32+2*pad, 1)
				got, _ := testPlane(16, 16, 1, 16)
				want, _ := testPlane(16, 16, 1, 16)
				convolve2D8GoSIMD(got, ref, 0, 0, pad, pad, 16, 16, tbl[sx], tbl[sy])
				convolve2D8PureGo(want, ref, 0, 0, pad, pad, 16, 16, tbl[sx], tbl[sy])
				diffPlanes8(t, got, want, 16, 16, "2Dphase", tbl[sx], tbl[sy])
			}
		}
	}
}

func TestConvolve2D8ClampedEdgeSplitGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d8a11e))
	widths := []int{16, 24, 32}
	heights := []int{4, 8, 16, 32}
	phasePairs := [][2][filterTaps]int16{
		{subpelFilters8[3], subpelFilters8[5]},
		{subpelFilters8Smooth[6], subpelFilters8Smooth[11]},
		{subpelFilters8Sharp[9], subpelFilters8Sharp[13]},
		{bilinearFilters[7], bilinearFilters[2]},
	}

	for _, w := range widths {
		for _, h := range heights {
			for _, kernels := range phasePairs {
				for _, edge := range []string{"left", "right"} {
					const refW = 96
					refH := h + 2*filterTaps
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := 1
					if edge == "right" {
						refX = refW - w - 3
					}
					refY := filterTaps
					got, _ := testPlane(w, h, 1, w)
					gotScratch, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					var scratch ConvolveScratch
					if !convolve2D8ClampedEdgeSplitGoSIMDWithScratch(got, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], nil) {
						t.Fatalf("2D8horizontal-edge GoSIMD split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					if !convolve2D8ClampedEdgeSplitGoSIMDWithScratch(gotScratch, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratch) {
						t.Fatalf("2D8horizontal-edge GoSIMD scratch split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolve2D8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					diffPlanes8(t, got, want, w, h, "2Dclamped-edge", kernels[0], kernels[1])
					diffPlanes8(t, gotScratch, want, w, h, "2Dclamped-edge-scratch", kernels[0], kernels[1])
				}
			}
		}
	}
}

func TestConvolveGoSIMDZeroAlloc(t *testing.T) {
	const pad = filterTaps
	rng := rand.New(rand.NewSource(1))
	ref := randPlane(rng, 32+2*pad, 1)
	dst, _ := testPlane(32, 32, 1, 32)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	if a := testing.AllocsPerRun(20, func() {
		convolve2D8GoSIMD(dst, ref, 0, 0, pad, pad, 32, 32, xk, yk)
	}); a != 0 {
		t.Fatalf("convolve2D8GoSIMD allocated %v times, want 0", a)
	}
	if a := testing.AllocsPerRun(20, func() {
		convolveX8GoSIMD(dst, ref, 0, 0, pad, pad, 32, 32, xk)
	}); a != 0 {
		t.Fatalf("convolveX8GoSIMD allocated %v times, want 0", a)
	}
}

func diffPlanes8(t *testing.T, got, want frame.Plane, w, h int, tag string, xk, yk [filterTaps]int16) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.Pix[y*got.Stride+x]
			e := want.Pix[y*want.Stride+x]
			if g != e {
				t.Fatalf("%s w=%d h=%d (%d,%d): GoSIMD=%d PureGo=%d xk=%v yk=%v", tag, w, h, x, y, g, e, xk, yk)
			}
		}
	}
}

func isFourTap(k [filterTaps]int16) bool {
	return k[0] == 0 && k[1] == 0 && k[6] == 0 && k[7] == 0
}

func TestConvolve8HalfExtremesAndExactWindows(t *testing.T) {
	for _, tbl := range avx2FilterTables() {
		for phase := 0; phase < 16; phase++ {
			k := tbl[phase]
			for _, shape := range [][2]int{{4, 4}, {8, 16}, {16, 8}, {32, 32}, {128, 128}} {
				w, h := shape[0], shape[1]
				for pattern := 0; pattern < 3; pattern++ {
					stride := w + 9 // odd stride, with seven horizontal tap samples
					ref := frame.Plane{Pix: make([]byte, (h+6)*stride+w+7), Stride: stride, Width: w + 7, Height: h + 7}
					for y := 0; y < ref.Height; y++ {
						for x := 0; x < ref.Width; x++ {
							if pattern == 1 || pattern == 2 && (x+y)&1 != 0 {
								ref.Pix[y*stride+x] = 255
							}
						}
					}
					mkDst := func() frame.Plane {
						return frame.Plane{Pix: make([]byte, (h-1)*(w+1)+w), Stride: w + 1, Width: w, Height: h}
					}
					var scratch ConvolveScratch
					for axis := 0; axis < 3; axis++ {
						got, want := mkDst(), mkDst()
						switch axis {
						case 0:
							convolveX8GoSIMD(got, ref, 0, 0, 3, 3, w, h, k)
							convolveX8PureGo(want, ref, 0, 0, 3, 3, w, h, k)
						case 1:
							convolveY8GoSIMD(got, ref, 0, 0, 3, 3, w, h, k)
							convolveY8PureGo(want, ref, 0, 0, 3, 3, w, h, k)
						default:
							convolve2D8GoSIMDWithScratch(got, ref, 0, 0, 3, 3, w, h, k, k, &scratch)
							convolve2D8PureGo(want, ref, 0, 0, 3, 3, w, h, k, k)
						}
						for y := 0; y < h; y++ {
							for x := 0; x < w; x++ {
								idx := y*(w+1) + x
								if got.Pix[idx] != want.Pix[idx] {
									t.Fatalf("axis=%d phase=%d shape=%dx%d pattern=%d pixel=(%d,%d): got=%d want=%d", axis, phase, w, h, pattern, x, y, got.Pix[idx], want.Pix[idx])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestConvolve8HalfRejectsUnsafeTaps(t *testing.T) {
	for _, k := range [][filterTaps]int16{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{128, 128, 128, 128, 128, 128, 128, 128},
		{-128, -128, -128, -128, -128, -128, -128, -128},
	} {
		if _, ok := u8HalfTaps(&k, 8194); ok {
			t.Fatalf("unsafe taps accepted: %v", k)
		}
	}
}

// These tests assert the width-4 GoSIMD convolve kernels (X, Y, 2D, plus their
// clamped wrappers) are bit-identical to the pure-Go reference. Width 4 is the
// most common inter shape: every 4:2:0 chroma block of an 8x8 luma block is 4x4
// and 4xN luma blocks are frequent, so a byte-exact 4-lane kernel is what closes
// the biggest inter-clip gap.

// w4FilterTables returns one 8-tap and one 4-tap (zero end taps) table so the
// sweep exercises both the full MAC and the 4-tap-kernel-contributes-nothing
// path through the same asm.
func w4FilterTables() (eight [][16][filterTaps]int16, four [][16][filterTaps]int16) {
	eight = [][16][filterTaps]int16{subpelFilters8, subpelFilters8Smooth, subpelFilters8Sharp}
	four = [][16][filterTaps]int16{subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	return eight, four
}

// TestConvolveW4GoSIMDMatchesPureGo sweeps the non-clamped width-4 X / Y / 2D GoSIMD
// kernels across heights 4/8/16, all 16 subpel phases, 8-tap and 4-tap kernels,
// over deterministic and random reference pixels.
func TestConvolveW4GoSIMDMatchesPureGo(t *testing.T) {
	const pad = filterTaps
	const w = 4
	heights := []int{4, 8, 16}
	eightTables, fourTables := w4FilterTables()
	rng := rand.New(rand.NewSource(0x04f4beef))

	makeRef := func(side int, randomize bool) frame.Plane {
		refSide := side + 2*pad
		r, _ := testPlane(refSide, refSide, 1, refSide)
		if randomize {
			for i := range r.Pix {
				r.Pix[i] = byte(rng.Intn(256))
			}
		} else {
			fillMotionTestPlane(r)
		}
		return r
	}

	checkX := func(t *testing.T, h int, k [filterTaps]int16, randomize bool) {
		ref := makeRef(maxInt2(w, h), randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolveX8GoSIMD(got, ref, 0, 0, pad, pad, w, h, k)
		convolveX8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
		assertClampedEqual(t, got, want, w, h)
	}
	checkY := func(t *testing.T, h int, k [filterTaps]int16, randomize bool) {
		ref := makeRef(maxInt2(w, h), randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolveY8GoSIMD(got, ref, 0, 0, pad, pad, w, h, k)
		convolveY8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
		assertClampedEqual(t, got, want, w, h)
	}
	check2D := func(t *testing.T, h int, xk, yk [filterTaps]int16, randomize bool) {
		ref := makeRef(maxInt2(w, h), randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolve2D8GoSIMD(got, ref, 0, 0, pad, pad, w, h, xk, yk)
		convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, xk, yk)
		assertClampedEqual(t, got, want, w, h)
	}

	for _, h := range heights {
		for _, randomize := range []bool{false, true} {
			// 8-tap: sweep all 16 phases for the first table; representative for rest.
			for ti, tbl := range eightTables {
				phases := []int{3, 5}
				if ti == 0 {
					phases = nil
					for p := 0; p < 16; p++ {
						phases = append(phases, p)
					}
				}
				for _, p := range phases {
					checkX(t, h, tbl[p], randomize)
					checkY(t, h, tbl[p], randomize)
				}
				for _, sx := range []int{1, 3, 7, 12} {
					for _, sy := range []int{2, 5, 9, 15} {
						check2D(t, h, tbl[sx], tbl[sy], randomize)
					}
				}
			}
			// 4-tap kernels (zero end taps) through the same asm.
			for _, tbl := range fourTables {
				for _, p := range []int{1, 3, 6, 11, 14} {
					checkX(t, h, tbl[p], randomize)
					checkY(t, h, tbl[p], randomize)
				}
				check2D(t, h, tbl[3], tbl[5], randomize)
				check2D(t, h, tbl[7], tbl[12], randomize)
			}
			// Mixed 4-tap X / 8-tap Y (and vice versa) for 2D.
			check2D(t, h, subpelFilters4[3], subpelFilters8[5], randomize)
			check2D(t, h, subpelFilters8[3], subpelFilters4[5], randomize)
		}
	}
}

// TestConvolveW4ClampedGoSIMDMatchesPureGo asserts the width-4 *ClampedGoSIMD
// wrappers match the clamped pure-Go reference for interior (resident halo) and
// genuine edge placements (where the tap window falls off the frame).
func TestConvolveW4ClampedGoSIMDMatchesPureGo(t *testing.T) {
	const planeW, planeH = 48, 48
	ref, _ := testPlane(planeW, planeH, 1, planeW)
	fillMotionTestPlane(ref)

	xk8 := subpelFilters8[3]
	yk8 := subpelFilters8[5]
	xk4 := subpelFilters4[3]
	yk4 := subpelFilters4[5]

	type wcase struct {
		name       string
		refX, refY int
		height     int
		four       bool
	}
	cases := []wcase{
		// Interior: full tap window resident, clamp is a no-op -> hits the GoSIMD
		// resident fast path.
		{"interior_4x4_8tap", 20, 20, 4, false},
		{"interior_4x8_8tap", 20, 16, 8, false},
		{"interior_4x16_8tap", 20, 12, 16, false},
		{"interior_4x4_4tap", 20, 20, 4, true},
		{"interior_4x16_4tap", 20, 12, 16, true},
		// Genuine edges -> off-frame taps -> pure-Go clamped fallback.
		{"corner_4x4_8tap", 0, 0, 4, false},
		{"corner_4x16_4tap", 0, 0, 16, true},
		{"left_4x8_8tap", 0, 20, 8, false},
		{"top_4x8_8tap", 20, 0, 8, false},
		{"br_4x4_8tap", planeW - 4, planeH - 4, 4, false},
		{"right_straddle_4x8_8tap", planeW - 2, 16, 8, false},
		{"bottom_straddle_4x8_8tap", 16, planeH - 2, 8, false},
	}

	pick := func(four bool, eight, fourK [filterTaps]int16) [filterTaps]int16 {
		if four {
			return fourK
		}
		return eight
	}

	for _, c := range cases {
		xk := pick(c.four, xk8, xk4)
		yk := pick(c.four, yk8, yk4)
		const w = 4
		t.Run("X/"+c.name, func(t *testing.T) {
			got, _ := testPlane(w, c.height, 1, w)
			want, _ := testPlane(w, c.height, 1, w)
			convolveX8ClampedGoSIMD(got, ref, 0, 0, c.refX, c.refY, w, c.height, xk)
			convolveX8ClampedPureGo(want, ref, 0, 0, c.refX, c.refY, w, c.height, xk)
			assertClampedEqual(t, got, want, w, c.height)
		})
		t.Run("Y/"+c.name, func(t *testing.T) {
			got, _ := testPlane(w, c.height, 1, w)
			want, _ := testPlane(w, c.height, 1, w)
			convolveY8ClampedGoSIMD(got, ref, 0, 0, c.refX, c.refY, w, c.height, yk)
			convolveY8ClampedPureGo(want, ref, 0, 0, c.refX, c.refY, w, c.height, yk)
			assertClampedEqual(t, got, want, w, c.height)
		})
		t.Run("2D/"+c.name, func(t *testing.T) {
			got, _ := testPlane(w, c.height, 1, w)
			want, _ := testPlane(w, c.height, 1, w)
			convolve2D8ClampedGoSIMD(got, ref, 0, 0, c.refX, c.refY, w, c.height, xk, yk)
			convolve2D8ClampedPureGo(want, ref, 0, 0, c.refX, c.refY, w, c.height, xk, yk)
			assertClampedEqual(t, got, want, w, c.height)
		})
	}
}

func maxInt2(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func testHBDZeroEndpointKernel(kernel [filterTaps]int16) bool {
	return kernel[0] == 0 && kernel[filterTaps-1] == 0
}

func TestHBDGoSIMDDispatchBound(t *testing.T) {
	if !hbdSIMDAvailable() {
		t.Skip("Go SIMD high-bit-depth kernels are unavailable on this CPU")
	}
	cases := []struct {
		name string
		got  any
		want any
	}{
		{"convolve-X", convolveXHighBDImpl, convolveXHighBDGoSIMD},
		{"convolve-Y", convolveYHighBDImpl, convolveYHighBDGoSIMD},
		{"convolve-2D", convolve2DHighBDImpl, convolve2DHighBDGoSIMD},
		{"convolve-X-clamped", convolveXHighBDClampedImpl, convolveXHighBDClampedGoSIMD},
		{"convolve-Y-clamped", convolveYHighBDClampedImpl, convolveYHighBDClampedGoSIMD},
		{"convolve-2D-clamped", convolve2DHighBDClampedImpl, convolve2DHighBDClampedGoSIMD},
		{"compound-X", predictInterCompoundRefHighBDToConvBufXResidentImpl, predictInterCompoundRefHighBDToConvBufXResidentGoSIMD},
		{"compound-Y", predictInterCompoundRefHighBDToConvBufYResidentImpl, predictInterCompoundRefHighBDToConvBufYResidentGoSIMD},
		{"compound-2D", predictInterCompoundRefHighBDToConvBuf2DResidentImpl, predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD},
		{"compound-2D-clamped", predictInterCompoundRefHighBDToConvBuf2DClampedImpl, predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD},
	}
	for _, tc := range cases {
		if reflect.ValueOf(tc.got).Pointer() != reflect.ValueOf(tc.want).Pointer() {
			t.Errorf("%s dispatch does not select the Go SIMD candidate", tc.name)
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapMatchesPureGo(t *testing.T) {
	tables := []struct {
		name string
		k    [16][filterTaps]int16
	}{
		{"regular8", subpelFilters8},
		{"smooth8", subpelFilters8Smooth},
		{"regular4", subpelFilters4},
		{"smooth4", subpelFilters4Smooth},
		{"bilinear", bilinearFilters},
		{"sharp", subpelFilters8Sharp},
	}
	sizes := []struct{ w, h int }{{8, 1}, {16, 7}, {32, 16}, {64, 9}}
	rng := rand.New(rand.NewSource(0x6bd6))

	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		for _, table := range tables {
			for phase := range 16 {
				xk := table.k[phase]
				yk := table.k[(phase+5)&15]
				xOK := testHBDZeroEndpointKernel(xk)
				yOK := testHBDZeroEndpointKernel(yk)
				if !xOK || !yOK {
					continue
				}
				for _, size := range sizes {
					side := size.w
					if size.h > side {
						side = size.h
					}
					ref := makeHighBDRef(side, filterTaps, max, true, rng)
					got2D, _ := testPlane(size.w, size.h, 2, (size.w+3)*2)
					want2D, _ := testPlane(size.w, size.h, 2, (size.w+3)*2)
					scratch := &ConvolveScratch{}
					for i := range scratch.imHBD {
						scratch.imHBD[i] = int32(0x5a5a0000 | i&0xffff)
					}
					convolve2DHighBDGoSIMDWithScratch(got2D, ref, bd, max, 0, 0, filterTaps, filterTaps, size.w, size.h, xk, yk, scratch)
					convolve2DHighBDPureGo(want2D, ref, bd, max, 0, 0, filterTaps, filterTaps, size.w, size.h, xk, yk)
					eqHighBDBlock(t, got2D, want2D, size.w, size.h, "GoSIMD-2D", bd, table.name, phase, size)
				}
			}
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapClampedMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bd7))
	const refW, refH = 37, 29
	const w, h = 16, 16
	positions := [][2]int{{0, 0}, {-2, -1}, {refW - w + 1, refH - h + 2}}
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		ref := makeGoSIMDHighBDPlane(refW, refH, max, rng, 3, 1)
		for _, pos := range positions {
			got2D, _ := testPlane(w, h, 2, (w+2)*2)
			want2D, _ := testPlane(w, h, 2, (w+2)*2)
			convolve2DHighBDClampedGoSIMD(got2D, ref, bd, max, 0, 0, pos[0], pos[1], w, h, xk, yk)
			convolve2DHighBDClampedPureGo(want2D, ref, bd, max, 0, 0, pos[0], pos[1], w, h, xk, yk)
			eqHighBDBlock(t, got2D, want2D, w, h, "GoSIMD-2D-clamped", bd, pos)
		}
	}
}

func TestConvolveHighBDGoSIMD2DZeroEndpointFourTapAndBilinearPhases(t *testing.T) {
	tables := []struct {
		name string
		k    [16][filterTaps]int16
	}{
		{"regular4", subpelFilters4},
		{"smooth4", subpelFilters4Smooth},
		{"bilinear", bilinearFilters},
	}
	const w, h = 32, 16
	rng := rand.New(rand.NewSource(0x6bde))

	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		ref := makeHighBDRef(64, filterTaps, max, true, rng)
		for _, table := range tables {
			for phase := range 16 {
				xk, yk := table.k[phase], table.k[(phase+7)&15]
				if !testHBDZeroEndpointKernel(xk) || !testHBDZeroEndpointKernel(yk) {
					t.Fatalf("%s phase pair %d/%d unexpectedly has a nonzero endpoint", table.name, phase, (phase+7)&15)
				}
				got, _ := testPlane(w, h, 2, (w+3)*2)
				want, _ := testPlane(w, h, 2, (w+3)*2)
				scratch := &ConvolveScratch{}
				convolve2DHighBDGoSIMDWithScratch(got, ref, bd, max, 0, 0, filterTaps, filterTaps, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(want, ref, bd, max, 0, 0, filterTaps, filterTaps, w, h, xk, yk)
				eqHighBDBlock(t, got, want, w, h, "GoSIMD-zero-endpoint", bd, table.name, phase)
			}
		}
	}

	// Mixed table pairs exercise zero-endpoint coefficients independently on
	// the horizontal and vertical pass, including the identity phase.
	for _, pair := range []struct {
		name string
		x, y [filterTaps]int16
	}{
		{"regular4-smooth4", subpelFilters4[1], subpelFilters4Smooth[14]},
		{"bilinear-regular4", bilinearFilters[1], subpelFilters4[15]},
		{"smooth4-bilinear", subpelFilters4Smooth[0], bilinearFilters[8]},
	} {
		max, _ := highBDMax(10)
		ref := makeHighBDRef(64, filterTaps, max, true, rng)
		got, _ := testPlane(w, h, 2, (w+1)*2+1)
		want, _ := testPlane(w, h, 2, (w+1)*2+1)
		convolve2DHighBDGoSIMD(got, ref, 10, max, 0, 0, filterTaps, filterTaps, w, h, pair.x, pair.y)
		convolve2DHighBDPureGo(want, ref, 10, max, 0, 0, filterTaps, filterTaps, w, h, pair.x, pair.y)
		eqHighBDBlock(t, got, want, w, h, "GoSIMD-zero-endpoint-mixed", pair.name)
	}
}

func TestConvolveHighBDGoSIMD2DSixTapExactWindowAndMaxScratch(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bdb))
	const bd = 12
	max, _ := highBDMax(bd)
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	for _, size := range []struct{ w, h int }{{16, 16}, {maxBlockSize, maxBlockSize}} {
		// The source has only the six-tap footprint: width+5 by height+5.
		// Odd byte stride and a one-byte base offset exercise unaligned vector
		// loads/stores while the scalar oracle gets a one-pixel zero-coefficient
		// border around the exact samples it consumes.
		exact := makeGoSIMDHighBDPlane(size.w+5, size.h+5, max, rng, 1, 1)
		padded := padGoSIMDHighBDPlane(exact)
		got2D := makeGoSIMDHighBDPlane(size.w, size.h, max, rng, 1, 1)
		want2D := makeGoSIMDHighBDPlane(size.w, size.h, max, rng, 1, 1)
		scratch := &ConvolveScratch{}
		convolve2DHighBDGoSIMDWithScratch(got2D, exact, bd, max, 0, 0, 2, 2, size.w, size.h, xk, yk, scratch)
		convolve2DHighBDPureGo(want2D, padded, bd, max, 0, 0, 3, 3, size.w, size.h, xk, yk)
		eqHighBDBlock(t, got2D, want2D, size.w, size.h, "GoSIMD-2D-exact-window", size)
	}
}

func TestCompoundHighBDGoSIMD2DSixTapMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bd8))
	sizes := []struct{ w, h int }{{16, 16}, {32, 16}}
	pairs := []struct {
		name string
		x, y [16][filterTaps]int16
	}{
		{"regular8-smooth8", subpelFilters8, subpelFilters8Smooth},
		{"regular4-smooth4", subpelFilters4, subpelFilters4Smooth},
		{"bilinear-regular4", bilinearFilters, subpelFilters4},
		{"smooth4-bilinear", subpelFilters4Smooth, bilinearFilters},
		{"sharp-identity-smooth", subpelFilters8Sharp, subpelFilters8Smooth},
	}
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		round0 := compoundRound0(bd)
		offsetBits := int(bd) + 2*filterBits - round0
		ref := makeHighBDRef(40, filterTaps, max, true, rng)
		for _, pair := range pairs {
			for phase := range 16 {
				xk := pair.x[phase]
				yk := pair.y[(phase+3)&15]
				if !testHBDZeroEndpointKernel(xk) || !testHBDZeroEndpointKernel(yk) {
					continue
				}
				for _, size := range sizes {
					got2D, want2D := make([]uint16, size.w*size.h), make([]uint16, size.w*size.h)
					var gotIM, wantIM compoundIM
					predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got2D, ref, filterTaps, filterTaps, size.w, size.h, xk, yk, round0, offsetBits, int(bd), &gotIM)
					predictInterCompoundRefHighBDToConvBuf2DResident(want2D, ref, filterTaps, filterTaps, size.w, size.h, xk, yk, round0, offsetBits, int(bd), &wantIM)
					assertCompoundHighBDBlockEqual(t, got2D, want2D, "GoSIMD-compound-2D", bd, struct {
						filter string
						phase  int
					}{pair.name, phase}, size)
				}
			}
		}
	}
}

func TestCompoundHighBDGoSIMD2DSixTapClampedMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bda))
	const refW, refH, w, h = 37, 29, 16, 16
	positions := [][2]int{{0, 0}, {-2, -1}, {refW - w + 1, refH - h + 2}}
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		round0 := compoundRound0(bd)
		offsetBits := int(bd) + 2*filterBits - round0
		ref := makeGoSIMDHighBDPlane(refW, refH, max, rng, 3, 1)
		for _, pos := range positions {
			got2D, want2D := make([]uint16, w*h), make([]uint16, w*h)
			var gotIM, wantIM compoundIM
			predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(got2D, ref, pos[0], pos[1], w, h, xk, yk, round0, offsetBits, int(bd), &gotIM, nil)
			predictInterCompoundRefHighBDToConvBuf2DClamped(want2D, ref, pos[0], pos[1], w, h, xk, yk, round0, offsetBits, int(bd), &wantIM)
			assertCompoundHighBDBlockEqual(t, got2D, want2D, "GoSIMD-compound-2D-clamped", bd, pos[0], struct{ w, h int }{w, h})
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapZeroAlloc(t *testing.T) {
	const pad = filterTaps
	max, _ := highBDMax(10)
	ref := makeHighBDRef(32, pad, max, true, rand.New(rand.NewSource(0x6bd9)))
	dst, _ := testPlane(32, 16, 2, 70)
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	scratch := &ConvolveScratch{}
	var im compoundIM
	out := make([]uint16, 32*16)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	cases := []struct {
		name string
		fn   func()
	}{
		{"2D-no-scratch", func() { convolve2DHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 16, xk, yk) }},
		{"2D-scratch", func() { convolve2DHighBDGoSIMDWithScratch(dst, ref, 10, max, 0, 0, pad, pad, 32, 16, xk, yk, scratch) }},
		{"compound-2D", func() {
			predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, pad, pad, 32, 16, xk, yk, round0, offsetBits, 10, &im)
		}},
	}
	for _, tc := range cases {
		if allocs := testing.AllocsPerRun(20, tc.fn); allocs != 0 {
			t.Errorf("%s allocated %v times, want zero", tc.name, allocs)
		}
	}
}

var hbd6TapBenchmarkSink uint16

type hbdSIMDBenchFilter struct {
	name string
	x, y [filterTaps]int16
}

func hbdSIMDBenchFilters() []hbdSIMDBenchFilter {
	return []hbdSIMDBenchFilter{
		{"regular8", subpelFilters8[6], subpelFilters8Smooth[9]},
		{"smooth8", subpelFilters8Smooth[6], subpelFilters8Smooth[9]},
		{"regular4", subpelFilters4[6], subpelFilters4[9]},
		{"smooth4", subpelFilters4Smooth[6], subpelFilters4Smooth[9]},
		{"bilinear", bilinearFilters[6], bilinearFilters[9]},
	}
}

// BenchmarkMotionHBD6TapConvolve compares the resident six-slot Go SIMD kernel
// and its production dispatch wrapper with GoSIMD across common filter families
// and block shapes. The direct variant isolates kernel cost; the dispatched
// variant includes shape and endpoint checks.
func BenchmarkMotionHBD6TapConvolve(b *testing.B) {
	const bd = 10
	max, _ := highBDMax(bd)
	const pad = filterTaps
	ref := makeHighBDRef(maxBlockSize, pad, max, true, rand.New(rand.NewSource(0x6bdc)))
	sizes := []struct{ w, h int }{
		{8, 1}, {8, 2}, {8, 4}, {8, 8}, {8, 16}, {8, 32},
		{16, 1}, {16, 4}, {16, 8}, {16, 16},
		{32, 4}, {32, 8}, {32, 16}, {32, 32},
		{64, 64}, {128, 128},
	}
	for _, filter := range hbdSIMDBenchFilters() {
		filter := filter
		b.Run(filter.name, func(b *testing.B) {
			for _, size := range sizes {
				size := size
				for _, variant := range []string{"GoSIMD"} {
					variant := variant
					name := fmt.Sprintf("2D/%s/W%dH%d", variant, size.w, size.h)
					b.Run(name, func(b *testing.B) {
						dst, _ := testPlane(size.w, size.h, 2, size.w*2)
						scratch := &ConvolveScratch{}
						b.ReportAllocs()
						b.SetBytes(int64(size.w * size.h * 2))
						for b.Loop() {
							convolve2DHighBDGoSIMDWithScratch(dst, ref, bd, max, 0, 0, pad, pad, size.w, size.h, filter.x, filter.y, scratch)
						}
						hbd6TapBenchmarkSink = getSample(dst, 2, size.w-1, size.h-1)
					})
				}
			}
		})
	}
}

func BenchmarkMotionHBD6TapCompound(b *testing.B) {
	const bd = 10
	max, _ := highBDMax(bd)
	const pad = filterTaps
	ref := makeHighBDRef(maxBlockSize, pad, max, true, rand.New(rand.NewSource(0x6bdd)))
	round0 := compoundRound0(bd)
	offsetBits := bd + 2*filterBits - round0
	sizes := []struct{ w, h int }{
		{8, 1}, {8, 2}, {8, 4}, {8, 8}, {8, 16}, {8, 32},
		{16, 1}, {16, 4}, {16, 8}, {16, 16},
		{32, 4}, {32, 8}, {32, 16}, {32, 32},
		{64, 64}, {128, 128},
	}
	for _, filter := range hbdSIMDBenchFilters() {
		filter := filter
		b.Run(filter.name, func(b *testing.B) {
			for _, size := range sizes {
				size := size
				for _, variant := range []string{"GoSIMD"} {
					variant := variant
					name := fmt.Sprintf("2D/%s/W%dH%d", variant, size.w, size.h)
					b.Run(name, func(b *testing.B) {
						out := make([]uint16, size.w*size.h)
						var im compoundIM
						b.ReportAllocs()
						b.SetBytes(int64(size.w * size.h * 2))
						for b.Loop() {
							predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, pad, pad, size.w, size.h, filter.x, filter.y, round0, offsetBits, bd, &im)
						}
						hbd6TapBenchmarkSink = out[len(out)-1]
					})
				}
			}
		})
	}
}

func assertCompoundHighBDBlockEqual(t *testing.T, got, want []uint16, tag string, bd uint8, phase any, size struct{ w, h int }) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: bd=%d phase=%v size=%dx%d sample=%d got=%d want=%d", tag, bd, phase, size.w, size.h, i, got[i], want[i])
		}
	}
}

func makeGoSIMDHighBDPlane(width, height int, max uint16, rng *rand.Rand, strideExtra, baseOffset int) frame.Plane {
	stride := width*2 + strideExtra
	pix := make([]byte, baseOffset+stride*height)
	plane := frame.Plane{Pix: pix[baseOffset:], Stride: stride, Width: width, Height: height}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			var value uint16
			switch (x + 3*y) % 3 {
			case 0:
				value = 0
			case 1:
				value = max
			default:
				value = uint16(rng.Intn(int(max) + 1))
			}
			storeHighBDSample(plane, x, y, value)
		}
	}
	return plane
}

func padGoSIMDHighBDPlane(src frame.Plane) frame.Plane {
	width, height := src.Width+2, src.Height+2
	stride := width * 2
	out := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			storeHighBDSample(out, x+1, y+1, loadHighBDSample(src, x, y))
		}
	}
	return out
}

func makeHighBDRef(side, pad int, max uint16, randomize bool, rng *rand.Rand) frame.Plane {
	refSide := side + 2*pad
	r, _ := testPlane(refSide, refSide, 2, refSide*2)
	if randomize {
		for y := 0; y < refSide; y++ {
			for x := 0; x < refSide; x++ {
				setSample(r, 2, x, y, uint16(rng.Intn(int(max)+1)))
			}
		}
	} else {
		fillHighBDMotionTestPlane(r, max)
	}
	return r
}

func eqHighBDBlock(t *testing.T, got, want frame.Plane, w, h int, tag string, ctx ...any) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := getSample(got, 2, x, y)
			e := getSample(want, 2, x, y)
			if g != e {
				t.Fatalf("%s (%d,%d): GoSIMD=%d PureGo=%d ctx=%v", tag, x, y, g, e, ctx)
			}
		}
	}
}

// TestConvolveHighBDGoSIMDMatchesPureGo asserts the high-bit-depth GoSIMD X/Y/2D
// convolves are bit-identical to their pure-Go references for every width/height
// in 4..64, every subpel phase of each filter type, at bit depths 10 and 12,
// over deterministic and random reference pixels.
func TestConvolveHighBDGoSIMDMatchesPureGo(t *testing.T) {
	tables := convolve2DKernelTables()
	rng := rand.New(rand.NewSource(0x4b1d))
	const pad = filterTaps
	sweep := []int{4, 8, 12, 16, 24, 32, 48, 64}
	bds := []uint8{10, 12}

	for _, bd := range bds {
		max, _ := highBDMax(bd)
		// Full size sweep with a representative phase per filter type.
		for _, tbl := range tables {
			xk := tbl[3]
			yk := tbl[5]
			for _, w := range sweep {
				for _, h := range sweep {
					for _, randomize := range []bool{false, true} {
						side := w
						if h > side {
							side = h
						}
						ref := makeHighBDRef(side, pad, max, randomize, rng)
						// X
						gx, _ := testPlane(w, h, 2, w*2)
						ex, _ := testPlane(w, h, 2, w*2)
						convolveXHighBDGoSIMD(gx, ref, bd, max, 0, 0, pad, pad, w, h, xk)
						convolveXHighBDPureGo(ex, ref, bd, max, 0, 0, pad, pad, w, h, xk)
						eqHighBDBlock(t, gx, ex, w, h, "X", bd, w, h)
						// Y
						gy, _ := testPlane(w, h, 2, w*2)
						ey, _ := testPlane(w, h, 2, w*2)
						convolveYHighBDGoSIMD(gy, ref, bd, max, 0, 0, pad, pad, w, h, yk)
						convolveYHighBDPureGo(ey, ref, bd, max, 0, 0, pad, pad, w, h, yk)
						eqHighBDBlock(t, gy, ey, w, h, "Y", bd, w, h)
						// 2D
						g2, _ := testPlane(w, h, 2, w*2)
						e2, _ := testPlane(w, h, 2, w*2)
						convolve2DHighBDGoSIMD(g2, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
						convolve2DHighBDPureGo(e2, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
						eqHighBDBlock(t, g2, e2, w, h, "2D", bd, w, h)
					}
				}
			}
		}

		// All subpel phases per filter type on a fixed 16x16 shape.
		ref := makeHighBDRef(16, pad, max, false, rng)
		for _, tbl := range tables {
			for sp := 0; sp < 16; sp++ {
				k := tbl[sp]
				gx, _ := testPlane(16, 16, 2, 32)
				ex, _ := testPlane(16, 16, 2, 32)
				convolveXHighBDGoSIMD(gx, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				convolveXHighBDPureGo(ex, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				eqHighBDBlock(t, gx, ex, 16, 16, "Xphase", bd, sp)

				gy, _ := testPlane(16, 16, 2, 32)
				ey, _ := testPlane(16, 16, 2, 32)
				convolveYHighBDGoSIMD(gy, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				convolveYHighBDPureGo(ey, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				eqHighBDBlock(t, gy, ey, 16, 16, "Yphase", bd, sp)

				g2, _ := testPlane(16, 16, 2, 32)
				e2, _ := testPlane(16, 16, 2, 32)
				convolve2DHighBDGoSIMD(g2, ref, bd, max, 0, 0, pad, pad, 16, 16, k, k)
				convolve2DHighBDPureGo(e2, ref, bd, max, 0, 0, pad, pad, 16, 16, k, k)
				eqHighBDBlock(t, g2, e2, 16, 16, "2Dphase", bd, sp)
			}
		}
	}
}

// TestConvolveHighBDGoSIMDZeroAlloc asserts the HBD GoSIMD wrappers allocate nothing
// on the fast (asm) path.
func TestConvolveHighBDGoSIMDZeroAlloc(t *testing.T) {
	const pad = filterTaps
	max, _ := highBDMax(10)
	rng := rand.New(rand.NewSource(7))
	ref := makeHighBDRef(32, pad, max, true, rng)
	dst, _ := testPlane(32, 32, 2, 64)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	cases := []struct {
		name string
		fn   func()
	}{
		{"X", func() { convolveXHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, xk) }},
		{"Y", func() { convolveYHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, yk) }},
		{"2D", func() { convolve2DHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, xk, yk) }},
	}
	for _, c := range cases {
		if allocs := testing.AllocsPerRun(20, c.fn); allocs != 0 {
			t.Errorf("%s HBD GoSIMD allocated %v times, want 0", c.name, allocs)
		}
	}
}

// TestConvolveClampedGoSIMDMatchesPureGo asserts the edge-clamped GoSIMD wrappers
// (8-bit and high-bit-depth) stay bit-identical to the pure-Go clamped
// references at genuine frame edges, where the tap window falls off the plane.
// It also covers the in-bounds case where the wrapper routes to the fast GoSIMD
// kernel.
func TestConvolveClampedHighBDGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc1a))
	sizes := []int{4, 8, 12, 16, 32}
	// High-bit-depth clamped at bd 10 and 12.
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		for _, w := range sizes {
			for _, h := range sizes {
				side := w
				if h > side {
					side = h
				}
				ref, _ := testPlane(side, side, 2, side*2)
				for y := 0; y < side; y++ {
					for x := 0; x < side; x++ {
						setSample(ref, 2, x, y, uint16(rng.Intn(int(max)+1)))
					}
				}
				xk := subpelFilters8[6]
				yk := subpelFilters8[9]
				for _, org := range []int{0, 1} {
					gx, _ := testPlane(w, h, 2, w*2)
					ex, _ := testPlane(w, h, 2, w*2)
					convolveXHighBDClampedGoSIMD(gx, ref, bd, max, 0, 0, org, org, w, h, xk)
					convolveXHighBDClampedPureGo(ex, ref, bd, max, 0, 0, org, org, w, h, xk)
					eqHighBDBlock(t, gx, ex, w, h, "XHBDclamped", bd, w, h, org)

					gy, _ := testPlane(w, h, 2, w*2)
					ey, _ := testPlane(w, h, 2, w*2)
					convolveYHighBDClampedGoSIMD(gy, ref, bd, max, 0, 0, org, org, w, h, yk)
					convolveYHighBDClampedPureGo(ey, ref, bd, max, 0, 0, org, org, w, h, yk)
					eqHighBDBlock(t, gy, ey, w, h, "YHBDclamped", bd, w, h, org)

					g2, _ := testPlane(w, h, 2, w*2)
					e2, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDClampedGoSIMD(g2, ref, bd, max, 0, 0, org, org, w, h, xk, yk)
					convolve2DHighBDClampedPureGo(e2, ref, bd, max, 0, 0, org, org, w, h, xk, yk)
					eqHighBDBlock(t, g2, e2, w, h, "2DHBDclamped", bd, w, h, org)
				}
			}
		}
	}
}

// TestConvolveHighBDClampedEmuEdgeMatchesPureGo drives the HBD edge-clamped
// GoSIMD wrappers over tap windows that overhang the reference plane on every
// side (and past both corners), forcing the emu_edge halo materialization
// (emuEdgeWindow16). Every shape must stay bit-identical to the pure-Go
// per-tap-clamping reference at bit depths 10 and 12, including the max/zero
// saturation edge values.
func TestConvolveHighBDClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16bced9e))
	sizes := []int{4, 8, 12, 16, 32, 64}
	const refW, refH = 24, 20
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		ref, _ := testPlane(refW, refH, 2, refW*2)
		for y := 0; y < refH; y++ {
			for x := 0; x < refW; x++ {
				v := uint16(rng.Intn(int(max) + 1))
				// Seed the corners with the saturation extremes so the
				// clamped-replication halo carries 0 and max samples.
				if (x == 0 || x == refW-1) && (y == 0 || y == refH-1) {
					if (x+y)&1 == 0 {
						v = 0
					} else {
						v = max
					}
				}
				setSample(ref, 2, x, y, v)
			}
		}
		for _, w := range sizes {
			for _, h := range sizes {
				// Offsets that push the tap window past each plane boundary and
				// diagonally past the corners, plus an interior case.
				offs := [][2]int{
					{-w, -h}, {-3, -3}, {2, -5}, {refW - 2, -1},
					{refW - 1, 5}, {refW, refH}, {5, refH - 2},
					{-4, refH - 3}, {-6, 4}, {4, 4},
				}
				xk := subpelFilters8[6]
				yk := subpelFilters8[9]
				for _, o := range offs {
					rx, ry := o[0], o[1]
					gx, _ := testPlane(w, h, 2, w*2)
					ex, _ := testPlane(w, h, 2, w*2)
					convolveXHighBDClampedGoSIMD(gx, ref, bd, max, 0, 0, rx, ry, w, h, xk)
					convolveXHighBDClampedPureGo(ex, ref, bd, max, 0, 0, rx, ry, w, h, xk)
					eqHighBDBlock(t, gx, ex, w, h, "XHBDemu", bd, w, h, o)

					gy, _ := testPlane(w, h, 2, w*2)
					ey, _ := testPlane(w, h, 2, w*2)
					convolveYHighBDClampedGoSIMD(gy, ref, bd, max, 0, 0, rx, ry, w, h, yk)
					convolveYHighBDClampedPureGo(ey, ref, bd, max, 0, 0, rx, ry, w, h, yk)
					eqHighBDBlock(t, gy, ey, w, h, "YHBDemu", bd, w, h, o)

					g2, _ := testPlane(w, h, 2, w*2)
					e2, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDClampedGoSIMD(g2, ref, bd, max, 0, 0, rx, ry, w, h, xk, yk)
					convolve2DHighBDClampedPureGo(e2, ref, bd, max, 0, 0, rx, ry, w, h, xk, yk)
					eqHighBDBlock(t, g2, e2, w, h, "2DHBDemu", bd, w, h, o)
				}
			}
		}
	}
}

// TestConvolveHighBDClampedEmuEdgeZeroAlloc proves the emu_edge halo scratch
// stays on the stack (the func-ptr dispatch slot must not force it to the heap).
func TestConvolveHighBDClampedEmuEdgeZeroAlloc(t *testing.T) {
	max, _ := highBDMax(10)
	ref, _ := testPlane(24, 20, 2, 24*2)
	for y := 0; y < 20; y++ {
		for x := 0; x < 24; x++ {
			setSample(ref, 2, x, y, uint16((x*7+y*3)&int(max)))
		}
	}
	dst, _ := testPlane(64, 64, 2, 64*2)
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	check := func(name string, fn func()) {
		if a := testing.AllocsPerRun(20, fn); a != 0 {
			t.Fatalf("%s allocated %v times, want 0", name, a)
		}
	}
	check("2DHBDemu", func() {
		convolve2DHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, xk, yk)
	})
	check("XHBDemu", func() {
		convolveXHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, xk)
	})
	check("YHBDemu", func() {
		convolveYHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, yk)
	})
}

// BenchmarkConvolveHighBDClampedEmuEdge measures the edge-overhanging HBD 2D
// clamped convolve: the GoSIMD emu_edge path against the pure-Go per-tap clamp.
func BenchmarkConvolveHighBDClampedEmuEdge(b *testing.B) {
	max, _ := highBDMax(10)
	ref, _ := testPlane(24, 20, 2, 24*2)
	for y := 0; y < 20; y++ {
		for x := 0; x < 24; x++ {
			setSample(ref, 2, x, y, uint16((x*7+y*3)&int(max)))
		}
	}
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	for _, w := range []int{8, 16, 32, 64} {
		dst, _ := testPlane(w, w, 2, w*2)
		b.Run("neon_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				convolve2DHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, w, w, xk, yk)
			}
		})
		b.Run("purego_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				convolve2DHighBDClampedPureGo(dst, ref, 10, max, 0, 0, -3, -3, w, w, xk, yk)
			}
		})
	}
}

func TestConvolveHighBDGoSIMDXYMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6BD0FACE))
	const pad = filterTaps
	sizes := []int{4, 8, 12, 16, 24, 32}
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		for _, tbl := range avx2FilterTables() {
			for ph := 0; ph < 16; ph++ {
				k := tbl[ph]
				for _, w := range sizes {
					for _, h := range []int{1, 4, 8, 16} {
						side := w
						if h > side {
							side = h
						}
						ref := randPlaneHBD(rng, side+2*pad, max)
						// X
						gx, _ := testPlane(w, h, 2, w*2)
						wx, _ := testPlane(w, h, 2, w*2)
						convolveXHighBDGoSIMD(gx, ref, bd, max, 0, 0, pad, pad, w, h, k)
						convolveXHighBDPureGo(wx, ref, bd, max, 0, 0, pad, pad, w, h, k)
						diffPlanesHBD(t, gx, wx, w, h, "Xhbd", bd)
						// Y
						gy, _ := testPlane(w, h, 2, w*2)
						wy, _ := testPlane(w, h, 2, w*2)
						convolveYHighBDGoSIMD(gy, ref, bd, max, 0, 0, pad, pad, w, h, k)
						convolveYHighBDPureGo(wy, ref, bd, max, 0, 0, pad, pad, w, h, k)
						diffPlanesHBD(t, gy, wy, w, h, "Yhbd", bd)
					}
				}
			}
		}
	}
}

func TestConvolveYHighBDGoSIMDSixTapSlidingExactWindow(t *testing.T) {
	// Phase three has six nonzero taps. The last row contains only the
	// requested samples, while both strides are odd byte counts.
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		for _, w := range []int{8, 16} {
			for _, h := range []int{1, 4, 16} {
				refW, refH := w+6, h+7
				refStr := refW*2 + 1
				ref := frame.Plane{Pix: make([]byte, (refH-1)*refStr+refW*2), Stride: refStr, Width: refW, Height: refH}
				for y := 0; y < refH; y++ {
					for x := 0; x < refW; x++ {
						value := uint16((x*173 + y*911) & int(max))
						if (x+y)%5 == 0 {
							value = max
						} else if (x+y)%7 == 0 {
							value = 0
						}
						storeHighBDSample(ref, x, y, value)
					}
				}
				dstStr := w*2 + 1
				got := frame.Plane{Pix: make([]byte, (h-1)*dstStr+w*2), Stride: dstStr, Width: w, Height: h}
				want := frame.Plane{Pix: make([]byte, len(got.Pix)), Stride: dstStr, Width: w, Height: h}
				convolveYHighBDGoSIMD(got, ref, bd, max, 0, 0, 3, 3, w, h, subpelFilters8[3])
				convolveYHighBDPureGo(want, ref, bd, max, 0, 0, 3, 3, w, h, subpelFilters8[3])
				diffPlanesHBD(t, got, want, w, h, "six-tap exact window", bd)
			}
		}
	}
}

func TestConvolve2DHighBDGoSIMDSweepMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x123abc))
	const pad = filterTaps
	sizes := []int{4, 8, 16, 32}
	tables := avx2FilterTables()
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		// size sweep
		for _, tbl := range tables {
			xk := tbl[3]
			yk := tbl[5]
			for _, w := range sizes {
				for _, h := range []int{4, 8, 16} {
					side := w
					if h > side {
						side = h
					}
					ref := randPlaneHBD(rng, side+2*pad, max)
					g, _ := testPlane(w, h, 2, w*2)
					wn, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDGoSIMD(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
					convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
					diffPlanesHBD(t, g, wn, w, h, "2Dhbd", bd)
				}
			}
		}
		// all phases on a fixed shape
		for _, tbl := range tables {
			for sx := 0; sx < 16; sx++ {
				for sy := 0; sy < 16; sy++ {
					ref := randPlaneHBD(rng, 16+2*pad, max)
					g, _ := testPlane(8, 8, 2, 16)
					wn, _ := testPlane(8, 8, 2, 16)
					convolve2DHighBDGoSIMD(g, ref, bd, max, 0, 0, pad, pad, 8, 8, tbl[sx], tbl[sy])
					convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, 8, 8, tbl[sx], tbl[sy])
					diffPlanesHBD(t, g, wn, 8, 8, "2Dhbdphase", bd)
				}
			}
		}
	}
}

// TestConvolveGoSIMDZeroAlloc asserts the GoSIMD fast paths allocate nothing.

// TestConvolve2DHighBDGoSIMDScratchSweepMatchesPureGo asserts the scratch-carrying
// HBD 2D GoSIMD convolve stays bit-identical to the pure-Go reference with a
// deliberately poisoned scratch, proving every intermediate sample read is
// written first. Kernels are called directly (never gated on cpu.Detected) so
// the sweep runs under Rosetta too.
func TestConvolve2DHighBDGoSIMDScratchSweepMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xa7c2d))
	const pad = filterTaps
	scratch := &ConvolveScratch{}
	poison := func() {
		for i := range scratch.imHBD {
			scratch.imHBD[i] = int32(0x7bad0000) | int32(i&0xffff)
		}
	}
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		for _, w := range []int{4, 8, 16, 32, 64, 128} {
			for _, h := range []int{4, 8, 16, 32} {
				side := w
				if h > side {
					side = h
				}
				ref := randPlaneHBD(rng, side+2*pad, max)
				g, _ := testPlane(w, h, 2, w*2)
				wn, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDGoSIMDWithScratch(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				diffPlanesHBD(t, g, wn, w, h, "2Dhbd-scratch", bd)

				gc, _ := testPlane(w, h, 2, w*2)
				wc, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedGoSIMDWithScratch(gc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedPureGo(wc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				diffPlanesHBD(t, gc, wc, w, h, "2Dhbd-scratch-clamped", bd)
			}
		}
	}
}

// TestConvolveScale2D8Step2MatchesScalar checks the 2:1 phase-split kernel
// against the scalar scaled convolve for every filter, block shape and
// sub-pel start phase.
func TestConvolveScale2D8Step2MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const refW, refH = 200, 180
	ref := frame.Plane{Width: refW, Height: refH, Stride: refW + 16, Pix: make([]byte, (refW+16)*refH+64)}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	sizes := []int{1, 2, 3, 4, 7, 8, 12, 15, 16, 24, 31, 32, 48, 63, 64, 96, 127, 128}
	var im, imRef scaledIM
	checked := 0
	for _, fx := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
		for _, fy := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
			for _, w := range sizes {
				for _, h := range sizes {
					xTable, err := SubpelKernelTableFor(fx, w)
					if err != nil {
						t.Fatal(err)
					}
					yTable, err := SubpelKernelTableFor(fy, h)
					if err != nil {
						t.Fatal(err)
					}
					for trial := 0; trial < 3; trial++ {
						startX := int64(3<<ScaleSubpelBits) + rng.Int63n(int64(refW-w/2-12)<<ScaleSubpelBits)
						startY := int64(3<<ScaleSubpelBits) + rng.Int63n(int64(refH-h/2-12)<<ScaleSubpelBits)
						imH, ok := scaledIMHeight(h, startY, scaledStep2Half)
						if !ok || !scaledRefRegionFits(ref, w, imH, startX, scaledStep2Half, startY) {
							continue
						}
						got := frame.Plane{Width: w, Height: h, Stride: w + 5, Pix: make([]byte, (w+5)*h)}
						want := frame.Plane{Width: w, Height: h, Stride: w + 5, Pix: make([]byte, (w+5)*h)}
						if !convolveScale2D8Step2GoSIMD(got, ref, 0, 0, w, h, startX, startY, xTable, yTable, imH, &im) {
							continue
						}
						convolveScale2D8(want, ref, 0, 0, w, h, startX, scaledStep2Half, startY, scaledStep2Half, xTable, yTable, imH, &imRef)
						if !bytes.Equal(got.Pix, want.Pix) {
							t.Fatalf("filters %d/%d %dx%d start %d,%d: mismatch", fx, fy, w, h, startX, startY)
						}
						checked++
					}
				}
			}
		}
	}
	if checked < 1000 {
		t.Fatalf("checked only %d cases", checked)
	}
	t.Logf("checked %d cases", checked)
}

// BenchmarkConvolveScale2D8Step2 measures the 2:1 phase-split scaled kernel.
func BenchmarkConvolveScale2D8Step2(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	const refW, refH = 200, 180
	ref := frame.Plane{Width: refW, Height: refH, Stride: refW + 16, Pix: make([]byte, (refW+16)*refH+64)}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	var im scaledIM
	for _, w := range []int{8, 16, 32} {
		xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, w)
		yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, w)
		startX := int64(20<<ScaleSubpelBits) + 37
		startY := int64(20<<ScaleSubpelBits) + 91
		imH, ok := scaledIMHeight(w, startY, scaledStep2Half)
		if !ok {
			b.Fatal("scaledIMHeight declined")
		}
		dst := frame.Plane{Width: w, Height: w, Stride: w, Pix: make([]byte, w*w)}
		if !convolveScale2D8Step2GoSIMD(dst, ref, 0, 0, w, w, startX, startY, xTable, yTable, imH, &im) {
			b.Fatal("convolveScale2D8Step2GoSIMD declined")
		}
		b.Run(fmt.Sprintf("%dx%d", w, w), func(b *testing.B) {
			for b.Loop() {
				convolveScale2D8Step2GoSIMD(dst, ref, 0, 0, w, w, startX, startY, xTable, yTable, imH, &im)
			}
		})
	}
}
