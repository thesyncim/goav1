//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"bytes"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"math/rand"
	"simd/archsimd"
	"slices"
	"testing"
)

// mc16TestRef builds a 16bpc reference plane (stride in samples) filled per
// pattern: 0 random, 1 all max, 2 all 0, 3 a 0/max checkerboard (the largest
// intermediate swings), 4 0/max column stripes aligned with the largest
// positive taps.
func mc16TestRef(rng *rand.Rand, w, h, stride, bitDepth, pattern int) frame.Plane {
	maxV := 1<<bitDepth - 1
	pix := make([]byte, 2*stride*h)
	for y := range h {
		for x := range stride {
			var v int
			switch pattern {
			case 0:
				v = rng.Intn(maxV + 1)
			case 1:
				v = maxV
			case 2:
				v = 0
			case 3:
				if (x+y)&1 == 0 {
					v = maxV
				}
			default:
				if x%3 != 1 {
					v = maxV
				}
			}
			o := 2 * (y*stride + x)
			pix[o], pix[o+1] = byte(v), byte(v>>8)
		}
	}
	return frame.Plane{Pix: pix, Stride: 2 * stride, Width: w, Height: h}
}

// mc16RefPrep is the per-tap-clamping scalar CONV_BUF reference.
func mc16RefPrep(out []uint16, ref frame.Plane, bitDepth, refX, refY, w, h, subX, subY int, filters InterpFilters) {
	xk, _ := interpKernel(filters.X, w, subX)
	yk, _ := interpKernel(filters.Y, h, subY)
	round0 := compoundRound0(uint8(bitDepth))
	offsetBits := bitDepth + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	switch {
	case subX != 0 && subY != 0:
		var im compoundIM
		predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, refX, refY, w, h, xk, yk, round0, offsetBits, bitDepth, &im)
	case subX != 0:
		predictInterCompoundRefHighBDToConvBufXClamped(out, ref, refX, refY, w, h, xk, round0, roundOffset)
	case subY != 0:
		predictInterCompoundRefHighBDToConvBufYClamped(out, ref, refX, refY, w, h, yk, round0, roundOffset)
	default:
		predictInterCompoundRefHighBDToConvBufCopyClamped(out, ref, refX, refY, w, h, round0, roundOffset)
	}
}

func TestMC16PrepFastMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	sizes := [][2]int{{4, 4}, {4, 8}, {4, 16}, {4, 2}, {4, 5}, {8, 5}, {16, 7}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 4}, {8, 16}, {16, 8}, {64, 64}, {128, 128}, {32, 8}, {16, 64}, {24, 6}, {8, 2}}
	const rw, rh = 200, 180
	scratch := new(CompoundConvolveScratch)
	for _, bd := range []int{10, 12} {
		for pattern := range 5 {
			ref := mc16TestRef(rng, rw, rh, rw+24, bd, pattern)
			for _, sz := range sizes {
				w, h := sz[0], sz[1]
				for iter := range 64 {
					filters := InterpFilters{X: mc8TestFilters[rng.Intn(len(mc8TestFilters))], Y: mc8TestFilters[rng.Intn(len(mc8TestFilters))]}
					subX, subY := rng.Intn(16), rng.Intn(16)
					switch iter % 8 {
					case 0:
						subX = 0
					case 1:
						subY = 0
					case 2:
						subX, subY = 0, 0
					}
					var refX, refY int
					switch iter % 4 {
					case 0:
						refX = 4 + rng.Intn(rw-w-12)
						refY = 4 + rng.Intn(rh-h-12)
					case 1:
						refX = -rng.Intn(w + 6)
						refY = -rng.Intn(h + 6)
					case 2:
						refX = rw - w + rng.Intn(w+6)
						refY = rh - h + rng.Intn(h+6)
					default:
						refX = rng.Intn(rw+2*w) - w - 3
						refY = rng.Intn(rh+2*h) - h - 3
					}
					want := make([]uint16, w*h+8)
					for i := range want {
						want[i] = uint16(i * 13)
					}
					got := slices.Clone(want)
					mc16RefPrep(want, ref, bd, refX, refY, w, h, subX, subY, filters)
					if !mc16PrepFast(got[:w*h], &ref, bd, refX, refY, w, h, w, h, subX, subY, filters, scratch) {
						if (w == 4 || w&7 == 0) && mc8AVX2 {
							t.Fatalf("mc16PrepFast declined bd=%d w=%d h=%d filters=%v sub=%d,%d", bd, w, h, filters, subX, subY)
						}
						continue
					}
					if !slices.Equal(got, want) {
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("bd %d pattern %d w=%d h=%d filters=%v sub=%d,%d ref=%d,%d: first mismatch at %d (x=%d y=%d) got %d want %d", bd, pattern, w, h, filters, subX, subY, refX, refY, i, i%w, i/w, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestMC16PrepFastEveryPhase sweeps every kind and phase pair (the six-tap
// kinds and the eight-tap sharp table, on either axis) over an in-plane
// random block at both depths.
func TestMC16PrepFastEveryPhase(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	scratch := new(CompoundConvolveScratch)
	kinds := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpBilinear, InterpMultiTapSharp}
	for _, bd := range []int{10, 12} {
		ref := mc16TestRef(rng, 96, 96, 104, bd, 0)
		for _, w := range []int{4, 8, 16} {
			for _, fx := range kinds {
				for _, fy := range kinds {
					filters := InterpFilters{X: fx, Y: fy}
					for subX := range 16 {
						for subY := range 16 {
							want := make([]uint16, w*w)
							got := make([]uint16, w*w)
							mc16RefPrep(want, ref, bd, 20, 30, w, w, subX, subY, filters)
							if !mc16PrepFast(got, &ref, bd, 20, 30, w, w, w, w, subX, subY, filters, scratch) {
								t.Fatalf("declined bd=%d w=%d filters=%v sub=%d,%d", bd, w, filters, subX, subY)
							}
							if !slices.Equal(got, want) {
								t.Fatalf("bd=%d w=%d filters=%v sub=%d,%d mismatch", bd, w, filters, subX, subY)
							}
						}
					}
				}
			}
		}
	}
}

// mc16RefPut is the per-tap-clamping scalar single-prediction reference.
func mc16RefPut(dst frame.Plane, ref frame.Plane, bitDepth, dstX, dstY, refX, refY, w, h, subX, subY int, filters InterpFilters) {
	xk, _ := interpKernel(filters.X, w, subX)
	yk, _ := interpKernel(filters.Y, h, subY)
	maxV, _ := highBDMax(uint8(bitDepth))
	switch {
	case subX != 0 && subY != 0:
		convolve2DHighBDClampedPureGo(dst, ref, uint8(bitDepth), maxV, dstX, dstY, refX, refY, w, h, xk, yk)
	case subX != 0:
		convolveXHighBDClampedPureGo(dst, ref, uint8(bitDepth), maxV, dstX, dstY, refX, refY, w, h, xk)
	default:
		convolveYHighBDClampedPureGo(dst, ref, uint8(bitDepth), maxV, dstX, dstY, refX, refY, w, h, yk)
	}
}

func TestMC16PutFastMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(18))
	sizes := [][2]int{{4, 4}, {4, 8}, {4, 16}, {4, 2}, {4, 5}, {8, 5}, {16, 7}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 4}, {8, 16}, {16, 8}, {64, 64}, {128, 128}, {32, 8}, {16, 64}, {24, 6}, {8, 2}}
	const rw, rh = 200, 180
	const dw, dh = 160, 150
	scratch := new(ConvolveScratch)
	for _, bd := range []int{10, 12} {
		for pattern := range 5 {
			ref := mc16TestRef(rng, rw, rh, rw+24, bd, pattern)
			for _, sz := range sizes {
				w, h := sz[0], sz[1]
				for iter := range 64 {
					filters := InterpFilters{X: mc8TestFilters[rng.Intn(len(mc8TestFilters))], Y: mc8TestFilters[rng.Intn(len(mc8TestFilters))]}
					subX, subY := 1+rng.Intn(15), 1+rng.Intn(15)
					switch iter % 4 {
					case 0:
						subX = 0
					case 1:
						subY = 0
					}
					var refX, refY int
					switch iter % 5 {
					case 0, 4:
						refX = 4 + rng.Intn(rw-w-12)
						refY = 4 + rng.Intn(rh-h-12)
					case 1:
						refX = -rng.Intn(w + 6)
						refY = -rng.Intn(h + 6)
					case 2:
						refX = rw - w + rng.Intn(w+6)
						refY = rh - h + rng.Intn(h+6)
					default:
						refX = rng.Intn(rw+2*w) - w - 3
						refY = rng.Intn(rh+2*h) - h - 3
					}
					dstX, dstY := rng.Intn(dw-w+1), rng.Intn(dh-h+1)
					want := frame.Plane{Pix: make([]byte, 2*(dw+8)*dh), Stride: 2 * (dw + 8), Width: dw, Height: dh}
					for i := range want.Pix {
						want.Pix[i] = byte(i * 7)
					}
					got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: want.Stride, Width: dw, Height: dh}
					mc16RefPut(want, ref, bd, dstX, dstY, refX, refY, w, h, subX, subY, filters)
					if !mc16PutFast(&got, &ref, bd, dstX, dstY, refX, refY, w, h, w, h, subX, subY, filters, scratch) {
						if (w == 4 || w&7 == 0) && subX|subY != 0 && mc8AVX2 {
							t.Fatalf("mc16PutFast declined bd=%d w=%d h=%d filters=%v sub=%d,%d", bd, w, h, filters, subX, subY)
						}
						continue
					}
					if !slices.Equal(got.Pix, want.Pix) {
						for i := range want.Pix {
							if got.Pix[i] != want.Pix[i] {
								y, x := i/want.Stride, (i%want.Stride)/2
								t.Fatalf("bd %d pattern %d w=%d h=%d filters=%v sub=%d,%d ref=%d,%d dst=%d,%d: first mismatch at x=%d y=%d", bd, pattern, w, h, filters, subX, subY, refX, refY, dstX, dstY, x, y)
							}
						}
					}
				}
			}
		}
	}
}

// TestMC16PutFastEveryPhase sweeps every kind and phase pair (including the
// eight-tap sharp table on either axis) through the put kernels: 2D at 10
// bits, the 1D passes at 10 and 12.
func TestMC16PutFastEveryPhase(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	scratch := new(ConvolveScratch)
	kinds := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpBilinear, InterpMultiTapSharp}
	const dw, dh = 40, 40
	for _, bd := range []int{10, 12} {
		for _, pattern := range []int{0, 3} {
			ref := mc16TestRef(rng, 96, 96, 104, bd, pattern)
			for _, sz := range [][2]int{{4, 8}, {4, 16}, {8, 8}, {16, 4}, {16, 16}, {8, 2}} {
				w, h := sz[0], sz[1]
				for _, fx := range kinds {
					for _, fy := range kinds {
						filters := InterpFilters{X: fx, Y: fy}
						for subX := range 16 {
							for subY := range 16 {
								if subX|subY == 0 || (bd != 10 && subX != 0 && subY != 0) {
									continue
								}
								want := frame.Plane{Pix: make([]byte, 2*dw*dh), Stride: 2 * dw, Width: dw, Height: dh}
								got := frame.Plane{Pix: make([]byte, 2*dw*dh), Stride: 2 * dw, Width: dw, Height: dh}
								mc16RefPut(want, ref, bd, 3, 5, 20, 30, w, h, subX, subY, filters)
								if !mc16PutFast(&got, &ref, bd, 3, 5, 20, 30, w, h, w, h, subX, subY, filters, scratch) {
									t.Fatalf("declined bd=%d %dx%d filters=%v sub=%d,%d", bd, w, h, filters, subX, subY)
								}
								if !slices.Equal(got.Pix, want.Pix) {
									t.Fatalf("bd=%d pattern=%d %dx%d filters=%v sub=%d,%d mismatch", bd, pattern, w, h, filters, subX, subY)
								}
							}
						}
					}
				}
			}
		}
	}
}

// mc16TestConvBufs fills two CONV_BUFs per mode: 0 the realistic CONV_BUF
// range, 1 equal extremes, 2 all zero, 3 the whole uint16 range.
func mc16TestConvBufs(rng *rand.Rand, n, mode int) ([]uint16, []uint16) {
	a := make([]uint16, n)
	b := make([]uint16, n)
	for i := range a {
		switch mode {
		case 0:
			a[i] = uint16(16000 + rng.Intn(36000))
			b[i] = uint16(16000 + rng.Intn(36000))
		case 1:
			a[i], b[i] = 65535, 65535
		case 2:
			a[i], b[i] = 0, 0
		default:
			a[i] = uint16(rng.Intn(65536))
			b[i] = uint16(rng.Intn(65536))
		}
	}
	return a, b
}

func mc16TestDst(w, h int) (want, got frame.Plane) {
	dw, dh := w+16, h+4
	want = frame.Plane{Pix: make([]byte, 2*dw*dh), Stride: 2 * dw, Width: dw, Height: dh}
	for i := range want.Pix {
		want.Pix[i] = byte(i * 5)
	}
	got = frame.Plane{Pix: slices.Clone(want.Pix), Stride: want.Stride, Width: dw, Height: dh}
	return want, got
}

func mc16BlendParams(bd int) (roundOffset, roundBits int, maxV uint16) {
	round0 := compoundRound0(uint8(bd))
	offsetBits := bd + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits = 2*filterBits - round0 - compoundRound1Bits
	maxV, _ = highBDMax(uint8(bd))
	return
}

func TestMC16BlendAvgFastMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(25))
	for _, bd := range []int{10, 12} {
		roundOffset, roundBits, maxV := mc16BlendParams(bd)
		for _, sz := range [][2]int{{4, 4}, {4, 2}, {4, 6}, {4, 8}, {4, 16}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {24, 8}, {32, 32}, {64, 64}, {128, 128}, {8, 2}, {8, 1}} {
			w, h := sz[0], sz[1]
			for iter := range 40 {
				a, b := mc16TestConvBufs(rng, w*h, iter%4)
				want, got := mc16TestDst(w, h)
				blendCompoundAvgHighBD(want, a, b, maxV, 5, 2, w, h, 8, 8, roundOffset, roundBits)
				if !mc16BlendAvgFast(&got, a, b, bd, 5, 2, w, h) {
					t.Fatalf("mc16BlendAvgFast declined %dx%d", w, h)
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("bd %d %dx%d iter %d: mismatch", bd, w, h, iter)
				}
			}
		}
	}
}

// mc16RefBlendMask is BlendCompoundMaskD16's high-bit-depth scalar loop.
func mc16RefBlendMask(dst frame.Plane, src0, src1 []uint16, bd, dstX, dstY, width, height int, mask []byte, maskStride int, subX, subY bool) {
	roundOffset, roundBits, maxV := mc16BlendParams(bd)
	for y := range height {
		for x := range width {
			m, _ := compoundMaskSample(mask, maskStride, y, x, subX, subY)
			i := y*width + x
			res := (m*int(src0[i]) + (compoundBlendA64MaxAlpha-m)*int(src1[i])) >> compoundBlendA64RoundBits
			res -= roundOffset
			res = roundPowerOfTwo(res, roundBits)
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(res, maxV))
		}
	}
}

// mc16RefDiffWtd is BuildDiffWtdMaskD16's scalar loop.
func mc16RefDiffWtd(mask []byte, maskStride int, src0, src1 []uint16, bd, width, height int, invert bool) {
	round := 2*filterBits - compoundRound0(uint8(bd)) - compoundRound1Bits + (bd - 8)
	for y := range height {
		for x := range width {
			i := y*width + x
			diff := int(src0[i]) - int(src1[i])
			if diff < 0 {
				diff = -diff
			}
			diff = roundPowerOfTwo(diff, round)
			m := min(max(38+diff/16, 0), compoundBlendA64MaxAlpha)
			if invert {
				m = compoundBlendA64MaxAlpha - m
			}
			mask[y*maskStride+x] = byte(m)
		}
	}
}

func TestMC16MaskBlendAndDiffWtdMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(26))
	for _, bd := range []int{10, 12} {
		for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 8}, {24, 8}, {40, 3}, {64, 64}, {128, 128}, {8, 32}, {8, 1}} {
			w, h := sz[0], sz[1]
			for iter := range 32 {
				a, b := mc16TestConvBufs(rng, w*h, iter%4)
				sub := iter&4 != 0
				mw, mh := w, h
				if sub {
					mw, mh = 2*w, 2*h
				}
				maskStride := mw + 8*(iter&1)
				mask := make([]byte, maskStride*mh+16)
				for i := range mask {
					mask[i] = byte(rng.Intn(65))
				}
				mask = mask[:maskStride*mh]
				want, got := mc16TestDst(w, h)
				mc16RefBlendMask(want, a, b, bd, 3, 1, w, h, mask, maskStride, sub, sub)
				if !mc16BlendMaskFast(got, a, b, bd, 3, 1, w, h, mask, maskStride, sub, sub) {
					t.Fatalf("mc16BlendMaskFast declined %dx%d sub=%v", w, h, sub)
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("bd %d blend %dx%d sub=%v iter %d: mismatch", bd, w, h, sub, iter)
				}
				for _, invert := range []bool{false, true} {
					wm := make([]byte, maskStride*h)
					gm := slices.Clone(wm)
					mc16RefDiffWtd(wm, maskStride, a, b, bd, w, h, invert)
					if !mc16DiffWtdMaskFast(gm, maskStride, a, b, bd, w, h, invert) {
						t.Fatalf("mc16DiffWtdMaskFast declined %dx%d", w, h)
					}
					if !bytes.Equal(gm, wm) {
						t.Fatalf("bd %d diffwtd %dx%d invert=%v iter %d: mismatch", bd, w, h, invert, iter)
					}
				}
			}
		}
	}
}

// The AVX2 differential tests mirror mc8_gosimd_arm64_test.go. They fail
// when the AVX2 kernels decline a covered shape.

func requireAVX2MC8(t testing.TB) {
	if !archsimd.X86.AVX2() {
		t.Skip("no AVX2")
	}
	if !mc8AVX2 {
		t.Fatal("AVX2 present but mc8AVX2 is off")
	}
}

// mc8TestRef builds a reference plane filled per pattern: 0 random, 1 all
// 255, 2 all 0, 3 a 0/255 checkerboard (stresses the int16 sums), 4 the 0/255
// column stripes aligned with the largest positive taps.
func mc8TestRef(rng *rand.Rand, w, h, stride, pattern int) frame.Plane {
	pix := make([]byte, stride*h)
	for y := range h {
		for x := range stride {
			var v byte
			switch pattern {
			case 0:
				v = byte(rng.Intn(256))
			case 1:
				v = 255
			case 2:
				v = 0
			case 3:
				if (x+y)&1 == 0 {
					v = 255
				}
			default:
				if x%3 != 1 {
					v = 255
				}
			}
			pix[y*stride+x] = v
		}
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

var mc8TestFilters = []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpBilinear, InterpMultiTapSharp}

func mc8RefPut(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters) {
	xk, _ := interpKernel(filters.X, filterW, subX)
	yk, _ := interpKernel(filters.Y, filterH, subY)
	switch {
	case subX != 0 && subY != 0:
		convolve2D8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, w, h, xk, yk)
	case subX != 0:
		convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, w, h, xk)
	default:
		convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, w, h, yk)
	}
}

func TestAMD64MC8PutFastMatchesPureGo(t *testing.T) {
	requireAVX2MC8(t)
	rng := rand.New(rand.NewSource(1))
	sizes := [][2]int{{4, 4}, {4, 8}, {4, 16}, {4, 2}, {4, 5}, {8, 5}, {16, 7}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 4}, {8, 2}, {8, 16}, {16, 8}, {24, 8}, {64, 64}, {128, 128}, {32, 8}, {64, 16}, {16, 64}, {8, 32}, {24, 9}, {40, 3}, {8, 1}, {16, 1}, {120, 5}, {56, 128}, {4, 1}, {4, 3}, {4, 6}, {4, 7}, {4, 32}, {4, 128}}
	const rw, rh = 200, 180
	for pattern := range 5 {
		ref := mc8TestRef(rng, rw, rh, rw+24, pattern)
		var scratch ConvolveScratch
		for _, sz := range sizes {
			w, h := sz[0], sz[1]
			for iter := range 60 {
				filters := InterpFilters{X: mc8TestFilters[rng.Intn(len(mc8TestFilters))], Y: mc8TestFilters[rng.Intn(len(mc8TestFilters))]}
				subX, subY := rng.Intn(16), rng.Intn(16)
				if subX == 0 && subY == 0 {
					subX = 1 + rng.Intn(15)
				}
				if iter%7 == 0 {
					subX = 0
					if subY == 0 {
						subY = 5
					}
				}
				if iter%7 == 1 {
					subY = 0
					if subX == 0 {
						subX = 9
					}
				}
				filterW, filterH := w, h
				if iter%5 == 0 {
					filterW, filterH = w*2, h*2
				}
				var refX, refY int
				switch iter % 4 {
				case 0: // interior
					refX = 4 + rng.Intn(rw-w-12)
					refY = 4 + rng.Intn(rh-h-12)
				case 1: // overhang left/top
					refX = -rng.Intn(w + 6)
					refY = -rng.Intn(h + 6)
				case 2: // overhang right/bottom
					refX = rw - w + rng.Intn(w+6)
					refY = rh - h + rng.Intn(h+6)
				default: // anywhere, including far outside
					refX = rng.Intn(rw+2*w) - w - 3
					refY = rng.Intn(rh+2*h) - h - 3
				}
				dw, dh := w+16, h+8
				want := frame.Plane{Pix: make([]byte, dw*dh), Stride: dw, Width: dw, Height: dh}
				for i := range want.Pix {
					want.Pix[i] = byte(i * 7)
				}
				got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: dw, Width: dw, Height: dh}
				mc8RefPut(want, ref, 3, 2, refX, refY, w, h, filterW, filterH, subX, subY, filters)
				if !Put8(&got, &ref, 3, 2, refX, refY, w, h, filterW, filterH, subX, subY, filters, &scratch) {
					kx, ky := mc8Kind(filters.X, filterW), mc8Kind(filters.Y, filterH)
					if _, cover := mc8Taps(w, kx, ky, subX, subY); cover && w%4 == 0 {
						t.Fatalf("Put8 declined w=%d h=%d filters=%v sub=%d,%d", w, h, filters, subX, subY)
					}
					continue
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("pattern %d w=%d h=%d filters=%v fsize=%dx%d sub=%d,%d ref=%d,%d: mismatch", pattern, w, h, filters, filterW, filterH, subX, subY, refX, refY)
				}
			}
		}
	}
}

func mc8RefPrep(out []uint16, ref frame.Plane, refX, refY, w, h, fw, fh, subX, subY int, filters InterpFilters) {
	xk, _ := interpKernel(filters.X, fw, subX)
	yk, _ := interpKernel(filters.Y, fh, subY)
	const offsetBits = 8 + 2*filterBits - compoundRound0Bits
	const roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	switch {
	case subX != 0 && subY != 0:
		predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, w, h, xk, yk, offsetBits, nil)
	case subX != 0:
		predictInterCompoundRef8ToConvBufXPureGo(out, ref, refX, refY, w, h, xk, roundOffset)
	case subY != 0:
		predictInterCompoundRef8ToConvBufYPureGo(out, ref, refX, refY, w, h, yk, compoundRound0Bits, roundOffset)
	default:
		predictInterCompoundRef8ToConvBufCopyPureGo(out, ref, refX, refY, w, h, compoundRound0Bits, roundOffset)
	}
}

func TestAMD64MC8PrepFastMatchesPureGo(t *testing.T) {
	requireAVX2MC8(t)
	rng := rand.New(rand.NewSource(2))
	sizes := [][2]int{{4, 4}, {4, 8}, {4, 16}, {4, 2}, {4, 5}, {8, 5}, {16, 7}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 4}, {8, 16}, {16, 8}, {64, 64}, {128, 128}, {32, 8}, {16, 64}, {24, 9}, {40, 3}, {8, 1}, {16, 1}, {120, 5}, {8, 128}, {4, 1}, {4, 3}, {4, 6}, {4, 7}, {4, 32}, {4, 128}}
	const rw, rh = 200, 180
	scratch := new(CompoundConvolveScratch)
	for pattern := range 5 {
		ref := mc8TestRef(rng, rw, rh, rw+24, pattern)
		for _, sz := range sizes {
			w, h := sz[0], sz[1]
			for iter := range 60 {
				filters := InterpFilters{X: mc8TestFilters[rng.Intn(len(mc8TestFilters))], Y: mc8TestFilters[rng.Intn(len(mc8TestFilters))]}
				subX, subY := rng.Intn(16), rng.Intn(16)
				switch iter % 8 {
				case 0:
					subX = 0
				case 1:
					subY = 0
				case 2:
					subX, subY = 0, 0
				}
				var refX, refY int
				switch iter % 4 {
				case 0:
					refX = 4 + rng.Intn(rw-w-12)
					refY = 4 + rng.Intn(rh-h-12)
				case 1:
					refX = -rng.Intn(w + 6)
					refY = -rng.Intn(h + 6)
				case 2:
					refX = rw - w + rng.Intn(w+6)
					refY = rh - h + rng.Intn(h+6)
				default:
					refX = rng.Intn(rw+2*w) - w - 3
					refY = rng.Intn(rh+2*h) - h - 3
				}
				want := make([]uint16, w*h+8)
				for i := range want {
					want[i] = uint16(i * 13)
				}
				got := slices.Clone(want)
				fw, fh := w, h
				if iter%5 == 3 {
					fw, fh = 2*w, 2*h
				}
				mc8RefPrep(want, ref, refX, refY, w, h, fw, fh, subX, subY, filters)
				if !mc8PrepFast(got[:w*h], &ref, refX, refY, w, h, fw, fh, subX, subY, filters, scratch) {
					kx, ky := mc8Kind(filters.X, fw), mc8Kind(filters.Y, fh)
					if _, cover := mc8Taps(w, kx, ky, subX, subY); cover && w%4 == 0 {
						t.Fatalf("mc8PrepFast declined w=%d h=%d filters=%v sub=%d,%d", w, h, filters, subX, subY)
					}
					continue
				}
				if !slices.Equal(got, want) {
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("pattern %d w=%d h=%d filters=%v sub=%d,%d ref=%d,%d: first mismatch at %d (x=%d y=%d) got %d want %d", pattern, w, h, filters, subX, subY, refX, refY, i, i%w, i/w, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestAMD64MC8BlendAvgFastMatchesPureGo(t *testing.T) {
	requireAVX2MC8(t)
	rng := rand.New(rand.NewSource(5))
	for _, sz := range [][2]int{{4, 4}, {4, 8}, {4, 16}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {24, 8}, {32, 32}, {64, 64}, {128, 128}, {8, 2}, {4, 3}, {4, 1}, {8, 7}, {8, 1}, {16, 3}, {40, 5}, {16, 1}} {
		w, h := sz[0], sz[1]
		for iter := range 40 {
			a := make([]uint16, w*h)
			b := make([]uint16, w*h)
			for i := range a {
				switch iter % 4 {
				case 0: // the full 8-bit CONV_BUF range (sharp 2D bounds)
					a[i] = uint16(2000 + rng.Intn(13400))
					b[i] = uint16(2000 + rng.Intn(13400))
				case 1:
					a[i], b[i] = 15356, 15356
				case 2:
					a[i], b[i] = 0, 0
				default:
					a[i] = uint16(6144 + 16*rng.Intn(256))
					b[i] = uint16(6144 + 16*rng.Intn(256))
				}
			}
			dw, dh := w+16, h+4
			want := frame.Plane{Pix: make([]byte, dw*dh), Stride: dw, Width: dw, Height: dh}
			for i := range want.Pix {
				want.Pix[i] = byte(i * 5)
			}
			got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: dw, Width: dw, Height: dh}
			blendCompoundAvg8PureGo(want, a, b, 5, 2, w, h, 8, 8, 6144, 4)
			if !mc8BlendAvgFast(&got, a, b, 5, 2, w, h) {
				t.Fatalf("mc8BlendAvgFast declined %dx%d", w, h)
			}
			if !bytes.Equal(got.Pix, want.Pix) {
				t.Fatalf("%dx%d iter %d: mismatch", w, h, iter)
			}
		}
	}
}

// mc8RefBlendMask is BlendCompoundMaskD16's 8-bit scalar loop.
func mc8RefBlendMask(dst frame.Plane, src0, src1 []uint16, dstX, dstY, width, height int, mask []byte, maskStride int, subX, subY bool) {
	for y := range height {
		for x := range width {
			m, _ := compoundMaskSample(mask, maskStride, y, x, subX, subY)
			i := y*width + x
			res := (m*int(src0[i]) + (compoundBlendA64MaxAlpha-m)*int(src1[i])) >> compoundBlendA64RoundBits
			res -= 6144
			res = roundPowerOfTwo(res, 4)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = clipPixel(res)
		}
	}
}

// mc8RefDiffWtd is BuildDiffWtdMaskD16's 8-bit scalar loop.
func mc8RefDiffWtd(mask []byte, maskStride int, src0, src1 []uint16, width, height int, invert bool) {
	for y := range height {
		for x := range width {
			i := y*width + x
			diff := int(src0[i]) - int(src1[i])
			if diff < 0 {
				diff = -diff
			}
			diff = roundPowerOfTwo(diff, 4)
			m := max(38+diff/16, 0)
			if m > compoundBlendA64MaxAlpha {
				m = compoundBlendA64MaxAlpha
			}
			if invert {
				m = compoundBlendA64MaxAlpha - m
			}
			mask[y*maskStride+x] = byte(m)
		}
	}
}

func TestAMD64MC8MaskBlendAndDiffWtdMatchScalar(t *testing.T) {
	requireAVX2MC8(t)
	rng := rand.New(rand.NewSource(8))
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 8}, {64, 64}, {128, 128}, {8, 32}, {4, 4}, {4, 8}, {4, 16}, {16, 3}, {16, 1}, {8, 2}, {32, 1}, {96, 4}} {
		w, h := sz[0], sz[1]
		for iter := range 30 {
			a := make([]uint16, w*h)
			b := make([]uint16, w*h)
			for i := range a {
				switch iter % 3 {
				case 0:
					a[i] = uint16(2000 + rng.Intn(13400))
					b[i] = uint16(2000 + rng.Intn(13400))
				case 1:
					a[i] = uint16(rng.Intn(65536))
					b[i] = uint16(rng.Intn(65536))
				default:
					a[i] = uint16(6144 + 16*rng.Intn(256))
					b[i] = uint16(6144 + 16*rng.Intn(256))
				}
			}
			invert := iter&1 != 0
			if iter%3 != 1 { // diff-wtd needs |a-b| + 8 to fit (8-bit CONV_BUF range)
				ms := w + 8
				want := make([]byte, ms*h+16)
				got := slices.Clone(want)
				mc8RefDiffWtd(want, ms, a, b, w, h, invert)
				if !mc8DiffWtdMaskFast(got, ms, a, b, w, h, invert) {
					t.Fatalf("diffwtd %dx%d declined", w, h)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("diffwtd %dx%d iter %d mismatch", w, h, iter)
				}
			}
			for _, sub := range []bool{false, true} {
				mw, mh := w, h
				if sub {
					mw, mh = 2*w, 2*h
				}
				ms := mw + 4
				mask := make([]byte, ms*mh+16)
				for i := range mask {
					mask[i] = byte(rng.Intn(65))
				}
				dw, dh := w+16, h+4
				want := frame.Plane{Pix: make([]byte, dw*dh), Stride: dw, Width: dw, Height: dh}
				for i := range want.Pix {
					want.Pix[i] = byte(i * 11)
				}
				got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: dw, Width: dw, Height: dh}
				mc8RefBlendMask(want, a, b, 3, 1, w, h, mask[:ms*mh], ms, sub, sub)
				if !mc8BlendMaskFast(got, a, b, 3, 1, w, h, mask[:ms*mh], ms, sub, sub) {
					t.Fatalf("mask blend %dx%d sub=%v declined", w, h, sub)
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("mask blend %dx%d sub=%v iter %d mismatch", w, h, sub, iter)
				}
			}
		}
	}
}

func TestAMD64MC8WAvgMatchesPureGo(t *testing.T) {
	requireAVX2MC8(t)
	rng := rand.New(rand.NewSource(10))
	weights := [][2]int{{8, 8}, {9, 7}, {7, 9}, {11, 5}, {5, 11}, {12, 4}, {4, 12}, {13, 3}, {3, 13}, {16, 0}, {0, 16}}
	for _, sz := range [][2]int{{4, 4}, {4, 8}, {4, 16}, {8, 2}, {8, 8}, {8, 32}, {16, 1}, {16, 3}, {16, 16}, {32, 8}, {64, 64}, {128, 128}, {96, 2}, {24, 4}, {8, 3}, {4, 6}} {
		w, h := sz[0], sz[1]
		for iter := range 24 {
			a := make([]uint16, w*h)
			b := make([]uint16, w*h)
			for i := range a {
				switch iter % 3 {
				case 0:
					a[i] = uint16(2000 + rng.Intn(13400))
					b[i] = uint16(2000 + rng.Intn(13400))
				case 1:
					a[i] = uint16(rng.Intn(65536))
					b[i] = uint16(rng.Intn(65536))
				default:
					a[i] = uint16(6144 + 16*rng.Intn(256))
					b[i] = uint16(6144 + 16*rng.Intn(256))
				}
			}
			wt := weights[iter%len(weights)]
			dw, dh := w+16, h+4
			want := frame.Plane{Pix: make([]byte, dw*dh), Stride: dw, Width: dw, Height: dh}
			for i := range want.Pix {
				want.Pix[i] = byte(i * 3)
			}
			got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: dw, Width: dw, Height: dh}
			blendCompoundAvg8PureGo(want, a, b, 5, 2, w, h, wt[0], wt[1], 6144, 4)
			blendCompoundAvg8AVX2(got, a, b, 5, 2, w, h, wt[0], wt[1], 6144, 4)
			if !bytes.Equal(got.Pix, want.Pix) {
				t.Fatalf("%dx%d weights %v iter %d: mismatch", w, h, wt, iter)
			}
		}
	}
	if !mc8BlendShapeOK(16, 16) || mc8BlendShapeOK(24, 4) || mc8BlendShapeOK(8, 3) {
		t.Fatal("mc8BlendShapeOK coverage changed")
	}
}

// TestAMD64MC8Dispatch checks that the public 8-bit predictors take the AVX2
// kernels and that the six-tap kinds match the arm64 set.
func TestAMD64MC8Dispatch(t *testing.T) {
	requireAVX2MC8(t)
	if want := [mc8KindCount]bool{true, true, false, true, true, true}; mc8SixTap != want {
		t.Fatalf("six-tap kinds = %v, want %v", mc8SixTap, want)
	}
	if want := [mc8KindCount]bool{true, true, true, true, true, true}; mc8EightTap != want {
		t.Fatalf("eight-tap kinds = %v, want %v", mc8EightTap, want)
	}
	rng := rand.New(rand.NewSource(9))
	ref := mc8TestRef(rng, 96, 96, 112, 0)
	dst := frame.Plane{Pix: make([]byte, 64*64), Stride: 64, Width: 64, Height: 64}
	var scratch ConvolveScratch
	sharp := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapRegular}
	for _, sub := range [][2]int{{5, 0}, {0, 7}, {3, 9}} {
		if !Put8(&dst, &ref, 0, 0, 20, 20, 16, 16, 16, 16, sub[0], sub[1], sharp, &scratch) {
			t.Fatalf("Put8 declined sharp 16x16 sub=%v", sub)
		}
		if !Put8(&dst, &ref, 0, 0, 20, 20, 16, 16, 16, 16, sub[0], sub[1], RegularFilters, &scratch) {
			t.Fatalf("Put8 declined 16x16 sub=%v", sub)
		}
		var cs CompoundConvolveScratch
		out := make([]uint16, 16*16)
		if !mc8PrepFast(out, &ref, 20, 20, 16, 16, 16, 16, sub[0], sub[1], RegularFilters, &cs) {
			t.Fatalf("mc8PrepFast declined 16x16 sub=%v", sub)
		}
	}
	a := make([]uint16, 64)
	if !mc8BlendAvgFast(&dst, a, a, 0, 0, 8, 8) {
		t.Fatal("mc8BlendAvgFast declined 8x8")
	}
	// The public predictor with the kernels on matches the general path.
	got := frame.Plane{Pix: make([]byte, 64*64), Stride: 64, Width: 64, Height: 64}
	want := frame.Plane{Pix: make([]byte, 64*64), Stride: 64, Width: 64, Height: 64}
	mv := Vector{Row: 37, Col: -21}
	if err := PredictTranslation8(&got, &ref, 8, 8, 32, 16, 32, 16, mv, false, false, RegularFilters, &scratch); err != nil {
		t.Fatal(err)
	}
	mc8AVX2 = false
	err := PredictTranslation8(&want, &ref, 8, 8, 32, 16, 32, 16, mv, false, false, RegularFilters, &scratch)
	mc8AVX2 = true
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Pix, want.Pix) {
		t.Fatal("PredictTranslation8 AVX2 and general paths differ")
	}
}

// TestAMD64MC8ZeroAlloc pins that the AVX2 predictors and blends do not
// allocate, including on emulated-edge blocks.
func TestAMD64MC8ZeroAlloc(t *testing.T) {
	requireAVX2MC8(t)
	rng := rand.New(rand.NewSource(12))
	ref := mc8TestRef(rng, 96, 96, 112, 0)
	dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	var scratch ConvolveScratch
	cs := new(CompoundConvolveScratch)
	b0, b1 := new(CompoundConvBuf), new(CompoundConvBuf)
	mask := make([]byte, 64*64)
	sharp := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}
	allocs := testing.AllocsPerRun(20, func() {
		for _, mv := range []Vector{{Row: 27, Col: 21}, {Row: -300, Col: 5}, {Row: 3, Col: 0}, {Row: 0, Col: 0}} {
			for _, sz := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}} {
				_ = PredictTranslation8(&dst, &ref, 8, 8, sz[0], sz[1], sz[0], sz[1], mv, false, false, RegularFilters, &scratch)
				_ = PredictTranslation8(&dst, &ref, 8, 8, sz[0], sz[1], sz[0], sz[1], mv, false, false, sharp, &scratch)
				_ = PrepTranslation8(b0, &ref, 8, 8, sz[0], sz[1], sz[0], sz[1], mv, false, false, RegularFilters, cs)
				_ = PrepTranslation8(b1, &ref, 8, 8, sz[0], sz[1], sz[0], sz[1], mv, false, false, sharp, cs)
				_ = BlendCompoundMaskD16(dst, b0, b1, 1, 8, 0, 0, sz[0], sz[1], mask, 64, false, false)
				_ = BuildDiffWtdMaskD16(mask, 64, b0, b1, 8, sz[0], sz[1], true)
				_ = BlendCompoundAvg(dst, b0, b1, 1, 8, 0, 0, sz[0], sz[1], 9, 7)
				_ = BlendCompoundAvg(dst, b0, b1, 1, 8, 0, 0, sz[0], sz[1], 8, 8)
			}
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v, want 0", allocs)
	}
}

// TestAMD64CopyFastMatchesClampedCopy checks the 8- and 16-bit put copies
// against the clamped reference copy over every AV1 block width (and a few
// widths that are not powers of two), in and out of the plane: they must
// decline exactly the blocks whose source leaves the plane and write exactly
// the block otherwise.
func TestAMD64CopyFastMatchesClampedCopy(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	const rw, rh = 200, 180
	for _, bps := range []int{1, 2} {
		rs := bps * (rw + 24)
		ref := frame.Plane{Pix: make([]byte, rs*rh), Stride: rs, Width: rw, Height: rh}
		for i := range ref.Pix {
			ref.Pix[i] = byte(rng.Intn(256))
		}
		for _, sz := range [][2]int{{4, 4}, {4, 1}, {4, 16}, {8, 8}, {12, 4}, {16, 16}, {24, 8}, {32, 32}, {48, 4}, {64, 64}, {96, 2}, {128, 128}} {
			w, h := sz[0], sz[1]
			for iter := range 40 {
				refX := rng.Intn(rw+2*w) - w
				refY := rng.Intn(rh+2*h) - h
				dw, dh := w+16, h+4
				want := frame.Plane{Pix: make([]byte, bps*(dw+3)*dh), Stride: bps * (dw + 3), Width: dw, Height: dh}
				for i := range want.Pix {
					want.Pix[i] = byte(i * 3)
				}
				got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: want.Stride, Width: dw, Height: dh}
				if err := copyPlaneBlockClamped(want, ref, bps, 7, 3, refX, refY, w, h); err != nil {
					t.Fatal(err)
				}
				inside := refX >= 0 && refY >= 0 && refX+w <= rw && refY+h <= rh
				var ok bool
				if bps == 1 {
					ok = mc8CopyFast(&got, &ref, 7, 3, refX, refY, w, h)
				} else {
					ok = mc16CopyFast(&got, &ref, 7, 3, refX, refY, w, h)
				}
				if ok != inside {
					t.Fatalf("bps %d %dx%d iter %d ref=%d,%d: handled=%v want %v", bps, w, h, iter, refX, refY, ok, inside)
				}
				if inside && !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("bps %d %dx%d iter %d ref=%d,%d: mismatch", bps, w, h, iter, refX, refY)
				}
			}
		}
	}
}

// TestWarpedFilterI8Exact guards the two properties the GoSIMD warp math depends
// on: the int8 narrow of warpedFilter is lossless, and every row still sums to
// 128 (the identity behind the horizontal SMULL bias fold).
func TestWarpedFilterI8Exact(t *testing.T) {
	for i := range warpedFilter {
		sum := 0
		minDot, maxDot := 0, 0
		for j := range warpedFilter[i] {
			if int16(warpedFilterI8[i][j]) != warpedFilter[i][j] {
				t.Fatalf("warpedFilterI8[%d][%d]=%d lossy vs %d", i, j, warpedFilterI8[i][j], warpedFilter[i][j])
			}
			sum += int(warpedFilter[i][j])
			v := int(warpedFilter[i][j])
			if v >= 0 {
				minDot -= 128 * v
				maxDot += 127 * v
			} else {
				minDot += 127 * v
				maxDot -= 128 * v
			}
		}
		if sum != 128 {
			t.Fatalf("warpedFilter[%d] sums to %d, want 128", i, sum)
		}
		if minDot < -32768 || maxDot+4 > 32767 {
			t.Fatalf("warpedFilter[%d] centred dot exceeds int16: [%d,%d]", i, minDot, maxDot+4)
		}
	}
}

func TestWarpHorizontal8AllPhasesAndMinimumWindow(t *testing.T) {
	ref := frame.Plane{Pix: make([]byte, 16*15), Stride: 16, Width: 16, Height: 15}
	for phase := range warpedFilter {
		for pattern := 0; pattern < 3; pattern++ {
			for y := 0; y < ref.Height; y++ {
				for x := 0; x < ref.Width; x++ {
					switch pattern {
					case 0:
						ref.Pix[y*16+x] = 0
					case 1:
						ref.Pix[y*16+x] = 255
					default:
						ref.Pix[y*16+x] = byte(((x + y) & 1) * 255)
					}
				}
			}
			sx := (phase - warpedPixelPrecShifts) << warpedDiffPrecBits
			var want, got warpTmp
			warpHorizontal8Resident(&want, ref, 8, sx, 7, 0, 0, 0, round0Bits, 8+filterBits-1)
			warpHorizontal8ResidentDispatch(&got, ref, 8, sx, 7, 0, 0, 0, round0Bits, 8+filterBits-1)
			if got != want {
				t.Fatalf("phase=%d pattern=%d mismatch", phase, pattern)
			}
		}
	}
}

func TestWarpVertical8AllPhasesAndExtremes(t *testing.T) {
	for phase := range warpedFilter {
		baseSY := (phase - warpedPixelPrecShifts) << warpedDiffPrecBits
		for pattern := 0; pattern < 3; pattern++ {
			var tmp warpTmp
			for k := range tmp {
				switch pattern {
				case 0:
					tmp[k] = 0
				case 1:
					tmp[k] = (1 << 14) - 1
				default:
					tmp[k] = int32((k & 1) * ((1 << 14) - 1))
				}
			}
			for _, gamma := range [...]int{0, 64} {
				if !warpVertFullOffsInRange(baseSY, gamma, 0) {
					continue
				}
				want := frame.Plane{Pix: make([]byte, 64), Stride: 8, Width: 8, Height: 8}
				got := frame.Plane{Pix: make([]byte, 64), Stride: 8, Width: 8, Height: 8}
				if gamma == 0 {
					warpVertical8FullGamma0(want, &tmp, 0, 0, 0, 0, baseSY, 0, round1Bits, 8+2*filterBits-round0Bits)
					warpVertical8FullGamma0Dispatch(got, &tmp, 0, 0, 0, 0, baseSY, 0, round1Bits, 8+2*filterBits-round0Bits)
				} else {
					warpVertical8Full(want, &tmp, 0, 0, 0, 0, baseSY, gamma, 0, round1Bits, 8+2*filterBits-round0Bits)
					warpVertical8FullDispatch(got, &tmp, 0, 0, 0, 0, baseSY, gamma, 0, round1Bits, 8+2*filterBits-round0Bits)
				}
				for k := range want.Pix {
					if got.Pix[k] != want.Pix[k] {
						t.Fatalf("phase=%d gamma=%d pattern=%d pixel=%d got=%d want=%d", phase, gamma, pattern, k, got.Pix[k], want.Pix[k])
					}
				}
			}
		}
	}
}

// residentRefPlane builds an 8-bit reference plane large enough that the
// resident horizontal window (and its one-byte right overshoot) is always in
// bounds for interior positions.
func residentRefPlane(rng *rand.Rand) frame.Plane {
	const w, h, stride = 80, 80, 96
	pix := make([]byte, stride*(h+2)) // +2 rows of border for the 16-byte load overshoot
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

func TestWarpHorizontal8ResidentGoSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5A1D0FF))
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1

	tested := 0
	for draw := 0; draw < 20000 && tested < 1500; draw++ {
		ref := residentRefPlane(rng)
		ix4 := 7 + rng.Intn(ref.Width-15) // keeps ix4>=7 and ix4+8<=Width
		iy4 := 7 + rng.Intn(ref.Height-15)
		sx4 := rng.Intn(1 << 16)
		sy4 := rng.Intn(1 << 16)
		alpha := rng.Intn(513) - 256
		beta := rng.Intn(513) - 256

		if !warpHorizResidentOffsInRange(sx4, alpha, beta) {
			continue
		}
		tested++

		var want, got warpTmp
		wantSY := warpHorizontal8Resident(&want, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
		gotSY := warpHorizontal8ResidentDispatch(&got, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)

		if gotSY != wantSY {
			t.Fatalf("draw %d: sy4 got %d want %d", draw, gotSY, wantSY)
		}
		if got != want {
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("draw %d ix4=%d iy4=%d sx4=%d alpha=%d beta=%d: tmp[%d]=%d want %d",
						draw, ix4, iy4, sx4, alpha, beta, i, got[i], want[i])
				}
			}
		}
	}
	if tested == 0 {
		t.Fatal("no in-range horizontal cases exercised the GoSIMD path")
	}
	t.Logf("compared %d in-range resident horizontal blocks", tested)
}

func TestWarpVertical8FullGoSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xBEEF77))
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits

	testedFull, testedGamma0 := 0, 0
	for draw := 0; draw < 40000 && (testedFull < 1200 || testedGamma0 < 400); draw++ {
		// Produce a realistic int32 tmp from a resident horizontal pass.
		ref := residentRefPlane(rng)
		ix4 := 7 + rng.Intn(ref.Width-15)
		iy4 := 7 + rng.Intn(ref.Height-15)
		hsx4 := rng.Intn(1 << 16)
		halpha := rng.Intn(257) - 128
		hbeta := rng.Intn(257) - 128
		if !warpHorizResidentOffsInRange(hsx4, halpha, hbeta) {
			continue
		}
		var tmp warpTmp
		warpHorizontal8Resident(&tmp, ref, ix4, hsx4, iy4, 0, halpha, hbeta, reduceBitsHoriz, offsetBitsHoriz)

		baseSY := rng.Intn(1 << 16)
		gamma := rng.Intn(257) - 128
		delta := rng.Intn(513) - 256

		gamma0 := draw%4 == 0
		if gamma0 {
			gamma = 0
		}
		if !warpVertFullOffsInRange(baseSY, gamma, delta) {
			continue
		}

		want, _ := testPlane(32, 32, 1, 32)
		got, _ := testPlane(32, 32, 1, 32)
		for i := range want.Pix {
			want.Pix[i] = 0xAA
			got.Pix[i] = 0xAA
		}

		if gamma0 {
			warpVertical8FullGamma0(want, &tmp, 8, 8, 0, 0, baseSY, delta, reduceBitsVert, offsetBitsVert)
			warpVertical8FullGamma0Dispatch(got, &tmp, 8, 8, 0, 0, baseSY, delta, reduceBitsVert, offsetBitsVert)
			testedGamma0++
		} else {
			warpVertical8Full(want, &tmp, 8, 8, 0, 0, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
			warpVertical8FullDispatch(got, &tmp, 8, 8, 0, 0, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
			testedFull++
		}

		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("draw %d gamma0=%v baseSY=%d gamma=%d delta=%d pix[%d]=%d want %d",
					draw, gamma0, baseSY, gamma, delta, i, got.Pix[i], want.Pix[i])
			}
		}
	}
	if testedFull == 0 || testedGamma0 == 0 {
		t.Fatalf("insufficient vertical coverage: full=%d gamma0=%d", testedFull, testedGamma0)
	}
	t.Logf("compared %d full + %d gamma0 vertical blocks", testedFull, testedGamma0)
}

func benchWarpHorizInputs() (frame.Plane, int, int, int, int, int, int) {
	rng := rand.New(rand.NewSource(1))
	ref := residentRefPlane(rng)
	return ref, 40, 40, 32768, 0, 96, -64
}

func benchWarpVertTmp() warpTmp {
	rng := rand.New(rand.NewSource(2))
	ref := residentRefPlane(rng)
	var tmp warpTmp
	warpHorizontal8Resident(&tmp, ref, 40, 32768, 40, 0, 96, -64, round0Bits, 8+filterBits-1)
	return tmp
}

func benchWarpVerticalInputs() (warpTmp, frame.Plane) {
	tmp := benchWarpVertTmp()
	dst, _ := testPlane(32, 32, 1, 32)
	return tmp, dst
}

func BenchmarkWarpVertical8FullGamma0Scalar(b *testing.B) {
	tmp, dst := benchWarpVerticalInputs()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpVertical8FullGamma0(dst, &tmp, 8, 8, 0, 0, 32768, -64, round1Bits, 8+2*filterBits-round0Bits)
	}
}

func BenchmarkWarpVertical8FullGamma0GoSIMD(b *testing.B) {
	tmp, dst := benchWarpVerticalInputs()
	if !warpVertFullOffsInRange(32768, 0, -64) {
		b.Skip("bench inputs out of range")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpVertical8FullGamma0Dispatch(dst, &tmp, 8, 8, 0, 0, 32768, -64, round1Bits, 8+2*filterBits-round0Bits)
	}
}

// BenchmarkAMD64Warp8 compares the AVX2 warp passes with the scalar ones on
// one resident 8x8 block.
func BenchmarkAMD64Warp8(b *testing.B) {
	ref, ix4, iy4, sx4, sy4, alpha, beta := benchWarpHorizInputs()
	vtmp, dst := benchWarpVerticalInputs()
	var tmp warpTmp
	for _, impl := range []string{"generic", "avx2"} {
		on := impl == "avx2"
		b.Run("pass=horizontal/impl="+impl, func(b *testing.B) {
			warpAVX2 = on
			defer func() { warpAVX2 = true }()
			for b.Loop() {
				warpHorizontal8ResidentDispatch(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, round0Bits, 8+filterBits-1)
			}
		})
		b.Run("pass=vertical/impl="+impl, func(b *testing.B) {
			warpAVX2 = on
			defer func() { warpAVX2 = true }()
			for b.Loop() {
				warpVertical8FullDispatch(dst, &vtmp, 8, 8, 0, 0, 32768, 96, -64, round1Bits, 8+2*filterBits-round0Bits)
			}
		})
	}
}

type warpHBDResidentKernel func(tmp *warpTmp, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int

var warpHBDResidentSIMDKernels = []struct {
	name string
	run  warpHBDResidentKernel
}{
	{name: "go-simd", run: warpHorizontalHighBDResidentDispatch},
}

func residentRefPlaneHighBD(rng *rand.Rand, bitDepth uint8, oddBase bool) frame.Plane {
	const width, height, stride = 96, 96, 208
	pix := make([]byte, stride*height+1)
	if oddBase {
		pix = pix[1:]
	}
	ref := frame.Plane{Pix: pix, Stride: stride, Width: width, Height: height}
	mask := uint16((1 << bitDepth) - 1)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			storeHighBDSample(ref, x, y, uint16(rng.Intn(int(mask)+1)))
		}
	}
	return ref
}

func TestWarpHorizontalHighBDGoSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5A1D0FF))
	phases := []int{-1 << 18, -257, -1, 0, 511, 32768, 65535, 1 << 18}
	steps := []int{-256, -96, -1, 0, 1, 96, 256}
	tested := 0
	for _, bitDepth := range []uint8{10, 12} {
		reduceBitsHoriz, _ := highBDRoundBits(bitDepth)
		offsetBitsHoriz := int(bitDepth) + filterBits - 1
		for _, oddBase := range []bool{false, true} {
			ref := residentRefPlaneHighBD(rng, bitDepth, oddBase)
			for pi, sx4 := range phases {
				for si, alpha := range steps {
					beta := steps[(pi+si)%len(steps)]
					ix4, iy4 := 24+(pi%7), 24+(si%7)
					var want warpTmp
					wantSY := warpHorizontalHighBDResident(&want, ref, ix4, sx4, iy4, 12345, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
					for _, kernel := range warpHBDResidentSIMDKernels {
						var got warpTmp
						gotSY := kernel.run(&got, ref, ix4, sx4, iy4, 12345, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
						if gotSY != wantSY || got != want {
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("kernel=%s bd=%d oddBase=%v ix4=%d iy4=%d sx4=%d alpha=%d beta=%d tmp[%d]=%d want %d",
										kernel.name, bitDepth, oddBase, ix4, iy4, sx4, alpha, beta, i, got[i], want[i])
								}
							}
							t.Fatalf("kernel=%s bd=%d oddBase=%v got sy %d want %d", kernel.name, bitDepth, oddBase, gotSY, wantSY)
						}
					}
					tested++
				}
			}
		}
	}
	if tested != 224 {
		t.Fatalf("tested %d cases, want 224", tested)
	}
}

func TestWarpHorizontalHighBDGoSIMDExtremesAndResidentBoundaries(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		maxSample := uint16((1 << bitDepth) - 1)
		reduceBitsHoriz, _ := highBDRoundBits(bitDepth)
		offsetBitsHoriz := int(bitDepth) + filterBits - 1
		for _, stride := range []int{48, 49} {
			for pattern := 0; pattern < 3; pattern++ {
				ref, _ := testPlane(24, 24, 2, stride)
				for y := 0; y < ref.Height; y++ {
					for x := 0; x < ref.Width; x++ {
						var sample uint16
						switch pattern {
						case 0:
							sample = 0
						case 1:
							sample = maxSample
						case 2:
							if (x+y)&1 != 0 {
								sample = maxSample
							}
						}
						storeHighBDSample(ref, x, y, sample)
					}
				}
				for _, phase := range []int{-1 << 18, -1, 0, 32768, 65535, 1 << 18} {
					for _, alpha := range []int{-256, -1, 0, 1, 256} {
						for _, beta := range []int{-256, 0, 256} {
							var want warpTmp
							wantSY := warpHorizontalHighBDResident(&want, ref, 16, phase, 16, 321, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
							for _, kernel := range warpHBDResidentSIMDKernels {
								var got warpTmp
								gotSY := kernel.run(&got, ref, 16, phase, 16, 321, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
								if gotSY != wantSY || got != want {
									t.Fatalf("kernel=%s bd=%d stride=%d pattern=%d phase=%d alpha=%d beta=%d got=%v want=%v", kernel.name, bitDepth, stride, pattern, phase, alpha, beta, got, want)
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestWarpHorizontalHighBDGoSIMDZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA110C))
	ref := residentRefPlaneHighBD(rng, 10, false)
	var tmp warpTmp
	const ix4, sx4, iy4, sy4, alpha, beta = 40, 32768, 40, 0, 96, -64
	for _, kernel := range warpHBDResidentSIMDKernels {
		allocs := testing.AllocsPerRun(1000, func() {
			kernel.run(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, 3, 10+filterBits-1)
		})
		if allocs != 0 {
			t.Errorf("resident Go SIMD horizontal %s allocated: %f", kernel.name, allocs)
		}
	}
}

func TestWarpHorizontalHighBDGoSIMDFullUint16AndExactFootprint(t *testing.T) {
	for filterIndex, filter := range warpedFilter {
		var sum, absSum int
		for _, coeff := range filter {
			sum += int(coeff)
			if coeff < 0 {
				absSum -= int(coeff)
			} else {
				absSum += int(coeff)
			}
		}
		if sum != 128 {
			t.Fatalf("warpedFilter[%d] sums to %d, want 128 for centered uint16 SIMD arithmetic", filterIndex, sum)
		}
		if absSum > 222 {
			t.Fatalf("warpedFilter[%d] L1=%d exceeds centered SIMD accumulator proof", filterIndex, absSum)
		}
	}

	const width, height, stride = 15, 15, 30
	ref := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	values := [...]uint16{0, 1, 32767, 32768, 65535}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			storeHighBDSample(ref, x, y, values[(3*x+2*y)%len(values)])
		}
	}
	for _, bitDepth := range []uint8{10, 12} {
		reduceBitsHoriz, _ := highBDRoundBits(bitDepth)
		offsetBitsHoriz := int(bitDepth) + filterBits - 1
		for _, sx4 := range []int{-1 << 18, -1, 0, 32768, 65535, 1 << 18} {
			var want warpTmp
			wantSY := warpHorizontalHighBDResident(&want, ref, 7, sx4, 7, 1234, 97, -31, reduceBitsHoriz, offsetBitsHoriz)
			for _, kernel := range warpHBDResidentSIMDKernels {
				var got warpTmp
				gotSY := kernel.run(&got, ref, 7, sx4, 7, 1234, 97, -31, reduceBitsHoriz, offsetBitsHoriz)
				if gotSY != wantSY || got != want {
					t.Fatalf("kernel=%s bd=%d sx4=%d exact 15x15 footprint got=%v want=%v", kernel.name, bitDepth, sx4, got, want)
				}
			}
		}
	}
}

func benchWarpHorizontalHighBDInputs() (frame.Plane, int, int, int, int, int, int) {
	rng := rand.New(rand.NewSource(1))
	ref := residentRefPlaneHighBD(rng, 10, false)
	const i, j = 16, 16
	const alpha, beta, gamma, delta = -320, -64, 0, -64
	ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, [6]int32{42805, -7571, 65230, -57, 0, 65509}, alpha, beta, gamma, delta, 0, 0)
	return ref, ix4, sx4, iy4, sy4, alpha, beta
}

// TestAMD64WarpVerticalHighBDMatchesScalar compares the AVX2 16bpc vertical
// warp pass (full and gamma-zero) with the scalar one over random
// intermediates and filter phases at 10 and 12 bits.
func TestAMD64WarpVerticalHighBDMatchesScalar(t *testing.T) {
	if !warpAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(91))
	tested := 0
	for draw := 0; draw < 4000; draw++ {
		bd := uint8(10)
		if draw&1 != 0 {
			bd = 12
		}
		round0, round1 := highBDRoundBits(bd)
		offsetV := int(bd) + 2*filterBits - round0
		max := uint16(1)<<bd - 1
		var tmp warpTmp
		lim := 1 << (int(bd) + 7 - round0 + 1)
		for i := range tmp {
			tmp[i] = int32(rng.Intn(2*lim) - lim/2)
		}
		baseSY := rng.Intn(1<<17) - 1<<16
		gamma, delta := rng.Intn(1<<13)-1<<12, rng.Intn(1<<13)-1<<12
		gamma0 := draw%4 >= 2
		if gamma0 {
			gamma = 0
		}
		if !warpVertFullOffsInRange(baseSY, gamma, delta) {
			continue
		}
		mk := func() frame.Plane {
			p := frame.Plane{Pix: make([]byte, 2*24*24), Stride: 2 * 24, Width: 24, Height: 24}
			for i := range p.Pix {
				p.Pix[i] = 0xa5
			}
			return p
		}
		want, got := mk(), mk()
		if gamma0 {
			warpVerticalHighBDFullGamma0(want, bd, max, &tmp, 8, 8, 1, 2, baseSY, delta, round1, offsetV)
			warpVerticalHighBDFullGamma0Dispatch(got, bd, max, &tmp, 8, 8, 1, 2, baseSY, delta, round1, offsetV)
		} else {
			warpVerticalHighBDFull(want, bd, max, &tmp, 8, 8, 1, 2, baseSY, gamma, delta, round1, offsetV)
			warpVerticalHighBDFullDispatch(got, bd, max, &tmp, 8, 8, 1, 2, baseSY, gamma, delta, round1, offsetV)
		}
		tested++
		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("draw %d bd %d gamma0=%v: byte %d got %d want %d", draw, bd, gamma0, i, got.Pix[i], want.Pix[i])
			}
		}
	}
	if tested < 1000 {
		t.Fatalf("only %d blocks tested", tested)
	}
}

// BenchmarkAMD64WarpHBD compares the AVX2 10-bit warp passes with the
// scalar ones on one resident 8x8 block.
func BenchmarkAMD64WarpHBD(b *testing.B) {
	ref, ix4, sx4, iy4, sy4, alpha, beta := benchWarpHorizontalHighBDInputs()
	var tmp warpTmp
	warpHorizontalHighBDResident(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, 3, 10+filterBits-1)
	dst := frame.Plane{Pix: make([]byte, 2*24*24), Stride: 2 * 24, Width: 24, Height: 24}
	var htmp warpTmp
	for _, impl := range []string{"generic", "avx2"} {
		on := impl == "avx2"
		b.Run("pass=horizontal/impl="+impl, func(b *testing.B) {
			warpAVX2 = on
			defer func() { warpAVX2 = true }()
			for b.Loop() {
				warpHorizontalHighBDResidentDispatch(&htmp, ref, ix4, sx4, iy4, sy4, alpha, beta, 3, 10+filterBits-1)
			}
		})
		b.Run("pass=vertical/impl="+impl, func(b *testing.B) {
			warpAVX2 = on
			defer func() { warpAVX2 = true }()
			for b.Loop() {
				warpVerticalHighBDFullDispatch(dst, 10, 1023, &tmp, 8, 8, 0, 0, 32768, 96, -64, 11, 10+2*filterBits-3)
			}
		})
	}
}
