// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
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
