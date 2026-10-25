// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"bytes"
	"math/rand"
	"simd/archsimd"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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
				if !mc8PutFast(&got, &ref, 3, 2, refX, refY, w, h, filterW, filterH, subX, subY, filters, &scratch) {
					kx, ky := mc8Kind(filters.X, filterW), mc8Kind(filters.Y, filterH)
					if _, cover := mc8Taps(w, kx, ky, subX, subY); cover && w%4 == 0 {
						t.Fatalf("mc8PutFast declined w=%d h=%d filters=%v sub=%d,%d", w, h, filters, subX, subY)
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
					if _, cover := mc8Taps(w, kx, ky, subX, subY); cover && subX|subY != 0 && w%4 == 0 {
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
		if !mc8PutFast(&dst, &ref, 0, 0, 20, 20, 16, 16, 16, 16, sub[0], sub[1], sharp, &scratch) {
			t.Fatalf("mc8PutFast declined sharp 16x16 sub=%v", sub)
		}
		if !mc8PutFast(&dst, &ref, 0, 0, 20, 20, 16, 16, 16, 16, sub[0], sub[1], RegularFilters, &scratch) {
			t.Fatalf("mc8PutFast declined 16x16 sub=%v", sub)
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
