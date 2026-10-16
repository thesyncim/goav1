// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"bytes"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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

func TestMC8PutFastMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	sizes := [][2]int{{4, 4}, {4, 8}, {4, 16}, {4, 2}, {4, 5}, {8, 5}, {16, 7}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 4}, {8, 2}, {8, 16}, {16, 8}, {24, 8}, {64, 64}, {128, 128}, {32, 8}, {64, 16}, {16, 64}, {8, 32}}
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
					if (subX == 0 || mc8SixTap[kx]) && (subY == 0 || mc8SixTap[ky]) && (w%8 == 0 || w == 4) {
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

func mc8RefPrep(out []uint16, ref frame.Plane, refX, refY, w, h, subX, subY int, filters InterpFilters) {
	xk, _ := interpKernel(filters.X, w, subX)
	yk, _ := interpKernel(filters.Y, h, subY)
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

func TestMC8PrepFastMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	sizes := [][2]int{{4, 4}, {4, 8}, {4, 16}, {4, 2}, {4, 5}, {8, 5}, {16, 7}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 4}, {8, 16}, {16, 8}, {64, 64}, {128, 128}, {32, 8}, {16, 64}}
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
				mc8RefPrep(want, ref, refX, refY, w, h, subX, subY, filters)
				if !mc8PrepFast(got[:w*h], &ref, refX, refY, w, h, subX, subY, filters, scratch) {
					kx, ky := mc8Kind(filters.X, w), mc8Kind(filters.Y, h)
					if (subX == 0 || mc8SixTap[kx]) && (subY == 0 || mc8SixTap[ky]) {
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

// TestMC8SixTapKinds pins which filter kinds run in the six-tap window.
func TestMC8SixTapKinds(t *testing.T) {
	want := [mc8KindCount]bool{true, true, false, true, true, true}
	if mc8SixTap != want {
		t.Fatalf("six-tap kinds = %v, want %v", mc8SixTap, want)
	}
}

func TestMC8CopyFastMatchesClampedCopy(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const rw, rh = 200, 180
	ref := mc8TestRef(rng, rw, rh, rw+24, 0)
	for _, sz := range [][2]int{{4, 4}, {4, 16}, {8, 8}, {12, 4}, {16, 16}, {24, 8}, {32, 32}, {64, 64}, {128, 128}} {
		w, h := sz[0], sz[1]
		for iter := range 50 {
			refX := rng.Intn(rw+2*w) - w
			refY := rng.Intn(rh+2*h) - h
			dw, dh := w+16, h+4
			want := frame.Plane{Pix: make([]byte, dw*dh), Stride: dw, Width: dw, Height: dh}
			for i := range want.Pix {
				want.Pix[i] = byte(i * 3)
			}
			got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: dw, Width: dw, Height: dh}
			if err := copyPlaneBlockClamped(want, ref, 1, 7, 3, refX, refY, w, h); err != nil {
				t.Fatal(err)
			}
			inside := refX >= 0 && refY >= 0 && refX+w <= rw && refY+h <= rh
			if mc8CopyFast(&got, &ref, 7, 3, refX, refY, w, h) != inside {
				t.Fatalf("%dx%d iter %d ref=%d,%d: handled=%v want %v", w, h, iter, refX, refY, !inside, inside)
			}
			if inside && !bytes.Equal(got.Pix, want.Pix) {
				t.Fatalf("%dx%d iter %d ref=%d,%d: mismatch", w, h, iter, refX, refY)
			}
		}
	}
}

func TestMC8BlendAvgFastMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, sz := range [][2]int{{4, 4}, {4, 8}, {4, 16}, {8, 4}, {8, 8}, {16, 8}, {16, 16}, {24, 8}, {32, 32}, {64, 64}, {128, 128}, {8, 2}} {
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

func TestMC8BlendAvgFastDeclinesUnsupportedWidthsWithoutWriting(t *testing.T) {
	const h = 3
	for _, width := range []int{12, 20, 28, 36, 0, -4} {
		t.Run(fmt.Sprintf("width-%d", width), func(t *testing.T) {
			count := max(width, 0) * h
			a := make([]uint16, count)
			b := make([]uint16, count)
			for i := range count {
				a[i] = uint16(6144 + 16*((i*17+3)%256))
				b[i] = uint16(6144 + 16*((i*29+111)%256))
			}

			dstWidth := max(width, 0) + 8
			stride := dstWidth + 5
			pix := make([]byte, stride*(h+2))
			for i := range pix {
				pix[i] = 0xa5
			}
			got := frame.Plane{Pix: pix, Stride: stride, Width: dstWidth, Height: h + 2}
			if mc8BlendAvgFast(&got, a, b, 3, 1, width, h) {
				t.Fatalf("mc8BlendAvgFast accepted unsupported width %d", width)
			}
			for i, v := range got.Pix {
				if v != 0xa5 {
					t.Fatalf("declined width %d changed destination byte %d to %d", width, i, v)
				}
			}
		})
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

func TestMC8MaskBlendAndDiffWtdMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {8, 4}, {16, 8}, {64, 64}, {128, 128}, {8, 32}} {
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
