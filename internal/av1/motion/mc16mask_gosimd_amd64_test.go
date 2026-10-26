// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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
