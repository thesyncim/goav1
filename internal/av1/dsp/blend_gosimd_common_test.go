// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import (
	"testing"
)

// blendSIMDCase runs one BlendA64Mask inner loop variant over the given args
// and returns the verdict and a copy of the destination.
func blendSIMDCase(fn func(blendA64MaskArgs) bool, a blendA64MaskArgs, dstLen int) (bool, []uint16) {
	dst := make([]uint16, dstLen)
	a.dst = dst
	return fn(a), dst
}

// TestBlendA64MaskSIMDMatchesPureGo compares the NEON-width Go SIMD inner loop
// with the scalar reference over every bit depth, both vectorised layouts, and
// the widths and heights the dispatcher can hand it, with random and extreme
// sample and weight values. The single-axis layouts route to the reference and
// are covered as well.
func TestBlendA64MaskSIMDMatchesPureGo(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, subX := range []bool{false, true} {
			for _, subY := range []bool{false, true} {
				for _, width := range []int{4, 8, 16, 32, 64} {
					for _, height := range []int{2, 4, 8, 16} {
						for variant := range 3 {
							rnd := newBlendRandom(uint32(width*7 + height*31 + int(bitDepth) + boolSeed(subX)*3 + boolSeed(subY)*5 + variant*101))
							mw := maskWidth(width, subX)
							mh := maskHeight(height, subY)
							src0 := make([]uint16, width*height)
							src1 := make([]uint16, width*height)
							mask := make([]uint8, mw*mh)
							fillBlendSamples(src0, max, rnd)
							fillBlendSamples(src1, max, rnd)
							fillBlendMask(mask, rnd)
							if variant == 1 {
								blendFillExtremeSamples(src0, max)
								blendFillExtremeSamples(src1, max)
								blendFillExtremeMask(mask)
							}
							args := blendA64MaskArgs{
								dst: nil, dstStride: width,
								src0: src0, src0Stride: width,
								src1: src1, src1Stride: width,
								mask: mask, maskStride: mw,
								width: width, height: height, max: max,
								subX: subX, subY: subY,
							}
							if variant == 2 {
								// Out-of-range weights that the verdict must still
								// reject identically.
								mask[len(mask)-1] = 255
							}
							gotOK, got := blendSIMDCase(blendA64MaskSIMD, args, width*height)
							wantOK, want := blendSIMDCase(blendA64MaskPureGo, args, width*height)
							if gotOK != wantOK {
								t.Fatalf("bd=%d %s w=%d h=%d variant=%d verdict: simd=%v ref=%v",
									bitDepth, blendLayoutName(subX, subY), width, height, variant, gotOK, wantOK)
							}
							if !wantOK {
								continue
							}
							for i := range got {
								if got[i] != want[i] {
									t.Fatalf("bd=%d %s w=%d h=%d variant=%d sample %d: simd=%d ref=%d",
										bitDepth, blendLayoutName(subX, subY), width, height, variant, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestBlendA64MaskSIMDRejectsOutOfRange confirms the SIMD path reports the same
// invalid verdict as the reference when a sample or mask weight is out of range
// in either vectorised layout, including values at or above 0x8000.
func TestBlendA64MaskSIMDRejectsOutOfRange(t *testing.T) {
	const width, height = 16, 8
	for _, subXY := range []bool{false, true} {
		mw := maskWidth(width, subXY)
		mh := maskHeight(height, subXY)
		newArgs := func() blendA64MaskArgs {
			return blendA64MaskArgs{
				dst: make([]uint16, width*height), dstStride: width,
				src0: make([]uint16, width*height), src0Stride: width,
				src1: make([]uint16, width*height), src1Stride: width,
				mask: make([]uint8, mw*mh), maskStride: mw,
				width: width, height: height, max: 0xfff,
				subX: subXY, subY: subXY,
			}
		}
		t.Run(blendLayoutName(subXY, subXY)+"-sample", func(t *testing.T) {
			for _, v := range []uint16{0x1000, 0x8000, 0xffff} {
				for _, second := range []bool{false, true} {
					a := newArgs()
					if second {
						a.src1[height*width-1] = v
					} else {
						a.src0[height*width-1] = v
					}
					if blendA64MaskSIMD(a) {
						t.Fatalf("SIMD accepted out-of-range sample %#x (src1=%v)", v, second)
					}
				}
			}
		})
		t.Run(blendLayoutName(subXY, subXY)+"-mask", func(t *testing.T) {
			a := newArgs()
			a.mask[0] = 65
			if subXY {
				a.mask[0] = 255
				a.mask[mw] = 255
			}
			if blendA64MaskSIMD(a) {
				t.Fatal("SIMD accepted out-of-range mask weight")
			}
		})
	}
}

// TestBlendA64MaskSIMDIsZeroAlloc keeps the Go SIMD inner loop allocation-free.
func TestBlendA64MaskSIMDIsZeroAlloc(t *testing.T) {
	const width, height = 32, 16
	src := make([]uint16, width*height)
	dst := make([]uint16, width*height)
	mask := make([]uint8, width*height*4)
	for i := range mask {
		mask[i] = uint8(i * 37)
	}
	for _, subXY := range []bool{false, true} {
		mw := maskWidth(width, subXY)
		a := blendA64MaskArgs{
			dst: dst, dstStride: width,
			src0: src, src0Stride: width,
			src1: src, src1Stride: width,
			mask: mask, maskStride: mw,
			width: width, height: height, max: 0x3ff,
			subX: subXY, subY: subXY,
		}
		allocs := testing.AllocsPerRun(100, func() {
			blendA64MaskSIMD(a)
		})
		if allocs != 0 {
			t.Fatalf("%s: SIMD inner loop allocated %f times per call", blendLayoutName(subXY, subXY), allocs)
		}
	}
}

// blendFillExtremeSamples sets samples to the bit-depth endpoints and the
// values around them.
func blendFillExtremeSamples(samples []uint16, max uint16) {
	pick := []uint16{0, 1, max - 1, max, max / 2}
	for i := range samples {
		samples[i] = pick[i%len(pick)]
	}
}

// blendFillExtremeMask sets weights to the valid endpoints and the first
// invalid value above them.
func blendFillExtremeMask(mask []uint8) {
	pick := []uint8{0, 1, 31, 32, 63, 64}
	for i := range mask {
		mask[i] = pick[i%len(pick)]
	}
}

func blendLayoutName(subX, subY bool) string {
	switch {
	case subX && subY:
		return "subxy"
	case subX:
		return "subx"
	case subY:
		return "suby"
	default:
		return "nosub"
	}
}

func boolSeed(b bool) int {
	if b {
		return 1
	}
	return 0
}
