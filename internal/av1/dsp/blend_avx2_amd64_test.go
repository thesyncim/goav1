// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

import "testing"

// TestBlendA64MaskAVX2MatchesPureGo runs the AVX2 inner loop and the pure-Go
// reference over identical valid inputs and asserts byte-for-byte equality, for
// the no-subsampling and 2x2 subsampled mask layouts at every supported bit
// depth and a sweep of block sizes (including the single-axis layouts, which
// the AVX2 wrapper routes to the reference).
//
// It calls blendA64MaskAVX2 directly rather than through the dispatch slot, and
// is deliberately NOT gated on cpu.Detected.AVX2: amd64 under Rosetta 2 reports
// AVX2 absent in CPUID yet still executes the instructions, which is exactly the
// configuration that exercises the kernel here.
func TestBlendA64MaskAVX2MatchesPureGo(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, subX := range []bool{false, true} {
			for _, subY := range []bool{false, true} {
				for _, width := range []int{8, 16, 32, 64} {
					for _, height := range []int{4, 8, 16} {
						rnd := newBlendRandom(uint32(width*7 + height*31 + int(bitDepth) + boolSeed(subX)*3 + boolSeed(subY)*5))
						mw := maskWidth(width, subX)
						mh := maskHeight(height, subY)
						src0 := make([]uint16, width*height)
						src1 := make([]uint16, width*height)
						mask := make([]uint8, mw*mh)
						fillBlendSamples(src0, max, rnd)
						fillBlendSamples(src1, max, rnd)
						fillBlendMask(mask, rnd)

						got := make([]uint16, width*height)
						want := make([]uint16, width*height)

						gotOK := blendA64MaskAVX2(blendA64MaskArgs{
							dst: got, dstStride: width,
							src0: src0, src0Stride: width,
							src1: src1, src1Stride: width,
							mask: mask, maskStride: mw,
							width: width, height: height, max: max,
							subX: subX, subY: subY,
						})
						wantOK := blendA64MaskPureGo(blendA64MaskArgs{
							dst: want, dstStride: width,
							src0: src0, src0Stride: width,
							src1: src1, src1Stride: width,
							mask: mask, maskStride: mw,
							width: width, height: height, max: max,
							subX: subX, subY: subY,
						})
						if gotOK != wantOK {
							t.Fatalf("bd=%d subX=%v subY=%v w=%d h=%d verdict: avx2=%v ref=%v", bitDepth, subX, subY, width, height, gotOK, wantOK)
						}
						for i := range got {
							if got[i] != want[i] {
								t.Fatalf("bd=%d subX=%v subY=%v w=%d h=%d sample %d: avx2=%d ref=%d", bitDepth, subX, subY, width, height, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestBlendA64MaskAVX2RejectsOutOfRange confirms the AVX2 path reports the same
// invalid verdict as the reference when a sample or mask weight is out of range,
// for both vectorised layouts. It also covers samples at and above 0x8000 in the
// high-bit-depth shapes, where a signed compare would wrongly accept them; the
// kernel uses an unsigned-saturating range test, so it must reject. Like the
// matches test it calls the kernel directly and is not gated on AVX2.
func TestBlendA64MaskAVX2RejectsOutOfRange(t *testing.T) {
	const width, height = 8, 8
	for _, subXY := range []bool{false, true} {
		mw := maskWidth(width, subXY)
		mh := maskHeight(height, subXY)
		newArgs := func(max uint16) blendA64MaskArgs {
			return blendA64MaskArgs{
				dst: make([]uint16, width*height), dstStride: width,
				src0: make([]uint16, width*height), src0Stride: width,
				src1: make([]uint16, width*height), src1Stride: width,
				mask: make([]uint8, mw*mh), maskStride: mw,
				width: width, height: height, max: max,
				subX: subXY, subY: subXY,
			}
		}
		t.Run(layoutName(subXY)+"-sample", func(t *testing.T) {
			a := newArgs(0xff)
			a.src0[5] = 0x100 // > max (0xff)
			assertRejected(t, a)
		})
		t.Run(layoutName(subXY)+"-sample-high", func(t *testing.T) {
			// 0x8000 is negative as a signed int16; an unsigned range test must
			// still reject it against a 12-bit max.
			a := newArgs(0xfff)
			a.src1[2] = 0x8000
			assertRejected(t, a)
		})
		t.Run(layoutName(subXY)+"-sample-max16", func(t *testing.T) {
			a := newArgs(0xfff)
			a.src0[7] = 0xffff
			assertRejected(t, a)
		})
		t.Run(layoutName(subXY)+"-mask", func(t *testing.T) {
			a := newArgs(0xff)
			a.mask[3] = 65 // > 64; in subXY this lifts the round2 above 64 too
			if subXY {
				a.mask[3] = 255
				a.mask[mw+3] = 255
			}
			assertRejected(t, a)
		})
	}
}

func assertRejected(t *testing.T, a blendA64MaskArgs) {
	t.Helper()
	if blendA64MaskAVX2(a) {
		t.Fatal("AVX2 accepted out-of-range input")
	}
	// The reference must agree that the input is invalid.
	if blendA64MaskPureGo(a) {
		t.Fatal("reference accepted the same input (test setup error)")
	}
}

func boolSeed(b bool) int {
	if b {
		return 1
	}
	return 0
}

func layoutName(subXY bool) string {
	if subXY {
		return "subxy"
	}
	return "nosub"
}

// TestBlendA64MaskAVX2IsZeroAlloc guards against the wrapper introducing heap
// traffic on the hot path.
func TestBlendA64MaskAVX2IsZeroAlloc(t *testing.T) {
	const width, height = 64, 64
	dst := make([]uint16, width*height)
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	mask := make([]uint8, maskWidth(width, true)*maskHeight(height, true))
	rnd := newBlendRandom(libaomBlendDeterministicSeed)
	fillBlendSamples(src0, 0xfff, rnd)
	fillBlendSamples(src1, 0xfff, rnd)
	fillBlendMask(mask, rnd)
	args := blendA64MaskArgs{
		dst: dst, dstStride: width,
		src0: src0, src0Stride: width,
		src1: src1, src1Stride: width,
		mask: mask, maskStride: maskWidth(width, true),
		width: width, height: height, max: 0xfff,
		subX: true, subY: true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if !blendA64MaskAVX2(args) {
			t.Fatal("unexpected invalid verdict")
		}
	})
	if allocs != 0 {
		t.Fatalf("blendA64MaskAVX2 allocated: %f", allocs)
	}
}
