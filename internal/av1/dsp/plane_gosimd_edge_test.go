// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import (
	"math/rand"
	"testing"
)

// planeEdgeDst lists destination samples at and around the clamp boundaries,
// including 16-bit values above every AV1 maximum up to 0xffff.
var planeEdgeDst = []uint16{0, 1, 127, 128, 254, 255, 256, 1022, 1023, 1024, 4094, 4095, 4096, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff}

// planeEdgeRes lists int16 residuals at and around the saturation boundaries.
var planeEdgeRes = []int16{-32768, -32767, -32766, -16384, -4096, -1024, -256, -255, -128, -1, 0, 1, 127, 128, 255, 256, 1023, 4095, 4096, 16384, 32766, 32767}

// planeEdgeRaw lists int32 raw transform samples at and around the rounding,
// int16-saturation and int32 boundaries.
var planeEdgeRaw = []int32{
	-1 << 31, -1<<31 + 1, -1<<31 + 7, -1<<31 + 8, -1<<31 + 9, -(32768 * 16) - 9, -(32768 * 16) - 8,
	-(32768 * 16) - 1, -(32768 * 16), -(32768 * 16) + 7, -(32768 * 16) + 8, -4096, -24, -9, -8, -7, -1,
	0, 1, 7, 8, 9, 23, 24, 4095, 32767*16 + 7, 32767*16 + 8, 32767*16 + 9, 32768 * 16, 1<<31 - 9,
	1<<31 - 8, 1<<31 - 7, 1<<31 - 1,
}

// planeEdgeLayout is one destination/source layout: the extra bytes between
// destination rows, the destination's byte offset into its allocation (odd
// offsets misalign 16-bit samples), and the extra source samples per row.
type planeEdgeLayout struct{ dstPad, dstOff, srcPad int }

var planeEdgeLayouts = []planeEdgeLayout{{0, 0, 0}, {1, 1, 0}, {3, 0, 4}, {16, 3, 1}, {0, 1, 8}, {64, 0, 32}}

// TestPlaneSIMDLayoutsMatchPureGo drives both kernels through the public
// trusted entry points over every kernel shape, heights one to nine (odd
// heights reach the paired-row tails), padded and misaligned destinations and
// source strides wider than the block.
func TestPlaneSIMDLayoutsMatchPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095, 0xffff}
		}
		for _, max := range maxes {
			for _, width := range []int{4, 8, 16, 24, 32, 40, 48, 64} {
				for height := 1; height <= 9; height++ {
					for _, lay := range planeEdgeLayouts {
						stride := width*bps + lay.dstPad
						srcStride := width + lay.srcPad
						alloc := planeSIMDBlock(rng, height*stride+lay.dstOff+1, 1)
						pix := alloc[lay.dstOff : lay.dstOff+(height-1)*stride+width*bps]
						res := make([]int16, (height-1)*srcStride+width)
						for i := range res {
							if rng.Intn(4) == 0 {
								res[i] = planeEdgeRes[rng.Intn(len(planeEdgeRes))]
							} else {
								res[i] = int16(rng.Intn(65536) - 32768)
							}
						}
						raw := make([]int32, (height-1)*srcStride+width)
						for i := range raw {
							if rng.Intn(2) == 0 {
								raw[i] = planeEdgeRaw[rng.Intn(len(planeEdgeRaw))]
							} else {
								raw[i] = int32(rng.Uint32())
							}
						}
						planeEdgeCheck(t, alloc, lay.dstOff, len(pix), stride, bps, max, width, height, res, raw, srcStride)
					}
				}
			}
		}
	}
}

// planeEdgeCheck runs both trusted entry points and both scalar references on
// identical copies of alloc and compares every byte, including the bytes
// outside the block that the kernels must leave untouched.
func planeEdgeCheck(t *testing.T, alloc []byte, off, n, stride, bps int, max uint16, width, height int, res []int16, raw []int32, srcStride int) {
	t.Helper()
	got := append([]byte(nil), alloc...)
	want := append([]byte(nil), alloc...)
	AddResidualPlaneBlockTrusted(got[off:off+n], stride, bps, max, width, height, res, srcStride)
	addResidualPlaneBlockPureGo(planeBlock{pix: want[off : off+n], stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, res, srcStride)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("residual bps=%d max=%#x w=%d h=%d stride=%d off=%d srcStride=%d: byte %d simd=%#x ref=%#x", bps, max, width, height, stride, off, srcStride, i, got[i], want[i])
		}
	}
	got = append(got[:0], alloc...)
	want = append(want[:0], alloc...)
	AddRawTransformPlaneBlockTrusted(got[off:off+n], stride, bps, max, width, height, raw, srcStride)
	addRawTransformPlaneBlockPureGo(planeBlock{pix: want[off : off+n], stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, raw, srcStride)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("raw bps=%d max=%#x w=%d h=%d stride=%d off=%d srcStride=%d: byte %d simd=%#x ref=%#x", bps, max, width, height, stride, off, srcStride, i, got[i], want[i])
		}
	}
}

// TestPlaneSIMDExhaustiveRawEdges pairs every destination edge value with
// every raw edge value, for each sample width, maximum and kernel width.
func TestPlaneSIMDExhaustiveRawEdges(t *testing.T) {
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095, 0xffff}
		}
		for _, max := range maxes {
			for _, width := range []int{4, 8, 16} {
				for _, d := range planeEdgeDst {
					if bps == 1 && d > 0xff {
						continue
					}
					for _, r := range planeEdgeRaw {
						pix := make([]byte, 3*width*bps)
						for i := range 3 * width {
							if bps == 1 {
								pix[i] = byte(d)
							} else {
								pix[2*i] = byte(d)
								pix[2*i+1] = byte(d >> 8)
							}
						}
						raw := make([]int32, 3*width)
						for i := range raw {
							raw[i] = r
						}
						planeSIMDCheckRaw(t, pix, width*bps, bps, max, width, 3, raw, width)
					}
				}
			}
		}
	}
}

// TestPlaneSIMDExhaustiveResidualEdgesWide extends the residual edge sweep to
// the 16-sample kernels and the 0xffff maximum.
func TestPlaneSIMDExhaustiveResidualEdgesWide(t *testing.T) {
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095, 0xffff}
		}
		for _, max := range maxes {
			for _, width := range []int{4, 8, 16, 24} {
				for _, d := range planeEdgeDst {
					if bps == 1 && d > 0xff {
						continue
					}
					for _, r := range planeEdgeRes {
						pix := make([]byte, 3*width*bps)
						for i := range 3 * width {
							if bps == 1 {
								pix[i] = byte(d)
							} else {
								pix[2*i] = byte(d)
								pix[2*i+1] = byte(d >> 8)
							}
						}
						res := make([]int16, 3*width)
						for i := range res {
							res[i] = r
						}
						planeSIMDCheckResidual(t, pix, width*bps, bps, max, width, 3, res, width)
					}
				}
			}
		}
	}
}

// TestPlaneSIMDEightBitOtherMaxima keeps 8-bit maxima other than 0xff, which
// AV1 never produces but the trusted entry points accept, on the reference
// semantics (including the byte truncation of maxima above 0xff).
func TestPlaneSIMDEightBitOtherMaxima(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	for _, max := range []uint16{0, 1, 200, 254, 256, 300, 0xffff} {
		for _, width := range []int{4, 8, 16} {
			const height = 5
			pix := planeSIMDBlock(rng, height, width)
			res := make([]int16, height*width)
			raw := make([]int32, height*width)
			for i := range res {
				res[i] = int16(rng.Intn(1200) - 600)
				raw[i] = int32(rng.Intn(1<<14) - 1<<13)
			}
			planeSIMDCheckResidual(t, pix, width, 1, max, width, height, res, width)
			planeSIMDCheckRaw(t, pix, width, 1, max, width, height, raw, width)
		}
	}
}
