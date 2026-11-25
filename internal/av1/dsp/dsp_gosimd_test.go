//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestAddConstantPlaneBlock8SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	residuals := []int16{-32768, -1000, -256, -255, -254, -128, -1, 0, 1, 127, 254, 255, 256, 1000, 32767}
	for _, w := range []int{4, 8, 12, 16, 24, 32, 40, 64} {
		for _, h := range []int{1, 2, 3, 4, 7, 8, 16, 64} {
			for i := 0; i < 24; i++ {
				r := int16(rng.Intn(65536) - 32768)
				if i < len(residuals) {
					r = residuals[i]
				}
				stride := w + rng.Intn(9)
				buf := make([]byte, (h-1)*stride+w+8)
				rng.Read(buf)
				want := append([]byte(nil), buf...)
				got := append([]byte(nil), buf...)
				addConstantPlaneBlock8PureGo(want, stride, w, h, r)
				addConstantPlaneBlock8SIMD(got, stride, w, h, r)
				if !bytes.Equal(got, want) {
					t.Fatalf("w=%d h=%d stride=%d r=%d: SIMD differs from the scalar reference", w, h, stride, r)
				}
			}
		}
	}
}

func TestAddConstantPlaneBlock16SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	residuals := []int16{-32768, -5000, -4096, -4095, -1024, -1023, -1, 0, 1, 1023, 1024, 4095, 4096, 5000, 32767}
	for _, maxV := range []uint16{1023, 4095, 65535} {
		for _, w := range []int{4, 8, 12, 16, 24, 32, 40, 64} {
			for _, h := range []int{1, 2, 3, 4, 7, 8, 16, 64} {
				for i := 0; i < 24; i++ {
					r := int16(rng.Intn(65536) - 32768)
					if i < len(residuals) {
						r = residuals[i]
					}
					stride := 2*w + 2*rng.Intn(9)
					buf := make([]byte, (h-1)*stride+2*w+8)
					rng.Read(buf)
					want := append([]byte(nil), buf...)
					got := append([]byte(nil), buf...)
					addConstantPlaneBlock16PureGo(want, stride, w, h, maxV, r)
					addConstantPlaneBlock16SIMD(got, stride, w, h, maxV, r)
					if !bytes.Equal(got, want) {
						t.Fatalf("max=%d w=%d h=%d stride=%d r=%d: SIMD differs from the scalar reference", maxV, w, h, stride, r)
					}
				}
			}
		}
	}
}

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

// planeSIMDBlock returns a block of height rows with stride bytes per row, filled
// from rng so that 16-bit samples cover the whole u16 range (including values
// above max, which the references clamp).
func planeSIMDBlock(rng *rand.Rand, height, stride int) []byte {
	pix := make([]byte, height*stride)
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	return pix
}

// planeSIMDCheckResidual runs addResidualPlaneBlockSIMD and the scalar reference
// over identical copies and reports the first byte that differs.
func planeSIMDCheckResidual(t *testing.T, pix []byte, stride int, bps int, max uint16, width int, height int, residual []int16, resStride int) {
	t.Helper()
	got := append([]byte(nil), pix...)
	want := append([]byte(nil), pix...)
	addResidualPlaneBlockSIMD(planeBlock{pix: got, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, residual, resStride)
	addResidualPlaneBlockPureGo(planeBlock{pix: want, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, residual, resStride)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("residual bps=%d max=%d w=%d h=%d stride=%d: byte %d simd=%#x ref=%#x", bps, max, width, height, stride, i, got[i], want[i])
		}
	}
}

// planeSIMDCheckRaw runs addRawTransformPlaneBlockSIMD and the scalar reference
// over identical copies and reports the first byte that differs.
func planeSIMDCheckRaw(t *testing.T, pix []byte, stride int, bps int, max uint16, width int, height int, raw []int32, rawStride int) {
	t.Helper()
	got := append([]byte(nil), pix...)
	want := append([]byte(nil), pix...)
	addRawTransformPlaneBlockSIMD(planeBlock{pix: got, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, raw, rawStride)
	addRawTransformPlaneBlockPureGo(planeBlock{pix: want, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, raw, rawStride)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("raw bps=%d max=%d w=%d h=%d stride=%d: byte %d simd=%#x ref=%#x", bps, max, width, height, stride, i, got[i], want[i])
		}
	}
}

// planeSIMDShapes lists the widths the Go SIMD kernels accept, plus widths they
// route to the reference, and every height from one to five so that odd row
// counts reach the paired-row tail.
var planeSIMDShapes = []struct{ width, height int }{
	{4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5},
	{8, 1}, {8, 4}, {8, 8},
	{12, 3}, {16, 2}, {16, 8}, {32, 4}, {64, 2}, {5, 2},
}

// TestAddResidualPlaneBlockSIMDMatchesPureGo sweeps widths, heights, strides and
// both sample widths and bit depths with random destinations and residuals.
func TestAddResidualPlaneBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095, 0xffff}
		}
		for _, max := range maxes {
			for _, shape := range planeSIMDShapes {
				for _, pad := range []int{0, 3} {
					stride := shape.width*bps + pad
					for trial := range 16 {
						pix := planeSIMDBlock(rng, shape.height, stride)
						res := make([]int16, shape.height*shape.width+8)
						for i := range res {
							switch {
							case trial%4 == 0 && rng.Intn(3) == 0:
								res[i] = []int16{-32768, -32767, -4096, -1, 0, 1, 4096, 32767}[rng.Intn(8)]
							default:
								res[i] = int16(rng.Intn(65536) - 32768)
							}
						}
						planeSIMDCheckResidual(t, pix, stride, bps, max, shape.width, shape.height, res, shape.width)
					}
				}
			}
		}
	}
}

// TestAddRawTransformPlaneBlockSIMDMatchesPureGo sweeps the same shapes with raw
// int32 inverse-transform samples across their full range.
func TestAddRawTransformPlaneBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	edges := []int32{
		-1 << 31, -1<<31 + 1, -1<<31 + 7, -1<<31 + 8, -(1 << 20), -4096, -129, -128, -121, -120,
		-9, -8, -7, -1, 0, 1, 7, 8, 9, 120, 121, 127, 128, 4095, 4096, 1 << 20, 1<<31 - 9,
		1<<31 - 8, 1<<31 - 1,
	}
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095}
		}
		for _, max := range maxes {
			for _, shape := range planeSIMDShapes {
				for _, pad := range []int{0, 5} {
					stride := shape.width*bps + pad
					for trial := range 16 {
						pix := planeSIMDBlock(rng, shape.height, stride)
						raw := make([]int32, shape.height*shape.width+8)
						for i := range raw {
							if trial%2 == 0 {
								raw[i] = edges[rng.Intn(len(edges))]
							} else {
								raw[i] = int32(rng.Uint32())
							}
						}
						planeSIMDCheckRaw(t, pix, stride, bps, max, shape.width, shape.height, raw, shape.width)
					}
				}
			}
		}
	}
}

// TestPlaneSIMDExhaustiveResidualEdges pairs every destination edge value with
// every residual edge value on a single row, for each sample width and maximum.
func TestPlaneSIMDExhaustiveResidualEdges(t *testing.T) {
	dstEdges := []uint16{0, 1, 127, 128, 255, 256, 1023, 1024, 4095, 4096, 0x7fff, 0x8000, 0xfffe, 0xffff}
	resEdges := []int16{-32768, -32767, -4096, -1024, -256, -128, -1, 0, 1, 127, 128, 255, 256, 4095, 4096, 32766, 32767}
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095}
		}
		for _, max := range maxes {
			for _, width := range []int{8, 4} {
				for _, d := range dstEdges {
					for _, r := range resEdges {
						pix := make([]byte, 2*width*bps)
						res := make([]int16, 2*width)
						for i := range 2 * width {
							if bps == 1 {
								pix[i] = byte(d)
							} else {
								pix[2*i] = byte(d)
								pix[2*i+1] = byte(d >> 8)
							}
							res[i] = r
						}
						planeSIMDCheckResidual(t, pix, width*bps, bps, max, width, 2, res, width)
					}
				}
			}
		}
	}
}

// TestPlaneSIMDRoutesOtherWidthsToReference confirms widths the SIMD kernels do
// not vectorise still produce the reference output through the same entry point.
func TestPlaneSIMDRoutesOtherWidthsToReference(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, width := range []int{1, 2, 3, 5, 6, 7, 9, 12} {
		pix := planeSIMDBlock(rng, 3, width*2+1)
		res := make([]int16, 3*width)
		for i := range res {
			res[i] = int16(rng.Intn(2000) - 1000)
		}
		planeSIMDCheckResidual(t, pix, width*2+1, 2, 4095, width, 3, res, width)
		raw := make([]int32, 3*width)
		for i := range raw {
			raw[i] = int32(rng.Intn(1<<20) - 1<<19)
		}
		planeSIMDCheckRaw(t, pix, width+1, 1, 255, width, 3, raw, width)
	}
}

// TestPlaneSIMDIsZeroAlloc keeps both Go SIMD kernels allocation-free.
func TestPlaneSIMDIsZeroAlloc(t *testing.T) {
	const width, height = 16, 8
	pix := make([]byte, height*width*2)
	res := make([]int16, height*width)
	raw := make([]int32, height*width)
	for _, bps := range []int{1, 2} {
		block := planeBlock{pix: pix, stride: width * 2, width: width, height: height, rowBytes: width * bps}
		allocs := testing.AllocsPerRun(200, func() {
			addResidualPlaneBlockSIMD(block, bps, 255, width, res, width)
			addRawTransformPlaneBlockSIMD(block, bps, 255, width, raw, width)
		})
		if allocs != 0 {
			t.Fatalf("bps=%d: Go SIMD plane kernels allocated %f times per call", bps, allocs)
		}
	}
}
