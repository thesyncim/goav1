// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// init binds the high-bit-depth convolves to the Go SIMD kernels. The clamped
// variants run the resident kernels over an emulated-edge window.
func init() {
	if !hbdSIMDAvailable() {
		return
	}
	convolveXHighBDImpl = convolveXHighBDGoSIMD
	convolveYHighBDImpl = convolveYHighBDGoSIMD
	convolve2DHighBDImpl = convolve2DHighBDGoSIMD
	convolveXHighBDClampedImpl = convolveXHighBDClampedGoSIMD
	convolveYHighBDClampedImpl = convolveYHighBDClampedGoSIMD
	convolve2DHighBDClampedImpl = convolve2DHighBDClampedGoSIMD
	convolve2DHighBDWithScratchImpl = convolve2DHighBDGoSIMDWithScratch
	convolve2DHighBDClampedWithScratchImpl = convolve2DHighBDClampedGoSIMDWithScratch
}

// Go SIMD high-bit-depth (10/12-bit) convolve wrappers. Samples are
// little-endian uint16; the round-0/round-1 shift amounts are bit-depth
// dependent (highBDRoundBits): round0=3,round1=11 for bd<=10 and round0=5,
// round1=9 for bd==12. The shifts are applied as arithmetic right shifts of
// int32 lanes, so a single code path covers every bit depth.
//
// The kernels handle every width that is a multiple of 4 (the tails of 4 and
// 12-lane blocks use a four-sample window). Widths that are not a multiple of 4
// (none occur in AV1) fall back to pure-Go. Final clipping is a vector
// max(0)/min(max) clamp because the upper bound is bit-depth dependent, not a
// fixed 8-/16-bit saturation.

// convolveHighBDGoSIMDCtx is the calling context of the high-bit-depth convolve
// kernels. dst and ref are little-endian byte planes starting at the first
// destination sample and the first sample of the trimmed tap span; kernel and
// xKern hold the trimmed taps zero-padded to filterTaps.
type convolveHighBDGoSIMDCtx struct {
	dst    []byte
	ref    []byte
	kernel [filterTaps]int16 // vertical kernel (1D Y) or shared single kernel
	xKern  [filterTaps]int16 // 2D only: horizontal kernel
	dstStr int
	refStr int
	width  int
	height int
	im     []int32 // 2D only: int32 intermediate scratch
	imStr  int     // 2D only: intermediate row stride in int32 elements
	round0 int     // round-0 shift bits (1D Y: single FILTER_BITS shift)
	round1 int     // round-1 shift bits (2D vertical) or final "bits" (1D X)
	bits   int     // 2D only: final round shift
	rndOff int32   // 2D only: roundOffset
	xBias  int32   // 2D only: 1 << (bd + FILTER_BITS - 1)
	yBias  int32   // 2D only: 1 << offsetBits
	maxVal int32   // clip upper bound (1<<bd)-1
	tapsX  int     // X tap count of the trimmed span (1D X and 2D)
	tapsY  int     // Y tap count of the trimmed span (1D Y and 2D)
}

func convolveXHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 4 || width%4 != 0 {
		convolveXHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	lo, n := hbdTapSpan(&kernel)
	ctx := convolveHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		ref:    ref.Pix[refY*ref.Stride+(refX-fo+lo)*2:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		round0: round0,
		round1: bits,
		maxVal: int32(max),
		tapsX:  n,
	}
	convolveXHighBDKernel(&ctx)
}

func convolveYHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_ = bitDepth
	if width < 4 || width%4 != 0 {
		convolveYHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	lo, n := hbdTapSpan(&kernel)
	ctx := convolveHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		ref:    ref.Pix[(refY-fo+lo)*ref.Stride+refX*2:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		round0: filterBits, // single rounding shift by FILTER_BITS
		maxVal: int32(max),
		tapsY:  n,
	}
	convolveYHighBDKernel(&ctx)
}

// convolve2DHighBDGoSIMDIMStride matches the imStride const in
// convolve2DHighBDPureGo (maxBlockSize) so the vertical pass walks the same rows.
const convolve2DHighBDGoSIMDIMStride = maxBlockSize

func convolve2DHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDGoSIMDWithScratch is convolve2DHighBDGoSIMD with optional
// caller-owned scratch for the int32 intermediate block. With scratch the
// per-call zero-fill of the ~69KB stack array disappears (the kernel overwrites
// every intermediate sample it reads); without scratch the stack array keeps
// the historical behavior for callers outside the framework decode path.
func convolve2DHighBDGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width < 4 || width%4 != 0 {
		convolve2DHighBDPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if scratch != nil {
		convolve2DHighBDGoSIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch.imHBD[:])
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DHighBDGoSIMDIMStride]int32
	convolve2DHighBDGoSIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, im[:])
}

func convolve2DHighBDGoSIMDWithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im []int32) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	round0, round1 := highBDRoundBits(bitDepth)
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	xBias := 1 << (int(bitDepth) + filterBits - 1)
	yBias := 1 << offsetBits

	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	ctx := convolveHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX)*2:],
		kernel: hbdTrimTaps(&yKernel, loY, nY),
		xKern:  hbdTrimTaps(&xKernel, loX, nX),
		tapsX:  nX,
		tapsY:  nY,
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		im:     im,
		imStr:  convolve2DHighBDGoSIMDIMStride,
		round0: round0,
		round1: round1,
		bits:   bits,
		rndOff: int32(roundOffset),
		xBias:  int32(xBias),
		yBias:  int32(yBias),
		maxVal: int32(max),
	}
	convolve2DHighBDKernel(&ctx)
}

// The HBD clamped wrappers mirror the 8-bit clamped wrappers: when the whole
// tap window is resident the clamp is a no-op and the result is bit-identical
// to the resident kernel. When the window overhangs the plane, dav1d's
// reconstruction (src/recon_tmpl.c mc()) materializes the clamped halo once via
// emu_edge (src/mc_tmpl.c emu_edge_c) and re-runs the resident kernel over it,
// bit-identical to the per-tap-clamping pure-Go reference. Only widths the
// kernels cannot vectorize (width%4 != 0, which does not occur for AV1
// luma/chroma inter blocks) fall back to pure-Go.

func convolveXHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-fo, refY, width+filterTaps-1, height) {
		convolveXHighBDGoSIMD(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 4 && width%4 == 0 {
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolveXHighBDGoSIMD(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveXHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveYHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX, refY-fo, width, height+filterTaps-1) {
		convolveYHighBDGoSIMD(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 4 && width%4 == 0 {
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolveYHighBDGoSIMD(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveYHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolve2DHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDClampedGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDClampedGoSIMDWithScratch is convolve2DHighBDClampedGoSIMD with
// optional caller-owned scratch for both the int32 intermediate block and the
// 16bpc emulated-edge window, so hot decode paths zero-fill neither per call.
func convolve2DHighBDClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		convolve2DHighBDGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if width >= 4 && width%4 == 0 {
		if scratch != nil {
			emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &scratch.edge16)
			convolve2DHighBDGoSIMDWithScratch(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, scratch)
			return
		}
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolve2DHighBDGoSIMD(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel)
		return
	}
	convolve2DHighBDClampedPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}
