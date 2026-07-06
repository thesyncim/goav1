// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// NEON-accelerated high-bit-depth (10/12-bit) convolve kernels. Samples are
// little-endian uint16; the round-0/round-1 shift amounts are bit-depth
// dependent (highBDRoundBits): round0=3,round1=11 for bd<=10 and round0=5,
// round1=9 for bd==12. The asm receives the shifts as negative lane values and
// applies them with SRSHL (signed rounding shift left == roundPowerOfTwo for a
// negative amount), so a single code path covers every bit depth without
// per-shift WORD-immediate variants.
//
// The asm processes 4 samples per iteration (one 4x int32 accumulator). Block
// widths that are not a multiple of 4 (none occur in AV1) fall back to pure-Go.
// Final clipping is a vector max(0)/min(max) clamp because the upper bound is
// bit-depth dependent, not a fixed 8-/16-bit saturation.

// convolveHighBDNEONCtx is the asm calling context for the high-bit-depth
// convolves. Field order and sizes are part of the ABI shared with
// convolve_highbd_neon_arm64.s; do not reorder.
type convolveHighBDNEONCtx struct {
	dst    *byte  // first destination sample (2 bytes/sample)
	ref    *byte  // first reference sample of the tap window
	kernel *int16 // vertical kernel (1D Y) or shared single kernel
	xKern  *int16 // 2D only: horizontal kernel
	dstStr uintptr
	refStr uintptr
	width  uintptr
	height uintptr
	im     *int32  // 2D only: int32 intermediate scratch
	imStr  uintptr // 2D only: intermediate row stride in int32 elements
	round0 uintptr // round-0 shift bits (1D Y: single FILTER_BITS shift)
	round1 uintptr // round-1 shift bits (2D vertical) or final "bits" (1D X)
	bits   uintptr // 2D only: final round shift
	rndOff uintptr // 2D only: roundOffset
	xBias  uintptr // 2D only: 1 << (bd + FILTER_BITS - 1)
	yBias  uintptr // 2D only: 1 << offsetBits
	maxVal uintptr // clip upper bound (1<<bd)-1
}

//go:noescape
func convolveXHighBDNEONAsm(ctx *convolveHighBDNEONCtx)

//go:noescape
func convolveYHighBDNEONAsm(ctx *convolveHighBDNEONCtx)

//go:noescape
func convolve2DHighBDNEONAsm(ctx *convolveHighBDNEONCtx)

func convolveXHighBDNEON(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 4 || width%4 != 0 {
		convolveXHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	k := kernel
	ctx := convolveHighBDNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX*2],
		ref:    &ref.Pix[refY*ref.Stride+(refX-fo)*2],
		kernel: &k[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		round0: uintptr(round0),
		round1: uintptr(bits),
		maxVal: uintptr(max),
	}
	convolveXHighBDNEONAsm(&ctx)
}

func convolveYHighBDNEON(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_ = bitDepth
	if width < 4 || width%4 != 0 {
		convolveYHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	k := kernel
	ctx := convolveHighBDNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX*2],
		ref:    &ref.Pix[(refY-fo)*ref.Stride+refX*2],
		kernel: &k[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		round0: uintptr(filterBits), // single rounding shift by FILTER_BITS
		maxVal: uintptr(max),
	}
	convolveYHighBDNEONAsm(&ctx)
}

// convolve2DHighBDNEONIMStride matches the imStride const in
// convolve2DHighBDPureGo (maxBlockSize) so the vertical pass walks the same rows.
const convolve2DHighBDNEONIMStride = maxBlockSize

func convolve2DHighBDNEON(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDNEONWithScratch is convolve2DHighBDNEON with optional
// caller-owned scratch for the int32 intermediate block. With scratch the
// per-call zero-fill of the ~69KB stack array disappears (the asm overwrites
// every intermediate sample it reads); without scratch the stack array keeps
// the historical behavior for callers outside the framework decode path.
func convolve2DHighBDNEONWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width < 4 || width%4 != 0 {
		convolve2DHighBDPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if scratch != nil {
		convolve2DHighBDNEONWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.imHBD[0])
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DHighBDNEONIMStride]int32
	convolve2DHighBDNEONWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im[0])
}

func convolve2DHighBDNEONWithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *int32) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	round0, round1 := highBDRoundBits(bitDepth)
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	xBias := 1 << (int(bitDepth) + filterBits - 1)
	yBias := 1 << offsetBits

	xk := xKernel
	yk := yKernel
	ctx := convolveHighBDNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX*2],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+(refX-foX)*2],
		kernel: &yk[0],
		xKern:  &xk[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     im,
		imStr:  uintptr(convolve2DHighBDNEONIMStride),
		round0: uintptr(round0),
		round1: uintptr(round1),
		bits:   uintptr(bits),
		rndOff: uintptr(roundOffset),
		xBias:  uintptr(xBias),
		yBias:  uintptr(yBias),
		maxVal: uintptr(max),
	}
	convolve2DHighBDNEONAsm(&ctx)
}

// The HBD *ClampedNEON wrappers mirror the 8-bit clamped wrappers: when the
// whole tap window is resident the clamp is a no-op and the result is
// bit-identical to the non-clamped NEON path. When the window overhangs the
// plane, dav1d's reconstruction (src/recon_tmpl.c mc()) materializes the
// clamped halo once via emu_edge (src/mc_tmpl.c emu_edge_c) and re-runs the
// plain resident 8tap kernel over it — still NEON, and bit-identical to the
// per-tap-clamping pure-Go reference. Only widths the NEON kernels cannot
// vectorize (width%4 != 0, which does not occur for AV1 luma/chroma inter
// blocks) fall back to pure-Go.

func convolveXHighBDClampedNEON(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-fo, refY, width+filterTaps-1, height) {
		convolveXHighBDNEON(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 4 && width%4 == 0 {
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolveXHighBDNEON(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveXHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveYHighBDClampedNEON(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX, refY-fo, width, height+filterTaps-1) {
		convolveYHighBDNEON(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 4 && width%4 == 0 {
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolveYHighBDNEON(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveYHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolve2DHighBDClampedNEON(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDClampedNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDClampedNEONWithScratch is convolve2DHighBDClampedNEON with
// optional caller-owned scratch for both the int32 intermediate block and the
// 16bpc emulated-edge window, so hot decode paths zero-fill neither per call.
func convolve2DHighBDClampedNEONWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		convolve2DHighBDNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if width >= 4 && width%4 == 0 {
		if scratch != nil {
			emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &scratch.edge16)
			convolve2DHighBDNEONWithScratch(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, scratch)
			return
		}
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolve2DHighBDNEON(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel)
		return
	}
	convolve2DHighBDClampedPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}
