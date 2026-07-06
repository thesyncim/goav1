// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2-accelerated high-bit-depth (10/12-bit) convolve kernels. Samples are
// little-endian uint16; the round-0/round-1 shift amounts are bit-depth
// dependent (highBDRoundBits): round0=3,round1=11 for bd<=10 and round0=5,
// round1=9 for bd==12. The shifts and biases are passed in via the context so a
// single code path covers every bit depth.
//
// The asm widens uint16 samples to int32 (VPMOVZXWD), multiplies by an int32
// broadcast tap (VPMULLD) and accumulates (VPADDD), then applies the staged
// round (VPADDD bias + VPSRAD) exactly like roundPowerOfTwo. Final clipping is a
// VPMAXSD(0)/VPMINSD(max) clamp because the upper bound is bit-depth dependent.

// convolveHighBDAVX2Ctx is the asm calling context for the HBD convolves. Field
// order and sizes are part of the ABI shared with convolve_highbd_avx2_amd64.s;
// do not reorder.
type convolveHighBDAVX2Ctx struct {
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
func convolveXHighBDAVX2Asm(ctx *convolveHighBDAVX2Ctx)

//go:noescape
func convolveYHighBDAVX2Asm(ctx *convolveHighBDAVX2Ctx)

//go:noescape
func convolve2DHighBDAVX2Asm(ctx *convolveHighBDAVX2Ctx)

func convolveXHighBDAVX2(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 4 || width%4 != 0 {
		convolveXHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	k := kernel
	ctx := convolveHighBDAVX2Ctx{
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
	convolveXHighBDAVX2Asm(&ctx)
}

func convolveYHighBDAVX2(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_ = bitDepth
	if width < 4 || width%4 != 0 {
		convolveYHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	k := kernel
	ctx := convolveHighBDAVX2Ctx{
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
	convolveYHighBDAVX2Asm(&ctx)
}

// convolve2DHighBDAVX2IMStride matches the imStride const in
// convolve2DHighBDPureGo (maxBlockSize) so the vertical pass walks the same rows.
const convolve2DHighBDAVX2IMStride = maxBlockSize

func convolve2DHighBDAVX2(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDAVX2WithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDAVX2WithScratch is convolve2DHighBDAVX2 with optional
// caller-owned scratch for the int32 intermediate block, mirroring the NEON
// variant: with scratch the per-call zero-fill of the ~69KB stack array
// disappears; without scratch the stack array keeps the historical behavior.
func convolve2DHighBDAVX2WithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width < 4 || width%4 != 0 {
		convolve2DHighBDPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if scratch != nil {
		convolve2DHighBDAVX2WithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.imHBD[0])
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DHighBDAVX2IMStride]int32
	convolve2DHighBDAVX2WithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im[0])
}

func convolve2DHighBDAVX2WithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *int32) {
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
	ctx := convolveHighBDAVX2Ctx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX*2],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+(refX-foX)*2],
		kernel: &yk[0],
		xKern:  &xk[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     im,
		imStr:  uintptr(convolve2DHighBDAVX2IMStride),
		round0: uintptr(round0),
		round1: uintptr(round1),
		bits:   uintptr(bits),
		rndOff: uintptr(roundOffset),
		xBias:  uintptr(xBias),
		yBias:  uintptr(yBias),
		maxVal: uintptr(max),
	}
	convolve2DHighBDAVX2Asm(&ctx)
}

func convolveXHighBDClampedAVX2(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-fo, refY, width+filterTaps-1, height) {
		convolveXHighBDAVX2(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	convolveXHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveYHighBDClampedAVX2(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX, refY-fo, width, height+filterTaps-1) {
		convolveYHighBDAVX2(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	convolveYHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolve2DHighBDClampedAVX2(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDClampedAVX2WithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDClampedAVX2WithScratch is convolve2DHighBDClampedAVX2 with
// optional caller-owned scratch for the int32 intermediate block.
func convolve2DHighBDClampedAVX2WithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		convolve2DHighBDAVX2WithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	convolve2DHighBDClampedPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}
