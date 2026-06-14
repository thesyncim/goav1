// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

type compoundCopy8NEONCtx struct {
	dst         *uint16
	ref         *byte
	refStr      uintptr
	width       uintptr
	height      uintptr
	roundOffset uintptr
}

type compoundFilter8NEONCtx struct {
	dst         *uint16
	ref         *byte
	kernel      *int16
	refStr      uintptr
	width       uintptr
	height      uintptr
	roundOffset uintptr
}

type compound2D8NEONCtx struct {
	dst    *uint16
	ref    *byte
	kernel *int16
	xKern  *int16
	refStr uintptr
	width  uintptr
	height uintptr
	im     *int16
	imStr  uintptr
}

//go:noescape
func compoundCopy8NEONAsm(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundCopy8NEONAsmW4(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundX8NEONAsm(ctx *compoundFilter8NEONCtx)

//go:noescape
func compoundX8NEONAsmW4(ctx *compoundFilter8NEONCtx)

//go:noescape
func compoundY8NEONAsm(ctx *compoundFilter8NEONCtx)

//go:noescape
func compoundY8NEONAsmW4(ctx *compoundFilter8NEONCtx)

//go:noescape
func compound2D8NEONAsm(ctx *compound2D8NEONCtx)

//go:noescape
func compound2D8NEONAsmW4(ctx *compound2D8NEONCtx)

func predictInterCompoundRef8ToConvBufCopyNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if round0 != compoundRound0Bits ||
		!planeRegionFits(ref, 1, refX, refY, width, height) {
		predictInterCompoundRef8ToConvBufCopyPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	ctx := compoundCopy8NEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[refY*ref.Stride+refX],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		roundOffset: uintptr(roundOffset),
	}
	if width == 4 {
		compoundCopy8NEONAsmW4(&ctx)
		return
	}
	if width < 8 || width%8 != 0 {
		predictInterCompoundRef8ToConvBufCopyPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	compoundCopy8NEONAsm(&ctx)
}

func predictInterCompoundRef8ToConvBufXNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	// The horizontal asm loads one extra byte in the last 8-column group while
	// forming the slide register. Require that byte to be resident too.
	if !(width == 4 || (width >= 8 && width%8 == 0)) ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		predictInterCompoundRef8ToConvBufXPureGo(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	k := kernel
	ctx := compoundFilter8NEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[refY*ref.Stride+refX-fo],
		kernel:      &k[0],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		roundOffset: uintptr(roundOffset),
	}
	if width == 4 {
		compoundX8NEONAsmW4(&ctx)
		return
	}
	compoundX8NEONAsm(&ctx)
}

func predictInterCompoundRef8ToConvBufYNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if round0 != compoundRound0Bits ||
		!(width == 4 || (width >= 8 && width%8 == 0)) ||
		!planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		predictInterCompoundRef8ToConvBufYPureGo(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	k := kernel
	ctx := compoundFilter8NEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[(refY-fo)*ref.Stride+refX],
		kernel:      &k[0],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		roundOffset: uintptr(roundOffset),
	}
	if width == 4 {
		compoundY8NEONAsmW4(&ctx)
		return
	}
	compoundY8NEONAsm(&ctx)
}

func predictInterCompoundRef8ToConvBuf2DNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, scratch *CompoundConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	// The horizontal asm consumes one extra byte in the last width>=8 group
	// while forming slide vectors, so require that byte to be resident.
	if offsetBits != 19 ||
		!(width == 4 || (width >= 8 && width%8 == 0)) ||
		!planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	xk := xKernel
	yk := yKernel
	if width == 4 {
		const w4Stride = 4
		if scratch != nil {
			predictInterCompoundRef8ToConvBuf2DNEONWithIMStride(out, ref, refX, refY, width, height, xk, yk, &scratch.im8[0], w4Stride)
			return
		}
		var im [(maxBlockSize + filterTaps - 1) * w4Stride]int16
		predictInterCompoundRef8ToConvBuf2DNEONWithIMStride(out, ref, refX, refY, width, height, xk, yk, &im[0], w4Stride)
		return
	}
	if scratch != nil {
		predictInterCompoundRef8ToConvBuf2DNEONWithIM(out, ref, refX, refY, width, height, xk, yk, &scratch.im8)
		return
	}
	var im compoundIM16
	predictInterCompoundRef8ToConvBuf2DNEONWithIM(out, ref, refX, refY, width, height, xk, yk, &im)
}

func predictInterCompoundRef8ToConvBuf2DNEONWithIM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *compoundIM16) {
	predictInterCompoundRef8ToConvBuf2DNEONWithIMStride(out, ref, refX, refY, width, height, xKernel, yKernel, &im[0], maxBlockSize)
}

func predictInterCompoundRef8ToConvBuf2DNEONWithIMStride(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *int16, imStride int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	ctx := compound2D8NEONCtx{
		dst:    &out[0],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+refX-foX],
		kernel: &yKernel[0],
		xKern:  &xKernel[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     im,
		imStr:  uintptr(imStride),
	}
	if width == 4 {
		compound2D8NEONAsmW4(&ctx)
		return
	}
	compound2D8NEONAsm(&ctx)
}

func init() {
	if cpu.Detected.NEON {
		predictInterCompoundRef8ToConvBuf2DImpl = predictInterCompoundRef8ToConvBuf2DNEON
		predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyNEON
		predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXNEON
		predictInterCompoundRef8ToConvBufYImpl = predictInterCompoundRef8ToConvBufYNEON
	}
}
