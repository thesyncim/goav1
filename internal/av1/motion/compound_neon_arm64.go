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

type compoundFilterHighBDNEONCtx struct {
	dst         *uint16
	ref         *byte
	kernel      *int16
	refStr      uintptr
	width       uintptr
	height      uintptr
	round0      uintptr
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

type compound2DHighBDNEONCtx struct {
	dst    *uint16
	ref    *byte
	kernel *int16
	xKern  *int16
	refStr uintptr
	width  uintptr
	height uintptr
	im     *int32
	imStr  uintptr
	round0 uintptr
	xBias  uintptr
	yBias  uintptr
}

type compoundBlend8NEONCtx struct {
	dst    *byte
	src0   *uint16
	src1   *uint16
	dstStr uintptr
	width  uintptr
	height uintptr
	fwd    uintptr
	bck    uintptr
	bias   uintptr
}

//go:noescape
func blendCompoundAvg8NEONAsm(ctx *compoundBlend8NEONCtx)

//go:noescape
func blendCompoundAvg8NEONAsmW4(ctx *compoundBlend8NEONCtx)

//go:noescape
func compoundCopy8NEONAsm(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundCopy8NEONAsmW4(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundCopyHighBDNEONAsmS4(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundCopyHighBDNEONAsmS2(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundX8NEONAsm(ctx *compoundFilter8NEONCtx)

//go:noescape
func compoundXHighBDNEONAsm(ctx *compoundFilterHighBDNEONCtx)

//go:noescape
func compoundXHighBDNEONAsmW4(ctx *compoundFilterHighBDNEONCtx)

//go:noescape
func compoundYHighBDNEONAsm(ctx *compoundFilterHighBDNEONCtx)

//go:noescape
func compoundYHighBDNEONAsmW4(ctx *compoundFilterHighBDNEONCtx)

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

//go:noescape
func compound2DHighBDNEONAsm(ctx *compound2DHighBDNEONCtx)

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

func predictInterCompoundRefHighBDToConvBufCopyResidentNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 2, refX, refY, width, height) {
		predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	ctx := compoundCopy8NEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[refY*ref.Stride+refX*2],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		roundOffset: uintptr(roundOffset),
	}
	if round0 == compoundRound0Bits {
		compoundCopyHighBDNEONAsmS4(&ctx)
		return
	}
	compoundCopyHighBDNEONAsmS2(&ctx)
}

func predictInterCompoundRefHighBDToConvBufXResidentNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if round0 != compoundRound0Bits && round0 != compoundRound0Bits+2 {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	k := kernel
	if width == 4 {
		if k[0] != 0 || k[1] != 0 || k[6] != 0 || k[7] != 0 {
			predictInterCompoundRefHighBDToConvBufXResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
			return
		}
		ctx := compoundFilterHighBDNEONCtx{
			dst:         &out[0],
			ref:         &ref.Pix[refY*ref.Stride+(refX-1)*2],
			kernel:      &k[2],
			refStr:      uintptr(ref.Stride),
			width:       uintptr(width),
			height:      uintptr(height),
			round0:      uintptr(round0),
			roundOffset: uintptr(roundOffset),
		}
		compoundXHighBDNEONAsmW4(&ctx)
		return
	}
	if width < 8 || width%8 != 0 {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	ctx := compoundFilterHighBDNEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[refY*ref.Stride+(refX-fo)*2],
		kernel:      &k[0],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		round0:      uintptr(round0),
		roundOffset: uintptr(roundOffset),
	}
	compoundXHighBDNEONAsm(&ctx)
}

func predictInterCompoundRefHighBDToConvBufYResidentNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if round0 != compoundRound0Bits && round0 != compoundRound0Bits+2 {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	if !(width == 4 || (width >= 8 && width%8 == 0)) {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	k := kernel
	ctx := compoundFilterHighBDNEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[(refY-fo)*ref.Stride+refX*2],
		kernel:      &k[0],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		round0:      uintptr(round0),
		roundOffset: uintptr(roundOffset),
	}
	if width == 4 {
		compoundYHighBDNEONAsmW4(&ctx)
		return
	}
	compoundYHighBDNEONAsm(&ctx)
}

func predictInterCompoundRefHighBDToConvBuf2DResidentNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	// The horizontal slide loads one extra uint16 sample in the last 4-column
	// group. Keep exact resident/clamped behavior by falling back when that
	// single extra sample is not available.
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		predictInterCompoundRefHighBDToConvBuf2DResident(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	xk := xKernel
	yk := yKernel
	ctx := compound2DHighBDNEONCtx{
		dst:    &out[0],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+(refX-foX)*2],
		kernel: &yk[0],
		xKern:  &xk[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     &im[0],
		imStr:  uintptr(maxBlockSize),
		round0: uintptr(round0),
		xBias:  uintptr(1 << (bitDepth + filterBits - 1)),
		yBias:  uintptr(1 << offsetBits),
	}
	compound2DHighBDNEONAsm(&ctx)
}

// predictInterCompoundRefHighBDToConvBuf2DClampedNEON handles the
// edge-overhanging HBD compound 2D convolve. Following dav1d's reconstruction
// model (src/recon_tmpl.c mc(), src/mc_tmpl.c emu_edge_c), it materializes the
// clamped tap-window halo once (emuEdgeWindow16) and re-runs the resident NEON
// joint-convolve over the resident window. Bit-identical to the pure-Go
// per-tap-clamping reference (predictInterCompoundRefHighBDToConvBuf2DClamped).
// Only width%4 != 0 shapes, which do not occur for AV1 inter blocks, take
// pure-Go.
func predictInterCompoundRefHighBDToConvBuf2DClampedNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	if width < 4 || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	var edge emuEdge16Buf
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
	predictInterCompoundRefHighBDToConvBuf2DResidentNEON(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
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
		!(width == 4 || (width >= 8 && width%8 == 0)) {
		predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	if !planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		if scratch == nil {
			predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
			return
		}
		// dav1d src/recon_tmpl.c mc(): out-of-bounds references materialize
		// the clamped halo once (emu_edge) and rerun the plain kernel over
		// the resident window.
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		predictInterCompoundRef8ToConvBuf2DNEON(out, emu, emuX, emuY, width, height, xKernel, yKernel, offsetBits, scratch)
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

// blendCompoundAvg8NEON is the NEON binding for the 8-bit compound average /
// dist-wtd blend (av1_dist_wtd_convolve_* do_average branch); the kernel is a
// port of dav1d's avg/w_avg (src/mc_tmpl.c, src/arm/64/mc.S) adapted to
// libaom's offset CONV_BUF domain. The pure-Go reference computes
//
//	res = clip8(round((s0*fwd + s1*bck) >> DIST_PRECISION_BITS - roundOffset, roundBits))
//
// For roundBits == 4 (the only value reachable at 8-bit, where round0 == 3)
// this folds exactly to
//
//	res = clip8((s0*fwd + s1*bck - ((roundOffset - 8) << 4)) >> 8)
//
// using nested floor-division identities (the >>4 and the rounded >>4 combine
// into one arithmetic >>8 against a pre-shifted bias), so the asm is
// bit-identical to the scalar loop.
func blendCompoundAvg8NEON(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	if roundBits != 4 || roundOffset < 8 ||
		fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 ||
		height <= 0 {
		blendCompoundAvg8PureGo(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	ctx := compoundBlend8NEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		src0:   &src0[0],
		src1:   &src1[0],
		dstStr: uintptr(dst.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		fwd:    uintptr(fwdOffset),
		bck:    uintptr(bckOffset),
		bias:   uintptr((roundOffset - 8) << 4),
	}
	switch {
	case width >= 8 && width%8 == 0:
		blendCompoundAvg8NEONAsm(&ctx)
	case width == 4 && height%2 == 0:
		blendCompoundAvg8NEONAsmW4(&ctx)
	default:
		blendCompoundAvg8PureGo(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
	}
}

func init() {
	if cpu.Detected.NEON {
		blendCompoundAvg8Impl = blendCompoundAvg8NEON
		predictInterCompoundRef8ToConvBuf2DImpl = predictInterCompoundRef8ToConvBuf2DNEON
		predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyNEON
		predictInterCompoundRefHighBDToConvBufCopyResidentImpl = predictInterCompoundRefHighBDToConvBufCopyResidentNEON
		predictInterCompoundRefHighBDToConvBuf2DResidentImpl = predictInterCompoundRefHighBDToConvBuf2DResidentNEON
		predictInterCompoundRefHighBDToConvBuf2DClampedImpl = predictInterCompoundRefHighBDToConvBuf2DClampedNEON
		predictInterCompoundRefHighBDToConvBufXResidentImpl = predictInterCompoundRefHighBDToConvBufXResidentNEON
		predictInterCompoundRefHighBDToConvBufYResidentImpl = predictInterCompoundRefHighBDToConvBufYResidentNEON
		predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXNEON
		predictInterCompoundRef8ToConvBufYImpl = predictInterCompoundRef8ToConvBufYNEON
		if cpu.Detected.I8MM {
			predictInterCompoundRef8ToConvBuf2DImpl = predictInterCompoundRef8ToConvBuf2DI8MM
			predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXI8MM
			predictInterCompoundRef8ToConvBufYImpl = predictInterCompoundRef8ToConvBufYI8MM
		}
		if cpu.Detected.DOTPROD {
			predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXDotProd
		}
	}
}
