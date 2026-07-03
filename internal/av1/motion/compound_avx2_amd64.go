// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2-accelerated 8-bit compound inter-prediction kernels. These mirror the
// arm64 NEON variants (compound_neon_arm64.go) slot for slot and are bit-exact
// with the *PureGo references in compound.go.
//
// dav1d reference: src/x86/mc_avx2.asm (the w_avg / avg blend and the 8tap
// prep/convbuf convolves). Unlike dav1d we keep libaom's offset CONV_BUF domain
// (each predictor is an un-rounded uint16 with roundOffset baked in), so the
// arithmetic matches the scalar loop rather than dav1d's PREP_BIAS int16 trick.
//
// Bit-exactness notes:
//   - Source bytes are zero-extended to int32 (VPMOVZXBD). The 8-tap MAC uses
//     VPMULLD by an int32-broadcast tap + VPADDD; int32 never overflows for the
//     8-tap subpel sums so the accumulator matches the pure-Go int sum exactly.
//     4-tap kernels (k0=k1=k6=k7=0) fall out of the same 8-tap MAC unchanged, so
//     no fourTap special case is needed.
//   - roundPowerOfTwo(v,b) = (v+(1<<(b-1)))>>b is VPADDD bias + arithmetic
//     VPSRAD b, matching Go's signed >>.
//   - The CONV_BUF store is a *truncating* uint16 narrow (Go uint16(), mod 2^16,
//     not saturating): VPSHUFB gathers the low word of each int32 lane then a
//     VPERMQ $0xD8 packs the 8 valid words for a 16-byte store.
//   - The blend writes final pixels, so it clips with VPACKSSDW (s32->s16) then
//     VPACKUSWB (s16->u8) == clamp to [0,255].
//
// The asm processes 8 output columns per group and requires width%8==0; other
// shapes (including width==4) route to the pure-Go reference, which keeps the
// asm simple and the byte-exactness contract easy to audit.

// compoundConvAVX2Ctx is the asm calling context for the convbuf convolves.
// Field order/sizes are part of the ABI shared with compound_avx2_amd64.s and
// compound_highbd_avx2_amd64.s; do not reorder.
type compoundConvAVX2Ctx struct {
	dst    *uint16 // first CONV_BUF output sample (row-major, width-strided)
	ref    *byte   // first reference sample of the tap window
	kernel *int16  // vertical kernel (Y/2D) or single kernel (X)
	xKern  *int16  // 2D only: horizontal kernel
	refStr uintptr
	width  uintptr
	height uintptr
	im     *int32  // 2D only: int32 intermediate scratch
	imStr  uintptr // 2D only: intermediate row stride in int32 elements
	round0 uintptr // round-0 shift bits (HBD; 8-bit is fixed in asm)
	scale  uintptr // vertical/copy multiplicative scale
	rndOff uintptr // roundOffset added to each CONV_BUF sample
	xBias  uintptr // 2D only: 1 << (bd + FILTER_BITS - 1)
	yBias  uintptr // 2D only: 1 << offsetBits
}

// compoundBlendAVX2Ctx is the asm calling context for the 8-bit compound blend.
type compoundBlendAVX2Ctx struct {
	dst    *byte // first destination pixel
	src0   *uint16
	src1   *uint16
	dstStr uintptr
	width  uintptr
	height uintptr
	fwd    uintptr
	bck    uintptr
	rndOff uintptr
}

//go:noescape
func blendCompoundAvg8AVX2Asm(ctx *compoundBlendAVX2Ctx)

//go:noescape
func compoundCopy8AVX2Asm(ctx *compoundConvAVX2Ctx)

//go:noescape
func compoundX8AVX2Asm(ctx *compoundConvAVX2Ctx)

//go:noescape
func compoundY8AVX2Asm(ctx *compoundConvAVX2Ctx)

//go:noescape
func compound2D8AVX2Asm(ctx *compoundConvAVX2Ctx)

func blendCompoundAvg8AVX2(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	if roundBits != 4 || width < 8 || width%8 != 0 || height <= 0 {
		blendCompoundAvg8PureGo(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	ctx := compoundBlendAVX2Ctx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		src0:   &src0[0],
		src1:   &src1[0],
		dstStr: uintptr(dst.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		fwd:    uintptr(fwdOffset),
		bck:    uintptr(bckOffset),
		rndOff: uintptr(roundOffset),
	}
	blendCompoundAvg8AVX2Asm(&ctx)
}

func predictInterCompoundRef8ToConvBufCopyAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if round0 != compoundRound0Bits || width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX, refY, width, height) {
		predictInterCompoundRef8ToConvBufCopyPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	bits := 2*filterBits - compoundRound1Bits - round0
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[refY*ref.Stride+refX],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		scale:  uintptr(1 << bits),
		rndOff: uintptr(roundOffset),
	}
	compoundCopy8AVX2Asm(&ctx)
}

func predictInterCompoundRef8ToConvBufXAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	if width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps-1, height) {
		predictInterCompoundRef8ToConvBufXPureGo(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	k := kernel
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[refY*ref.Stride+refX-fo],
		kernel: &k[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		rndOff: uintptr(roundOffset),
	}
	compoundX8AVX2Asm(&ctx)
}

func predictInterCompoundRef8ToConvBufYAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if round0 != compoundRound0Bits || width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		predictInterCompoundRef8ToConvBufYPureGo(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	bits := filterBits - round0
	k := kernel
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[(refY-fo)*ref.Stride+refX],
		kernel: &k[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		scale:  uintptr(1 << bits),
		rndOff: uintptr(roundOffset),
	}
	compoundY8AVX2Asm(&ctx)
}

func predictInterCompoundRef8ToConvBuf2DAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, scratch *CompoundConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if offsetBits != 19 || width < 8 || width%8 != 0 {
		predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	if !planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		if scratch == nil {
			predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
			return
		}
		// dav1d src/recon_tmpl.c mc(): out-of-bounds references materialize the
		// clamped halo once (emu_edge) and rerun the resident kernel over it.
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		predictInterCompoundRef8ToConvBuf2DAVX2(out, emu, emuX, emuY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	if scratch != nil {
		predictInterCompoundRef8ToConvBuf2DAVX2WithIM(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, &scratch.im)
		return
	}
	var im compoundIM
	predictInterCompoundRef8ToConvBuf2DAVX2WithIM(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, &im)
}

func predictInterCompoundRef8ToConvBuf2DAVX2WithIM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, im *compoundIM) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	xk := xKernel
	yk := yKernel
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+refX-foX],
		kernel: &yk[0],
		xKern:  &xk[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     &im[0],
		imStr:  uintptr(maxBlockSize),
		xBias:  uintptr(1 << (8 + filterBits - 1)),
		yBias:  uintptr(1 << offsetBits),
	}
	compound2D8AVX2Asm(&ctx)
}

func init() {
	if cpu.Detected.AVX2 {
		blendCompoundAvg8Impl = blendCompoundAvg8AVX2
		predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyAVX2
		predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXAVX2
		predictInterCompoundRef8ToConvBufYImpl = predictInterCompoundRef8ToConvBufYAVX2
		predictInterCompoundRef8ToConvBuf2DImpl = predictInterCompoundRef8ToConvBuf2DAVX2
	}
}
