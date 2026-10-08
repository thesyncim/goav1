// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// init binds the high-bit-depth compound predictors to the Go SIMD kernels.
func init() {
	if !hbdSIMDAvailable() {
		return
	}
	blendCompoundAvgHighBDImpl = blendCompoundAvgHighBDGoSIMD
	predictInterCompoundRefHighBDToConvBufCopyResidentImpl = predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD
	predictInterCompoundRefHighBDToConvBuf2DResidentImpl = predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD
	predictInterCompoundRefHighBDToConvBuf2DClampedImpl = predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD
	predictInterCompoundRefHighBDToConvBufXResidentImpl = predictInterCompoundRefHighBDToConvBufXResidentGoSIMD
	predictInterCompoundRefHighBDToConvBufYResidentImpl = predictInterCompoundRefHighBDToConvBufYResidentGoSIMD
}

// compoundCopyHighBDGoSIMDCtx is the calling context of the unfiltered HBD
// compound copy. ref starts at the first sample and out is the CONV_BUF.
type compoundCopyHighBDGoSIMDCtx struct {
	out         []uint16
	ref         []byte
	refStr      int
	width       int
	height      int
	roundOffset int32
}

// compoundFilterHighBDGoSIMDCtx is the calling context of the one-dimensional
// HBD compound kernels. ref starts at the first sample of the trimmed tap span
// and kernel holds the taps zero-padded to filterTaps; taps counts the span.
type compoundFilterHighBDGoSIMDCtx struct {
	out         []uint16
	ref         []byte
	kernel      [filterTaps]int16
	refStr      int
	width       int
	height      int
	round0      int
	roundOffset int32
	taps        int
}

// compound2DHighBDGoSIMDCtx is the calling context of the separable 2D HBD
// compound kernel. ref starts at the first sample of the trimmed X span and the
// kernels hold their trimmed taps zero-padded to filterTaps.
type compound2DHighBDGoSIMDCtx struct {
	out    []uint16
	ref    []byte
	kernel [filterTaps]int16
	xKern  [filterTaps]int16
	refStr int
	width  int
	height int
	im     []int32
	imStr  int
	round0 int
	xBias  int32
	yBias  int32
	tapsX  int
	tapsY  int
}

// compoundBlendHighBDGoSIMDCtx is the calling context of the HBD distance
// weighted blend.
type compoundBlendHighBDGoSIMDCtx struct {
	dst    []byte // first destination sample (2 bytes/sample)
	src0   []uint16
	src1   []uint16
	dstStr int // destination stride in bytes
	width  int
	height int
	fwd    int
	bck    int
	maxVal int32 // clip upper bound (1<<bd)-1
}

func predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 2, refX, refY, width, height) {
		predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	ctx := compoundCopyHighBDGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[refY*ref.Stride+refX*2:],
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		roundOffset: int32(roundOffset),
	}
	compoundCopyHighBDKernel(&ctx, filterBits-round0)
}

func predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	lo, n := hbdTapSpan(&kernel)
	ctx := compoundFilterHighBDGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[refY*ref.Stride+(refX-fo+lo)*2:],
		kernel:      hbdTrimTaps(&kernel, lo, n),
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		round0:      round0,
		roundOffset: int32(roundOffset),
		taps:        n,
	}
	compoundXHighBDKernel(&ctx)
}

func predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	lo, n := hbdTapSpan(&kernel)
	ctx := compoundFilterHighBDGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[(refY-fo+lo)*ref.Stride+refX*2:],
		kernel:      hbdTrimTaps(&kernel, lo, n),
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		round0:      round0,
		roundOffset: int32(roundOffset),
		taps:        n,
	}
	compoundYHighBDKernel(&ctx)
}

func predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		predictInterCompoundRefHighBDToConvBuf2DResident(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	ctx := compound2DHighBDGoSIMDCtx{
		out:    out,
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX)*2:],
		kernel: hbdTrimTaps(&yKernel, loY, nY),
		xKern:  hbdTrimTaps(&xKernel, loX, nX),
		refStr: ref.Stride,
		width:  width,
		height: height,
		im:     im[:],
		imStr:  maxBlockSize,
		round0: round0,
		xBias:  int32(1 << (bitDepth + filterBits - 1)),
		yBias:  int32(1 << offsetBits),
		tapsX:  nX,
		tapsY:  nY,
	}
	compound2DHighBDKernel(&ctx)
}

// predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD handles the
// edge-overhanging HBD compound 2D convolve. Following dav1d's reconstruction
// model (src/recon_tmpl.c mc(), src/mc_tmpl.c emu_edge_c), it materializes the
// clamped tap-window halo once (emuEdgeWindow16) and re-runs the resident Go
// SIMD joint-convolve over the resident window. Bit-identical to the pure-Go
// per-tap-clamping reference (predictInterCompoundRefHighBDToConvBuf2DClamped).
// Only width%4 != 0 shapes, which do not occur for AV1 inter blocks, take
// pure-Go. edge optionally carries the caller-owned halo window so the ~38KB
// buffer is not zero-filled per call; nil keeps per-call stack storage.
func predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM, edge *emuEdge16Buf) {
	if width < 4 || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	if edge != nil {
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, edge)
		predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	var stackEdge emuEdge16Buf
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &stackEdge)
	predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
}

// blendCompoundAvgHighBDGoSIMD is the HBD distance-weighted compound average.
// The scalar reference is blendCompoundAvgHighBD.
func blendCompoundAvgHighBDGoSIMD(dst frame.Plane, src0 []uint16, src1 []uint16, max uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	if (roundBits != 4 && roundBits != 2) || roundOffset <= 0 ||
		fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 ||
		height <= 0 || width%4 != 0 {
		blendCompoundAvgHighBD(dst, src0, src1, max, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	ctx := compoundBlendHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		src0:   src0,
		src1:   src1,
		dstStr: dst.Stride,
		width:  width,
		height: height,
		fwd:    fwdOffset,
		bck:    bckOffset,
		maxVal: int32(max),
	}
	blendCompoundAvgHighBDKernel(&ctx, roundOffset, roundBits)
}
