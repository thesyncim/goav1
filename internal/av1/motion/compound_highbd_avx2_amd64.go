// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2-accelerated high-bit-depth (10/12-bit) compound inter-prediction
// kernels, bit-exact with the *HighBD*PureGo references in compound.go and
// mirroring the arm64 NEON dispatch slots.
//
// dav1d reference: src/x86/mc_avx2.asm (the highbd prep/convbuf 8tap path).
// Samples are little-endian uint16. round0 is bit-depth dependent (3 for
// bd<=10, 5 for bd==12), so it and the derived round-0 bias are passed in via
// the context and applied with a register-count VPSRAD. Each kernel processes
// 4 output columns per group into a 128-bit X register (4 int32 lanes).
//
// The CONV_BUF store is a *truncating* uint16 narrow (Go uint16(), mod 2^16):
// HBD compound values can exceed 2^16, so VPSHUFB gathers the low word of each
// int32 lane and VMOVQ stores 4 contiguous uint16 (never a saturating pack).

//go:noescape
func compoundCopyHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)

//go:noescape
func compoundXHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)

//go:noescape
func compoundYHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)

//go:noescape
func compound2DHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)

func predictInterCompoundRefHighBDToConvBufCopyResidentAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX, refY, width, height) {
		predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	bits := 2*filterBits - compoundRound1Bits - round0
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[refY*ref.Stride+refX*2],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		scale:  uintptr(1 << bits),
		rndOff: uintptr(roundOffset),
	}
	compoundCopyHighBDAVX2Asm(&ctx)
}

func predictInterCompoundRefHighBDToConvBufXResidentAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX-fo, refY, width+filterTaps-1, height) {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	bits := filterBits - compoundRound1Bits
	k := kernel
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[refY*ref.Stride+(refX-fo)*2],
		kernel: &k[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		round0: uintptr(round0),
		scale:  uintptr(1 << bits),
		rndOff: uintptr(roundOffset),
	}
	compoundXHighBDAVX2Asm(&ctx)
}

func predictInterCompoundRefHighBDToConvBufYResidentAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX, refY-fo, width, height+filterTaps-1) {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	bits := filterBits - round0
	k := kernel
	ctx := compoundConvAVX2Ctx{
		dst:    &out[0],
		ref:    &ref.Pix[(refY-fo)*ref.Stride+refX*2],
		kernel: &k[0],
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		scale:  uintptr(1 << bits),
		rndOff: uintptr(roundOffset),
	}
	compoundYHighBDAVX2Asm(&ctx)
}

func predictInterCompoundRefHighBDToConvBuf2DResidentAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		predictInterCompoundRefHighBDToConvBuf2DResident(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	xk := xKernel
	yk := yKernel
	ctx := compoundConvAVX2Ctx{
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
	compound2DHighBDAVX2Asm(&ctx)
}

// predictInterCompoundRefHighBDToConvBuf2DClampedAVX2 mirrors the NEON clamped
// path: materialize the clamped tap-window halo once (emu_edge, dav1d
// src/mc_tmpl.c emu_edge_c) and rerun the resident kernel over it. Shapes the
// asm cannot handle (width%4 != 0) route to the per-tap-clamping pure-Go
// reference.
// edge optionally carries the caller-owned halo window so the ~38KB buffer is
// not zero-filled per call; nil keeps per-call stack storage.
func predictInterCompoundRefHighBDToConvBuf2DClampedAVX2(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM, edge *emuEdge16Buf) {
	if width < 4 || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	if edge != nil {
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, edge)
		predictInterCompoundRefHighBDToConvBuf2DResidentAVX2(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	var stackEdge emuEdge16Buf
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &stackEdge)
	predictInterCompoundRefHighBDToConvBuf2DResidentAVX2(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
}

func init() {
	if cpu.Detected.AVX2 {
		predictInterCompoundRefHighBDToConvBufCopyResidentImpl = predictInterCompoundRefHighBDToConvBufCopyResidentAVX2
		predictInterCompoundRefHighBDToConvBufXResidentImpl = predictInterCompoundRefHighBDToConvBufXResidentAVX2
		predictInterCompoundRefHighBDToConvBufYResidentImpl = predictInterCompoundRefHighBDToConvBufYResidentAVX2
		predictInterCompoundRefHighBDToConvBuf2DResidentImpl = predictInterCompoundRefHighBDToConvBuf2DResidentAVX2
		predictInterCompoundRefHighBDToConvBuf2DClampedImpl = predictInterCompoundRefHighBDToConvBuf2DClampedAVX2
	}
}
