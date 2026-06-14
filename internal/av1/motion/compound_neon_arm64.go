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

type compoundX8NEONCtx struct {
	dst         *uint16
	ref         *byte
	kernel      *int16
	refStr      uintptr
	width       uintptr
	height      uintptr
	roundOffset uintptr
}

//go:noescape
func compoundCopy8NEONAsm(ctx *compoundCopy8NEONCtx)

//go:noescape
func compoundX8NEONAsm(ctx *compoundX8NEONCtx)

func predictInterCompoundRef8ToConvBufCopyNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if round0 != compoundRound0Bits ||
		width < 8 || width%8 != 0 ||
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
	compoundCopy8NEONAsm(&ctx)
}

func predictInterCompoundRef8ToConvBufXNEON(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	// The horizontal asm loads one extra byte in the last 8-column group while
	// forming the slide register. Require that byte to be resident too.
	if width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		predictInterCompoundRef8ToConvBufXPureGo(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	k := kernel
	ctx := compoundX8NEONCtx{
		dst:         &out[0],
		ref:         &ref.Pix[refY*ref.Stride+refX-fo],
		kernel:      &k[0],
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		roundOffset: uintptr(roundOffset),
	}
	compoundX8NEONAsm(&ctx)
}

func init() {
	if cpu.Detected.NEON {
		predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyNEON
		predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXNEON
	}
}
