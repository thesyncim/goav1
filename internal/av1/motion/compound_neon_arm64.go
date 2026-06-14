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

//go:noescape
func compoundCopy8NEONAsm(ctx *compoundCopy8NEONCtx)

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

func init() {
	if cpu.Detected.NEON {
		predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyNEON
	}
}
