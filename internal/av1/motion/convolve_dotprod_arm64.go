// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// compoundX8DotProdCtx carries the resident lowbd compound-X DOTPROD kernel
// arguments. Field offsets are mirrored by convolve_dotprod_arm64.s.
type compoundX8DotProdCtx struct {
	dst        *uint16
	ref        *byte
	kernel     *int16
	permute    *byte
	refStr     uintptr
	width      uintptr
	height     uintptr
	correction uintptr
}

//go:noescape
func compoundX8DotProdAsm(ctx *compoundX8DotProdCtx)

// SVT ASM_NEON_DOTPROD/convolve_neon_dotprod.c: svt_kDotProdPermuteTbl.
var compoundX8DotProdPermute = [48]byte{
	0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
	4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
	8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14,
}

func predictInterCompoundRef8ToConvBufXDotProd(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	if !cpu.Detected.DOTPROD ||
		width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		predictInterCompoundRef8ToConvBufXNEON(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	k := kernel
	correction := ((128 << filterBits) + (roundOffset << compoundRound0Bits) + (1 << (compoundRound0Bits - 1))) / 2
	ctx := compoundX8DotProdCtx{
		dst:        &out[0],
		ref:        &ref.Pix[refY*ref.Stride+refX-fo],
		kernel:     &k[0],
		permute:    &compoundX8DotProdPermute[0],
		refStr:     uintptr(ref.Stride),
		width:      uintptr(width),
		height:     uintptr(height),
		correction: uintptr(correction),
	}
	compoundX8DotProdAsm(&ctx)
}
