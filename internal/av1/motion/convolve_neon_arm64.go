// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// NEON-accelerated 8-bit convolve kernels. The .s file implements the inner
// per-block loops for widths that are a multiple of 8 (i.e. every AV1 block
// width >= 8). The Go wrappers below resolve base pointers and route narrow
// (width 4) or 4-tap blocks to the pure-Go reference, which keeps the asm
// simple and the byte-exactness contract easy to audit.
//
// The asm matches the pure-Go reference bit-for-bit: NEON SRSHR performs
// (v + (1<<(shift-1))) >> shift with a signed shift, identical to
// roundPowerOfTwo; SQXTN+SQXTUN clamp to [0,255] identical to clipPixel.

// convolveNEONCtx is the asm calling context. Field order and sizes are part of
// the ABI shared with convolve_neon_arm64.s; do not reorder.
type convolveNEONCtx struct {
	dst    *byte // first destination pixel
	ref    *byte // first reference pixel of the tap window
	kernel *int16
	xKern  *int16 // 2D only: horizontal kernel; kernel above is the vertical kernel
	dstStr uintptr
	refStr uintptr
	width  uintptr
	height uintptr
}

//go:noescape
func convolveX8NEONAsm(ctx *convolveNEONCtx)

//go:noescape
func convolveY8NEONAsm(ctx *convolveNEONCtx)

func isFourTap(k [filterTaps]int16) bool {
	return k[0] == 0 && k[1] == 0 && k[6] == 0 && k[7] == 0
}

func convolveX8NEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 8 || width%8 != 0 || isFourTap(kernel) {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	k := kernel
	ctx := convolveNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[refY*ref.Stride+refX-fo],
		kernel: &k[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
	}
	convolveX8NEONAsm(&ctx)
}

func convolveY8NEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 8 || width%8 != 0 || isFourTap(kernel) {
		convolveY8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	k := kernel
	ctx := convolveNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[(refY-fo)*ref.Stride+refX],
		kernel: &k[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
	}
	convolveY8NEONAsm(&ctx)
}
