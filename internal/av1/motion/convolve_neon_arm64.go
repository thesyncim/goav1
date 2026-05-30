// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// NEON-accelerated 8-bit convolve kernels. The .s file implements the inner
// per-block loops for widths that are a multiple of 8 (i.e. every AV1 block
// width >= 8) plus dedicated 4-lane kernels for width-4 blocks (4xN luma and
// every 4:2:0 chroma block of an 8x8 luma block). The Go wrappers below resolve
// base pointers and route any remaining narrow / 4-tap shapes the asm does not
// cover to the pure-Go reference, keeping the byte-exactness contract auditable.
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
	im     *int16  // 2D only: int16 intermediate scratch (imH rows of imStride)
	imStr  uintptr // 2D only: intermediate row stride in int16 elements
}

//go:noescape
func convolveX8NEONAsm(ctx *convolveNEONCtx)

//go:noescape
func convolveY8NEONAsm(ctx *convolveNEONCtx)

//go:noescape
func convolve2D8NEONAsm(ctx *convolveNEONCtx)

// Width-4 variants for AV1's narrowest inter blocks (4xN luma and every 4:2:0
// chroma block of an 8x8 luma block). They compute 4 output pixels per row with
// the low halves of the NEON registers and are bit-exact with the pure-Go
// reference. width is implicitly 4; the asm ignores the width field.

//go:noescape
func convolveX8NEONAsmW4(ctx *convolveNEONCtx)

//go:noescape
func convolveY8NEONAsmW4(ctx *convolveNEONCtx)

//go:noescape
func convolve2D8NEONAsmW4(ctx *convolveNEONCtx)

func isFourTap(k [filterTaps]int16) bool {
	return k[0] == 0 && k[1] == 0 && k[6] == 0 && k[7] == 0
}

func convolveX8NEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	// width==4 uses the dedicated 4-lane kernel (it runs the full 8-tap MAC, so
	// 4-tap kernels are handled directly). Other narrow / non-multiple-of-8
	// widths fall back to pure-Go.
	if !(width == 4 || (width >= 8 && width%8 == 0)) {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 8 && isFourTap(kernel) {
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
	if width == 4 {
		convolveX8NEONAsmW4(&ctx)
		return
	}
	convolveX8NEONAsm(&ctx)
}

func convolveY8NEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if !(width == 4 || (width >= 8 && width%8 == 0)) {
		convolveY8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 8 && isFourTap(kernel) {
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
	if width == 4 {
		convolveY8NEONAsmW4(&ctx)
		return
	}
	convolveY8NEONAsm(&ctx)
}

// convolve2DNEONIMStride is the int16 element stride of the intermediate buffer.
// It must match the imStride const in convolve2D8PureGo (maxBlockSize) so the
// vertical pass walks the same rows.
const convolve2DNEONIMStride = maxBlockSize

// convolve2D8NEON is the both-axes-fractional 8-bit inter-prediction convolve.
// It matches convolve2D8PureGo bit-for-bit: a horizontal 8-tap pass produces an
// int16 intermediate (rounded by round0Bits with the 1<<(8+filterBits-1) bias),
// then a vertical 8-tap pass over the intermediate applies round1Bits, the
// roundOffset and the [0,255] clip. Only width<8 (or non-multiple-of-8) blocks
// fall back to the pure-Go reference; the asm always runs the full 8-tap MAC,
// so 4-tap kernels (which merely zero the end taps) are handled directly and
// stay bit-exact because the zeroed taps contribute nothing to the sum.
func convolve2D8NEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	xk := xKernel
	yk := yKernel
	if width == 4 {
		// Width-4 uses a tightly-sized intermediate (stride 4) so the per-call
		// zeroing of the scratch is ~1KB rather than the full 135KB the width>=8
		// path needs. The asm walks the rows at imStr int16 elements.
		const w4Stride = 4
		var im [(maxBlockSize + filterTaps - 1) * w4Stride]int16
		ctx := convolveNEONCtx{
			dst:    &dst.Pix[dstY*dst.Stride+dstX],
			ref:    &ref.Pix[(refY-foY)*ref.Stride+refX-foX],
			kernel: &yk[0],
			xKern:  &xk[0],
			dstStr: uintptr(dst.Stride),
			refStr: uintptr(ref.Stride),
			width:  uintptr(width),
			height: uintptr(height),
			im:     &im[0],
			imStr:  uintptr(w4Stride),
		}
		convolve2D8NEONAsmW4(&ctx)
		return
	}
	if !(width >= 8 && width%8 == 0) {
		convolve2D8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	// Intermediate buffer: (height+filterTaps-1) rows of imStride int16. The
	// array is the same size as convolve2D8PureGo's stack im and does not escape.
	var im [(maxBlockSize + filterTaps - 1) * convolve2DNEONIMStride]int16
	ctx := convolveNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+refX-foX],
		kernel: &yk[0],
		xKern:  &xk[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     &im[0],
		imStr:  uintptr(convolve2DNEONIMStride),
	}
	convolve2D8NEONAsm(&ctx)
}

// The *ClampedNEON wrappers handle the edge-block convolves. When the whole tap
// window happens to lie inside the reference plane the clamp is a no-op and the
// result is bit-identical to the non-clamped fast path, so we route to the NEON
// asm. Otherwise (a tap coordinate genuinely falls off the frame) we fall back
// to the pure-Go clamped reference, which clamps each tap coordinate. This keeps
// the asm free of per-tap coordinate clamping while still accelerating the
// common case where an "edge" block's halo is actually resident.

func convolveX8ClampedNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// The width-4 horizontal kernel reads one byte past the resident tap window
	// (it loads bytes 8..11 to fill the slide register), so widen the fit check
	// by 1 sample for width 4 to keep that load resident.
	haloW := width + filterTaps - 1
	if width == 4 {
		haloW++
	}
	if neonWidth && !(width >= 8 && isFourTap(kernel)) &&
		planeRegionFits(ref, 1, refX-fo, refY, haloW, height) {
		convolveX8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveY8ClampedNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if neonWidth && !(width >= 8 && isFourTap(kernel)) &&
		planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		convolveY8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolve2D8ClampedNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// The width-4 horizontal pass reads one byte past the resident window; widen
	// the column halo by 1 sample for width 4 so that load stays resident.
	haloW := width + filterTaps - 1
	if width == 4 {
		haloW++
	}
	if neonWidth &&
		planeRegionFits(ref, 1, refX-foX, refY-foY, haloW, height+filterTaps-1) {
		convolve2D8NEON(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	convolve2D8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
}
