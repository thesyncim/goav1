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
// base pointers and route any remaining narrow shapes the asm does not cover to
// the pure-Go reference, keeping the byte-exactness contract auditable.
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

func convolveX8NEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	// width==4 uses the dedicated 4-lane kernel. Other narrow /
	// non-multiple-of-8 widths fall back to pure-Go.
	if !(width == 4 || (width >= 8 && width%8 == 0)) {
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
	convolve2D8NEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8NEONWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	xk := xKernel
	yk := yKernel
	if width == 4 {
		// Width-4 uses a tightly-sized intermediate (stride 4) so the per-call
		// zeroing of the scratch is ~1KB rather than the full 135KB the width>=8
		// path needs. The asm walks the rows at imStr int16 elements.
		const w4Stride = 4
		if scratch != nil {
			convolve2D8NEONWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xk, yk, &scratch.im[0], w4Stride)
			return
		}
		var im [(maxBlockSize + filterTaps - 1) * w4Stride]int16
		convolve2D8NEONWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xk, yk, &im[0], w4Stride)
		return
	}
	if !(width >= 8 && width%8 == 0) {
		convolve2D8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	// Intermediate buffer: (height+filterTaps-1) rows of imStride int16. The
	// array is the same size as convolve2D8PureGo's stack im and does not escape.
	if scratch != nil {
		convolve2D8NEONWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xk, yk, &scratch.im[0], convolve2DNEONIMStride)
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DNEONIMStride]int16
	convolve2D8NEONWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xk, yk, &im[0], convolve2DNEONIMStride)
}

func convolve2D8NEONWithIM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xk [filterTaps]int16, yk [filterTaps]int16, im *int16, imStride int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	ctx := convolveNEONCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[(refY-foY)*ref.Stride+refX-foX],
		kernel: &yk[0],
		xKern:  &xk[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
		im:     im,
		imStr:  uintptr(imStride),
	}
	if width == 4 {
		convolve2D8NEONAsmW4(&ctx)
		return
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
	// Both the width>=8 and width-4 horizontal kernels physically load one sample
	// past the consumed tap window: the width>=8 loop does a 16-byte ld1 for its
	// last 8-column group, and the width-4 kernel loads bytes 8..11 to fill the
	// slide register. That trailing sample never feeds an output, but it must be
	// resident, so reserve width+filterTaps samples in the fit check.
	haloW := width + filterTaps
	if neonWidth && planeRegionFits(ref, 1, refX-fo, refY, haloW, height) {
		convolveX8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveX8HorizontalEdgeNEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveY8ClampedNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if neonWidth && planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		convolveY8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveY8VerticalEdgeNEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

// convolveX8ClampedNEONWithScratch is convolveX8ClampedNEON with a dav1d-style
// emu_edge fast path (src/recon_tmpl.c mc()): when the tap window is not
// resident, the clamped halo is materialized once into the caller scratch and
// the plain X kernel runs over it.
func convolveX8ClampedNEONWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16, scratch *ConvolveScratch) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if scratch != nil && neonWidth {
		// The horizontal kernels load one sample past the consumed tap window
		// (16-byte ld1 for width>=8; bytes 8..11 for width-4); reserve
		// width+filterTaps so that trailing sample stays resident.
		haloW := width + filterTaps
		if !planeRegionFits(ref, 1, refX-fo, refY, haloW, height) {
			emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
			convolveX8Impl(dst, emu, dstX, dstY, emuX, emuY, width, height, kernel)
			return
		}
	}
	convolveX8ClampedNEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

// convolveY8ClampedNEONWithScratch is convolveY8ClampedNEON with the same
// emu_edge fast path as convolveX8ClampedNEONWithScratch.
func convolveY8ClampedNEONWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16, scratch *ConvolveScratch) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if scratch != nil && neonWidth &&
		!planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		convolveY8Impl(dst, emu, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveY8ClampedNEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveX8HorizontalEdgeNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
	fo := filterTaps/2 - 1
	if !planeRegionFits(ref, 1, 0, refY, ref.Width, height) {
		return false
	}
	xLo, xHi := clampedXInterior(refX-fo, filterTaps, ref.Width, width)
	if xHi <= xLo {
		return false
	}
	if xLo > 0 {
		convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, xLo, height, kernel)
	}
	start := xLo
	didNEON := false
	for start < xHi {
		remaining := xHi - start
		if remaining >= 8 {
			chunk := remaining &^ 7
			// clampedXInterior only guarantees the consumed taps are resident;
			// the last 8-column group's 16-byte load reaches one sample further
			// (chunk+filterTaps in total). Shrink the NEON span by whole groups
			// until that physical reach is in-bounds so the asm never over-reads;
			// any dropped columns fall to the scalar tail below.
			for chunk >= 8 && !planeRegionFits(ref, 1, refX+start-fo, refY, chunk+filterTaps, height) {
				chunk -= 8
			}
			if chunk < 8 {
				break
			}
			convolveX8NEON(dst, ref, dstX+start, dstY, refX+start, refY, chunk, height, kernel)
			start += chunk
			didNEON = true
			continue
		}
		if remaining >= 4 {
			if !planeRegionFits(ref, 1, refX+start-fo, refY, filterTaps+4, height) {
				break
			}
			convolveX8NEON(dst, ref, dstX+start, dstY, refX+start, refY, 4, height, kernel)
			start += 4
			didNEON = true
			continue
		}
		break
	}
	if !didNEON {
		return false
	}
	if start < xHi {
		convolveX8ClampedPureGo(dst, ref, dstX+start, dstY, refX+start, refY, xHi-start, height, kernel)
	}
	if xHi < width {
		convolveX8ClampedPureGo(dst, ref, dstX+xHi, dstY, refX+xHi, refY, width-xHi, height, kernel)
	}
	return true
}

func convolveY8VerticalEdgeNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
	if !(width == 4 || (width >= 8 && width%8 == 0)) {
		return false
	}
	if !planeRegionFits(ref, 1, refX, 0, width, ref.Height) {
		return false
	}
	fo := filterTaps/2 - 1
	yLo, yHi := clampedXInterior(refY-fo, filterTaps, ref.Height, height)
	if yHi <= yLo {
		return false
	}
	if yLo > 0 {
		convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, yLo, kernel)
	}
	convolveY8NEON(dst, ref, dstX, dstY+yLo, refX, refY+yLo, width, yHi-yLo, kernel)
	if yHi < height {
		convolveY8ClampedPureGo(dst, ref, dstX, dstY+yHi, refX, refY+yHi, width, height-yHi, kernel)
	}
	return true
}

func convolve2D8ClampedNEON(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8ClampedNEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8ClampedNEONWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// The horizontal pass loads one sample past the consumed tap window (16-byte
	// ld1 for width>=8; bytes 8..11 for width-4); reserve width+filterTaps
	// columns so that trailing sample stays resident.
	haloW := width + filterTaps
	if neonWidth &&
		planeRegionFits(ref, 1, refX-foX, refY-foY, haloW, height+filterTaps-1) {
		convolve2D8NEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if neonWidth && scratch != nil {
		// dav1d src/recon_tmpl.c mc(): out-of-bounds references materialize
		// the clamped halo once (emu_edge) and run the plain 8tap kernel over
		// the resident window.
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		convolve2D8NEONWithScratch(dst, emu, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, scratch)
		return
	}
	if convolve2D8ClampedEdgeSplitNEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch) {
		return
	}
	convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}

func convolve2D8ClampedEdgeSplitNEONWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) bool {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	xLo, xHi := clampedXInterior(refX-foX, filterTaps, ref.Width, width)
	yLo, yHi := clampedXInterior(refY-foY, filterTaps, ref.Height, height)
	if xHi <= xLo || yHi <= yLo {
		return false
	}

	var starts [16]int
	var widths [16]int
	n := 0
	for start := xLo; start < xHi && n < len(starts); {
		remaining := xHi - start
		if remaining >= 8 {
			chunk := remaining &^ 7
			// The 2D horizontal pass loads 16 bytes for its last 8-column group,
			// reaching chunk+filterTaps source samples across the midH+filterTaps-1
			// halo rows. Shrink by whole groups until that physical reach is
			// resident so the asm never over-reads; dropped columns go to the
			// scalar tail below.
			for chunk >= 8 && !planeRegionFits(ref, 1, refX+start-foX, refY+yLo-foY, chunk+filterTaps, yHi-yLo+filterTaps-1) {
				chunk -= 8
			}
			if chunk < 8 {
				break
			}
			starts[n], widths[n] = start, chunk
			n++
			start += chunk
			continue
		}
		if remaining >= 4 {
			const w4Halo = 4 + filterTaps
			if !planeRegionFits(ref, 1, refX+start-foX, refY+yLo-foY, w4Halo, yHi-yLo+filterTaps-1) {
				break
			}
			starts[n], widths[n] = start, 4
			n++
			start += 4
			continue
		}
		break
	}
	if n == 0 {
		return false
	}

	if yLo > 0 {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, yLo, xKernel, yKernel, scratch)
	}
	midH := yHi - yLo
	if xLo > 0 {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY+yLo, refX, refY+yLo, xLo, midH, xKernel, yKernel, scratch)
	}
	coveredHi := xLo
	for i := 0; i < n; i++ {
		start := starts[i]
		chunk := widths[i]
		convolve2D8NEONWithScratch(dst, ref, dstX+start, dstY+yLo, refX+start, refY+yLo, chunk, midH, xKernel, yKernel, scratch)
		coveredHi = start + chunk
	}
	if coveredHi < width {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX+coveredHi, dstY+yLo, refX+coveredHi, refY+yLo, width-coveredHi, midH, xKernel, yKernel, scratch)
	}
	if yHi < height {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY+yHi, refX, refY+yHi, width, height-yHi, xKernel, yKernel, scratch)
	}
	return true
}
