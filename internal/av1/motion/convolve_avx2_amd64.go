// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2-accelerated convolve kernels. These mirror the arm64 NEON variants slot
// for slot: the .s file implements the inner per-block loops for the common
// width-multiple shapes; the Go wrappers below resolve base pointers and route
// narrow (8-bit width<8, HBD width<4) or 4-tap blocks to the pure-Go reference,
// which keeps the asm simple and the byte-exactness contract easy to audit.
//
// Bit-exactness with the *PureGo references:
//   - Source bytes are zero-extended to int16 (== uint8 load).
//   - The 8-tap MAC is done with VPMADDWD over interleaved (tap_k,tap_k+1)
//     int16 pairs. VPMADDWD computes s0*t0 + s1*t1 into int32 with full
//     widening, so the int32 accumulator never overflows and the sum matches
//     the pure-Go int accumulation exactly.
//   - roundPowerOfTwo(v,b) = (v + (1<<(b-1))) >> b is reproduced by adding the
//     bias constant (VPADDD) then an arithmetic right shift (VPSRAD). For b==0
//     the pure-Go path is a no-op, matched by skipping the shift.
//   - clipPixel is VPACKSSDW (s32->s16 saturate) then VPACKUSWB (s16->u8
//     saturate) == clamp to [0,255]. The truncating int16() narrow used for the
//     2D intermediate is VPACKSSDW only? No: the intermediate uses a plain
//     16-bit wrap (int16()), reproduced by masking/packing without saturation.

// convolveAVX2Ctx is the asm calling context for the 8-bit kernels. Field order
// and sizes are part of the ABI shared with convolve_avx2_amd64.s; do not
// reorder.
type convolveAVX2Ctx struct {
	dst    *byte // first destination pixel
	ref    *byte // first reference pixel of the tap window
	kernel *int16
	xKern  *int16 // 2D only: horizontal kernel; kernel above is the vertical kernel
	dstStr uintptr
	refStr uintptr
	width  uintptr
	height uintptr
	im     *int16  // 2D only: int16 intermediate scratch
	imStr  uintptr // 2D only: intermediate row stride in int16 elements
}

func isFourTap(k [filterTaps]int16) bool {
	return k[0] == 0 && k[1] == 0 && k[6] == 0 && k[7] == 0
}

//go:noescape
func convolveX8AVX2Asm(ctx *convolveAVX2Ctx)

//go:noescape
func convolveY8AVX2Asm(ctx *convolveAVX2Ctx)

//go:noescape
func convolve2D8AVX2Asm(ctx *convolveAVX2Ctx)

func convolveX8AVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 8 || width%8 != 0 || isFourTap(kernel) {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	k := kernel
	ctx := convolveAVX2Ctx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[refY*ref.Stride+refX-fo],
		kernel: &k[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
	}
	convolveX8AVX2Asm(&ctx)
}

func convolveY8AVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 8 || width%8 != 0 || isFourTap(kernel) {
		convolveY8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	k := kernel
	ctx := convolveAVX2Ctx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[(refY-fo)*ref.Stride+refX],
		kernel: &k[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
	}
	convolveY8AVX2Asm(&ctx)
}

// convolve2DAVX2IMStride matches the imStride const in convolve2D8PureGo
// (maxBlockSize) so the vertical pass walks the same rows.
const convolve2DAVX2IMStride = maxBlockSize

func convolve2D8AVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8AVX2WithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8AVX2WithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width < 8 || width%8 != 0 {
		convolve2D8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	xk := xKernel
	yk := yKernel
	if scratch != nil {
		convolve2D8AVX2WithIM(dst, ref, dstX, dstY, refX, refY, width, height, xk, yk, &scratch.im[0], convolve2DAVX2IMStride)
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DAVX2IMStride]int16
	convolve2D8AVX2WithIM(dst, ref, dstX, dstY, refX, refY, width, height, xk, yk, &im[0], convolve2DAVX2IMStride)
}

func convolve2D8AVX2WithIM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xk [filterTaps]int16, yk [filterTaps]int16, im *int16, imStride int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	ctx := convolveAVX2Ctx{
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
	convolve2D8AVX2Asm(&ctx)
}

// The *ClampedAVX2 wrappers mirror the NEON clamped wrappers: when the whole
// tap window is resident the clamp is a no-op and the result is bit-identical
// to the non-clamped fast path, otherwise fall back to the per-tap-clamping
// pure-Go reference.

func convolveX8ClampedAVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 8 && width%8 == 0 && !isFourTap(kernel) &&
		planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps-1, height) {
		convolveX8AVX2(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveX8HorizontalEdgeAVX2(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveY8ClampedAVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 8 && width%8 == 0 && !isFourTap(kernel) &&
		planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		convolveY8AVX2(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveY8VerticalEdgeAVX2(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveX8HorizontalEdgeAVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
	if width < 8 || width%8 != 0 || isFourTap(kernel) {
		return false
	}
	if !planeRegionFits(ref, 1, 0, refY, ref.Width, height) {
		return false
	}
	fo := filterTaps/2 - 1
	xLo, xHi := clampedXInterior(refX-fo, filterTaps, ref.Width, width)
	if xHi <= xLo {
		return false
	}
	if xLo > 0 {
		convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, xLo, height, kernel)
	}
	start := xLo
	didAVX2 := false
	for start < xHi {
		remaining := xHi - start
		if remaining < 8 {
			break
		}
		chunk := remaining &^ 7
		convolveX8AVX2(dst, ref, dstX+start, dstY, refX+start, refY, chunk, height, kernel)
		start += chunk
		didAVX2 = true
	}
	if !didAVX2 {
		return false
	}
	if start < width {
		convolveX8ClampedPureGo(dst, ref, dstX+start, dstY, refX+start, refY, width-start, height, kernel)
	}
	return true
}

func convolveY8VerticalEdgeAVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
	if width < 8 || width%8 != 0 || isFourTap(kernel) {
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
	convolveY8AVX2(dst, ref, dstX, dstY+yLo, refX, refY+yLo, width, yHi-yLo, kernel)
	if yHi < height {
		convolveY8ClampedPureGo(dst, ref, dstX, dstY+yHi, refX, refY+yHi, width, height-yHi, kernel)
	}
	return true
}

func convolve2D8ClampedAVX2(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8ClampedAVX2WithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8ClampedAVX2WithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if width >= 8 && width%8 == 0 &&
		planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		convolve2D8AVX2WithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if convolve2D8ClampedEdgeSplitAVX2WithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch) {
		return
	}
	convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}

func convolve2D8ClampedEdgeSplitAVX2WithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) bool {
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
		if remaining < 8 {
			break
		}
		chunk := remaining &^ 7
		starts[n], widths[n] = start, chunk
		n++
		start += chunk
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
		convolve2D8AVX2WithScratch(dst, ref, dstX+start, dstY+yLo, refX+start, refY+yLo, chunk, midH, xKernel, yKernel, scratch)
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
