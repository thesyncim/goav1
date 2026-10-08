// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Go-native SIMD 8-bit single-prediction convolves (X, Y and 2D, plus the
// clamped and emulated-edge wrappers). The kernels compute the scalar formulas
// of convolveX8PureGo, convolveY8PureGo and convolve2D8PureGo with the same
// int32 arithmetic as the compound kernels. The 2D intermediate is int32: for
// every valid AV1 kernel the scalar int16 intermediate never truncates, so the
// two agree bit for bit.

// convolve8GoSIMDCtx is the calling context of the 8-bit convolve kernels. dst
// and ref are byte planes starting at the first destination pixel and the first
// sample of the trimmed tap span; kernel and xKern hold the trimmed taps
// zero-padded to filterTaps.
type convolve8GoSIMDCtx struct {
	dst    []byte
	ref    []byte
	kernel [filterTaps]int16 // vertical kernel (1D Y) or shared single kernel
	xKern  [filterTaps]int16 // 2D only: horizontal kernel
	dstStr int
	refStr int
	width  int
	height int
	tapsX  int // X tap count of the trimmed span (1D X and 2D)
	tapsY  int // Y tap count of the trimmed span (1D Y and 2D)
	im     []int32
	imStr  int // 2D only: intermediate row stride in int32 elements
}

// The single-prediction 8-bit rounding: the first stage is round3 and the final
// stage is round4 (1D) for the 1D kernels. The 2D vertical stage uses the
// libaom offset (yBias, roundOffset) with round1 = 11 and no final shift.
const (
	convolve8Round1Bits  = round1Bits
	convolve8YBias       = 1 << (8 + 2*filterBits - round0Bits)
	convolve8RoundOffset = (1 << (8 + 2*filterBits - round0Bits - round1Bits)) + (1 << (8 + 2*filterBits - round0Bits - round1Bits - 1))
	convolve8Bits        = 2*filterBits - round0Bits - round1Bits
)

// convolveX8Kernel is the 1D horizontal 8-bit convolve: clip(round4(round3(sum))).
func convolveX8Kernel(ctx *convolve8GoSIMDCtx) {
	n := ctx.tapsX
	taps := hbdBroadcastTaps16(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	width, height := ctx.width, ctx.height
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumRowX(src[x:], &taps, n)
			lo = hbdClip(hbdRound(hbdRound(lo, round0Bits), filterBits-round0Bits), zero, maxV)
			hi = hbdClip(hbdRound(hbdRound(hi, round0Bits), filterBits-round0Bits), zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:], src[x:])
			lo, _ := u8SumRowX(win[:], &taps, n)
			lo = hbdClip(hbdRound(hbdRound(lo, round0Bits), filterBits-round0Bits), zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// convolveY8Kernel is the 1D vertical 8-bit convolve: clip(round7(sum)).
func convolveY8Kernel(ctx *convolve8GoSIMDCtx) {
	n := ctx.tapsY
	taps := hbdBroadcastTaps16(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	width, height := ctx.width, ctx.height
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumColY(src[x:], ctx.refStr, &taps, n)
			lo = hbdClip(hbdRound(lo, filterBits), zero, maxV)
			hi = hbdClip(hbdRound(hi, filterBits), zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			var win [8 * 16]byte
			for k := range n {
				copy(win[k*16:k*16+4], src[k*ctx.refStr+x:])
			}
			lo, _ := u8SumColY(win[:], 16, &taps, n)
			lo = hbdClip(hbdRound(lo, filterBits), zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// convolve2D8Kernel is the separable 8-bit 2D convolve. The horizontal pass is
// the 8-bit compound horizontal pass (round3 with bias 2^(8+6)); the vertical
// pass rounds with round1, removes roundOffset and clips to the pixel range.
func convolve2D8Kernel(ctx *convolve8GoSIMDCtx) {
	nx, ny := ctx.tapsX, ctx.tapsY
	xTaps := hbdBroadcastTaps16(&ctx.xKern)
	yTaps := hbdBroadcastTaps32(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	yBias := archsimd.BroadcastInt32x4(convolve8YBias)
	rndOff := archsimd.BroadcastInt32x4(convolve8RoundOffset)
	xBias := archsimd.BroadcastInt32x4(1 << (8 + filterBits - 1))
	width, height := ctx.width, ctx.height

	u8HorizontalIM(ctx.ref, ctx.refStr, ctx.im, ctx.imStr, width, height+ny-1, &xTaps, nx, xBias)
	for y := range height {
		dst := ctx.dst[y*ctx.dstStr:]
		col := ctx.im[y*ctx.imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8Int32(col[x:], ctx.imStr, &yTaps, ny)
			lo = hbdFinish2D(lo.Add(yBias), convolve8Round1Bits, rndOff, convolve8Bits, zero, maxV)
			hi = hbdFinish2D(hi.Add(yBias), convolve8Round1Bits, rndOff, convolve8Bits, zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			lo := hbdSum4Int32(col[x:], ctx.imStr, &yTaps, ny)
			lo = hbdFinish2D(lo.Add(yBias), convolve8Round1Bits, rndOff, convolve8Bits, zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// convolveX8GoSIMD is the 1D horizontal 8-bit convolve over a resident window.
func convolveX8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width%4 != 0 {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	lo, n := hbdTapSpan(&kernel)
	ctx := convolve8GoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		ref:    ref.Pix[refY*ref.Stride+refX-fo+lo:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		tapsX:  n,
	}
	convolveX8Kernel(&ctx)
}

// convolveY8GoSIMD is the 1D vertical 8-bit convolve over a resident window.
func convolveY8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width%4 != 0 {
		convolveY8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	lo, n := hbdTapSpan(&kernel)
	ctx := convolve8GoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		ref:    ref.Pix[(refY-fo+lo)*ref.Stride+refX:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		tapsY:  n,
	}
	convolveY8Kernel(&ctx)
}

// convolve2D8GoSIMD is convolve2D8GoSIMDWithScratch without caller scratch.
func convolve2D8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8GoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2D8GoSIMDWithScratch is the separable 8-bit 2D convolve. With scratch
// the int32 intermediate lives in the scratch (imHBD) and is not zero-filled
// per call.
func convolve2D8GoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width%4 != 0 {
		convolve2D8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	var im []int32
	if scratch != nil {
		im = scratch.imHBD[:]
	} else {
		var local [(maxBlockSize + filterTaps - 1) * maxBlockSize]int32
		im = local[:]
	}
	ctx := convolve8GoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX):],
		kernel: hbdTrimTaps(&yKernel, loY, nY),
		xKern:  hbdTrimTaps(&xKernel, loX, nX),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		tapsX:  nX,
		tapsY:  nY,
		im:     im,
		imStr:  maxBlockSize,
	}
	convolve2D8Kernel(&ctx)
}

// The *ClampedGoSIMD wrappers handle the edge-block convolves. When the whole tap
// window happens to lie inside the reference plane the clamp is a no-op and the
// result is bit-identical to the non-clamped fast path, so we route to the GoSIMD
// asm. Otherwise (a tap coordinate genuinely falls off the frame) we fall back
// to the pure-Go clamped reference, which clamps each tap coordinate. This keeps
// the asm free of per-tap coordinate clamping while still accelerating the
// common case where an "edge" block's halo is actually resident.

func convolveX8ClampedGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// Both the width>=8 and width-4 horizontal kernels physically load one sample
	// past the consumed tap window: the width>=8 loop does a 16-byte ld1 for its
	// last 8-column group, and the width-4 kernel loads bytes 8..11 to fill the
	// slide register. That trailing sample never feeds an output, but it must be
	// resident, so reserve width+filterTaps samples in the fit check.
	haloW := width + filterTaps
	if neonWidth && planeRegionFits(ref, 1, refX-fo, refY, haloW, height) {
		convolveX8GoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveX8HorizontalEdgeGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveY8ClampedGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if neonWidth && planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		convolveY8GoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveY8VerticalEdgeGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

// convolveX8ClampedGoSIMDWithScratch is convolveX8ClampedGoSIMD with a dav1d-style
// emu_edge fast path (src/recon_tmpl.c mc()): when the tap window is not
// resident, the clamped halo is materialized once into the caller scratch and
// the plain X kernel runs over it.
func convolveX8ClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16, scratch *ConvolveScratch) {
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
	convolveX8ClampedGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

// convolveY8ClampedGoSIMDWithScratch is convolveY8ClampedGoSIMD with the same
// emu_edge fast path as convolveX8ClampedGoSIMDWithScratch.
func convolveY8ClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16, scratch *ConvolveScratch) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if scratch != nil && neonWidth &&
		!planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		convolveY8Impl(dst, emu, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveY8ClampedGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveX8HorizontalEdgeGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
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
	didGoSIMD := false
	for start < xHi {
		remaining := xHi - start
		if remaining >= 8 {
			chunk := remaining &^ 7
			// clampedXInterior only guarantees the consumed taps are resident;
			// the last 8-column group's 16-byte load reaches one sample further
			// (chunk+filterTaps in total). Shrink the GoSIMD span by whole groups
			// until that physical reach is in-bounds so the asm never over-reads;
			// any dropped columns fall to the scalar tail below.
			for chunk >= 8 && !planeRegionFits(ref, 1, refX+start-fo, refY, chunk+filterTaps, height) {
				chunk -= 8
			}
			if chunk < 8 {
				break
			}
			convolveX8GoSIMD(dst, ref, dstX+start, dstY, refX+start, refY, chunk, height, kernel)
			start += chunk
			didGoSIMD = true
			continue
		}
		if remaining >= 4 {
			if !planeRegionFits(ref, 1, refX+start-fo, refY, filterTaps+4, height) {
				break
			}
			convolveX8GoSIMD(dst, ref, dstX+start, dstY, refX+start, refY, 4, height, kernel)
			start += 4
			didGoSIMD = true
			continue
		}
		break
	}
	if !didGoSIMD {
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

func convolveY8VerticalEdgeGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
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
	convolveY8GoSIMD(dst, ref, dstX, dstY+yLo, refX, refY+yLo, width, yHi-yLo, kernel)
	if yHi < height {
		convolveY8ClampedPureGo(dst, ref, dstX, dstY+yHi, refX, refY+yHi, width, height-yHi, kernel)
	}
	return true
}

func convolve2D8ClampedGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8ClampedGoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8ClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// The horizontal pass loads one sample past the consumed tap window (16-byte
	// ld1 for width>=8; bytes 8..11 for width-4); reserve width+filterTaps
	// columns so that trailing sample stays resident.
	haloW := width + filterTaps
	if neonWidth &&
		planeRegionFits(ref, 1, refX-foX, refY-foY, haloW, height+filterTaps-1) {
		convolve2D8GoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if neonWidth && scratch != nil {
		// dav1d src/recon_tmpl.c mc(): out-of-bounds references materialize
		// the clamped halo once (emu_edge) and run the plain 8tap kernel over
		// the resident window.
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		convolve2D8GoSIMDWithScratch(dst, emu, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, scratch)
		return
	}
	if convolve2D8ClampedEdgeSplitGoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch) {
		return
	}
	convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}

func convolve2D8ClampedEdgeSplitGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) bool {
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
		convolve2D8GoSIMDWithScratch(dst, ref, dstX+start, dstY+yLo, refX+start, refY+yLo, chunk, midH, xKernel, yKernel, scratch)
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

// init binds the 8-bit single-prediction convolves to the Go SIMD kernels. The
// scalar defaults in convolve_dispatch.go stay in place when Go SIMD is not
// available.
func init() {
	if !hbdSIMDAvailable() {
		return
	}
	convolveX8Impl = convolveX8GoSIMD
	convolveY8Impl = convolveY8GoSIMD
	convolve2D8Impl = convolve2D8GoSIMD
	convolve2D8WithScratchImpl = convolve2D8GoSIMDWithScratch
	convolveX8ClampedImpl = convolveX8ClampedGoSIMD
	convolveY8ClampedImpl = convolveY8ClampedGoSIMD
	convolve2D8ClampedImpl = convolve2D8ClampedGoSIMD
	convolve2D8ClampedWithScratchImpl = convolve2D8ClampedGoSIMDWithScratch
	convolveX8ClampedWithScratchImpl = convolveX8ClampedGoSIMDWithScratch
	convolveY8ClampedWithScratchImpl = convolveY8ClampedGoSIMDWithScratch
}

func isFourTap(k [filterTaps]int16) bool {
	return k[0] == 0 && k[1] == 0 && k[6] == 0 && k[7] == 0
}
