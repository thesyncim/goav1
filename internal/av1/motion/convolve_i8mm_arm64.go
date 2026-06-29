// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// compoundX8I8MMCtx carries the resident lowbd compound-X I8MM kernel
// arguments. Field offsets are mirrored by convolve_i8mm_arm64.s.
type compoundX8I8MMCtx struct {
	dst       *uint16
	ref       *byte
	filter    *byte
	permute   *byte
	refStr    uintptr
	width     uintptr
	height    uintptr
	roundShim uintptr
	f0        uintptr
}

// compoundY8I8MMCtx carries the resident lowbd compound-Y I8MM kernel
// arguments. Field offsets are mirrored by convolve_i8mm_arm64.s.
type compoundY8I8MMCtx struct {
	dst         *uint16
	ref         *byte
	filter      *byte
	merge       *byte
	dstStr      uintptr
	refStr      uintptr
	width       uintptr
	height      uintptr
	roundOffset uintptr
}

//go:noescape
func compoundX8I8MMAsm(ctx *compoundX8I8MMCtx)

//go:noescape
func compoundY8I8MMAsm(ctx *compoundY8I8MMCtx)

//go:noescape
func compoundY4TapI8MMAsm(ctx *compoundY8I8MMCtx)

// convolveX8I8MMCtx carries the resident lowbd single-prediction X I8MM kernel
// arguments. Field offsets are mirrored by convolve_i8mm_arm64.s.
type convolveX8I8MMCtx struct {
	dst     *byte
	ref     *byte
	filter  *byte
	permute *byte
	dstStr  uintptr
	refStr  uintptr
	width   uintptr
	height  uintptr
	f0      uintptr
}

//go:noescape
func convolveX8I8MMAsm(ctx *convolveX8I8MMCtx)

// convolveY8I8MMCtx carries the resident lowbd single-prediction Y I8MM kernel
// arguments. Field offsets are mirrored by convolve_i8mm_arm64.s.
type convolveY8I8MMCtx struct {
	dst    *byte
	ref    *byte
	filter *byte
	merge  *byte
	dstStr uintptr
	refStr uintptr
	width  uintptr
	height uintptr
}

//go:noescape
func convolveY8I8MMAsm(ctx *convolveY8I8MMCtx)

//go:noescape
func convolveY4TapI8MMAsm(ctx *convolveY8I8MMCtx)

// convolve2D8I8MMCtx carries the resident lowbd single-prediction 2D I8MM
// kernel arguments. Field offsets are mirrored by convolve_i8mm_arm64.s.
type convolve2D8I8MMCtx struct {
	dst     *byte
	ref     *byte
	xFilter *byte
	permute *byte
	yKernel *int16
	dstStr  uintptr
	refStr  uintptr
	width   uintptr
	height  uintptr
	im      *int16
	imStr   uintptr
	f0      uintptr
}

//go:noescape
func convolve2D8I8MMAsm(ctx *convolve2D8I8MMCtx)

// SVT ASM_NEON_I8MM/convolve_neon_i8mm.c: svt_kMatMul8PermuteTbl.
var convolveX8I8MMPermute = [32]byte{
	1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 10,
	5, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 13, 14,
}

// SVT ASM_NEON_DOTPROD/convolve_neon_dotprod.c: svt_kDotProdMergeBlockTbl.
var convolveY8I8MMMergeBlock = [48]byte{
	1, 2, 3, 16, 5, 6, 7, 20, 9, 10, 11, 24, 13, 14, 15, 28,
	2, 3, 16, 17, 6, 7, 20, 21, 10, 11, 24, 25, 14, 15, 28, 29,
	3, 16, 17, 18, 7, 20, 21, 22, 11, 24, 25, 26, 15, 28, 29, 30,
}

func convolveX8I8MMFilter(kernel [filterTaps]int16) (filter [16]byte, f0 uint8, ok bool) {
	for i := range kernel {
		if kernel[i]%2 != 0 {
			return [16]byte{}, 0, false
		}
	}
	if kernel[0] > 0 {
		return [16]byte{}, 0, false
	}
	tap0 := (-int(kernel[0])) >> 1
	if tap0 > 0xff {
		return [16]byte{}, 0, false
	}
	for i := 1; i < filterTaps; i++ {
		v := int(kernel[i] >> 1)
		if v < -128 || v > 127 {
			return [16]byte{}, 0, false
		}
		filter[i-1] = byte(int8(v))
		filter[i+8] = byte(int8(v))
	}
	return filter, uint8(tap0), true
}

func convolveY8I8MMFilter(kernel [filterTaps]int16) (filter [16]byte, taps int, ok bool) {
	for i := range kernel {
		if kernel[i]%2 != 0 {
			return [16]byte{}, 0, false
		}
	}
	if kernel[0] == 0 && kernel[1] == 0 && kernel[6] == 0 && kernel[7] == 0 {
		// SVT routes <=2-tap filters through NEON. Keep that split so the I8MM
		// entry covers the real 4-tap vertical family.
		if kernel[2] == 0 && kernel[5] == 0 {
			return [16]byte{}, 0, false
		}
		for i := range 4 {
			v := int(kernel[i+2] >> 1)
			if v < -128 || v > 127 {
				return [16]byte{}, 0, false
			}
			filter[i] = byte(int8(v))
		}
		return filter, 4, true
	}
	for i := range kernel {
		v := int(kernel[i] >> 1)
		if v < -128 || v > 127 {
			return [16]byte{}, 0, false
		}
		filter[i] = byte(int8(v))
	}
	return filter, 8, true
}

func predictInterCompoundRef8ToConvBufXI8MM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	if !cpu.Detected.I8MM ||
		width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		predictInterCompoundRef8ToConvBufXDotProd(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	filter, f0, ok := convolveX8I8MMFilter(kernel)
	if !ok {
		predictInterCompoundRef8ToConvBufXDotProd(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	roundShim := (roundOffset << (compoundRound0Bits - 1)) + (1 << ((compoundRound0Bits - 1) - 1))
	ctx := compoundX8I8MMCtx{
		dst:       &out[0],
		ref:       &ref.Pix[refY*ref.Stride+refX-fo],
		filter:    &filter[0],
		permute:   &convolveX8I8MMPermute[0],
		refStr:    uintptr(ref.Stride),
		width:     uintptr(width),
		height:    uintptr(height),
		roundShim: uintptr(roundShim),
		f0:        uintptr(f0),
	}
	compoundX8I8MMAsm(&ctx)
}

func predictInterCompoundRef8ToConvBufYI8MM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	if !cpu.Detected.I8MM ||
		round0 != compoundRound0Bits ||
		!(width == 4 || (width >= 8 && width%8 == 0)) ||
		height%4 != 0 ||
		!planeRegionFits(ref, 1, refX, refY-(filterTaps/2-1), width, height+filterTaps-1) {
		predictInterCompoundRef8ToConvBufYNEON(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	filter, taps, ok := convolveY8I8MMFilter(kernel)
	if !ok {
		predictInterCompoundRef8ToConvBufYNEON(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	fo := filterTaps/2 - 1
	if taps == 4 {
		fo = 1
	}
	ctx := compoundY8I8MMCtx{
		dst:         &out[0],
		ref:         &ref.Pix[(refY-fo)*ref.Stride+refX],
		filter:      &filter[0],
		merge:       &convolveY8I8MMMergeBlock[0],
		dstStr:      uintptr(width * 2),
		refStr:      uintptr(ref.Stride),
		width:       uintptr(width),
		height:      uintptr(height),
		roundOffset: uintptr(roundOffset),
	}
	if taps == 4 {
		compoundY4TapI8MMAsm(&ctx)
		return
	}
	compoundY8I8MMAsm(&ctx)
}

func convolveX8I8MM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if !cpu.Detected.I8MM || width < 8 || width%8 != 0 {
		convolveX8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	filter, f0, ok := convolveX8I8MMFilter(kernel)
	if !ok {
		convolveX8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	ctx := convolveX8I8MMCtx{
		dst:     &dst.Pix[dstY*dst.Stride+dstX],
		ref:     &ref.Pix[refY*ref.Stride+refX-fo],
		filter:  &filter[0],
		permute: &convolveX8I8MMPermute[0],
		dstStr:  uintptr(dst.Stride),
		refStr:  uintptr(ref.Stride),
		width:   uintptr(width),
		height:  uintptr(height),
		f0:      uintptr(f0),
	}
	convolveX8I8MMAsm(&ctx)
}

func convolveY8I8MM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if !cpu.Detected.I8MM ||
		!(width == 4 || (width >= 8 && width%8 == 0)) ||
		height%4 != 0 {
		convolveY8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	filter, taps, ok := convolveY8I8MMFilter(kernel)
	if !ok {
		convolveY8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	if taps == 4 {
		fo = 1
	}
	ctx := convolveY8I8MMCtx{
		dst:    &dst.Pix[dstY*dst.Stride+dstX],
		ref:    &ref.Pix[(refY-fo)*ref.Stride+refX],
		filter: &filter[0],
		merge:  &convolveY8I8MMMergeBlock[0],
		dstStr: uintptr(dst.Stride),
		refStr: uintptr(ref.Stride),
		width:  uintptr(width),
		height: uintptr(height),
	}
	if taps == 4 {
		convolveY4TapI8MMAsm(&ctx)
		return
	}
	convolveY8I8MMAsm(&ctx)
}

func convolve2D8I8MM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8I8MMWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8I8MMWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if !cpu.Detected.I8MM || width < 8 || width%8 != 0 {
		convolve2D8NEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	xFilter, f0, ok := convolveX8I8MMFilter(xKernel)
	if !ok {
		convolve2D8NEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	yk := yKernel
	if scratch != nil {
		convolve2D8I8MMWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xFilter, f0, yk, &scratch.im[0], convolve2DNEONIMStride)
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DNEONIMStride]int16
	convolve2D8I8MMWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xFilter, f0, yk, &im[0], convolve2DNEONIMStride)
}

func convolve2D8I8MMWithIM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xFilter [16]byte, f0 uint8, yk [filterTaps]int16, im *int16, imStride int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	ctx := convolve2D8I8MMCtx{
		dst:     &dst.Pix[dstY*dst.Stride+dstX],
		ref:     &ref.Pix[(refY-foY)*ref.Stride+refX-foX],
		xFilter: &xFilter[0],
		permute: &convolveX8I8MMPermute[0],
		yKernel: &yk[0],
		dstStr:  uintptr(dst.Stride),
		refStr:  uintptr(ref.Stride),
		width:   uintptr(width),
		height:  uintptr(height),
		im:      im,
		imStr:   uintptr(imStride),
		f0:      uintptr(f0),
	}
	convolve2D8I8MMAsm(&ctx)
}
