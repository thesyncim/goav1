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

//go:noescape
func compoundX8I8MMAsm(ctx *compoundX8I8MMCtx)

// SVT ASM_NEON_I8MM/convolve_neon_i8mm.c: svt_kMatMul8PermuteTbl.
var compoundX8I8MMPermute = [32]byte{
	1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 10,
	5, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 13, 14,
}

func compoundX8I8MMFilter(kernel [filterTaps]int16) (filter [16]byte, f0 uint8, ok bool) {
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

func predictInterCompoundRef8ToConvBufXI8MM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	if !cpu.Detected.I8MM ||
		width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		predictInterCompoundRef8ToConvBufXDotProd(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	filter, f0, ok := compoundX8I8MMFilter(kernel)
	if !ok {
		predictInterCompoundRef8ToConvBufXDotProd(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	roundShim := (roundOffset << (compoundRound0Bits - 1)) + (1 << ((compoundRound0Bits - 1) - 1))
	ctx := compoundX8I8MMCtx{
		dst:       &out[0],
		ref:       &ref.Pix[refY*ref.Stride+refX-fo],
		filter:    &filter[0],
		permute:   &compoundX8I8MMPermute[0],
		refStr:    uintptr(ref.Stride),
		width:     uintptr(width),
		height:    uintptr(height),
		roundShim: uintptr(roundShim),
		f0:        uintptr(f0),
	}
	compoundX8I8MMAsm(&ctx)
}
