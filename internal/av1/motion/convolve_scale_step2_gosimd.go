// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// scaledStep2Half is half a full sample in Q10 stepper units: the step of an
// exact 2:1 upscale, the ratio between spatial layers in SVC streams.
const scaledStep2Half = ScaleSubpelScale / 2

// convolveScale2D8Step2GoSIMD computes convolveScale2D8 for xStep == yStep ==
// 512, an exact 2:1 upscale. The filter phase then alternates between two
// kernels: output column x reads source column xi0 + x/2 (+1 for odd x when
// the odd phase starts one sample later) with kernel phase x&1. A 16-byte load
// and one byte permute per tap therefore gather eight output columns, and an
// alternating coefficient vector applies both phases at once. Each output row
// uses a single vertical kernel, so the vertical pass is an ordinary 8-tap
// filter across columns. Rounding is the scaled formula, so the result is
// bit-identical. It returns false when the taps are not all even or the source
// window does not allow 16-byte loads; the caller then runs the scalar path.
//
//go:nocheckptr
func convolveScale2D8Step2GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int,
	startX int64, startY int64, xTable, yTable *SubpelKernelTable, imH int, imBuf *scaledIM) bool {
	const fo = filterTaps/2 - 1
	k0 := &xTable[int(scaledSubpel(startX)>>ScaleExtraBits)]
	k1 := &xTable[int(scaledSubpel(startX+scaledStep2Half)>>ScaleExtraBits)]
	for t := range filterTaps {
		if k0[t]&1 != 0 || k1[t]&1 != 0 {
			return false
		}
	}
	xi0 := int(scaledIntFloor(startX)) - fo
	odd := int(scaledIntFloor(startX+scaledStep2Half)) - fo - xi0 // 0 or 1
	startRow := int(scaledIntFloor(startY)) - fo
	w8 := (width + 7) &^ 7
	if xi0 < 0 || startRow < 0 || imH <= 0 ||
		(startRow+imH-1)*ref.Stride+xi0+(w8-8)/2+16 > len(ref.Pix) {
		return false
	}

	// Lane x uses the halved coefficient of phase x&1.
	var coef [filterTaps]archsimd.Int16x8
	for t := range filterTaps {
		var c [8]int16
		for x := 0; x < 8; x += 2 {
			c[x], c[x+1] = k0[t]/2, k1[t]/2
		}
		coef[t] = archsimd.LoadInt16x8(c[:])
	}
	im := imBuf[:]
	base := unsafe.Pointer(unsafe.SliceData(ref.Pix))
	for y := range imH {
		scaledStep2HorizontalRow(unsafe.Add(base, (startRow+y)*ref.Stride+xi0), w8, odd, &coef, im[y*maxBlockSize:])
	}

	baseY := int(scaledIntFloor(startY))
	yRound := archsimd.BroadcastInt32x4(convolve8YBias + (1 << (round1Bits - 1)))
	right11 := archsimd.BroadcastInt32x4(-round1Bits)
	rndOff := archsimd.BroadcastInt32x4(convolve8RoundOffset)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	imBase := unsafe.Pointer(unsafe.SliceData(im))
	load := func(y, x int) archsimd.Int16x8 {
		return archsimd.LoadInt16x8((*[8]int16)(unsafe.Add(imBase, (y*maxBlockSize+x)*2))[:])
	}
	yPos := startY
	for y := range height {
		r := int(scaledIntFloor(yPos)) - baseY
		k := &yTable[int(scaledSubpel(yPos)>>ScaleExtraBits)]
		c0 := archsimd.BroadcastInt16x8(k[0])
		c1 := archsimd.BroadcastInt16x8(k[1])
		c2 := archsimd.BroadcastInt16x8(k[2])
		c3 := archsimd.BroadcastInt16x8(k[3])
		c4 := archsimd.BroadcastInt16x8(k[4])
		c5 := archsimd.BroadcastInt16x8(k[5])
		c6 := archsimd.BroadcastInt16x8(k[6])
		c7 := archsimd.BroadcastInt16x8(k[7])
		off := (dstY+y)*dst.Stride + dstX
		dstRow := dst.Pix[off : off+width : off+width]
		for x := 0; x < width; x += 8 {
			lo, hi := u8WideMAC8(zero, zero, load(r, x), c0)
			lo, hi = u8WideMAC8(lo, hi, load(r+1, x), c1)
			lo, hi = u8WideMAC8(lo, hi, load(r+2, x), c2)
			lo, hi = u8WideMAC8(lo, hi, load(r+3, x), c3)
			lo, hi = u8WideMAC8(lo, hi, load(r+4, x), c4)
			lo, hi = u8WideMAC8(lo, hi, load(r+5, x), c5)
			lo, hi = u8WideMAC8(lo, hi, load(r+6, x), c6)
			lo, hi = u8WideMAC8(lo, hi, load(r+7, x), c7)
			lo = u8Right32(lo.Add(yRound), right11, round1Bits).Sub(rndOff).Max(zero).Min(maxV)
			hi = u8Right32(hi.Add(yRound), right11, round1Bits).Sub(rndOff).Max(zero).Min(maxV)
			var tmp [16]byte
			u8PackClipped32(lo, hi).Store(tmp[:])
			copy(dstRow[x:], tmp[:min(8, width-x)])
		}
		yPos += scaledStep2Half
	}
	return true
}
