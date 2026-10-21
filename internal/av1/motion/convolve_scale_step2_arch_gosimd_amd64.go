// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"
)

// scaledStep2Perm holds, per tap and odd-phase offset, the byte shuffle that
// gathers the eight output columns of a 2:1 scaled row: lane x reads byte
// x/2 + (x&1)*odd + t of the load.
var scaledStep2Perm = func() (p [2][filterTaps][16]uint8) {
	for odd := range 2 {
		for t := range filterTaps {
			for x := range 8 {
				p[odd][t][x] = uint8(x>>1 + (x&1)*odd + t)
				p[odd][t][x+8] = p[odd][t][x]
			}
		}
	}
	return p
}()

// scaledStep2HorizontalRow filters one source row into eight output columns
// at a time for the 2:1 scaled convolve, gathering each tap's samples with a
// byte shuffle.
//
//go:nocheckptr
func scaledStep2HorizontalRow(src unsafe.Pointer, w8 int, odd int, coef *[filterTaps]archsimd.Int16x8, row []int16) {
	var perm [filterTaps]archsimd.Uint8x16
	for t := range filterTaps {
		perm[t] = archsimd.LoadUint8x16(scaledStep2Perm[odd][t][:])
	}
	xRound := archsimd.BroadcastInt16x8(8194)
	right2 := archsimd.BroadcastInt16x8(-2)
	for x := 0; x < w8; x += 8 {
		a := archsimd.LoadUint8x16((*[16]byte)(unsafe.Add(src, x/2))[:])
		sum := u8HalfMAC(a.Permute(perm[0]).ExtendLo8ToUint16().BitsToInt16(), coef[0], xRound)
		for t := 1; t < filterTaps; t++ {
			sum = u8HalfMAC(a.Permute(perm[t]).ExtendLo8ToUint16().BitsToInt16(), coef[t], sum)
		}
		u8Right16(sum, right2, 2).Store(row[x : x+8])
	}
}
