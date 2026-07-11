// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestI8MMDotProdUSIsUnsigned asserts that Int32x4.DotProdUS (VUSDOT) treats its
// first operand as UNSIGNED bytes, which the Go-native SIMD convolve kernels
// rely on to convolve unsigned pixels with signed filter taps without a -128
// bias. If this ever regresses the kernels would silently diverge for pixels
// >= 128.
func TestI8MMDotProdUSIsUnsigned(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	var yb [16]uint8
	for i := range yb {
		yb[i] = 200 // unsigned 200 == signed -56
	}
	var zb [16]int8
	for i := range zb {
		zb[i] = 1
	}
	y := archsimd.LoadUint8x16Array(&yb)
	z := archsimd.LoadInt8x16Array(&zb)
	var out [4]int32
	archsimd.BroadcastInt32x4(0).DotProdUS(y, z).StoreArray(&out)
	for lane, got := range out {
		if got != 4*200 { // 4 taps * 200
			t.Fatalf("DotProdUS lane %d = %d, want %d (first operand must be unsigned)", lane, got, 4*200)
		}
	}
}

// TestI8MMMatMulUSSignBug documents a tsgo-fork discrepancy: Int32x4.MatMulUS is
// documented (and intended) to emit VUSMMLA (unsigned x signed), but on this
// toolchain it emits SMMLA (signed x signed) -- its first operand is treated as
// SIGNED. The convolve horizontal pass therefore uses two VUSDOT per four
// outputs instead of one VUSMMLA per four; if MatMulUS is ever fixed to real
// USMMLA this test flips and the horizontal pass can be upgraded to halve its
// dot-product instruction count. It is skipped (not failed) so it never blocks
// CI, only records the observed behavior.
func TestI8MMMatMulUSSignBug(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	var yb [16]uint8
	yb[0] = 200 // unsigned 200 == signed -56
	var zb [16]int8
	zb[0] = 1
	y := archsimd.LoadUint8x16Array(&yb)
	z := archsimd.LoadInt8x16Array(&zb)
	var out [4]int32
	archsimd.BroadcastInt32x4(0).MatMulUS(y, z).StoreArray(&out)
	switch out[0] {
	case 200:
		t.Log("MatMulUS now behaves as USMMLA (unsigned first operand) -- the X/2D horizontal pass can be upgraded to USMMLA")
	case -56:
		t.Skip("known: MatMulUS emits SMMLA (signed first operand); convolve uses DotProdUS instead")
	default:
		t.Fatalf("MatMulUS lane0 = %d, expected 200 (USMMLA) or -56 (SMMLA)", out[0])
	}
}
