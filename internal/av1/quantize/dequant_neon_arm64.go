// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package quantize

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// NEON-accelerated per-column dequant kernel. The .s file implements the inner
// loop over a column of coefficients that share a single dequant scale; the Go
// wrapper here packages the scalar parameters into the asm calling context and
// binds the kernel at package init when NEON is present.
//
// Bit-exactness with dequantColumnPureGo (see dequant.go):
//   - each int16 coefficient's absolute value is taken with SQABS into int16
//     lanes; INT16_MIN cannot occur because AV1 coefficient levels are bounded
//     well within int16, but SQABS saturates rather than wrapping if it did.
//   - the product level*scale is formed with SMULL/SMULL2, widening the abs
//     int16 lanes against the broadcast int32 scale; the abs level and scale
//     are both non-negative and the product is < 2^30, so it never overflows
//     int32. AND with 0x00ffffff reproduces libaom's 24-bit tran_low_t mask.
//   - the masked product is arithmetically right-shifted by txScale (SSHL by a
//     negative count), the original sign reapplied, then clamped to the
//     [dqMin, dqMax] bound with SMIN/SMAX. Identical to dequantScalar.
//
// The QM (inverse quantization matrix) path and any unsupported shape stay on
// the pure-Go reference: this kernel is only bound for the flat-matrix column
// loop, which is what dequantizeBlockScaled dispatches to.

// dequantColumnNEONCtx is the asm calling context. Field order and sizes are
// part of the ABI shared with dequant_neon_arm64.s; do not reorder. The kernel
// processes blocks8 chunks of 8 coefficients each (blocks8 == n/8); the wrapper
// handles the remaining tail with the scalar reference.
type dequantColumnNEONCtx struct {
	dst     *int32
	coeff   *int16
	blocks8 uintptr
	scale   int64
	txeg    int64 // negated txScale, used as the SSHL right-shift count
	dqMin   int64
	dqMax   int64
}

//go:noescape
func dequantColumnNEONAsm(ctx *dequantColumnNEONCtx)

func dequantColumnNEON(dst []int32, coeff []int16, scale int32, txScale uint8, dqMin int32, dqMax int32) {
	n := len(coeff)
	if n == 0 {
		return
	}
	blocks8 := n &^ 7 // floor(n/8)*8
	if blocks8 > 0 {
		ctx := dequantColumnNEONCtx{
			dst:     &dst[0],
			coeff:   &coeff[0],
			blocks8: uintptr(blocks8 >> 3),
			scale:   int64(scale),
			txeg:    -int64(txScale),
			dqMin:   int64(dqMin),
			dqMax:   int64(dqMax),
		}
		dequantColumnNEONAsm(&ctx)
	}
	if blocks8 < n {
		dequantColumnPureGo(dst[blocks8:], coeff[blocks8:], scale, txScale, dqMin, dqMax)
	}
}

func init() {
	if cpu.Detected.NEON {
		dequantColumnImpl = dequantColumnNEON
	}
}
