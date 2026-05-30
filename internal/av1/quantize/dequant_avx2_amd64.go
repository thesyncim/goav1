// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package quantize

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// AVX2-accelerated per-column dequant kernel. The .s file implements the inner
// loop over a column of coefficients that share a single dequant scale; the Go
// wrapper here packages the scalar parameters into the asm calling context and
// binds the kernel at package init when AVX2 is present.
//
// Bit-exactness with dequantColumnPureGo (see dequant.go):
//   - eight int16 coefficients are sign-extended to int32 with VPMOVSXWD.
//   - the original sign is captured as an all-ones/all-zero mask with an
//     arithmetic VPSRAD by 31, so the magnitude can be reformed and re-signed
//     without a branch.
//   - the absolute value is taken with VPABSD (INT16_MIN-safe in int32 lanes;
//     AV1 coefficient levels never reach INT16_MIN, but VPABSD on int32 lanes
//     produces +32768 rather than wrapping if it did).
//   - the product level*scale is formed with VPMULLD against the broadcast
//     int32 scale. The abs level (<=32768) and scale (<2^15 in practice) keep
//     the product well within int32, matching the low 32 bits that libaom's
//     int64 multiply would feed through the 24-bit mask. VPAND with 0x00ffffff
//     reproduces libaom's tran_low_t mask.
//   - the masked product is arithmetically right-shifted by txScale with VPSRAD
//     using a scalar count in an XMM register. The masked product is
//     non-negative (top byte cleared), so arithmetic and logical shifts agree,
//     matching dequantScalar's `product >> txScale` on a non-negative product.
//   - the sign is reapplied as (level XOR signmask) - signmask (VPXOR/VPSUBD),
//     then the result is clamped to [dqMin, dqMax] with VPMINSD/VPMAXSD.
//     Identical to dequantScalar.
//
// The QM (inverse quantization matrix) path and any unsupported shape stay on
// the pure-Go reference: this kernel is only bound for the flat-matrix column
// loop, which is what dequantizeBlockScaled dispatches to.

// dequantColumnAVX2Ctx is the asm calling context. Field order and sizes are
// part of the ABI shared with dequant_avx2_amd64.s; do not reorder. The kernel
// processes blocks8 chunks of 8 coefficients each (blocks8 == n/8); the wrapper
// handles the remaining tail with the scalar reference.
type dequantColumnAVX2Ctx struct {
	dst     *int32
	coeff   *int16
	blocks8 uintptr
	scale   int64
	txScale int64 // right-shift count, loaded into an XMM for VPSRAD
	dqMin   int64
	dqMax   int64
}

//go:noescape
func dequantColumnAVX2Asm(ctx *dequantColumnAVX2Ctx)

//go:noescape
func cpuidQuantize(eax, ecx uint32) (a, b, c, d uint32)

//go:noescape
func xgetbvQuantize() (lo uint32)

// detectAVX2 probes CPUID/XGETBV for OS-enabled AVX2. It mirrors the standard
// sequence: leaf 1 must report OSXSAVE and AVX, XCR0 must enable both XMM and
// YMM state, and leaf 7 sub-leaf 0 must report the AVX2 bit.
func detectAVX2() bool {
	maxLeaf, _, _, _ := cpuidQuantize(0, 0)
	if maxLeaf < 7 {
		return false
	}
	_, _, ecx1, _ := cpuidQuantize(1, 0)
	const osxsaveBit = 1 << 27
	const avxBit = 1 << 28
	if ecx1&osxsaveBit == 0 || ecx1&avxBit == 0 {
		return false
	}
	// XCR0 bit 1 (SSE/XMM) and bit 2 (AVX/YMM) must both be set by the OS.
	const xcr0YMM = (1 << 1) | (1 << 2)
	if xgetbvQuantize()&xcr0YMM != xcr0YMM {
		return false
	}
	_, ebx7, _, _ := cpuidQuantize(7, 0)
	const avx2Bit = 1 << 5
	return ebx7&avx2Bit != 0
}

func dequantColumnAVX2(dst []int32, coeff []int16, scale int32, txScale uint8, dqMin int32, dqMax int32) {
	n := len(coeff)
	if n == 0 {
		return
	}
	blocks8 := n &^ 7 // floor(n/8)*8
	if blocks8 > 0 {
		ctx := dequantColumnAVX2Ctx{
			dst:     &dst[0],
			coeff:   &coeff[0],
			blocks8: uintptr(blocks8 >> 3),
			scale:   int64(scale),
			txScale: int64(txScale),
			dqMin:   int64(dqMin),
			dqMax:   int64(dqMax),
		}
		dequantColumnAVX2Asm(&ctx)
	}
	if blocks8 < n {
		dequantColumnPureGo(dst[blocks8:], coeff[blocks8:], scale, txScale, dqMin, dqMax)
	}
}

// hasAVX2 reports whether this process may execute the AVX2 column kernel.
//
// The shared cpu.Detected feature struct is the authoritative source; the
// dispatch binds when cpu.Detected.AVX2 is set. Because that package's amd64
// detector does not yet probe CPUID for AVX2, this file also runs a local
// CPUID/XGETBV check (detectAVX2) so the AVX2 kernel is used on real x86-64
// hardware that advertises and OS-enables AVX2. Under Apple Rosetta 2 the
// guest CPUID hides AVX (osxsave/avx report 0), so both sources are false and
// the dispatcher correctly falls back to pure-Go even though Rosetta can in
// fact execute the instructions.
func hasAVX2() bool {
	return cpu.Detected.AVX2 || detectAVX2()
}

func init() {
	if hasAVX2() {
		dequantColumnImpl = dequantColumnAVX2
	}
}
