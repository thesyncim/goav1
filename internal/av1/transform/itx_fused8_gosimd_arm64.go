// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itx_fused8: dav1d's pass structure (itx_fused32) for the 8-bit 8x8
// DCT_DCT block: one eight-row strip straight from the column-major
// coefficients with the DCT8 network across vectors, the inter-pass
// Round2(., 1) in registers, one 8x8 transpose, then the column network and
// the fused Round2(., 4) + add + saturate. The whole block stays in
// registers; nothing is stored between the passes.

// itx8Net is inverseDCT8 on eight lanes: in[k] is coefficient k, out[r]
// output r (the even half of inverseDCT16Col8SIMD16v2 with inputs 0..7).
func itx8Net(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0, in2, in4, in6 := in[0], in[2], in[4], in[6]
	t0 := itx16Rot(in0, in4, p2896, p2896, s12)
	t1 := itx16Rot(in0, in4, p2896, m2896, s12)
	t2 := itx16Rot(in2, in6, p1567, m3784, s12)
	t3 := itx16Rot(in2, in6, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	t4a := itx16Rot(in1, in7, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in5, in3, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in5, in3, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in1, in7, k(itx16Kp4017), k(itx16Kp799), s12)
	t4 := t4a.AddSaturated(t5a)
	t5a = t4a.SubSaturated(t5a)
	t7 := t7a.AddSaturated(t6a)
	t6a = t7a.SubSaturated(t6a)
	t5 := itx16Rot(t6a, t5a, p2896, m2896, s12)
	t6 := itx16Rot(t6a, t5a, p2896, p2896, s12)
	out[0], out[7] = d0.AddSaturated(t7), d0.SubSaturated(t7)
	out[1], out[6] = d1.AddSaturated(t6), d1.SubSaturated(t6)
	out[2], out[5] = d2.AddSaturated(t5), d2.SubSaturated(t5)
	out[3], out[4] = d3.AddSaturated(t4), d3.SubSaturated(t4)
}

// InverseAddDCT8x8 reconstructs one 8-bit 8x8 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride (8). pred may be dst. It reports false, with
// dst untouched, when the block is outside the kernel's contract or a guard
// declines; the caller then runs the staged path, which produces the same
// bytes. The kernels are generated (itx_fused_gen_arm64.go) per network
// form of the coded columns and rows.
func InverseAddDCT8x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 8 || activeCols <= 0 || activeCols > 8 ||
		coeffStride != 8 || len(coeff) < 64 || dstStride < 8 || predStride < 8 ||
		len(dst) < 7*dstStride+8 || len(pred) < 7*predStride+8 {
		return false
	}
	// The kernel switch is written here rather than in a dispatch function:
	// one call level less for the most frequent block.
	c := (*[64]int32)(coeff[:64])
	// Strided rows inside the extents checked above; same shape as the residual-add rows (checked rows +9-86%).
	dp, pp := unsafe.Pointer(unsafe.SliceData(dst)), unsafe.Pointer(unsafe.SliceData(pred))
	switch {
	case activeCols <= 4 && activeRows <= 4:
		return itxGenDCT8x8C4R4(c, dp, dstStride, pp, predStride)
	case activeCols <= 4:
		return itxGenDCT8x8C4R8(c, dp, dstStride, pp, predStride)
	case activeRows <= 4:
		return itxGenDCT8x8C8R4(c, dp, dstStride, pp, predStride)
	default:
		return itxGenDCT8x8C8R8(c, dp, dstStride, pp, predStride)
	}
}
