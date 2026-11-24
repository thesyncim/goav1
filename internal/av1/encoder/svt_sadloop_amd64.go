//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// svtSAD16xhx8 is the eight-position step of SVT-AV1's svt_sad_loop_kernel
// for 16/32-wide blocks (Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c and
// the NEON search_area_width % 8 == 0 branch): adds into acc the SADs of the 16-wide,
// rows-tall source block against ref+0..ref+7 with the source row shared.
// SumOf8AbsDiff is VPSADBW (SSE2/AVX2).
func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Uint64x2
	for r := 0; r < rows; r++ {
		// One window per row: the eight reference loads sit at constant
		// offsets inside it.
		so, ro := r*srcStride, r*refStride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16]))
		p := (*[23]uint8)(ref[ro : ro+23])
		a0 = a0.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))))
		a1 = a1.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))))
		a2 = a2.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))))
		a3 = a3.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))))
		a4 = a4.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))))
		a5 = a5.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))))
		a6 = a6.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))))
		a7 = a7.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))))
	}
	acc[0] += uint32(sumU64(a0))
	acc[1] += uint32(sumU64(a1))
	acc[2] += uint32(sumU64(a2))
	acc[3] += uint32(sumU64(a3))
	acc[4] += uint32(sumU64(a4))
	acc[5] += uint32(sumU64(a5))
	acc[6] += uint32(sumU64(a6))
	acc[7] += uint32(sumU64(a7))
}

// svtSAD32xhx8 is the 32-wide eight-position step: the two 16-column halves.
func svtSAD32xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	svtSAD16xhx8(acc, src, srcStride, ref, refStride, rows)
	svtSAD16xhx8(acc, src[16:], srcStride, ref[16:], refStride, rows)
}
