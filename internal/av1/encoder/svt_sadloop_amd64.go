//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtSAD16xhx8 is the eight-position step of SVT-AV1's svt_sad_loop_kernel
// for 16/32-wide blocks (Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c and
// the NEON search_area_width % 8 == 0 branch): adds into acc the SADs of the 16-wide,
// rows-tall source block against ref+0..ref+7 with the source row shared.
// SumOf8AbsDiff is VPSADBW (SSE2/AVX2).
func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	_ = src[(rows-1)*srcStride+15]
	_ = ref[(rows-1)*refStride+22]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Uint64x2
	for r := 0; r < rows; r++ {
		s := load16(unsafe.Add(sp, r*srcStride))
		p := unsafe.Add(rp, r*refStride)
		a0 = a0.Add(s.SumOf8AbsDiff(load16(p)))
		a1 = a1.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 1))))
		a2 = a2.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 2))))
		a3 = a3.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 3))))
		a4 = a4.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 4))))
		a5 = a5.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 5))))
		a6 = a6.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 6))))
		a7 = a7.Add(s.SumOf8AbsDiff(load16(unsafe.Add(p, 7))))
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
