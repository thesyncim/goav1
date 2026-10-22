//go:build goexperiment.simd && !purego && (arm64 || amd64)

package encoder

import "simd/archsimd"

// eightSAD32x32_64x64 is svt_ext_eight_sad_calculation_32x32_64x64_neon:
// compute_4sad adds the four 16x16 rows of each quadrant as vectors (lanes
// are the eight positions), the 64x64 row is the sum of the four 32x32
// rows, then each PU scans the eight positions for its first strict
// minimum.
func (me *realtimeSBMultiSizeME) eightSAD32x32_64x64(sad16 *[16][8]uint32, mvX, mvY int) {
	var sad32 [5][8]uint32
	var s64lo, s64hi archsimd.Uint32x4
	for q := range 4 {
		r := sad16[4*q : 4*q+4]
		lo := archsimd.LoadUint32x4Array((*[4]uint32)(r[0][0:4])).Add(archsimd.LoadUint32x4Array((*[4]uint32)(r[1][0:4]))).
			Add(archsimd.LoadUint32x4Array((*[4]uint32)(r[2][0:4])).Add(archsimd.LoadUint32x4Array((*[4]uint32)(r[3][0:4]))))
		hi := archsimd.LoadUint32x4Array((*[4]uint32)(r[0][4:8])).Add(archsimd.LoadUint32x4Array((*[4]uint32)(r[1][4:8]))).
			Add(archsimd.LoadUint32x4Array((*[4]uint32)(r[2][4:8])).Add(archsimd.LoadUint32x4Array((*[4]uint32)(r[3][4:8]))))
		lo.StoreArray((*[4]uint32)(sad32[q][0:4]))
		hi.StoreArray((*[4]uint32)(sad32[q][4:8]))
		s64lo, s64hi = s64lo.Add(lo), s64hi.Add(hi)
	}
	s64lo.StoreArray((*[4]uint32)(sad32[4][0:4]))
	s64hi.StoreArray((*[4]uint32)(sad32[4][4:8]))
	for q := range 5 {
		pu := 1 + q
		if q == 4 {
			pu = 0
		}
		best, bk := me.bestSAD[pu], -1
		for k, s := range sad32[q] {
			if s < best {
				best, bk = s, k
			}
		}
		if bk >= 0 {
			me.bestSAD[pu] = best
			me.bestMV[pu] = [2]int16{int16(mvX + bk), int16(mvY)}
		}
	}
}
