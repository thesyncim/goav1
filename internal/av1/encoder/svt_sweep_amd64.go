//go:build goexperiment.simd && amd64 && !purego

package encoder

import "unsafe"

// svtEightSAD8x8Sub is svt_ext_eight_sad_calculation_8x8_16x16 (SUB_SAD,
// Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c shape) for one 16x16 block:
// VPSADBW of a 16-byte row yields the left and right 8x8 row sums, so the
// four 8x8 SUB_SADs of a position are two accumulators; the bests update
// with strict '<' per 8x8 and sad16[k] is the four quadrants' sum.
func svtEightSAD8x8Sub(src, ref []byte, stride int, best *[4]uint32, bestMV *[4][2]int16, mvX, mvY int, sad16 *[8]uint32) {
	_ = src[14*stride+15]
	_ = ref[14*stride+22]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	s2 := 2 * stride
	a0 := load16(sp)
	a1 := load16(unsafe.Add(sp, s2))
	a2 := load16(unsafe.Add(sp, 2*s2))
	a3 := load16(unsafe.Add(sp, 3*s2))
	a4 := load16(unsafe.Add(sp, 4*s2))
	a5 := load16(unsafe.Add(sp, 5*s2))
	a6 := load16(unsafe.Add(sp, 6*s2))
	a7 := load16(unsafe.Add(sp, 7*s2))
	for k := range 8 {
		r := unsafe.Add(rp, k)
		top := a0.SumOf8AbsDiff(load16(r)).
			Add(a1.SumOf8AbsDiff(load16(unsafe.Add(r, s2)))).
			Add(a2.SumOf8AbsDiff(load16(unsafe.Add(r, 2*s2)))).
			Add(a3.SumOf8AbsDiff(load16(unsafe.Add(r, 3*s2))))
		bot := a4.SumOf8AbsDiff(load16(unsafe.Add(r, 4*s2))).
			Add(a5.SumOf8AbsDiff(load16(unsafe.Add(r, 5*s2)))).
			Add(a6.SumOf8AbsDiff(load16(unsafe.Add(r, 6*s2)))).
			Add(a7.SumOf8AbsDiff(load16(unsafe.Add(r, 7*s2))))
		q := [4]uint32{
			uint32(top.GetElem(0)) << 1, uint32(top.GetElem(1)) << 1,
			uint32(bot.GetElem(0)) << 1, uint32(bot.GetElem(1)) << 1,
		}
		mv := [2]int16{int16(mvX + k), int16(mvY)}
		for c, s := range q {
			if s < best[c] {
				best[c] = s
				bestMV[c] = mv
			}
		}
		sad16[k] = q[0] + q[1] + q[2] + q[3]
	}
}

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16 (SUB_SAD)
// over the superblock at src[0] / ref[0]: the 16x16 blocks in
// p_eight_sad16x16 order, each scored at eight positions, then the 16x16
// bests.
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		pu8 := 21 + n*4
		o := y16*stride + x16
		svtEightSAD8x8Sub(src[o:], ref[o:], stride, (*[4]uint32)(me.bestSAD[pu8:pu8+4]),
			(*[4][2]int16)(me.bestMV[pu8:pu8+4]), mvX, mvY, &sad16[n])
		for k, s := range sad16[n] {
			me.update(5+n, s, mvX+k, mvY)
		}
	}
}

func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
