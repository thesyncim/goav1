package encoder

// sweepSearchPoint8 is open_loop_me_get_eight_search_point_results_block
// (Source/Lib/Codec/motion_estimation.c) for the SUB_SAD sweep: the
// superblock scored at the eight reference positions (mvX..mvX+7, mvY) in
// SVT-AV1's data flow.
//
//   - svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16 with
//     sub_sad: per 16x16 block (svt_ext_eight_sad_calculation_8x8_16x16) it
//     scores the four 8x8s at all eight positions, keeps each 8x8's (best
//     SAD, best MV) with strict '<' (earliest position wins ties), stores the
//     eight 16x16 sums in p_eight_sad16x16[16][8] and updates the 16x16
//     bests.
//   - eightSAD32x32_64x64 is svt_ext_eight_sad_calculation_32x32_64x64.
//
// The 16x16 index n is p_eight_sad16x16's (the offsets[] walk): 32x32
// quadrant-major, raster within it. It reports false (nothing done) when the
// batch's reads would leave the slices.
func (me *realtimeSBMultiSizeME) sweepSearchPoint8(src, ref []byte, stride, srcBase, refBase, mvX, mvY int) bool {
	if srcBase < 0 || refBase < 0 || srcBase+63*stride+64 > len(src) || refBase+63*stride+64+8 > len(ref) {
		return false
	}
	var sad16 [16][8]uint32
	svtAllSAD8x8_16x16Sub(me, src[srcBase:srcBase+63*stride+64], ref[refBase:refBase+63*stride+64+8], stride, mvX, mvY, &sad16)
	me.eightSAD32x32_64x64(&sad16, mvX, mvY)
	return true
}

// sweepBlock16 is the SB offset of 16x16 block n (p_eight_sad16x16 order).
func sweepBlock16(n int) (x16, y16 int) {
	q, w := n>>2, n&3
	return (q&1)*32 + (w&1)*16, (q>>1)*32 + (w>>1)*16
}

// svtPackMV is SVT's packed motion vector, (y << 16) | (x & 0xFFFF); the
// bestMV [2]int16{x, y} pairs hold the same bits.
func svtPackMV(x, y int) uint32 {
	return uint32(uint16(int16(y)))<<16 | uint32(uint16(int16(x)))
}

// eightSAD32x32_64x64 is svt_ext_eight_sad_calculation_32x32_64x64
// (compute_sad_neon.c shape): the 32x32 SADs of the eight positions are the
// sums of their four 16x16 rows (compute_4sad), the 64x64 SADs the sums of
// the four 32x32s; each PU keeps its first strict minimum over the
// positions.
func (me *realtimeSBMultiSizeME) eightSAD32x32_64x64Generic(sad16 *[16][8]uint32, mvX, mvY int) {
	var sad32 [4][8]uint32
	var sad64 [8]uint32
	for q := range sad32 {
		a, b, c, d := &sad16[4*q], &sad16[4*q+1], &sad16[4*q+2], &sad16[4*q+3]
		for k := range sad32[q] {
			s := a[k] + b[k] + c[k] + d[k]
			sad32[q][k] = s
			sad64[k] += s
		}
	}
	for q := range sad32 {
		for k, s := range sad32[q] {
			me.update(1+q, s, mvX+k, mvY)
		}
	}
	for k, s := range sad64 {
		me.update(0, s, mvX+k, mvY)
	}
}

// svtAllSAD8x8_16x16SubScalar is the reference definition of
// svtAllSAD8x8_16x16Sub over the superblock at src[0] / ref[0].
func svtAllSAD8x8_16x16SubScalar(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		pu8 := 21 + n*4
		o := y16*stride + x16
		svtEightSAD8x8SubScalar(src[o:], ref[o:], stride, (*[4]uint32)(me.bestSAD[pu8:pu8+4]),
			(*[4][2]int16)(me.bestMV[pu8:pu8+4]), mvX, mvY, &sad16[n])
		for k, s := range sad16[n] {
			me.update(5+n, s, mvX+k, mvY)
		}
	}
}

// svtEightSAD8x8SubScalar is svt_ext_eight_sad_calculation_8x8_16x16 with
// sub_sad for one 16x16 block: for positions k = 0..7 (reference block at
// ref[k:]), the SUB_SAD of each 8x8 quadrant c (rows 0, 2, 4, 6 of the
// quadrant, sum doubled) updates (best[c], bestMV[c]) with strict '<', and
// sad16[k] is the four quadrants' sum.
func svtEightSAD8x8SubScalar(src, ref []byte, stride int, best *[4]uint32, bestMV *[4][2]int16, mvX, mvY int, sad16 *[8]uint32) {
	for k := range 8 {
		var sum uint32
		for c := range 4 {
			xo, yo := (c&1)*8, (c>>1)*8
			var s uint32
			for r := 0; r < 8; r += 2 {
				a := src[(yo+r)*stride+xo:]
				b := ref[(yo+r)*stride+xo+k:]
				for i := range 8 {
					d := int(a[i]) - int(b[i])
					if d < 0 {
						d = -d
					}
					s += uint32(d)
				}
			}
			s <<= 1
			if s < best[c] {
				best[c] = s
				bestMV[c] = [2]int16{int16(mvX + k), int16(mvY)}
			}
			sum += s
		}
		sad16[k] = sum
	}
}
