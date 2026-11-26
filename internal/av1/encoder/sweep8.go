package encoder

import "encoding/binary"

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

// svtPackMV is SVT's packed motion vector, (y << 16) | (x & 0xFFFF), the
// layout of realtimeSBMultiSizeME.bestMV.
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
			(*[4]uint32)(me.bestMV[pu8:pu8+4]), mvX, mvY, &sad16[n])
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
func svtEightSAD8x8SubScalar(src, ref []byte, stride int, best *[4]uint32, bestMV *[4]uint32, mvX, mvY int, sad16 *[8]uint32) {
	const loMsk = 0x00FF00FF00FF00FF
	var q0, q1, q2, q3 [8]uint64
	for r := 0; r < 8; r += 2 {
		// Top half (quadrants 0 and 1).
		off := r * stride
		a := src[off : off+16]
		b := ref[off : off+23]
		a0 := binary.LittleEndian.Uint64(a[0:8])
		a1 := binary.LittleEndian.Uint64(a[8:16])
		a0e, a0o := a0&loMsk, (a0>>8)&loMsk
		a1e, a1o := a1&loMsk, (a1>>8)&loMsk
		b0 := binary.LittleEndian.Uint64(b[0:8])
		b1 := binary.LittleEndian.Uint64(b[8:16])
		b2 := binary.LittleEndian.Uint64(b[15:23]) >> 8

		q0[0] += swarSAD4(a0e, b0&loMsk) + swarSAD4(a0o, (b0>>8)&loMsk)
		q1[0] += swarSAD4(a1e, b1&loMsk) + swarSAD4(a1o, (b1>>8)&loMsk)
		for k := uint(1); k < 8; k++ {
			sh := k * 8
			w0 := (b0 >> sh) | (b1 << (64 - sh))
			w1 := (b1 >> sh) | (b2 << (64 - sh))
			q0[k] += swarSAD4(a0e, w0&loMsk) + swarSAD4(a0o, (w0>>8)&loMsk)
			q1[k] += swarSAD4(a1e, w1&loMsk) + swarSAD4(a1o, (w1>>8)&loMsk)
		}

		// Bottom half (quadrants 2 and 3).
		off = (r + 8) * stride
		a = src[off : off+16]
		b = ref[off : off+23]
		a0 = binary.LittleEndian.Uint64(a[0:8])
		a1 = binary.LittleEndian.Uint64(a[8:16])
		a0e, a0o = a0&loMsk, (a0>>8)&loMsk
		a1e, a1o = a1&loMsk, (a1>>8)&loMsk
		b0 = binary.LittleEndian.Uint64(b[0:8])
		b1 = binary.LittleEndian.Uint64(b[8:16])
		b2 = binary.LittleEndian.Uint64(b[15:23]) >> 8

		q2[0] += swarSAD4(a0e, b0&loMsk) + swarSAD4(a0o, (b0>>8)&loMsk)
		q3[0] += swarSAD4(a1e, b1&loMsk) + swarSAD4(a1o, (b1>>8)&loMsk)
		for k := uint(1); k < 8; k++ {
			sh := k * 8
			w0 := (b0 >> sh) | (b1 << (64 - sh))
			w1 := (b1 >> sh) | (b2 << (64 - sh))
			q2[k] += swarSAD4(a0e, w0&loMsk) + swarSAD4(a0o, (w0>>8)&loMsk)
			q3[k] += swarSAD4(a1e, w1&loMsk) + swarSAD4(a1o, (w1>>8)&loMsk)
		}
	}
	for k := range 8 {
		mv := svtPackMV(mvX+k, mvY)
		s0 := swarReduce16(q0[k]) << 1
		if s0 < best[0] {
			best[0] = s0
			bestMV[0] = mv
		}
		s1 := swarReduce16(q1[k]) << 1
		if s1 < best[1] {
			best[1] = s1
			bestMV[1] = mv
		}
		s2 := swarReduce16(q2[k]) << 1
		if s2 < best[2] {
			best[2] = s2
			bestMV[2] = mv
		}
		s3 := swarReduce16(q3[k]) << 1
		if s3 < best[3] {
			best[3] = s3
			bestMV[3] = mv
		}
		sad16[k] = s0 + s1 + s2 + s3
	}
}
