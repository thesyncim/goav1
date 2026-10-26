//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// svtSubpelAVX2 gates the AVX2 sub-pel variance kernel.
var svtSubpelAVX2 = gosimdKernelsSupported()

// svtBilinearPredictArch is the scalar reference on amd64: only the fused
// variance below is on the encoder's path.
func svtBilinearPredictArch(scratch *svtSubpelScratch, ref []byte, refStride, bw, bh, xoff, yoff int) {
	svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
}

// svtSubpelVarianceArch is vfp->svf on AVX2, shaped like SVT-AV1's
// Source/Lib/ASM_AVX2/variance_avx2.c svt_aom_sub_pixel_variance{16,32,64}xh
// (filter_block_2rows / the half-pel _mm256_avg_epu8 branch): a zero phase
// skips its pass, the half-pel phase (taps 64/64) is a rounding average
// ((a+b+1)>>1 == (64a+64b+64)>>7), and the other phases multiply interleaved
// sample pairs by the packed tap pair with VPMADDUBSW and round by 7. Both
// passes keep 8-bit values, as the C reference does; they live here in int16
// lanes (sixteen columns of one row per vector, VPMOVZXBW), which is what the
// variance accumulation (variance_kernel_avx2: VPMADDWD squares, int16 sums)
// wants anyway. An 8-wide block runs the 16-column strip with the upper
// eight differences masked to zero.
func svtSubpelVarianceArch(scratch *svtSubpelScratch, ref []byte, refStride int, src []byte, srcStride, bw, bh, xoff, yoff int) uint32 {
	if !svtSubpelAVX2 || bw&7 != 0 || bw > 64 || bh <= 0 || bh > 64 || len(ref) < bh*refStride+bw+16 || len(src) < (bh-1)*srcStride+max(bw, 16) {
		svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
		_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
		return v
	}
	tx := svtBilTapsAVX2(xoff)
	ty := svtBilTapsAVX2(yoff)
	bias := archsimd.BroadcastInt16x16(64)
	ones := archsimd.BroadcastInt16x16(1)
	mask := archsimd.BroadcastInt16x16(-1)
	if bw == 8 {
		mask = archsimd.LoadInt16x16Array(&svtSubpelLo8Mask)
	}
	base := unsafe.Pointer(&ref[0])
	sbase := unsafe.Pointer(&src[0])
	var sse, sum32 archsimd.Int32x8
	for x := 0; x < bw; x += 16 {
		prev := svtBilRowHAVX2(unsafe.Add(base, x), xoff, tx, bias)
		var sum16 archsimd.Int16x16
		for r := 0; r < bh; r++ {
			cur := svtBilRowHAVX2(unsafe.Add(base, (r+1)*refStride+x), xoff, tx, bias)
			var out archsimd.Int16x16
			switch yoff {
			case 0:
				out = prev
			case 4:
				out = prev.AsUint16x16().Average(cur.AsUint16x16()).AsInt16x16()
			default:
				out = svtBilPairAVX2(prev, cur, ty, bias)
			}
			prev = cur
			s := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbase, r*srcStride+x))).ExtendToUint16().AsInt16x16()
			d := out.Sub(s).And(mask)
			sum16 = sum16.Add(d)
			sse = sse.Add(d.DotProductPairs(d))
		}
		// Each int16 lane holds at most 64 differences of magnitude <= 255.
		sum32 = sum32.Add(sum16.DotProductPairs(ones))
	}
	total := uint32(reduceInt32x8(sse))
	signed := reduceInt32x8(sum32)
	archsimd.ClearAVXUpperBits()
	return varianceFromStats(total, signed, uint(bits.TrailingZeros(uint(bw*bh))))
}

// svtSubpelLo8Mask keeps the eight low int16 lanes (an 8-wide block's columns).
var svtSubpelLo8Mask = [16]int16{-1, -1, -1, -1, -1, -1, -1, -1}

// svtBilTapsAVX2 is the VPMADDUBSW operand of a phase: the tap pair (f0, f1)
// as int8 bytes in every 16-bit lane. The non-zero, non-half phases have both
// taps in 16..112, so they fit int8 and no product sum saturates
// (255*128 < 2^15).
func svtBilTapsAVX2(off int) archsimd.Int8x32 {
	t := svtBilinearFilters2T[off&7]
	return archsimd.BroadcastInt16x16(int16(t[0] | t[1]<<8)).AsInt8x32()
}

// svtBilPairAVX2 is (a*f0 + b*f1 + 64) >> 7 on sixteen int16 lanes holding
// 8-bit values: the byte pairs (a, b) interleave as a | b<<8 per lane.
func svtBilPairAVX2(a, b archsimd.Int16x16, taps archsimd.Int8x32, bias archsimd.Int16x16) archsimd.Int16x16 {
	pairs := a.Or(b.ShiftAllLeft(8)).AsUint8x32()
	return pairs.DotProductPairsSaturated(taps).Add(bias).ShiftAllRight(7)
}

// svtBilRowHAVX2 is the first (horizontal) pass over the sixteen samples at
// p, as int16 lanes.
func svtBilRowHAVX2(p unsafe.Pointer, xoff int, taps archsimd.Int8x32, bias archsimd.Int16x16) archsimd.Int16x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendToUint16()
	switch xoff {
	case 0:
		return a.AsInt16x16()
	case 4:
		return a.Average(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 1))).ExtendToUint16()).AsInt16x16()
	}
	b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 1))).ExtendToUint16()
	return svtBilPairAVX2(a.AsInt16x16(), b.AsInt16x16(), taps, bias)
}
