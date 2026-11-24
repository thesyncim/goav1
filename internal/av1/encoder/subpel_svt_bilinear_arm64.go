//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/bits"
	"simd/archsimd"
)

// svtBilRowH is the first (horizontal) pass over the 16 samples at p[0:16]
// (the taps' second sample at p[1:17]).
func svtBilRowH(p *[17]uint8, xoff int, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8, sh7 archsimd.Int16x8) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))
	switch xoff {
	case 0:
		return a
	case 4:
		return a.Average(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])))
	}
	return svtBil16(a, archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])), f0, f1, bias, sh7)
}

// svtBil16 is var_filter_block2d_bil_w16's arithmetic on 16 lanes:
// (a*f0 + b*f1 + 64) >> 7. sh7 is the shift count (-7) built once by the
// caller: ShiftAllRight(7) would rebuild it (MOVD + INS + DUP) per use. The
// high half saturates as UQXTN2 (SetElem of the low 64 bits) rather than a
// second UQXTN joined by ZIP1.
// f0 and f1 are broadcasts, so their high halves equal them; naming the
// halves lets the high products fuse to UMULL2 instead of INS + UMULL.
func svtBil16(a, b, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8, sh7 archsimd.Int16x8) archsimd.Uint8x16 {
	lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).Shift(sh7).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	hi := a.HiToLo().MulWidenLo(f0.HiToLo()).Add(b.HiToLo().MulWidenLo(f1.HiToLo())).Add(bias).Shift(sh7).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	return lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s()
}

// svtSubpelVarianceArch is vfp->svf: the bilinear prediction of
// svtBilinearPredict folded straight into the residual statistics of
// svt_aom_variance (variance_neon.c variance_large_neon's accumulation:
// widened differences, int16 sums flushed to int32, squares in int32 lanes)
// instead of being stored and re-read. Each row is one bounds-checked
// window per array.
func svtSubpelVarianceArch(scratch *svtSubpelScratch, ref []byte, refStride int, src []byte, srcStride, bw, bh, xoff, yoff int) uint32 {
	if bw&7 != 0 || bw > 64 || bh <= 0 || len(ref) < bh*refStride+bw+16 || len(src) < (bh-1)*srcStride+max(bw, 16) {
		svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
		_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
		return v
	}
	fx0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][0]))
	fx1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][1]))
	fy0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[yoff][0]))
	fy1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[yoff][1]))
	bias := archsimd.BroadcastUint16x8(64)
	sh7 := archsimd.BroadcastInt16x8(-7)
	if uint(refStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(bh) > rowsMax {
		panic(errKernelArgs)
	}
	var sse0, sse1, sum32 archsimd.Int32x4
	for x := 0; x < bw; x += 16 {
		wide := x+16 <= bw
		prev := svtBilRowH((*[17]uint8)(ref[x:x+17]), xoff, fx0, fx1, bias, sh7)
		// The int16 sum flushes after every 32 rows of the strip rather than on a
		// per-row counter, whose conditional reset made the register allocator
		// swap the accumulator with its zero on every back edge. The integer
		// sums are exact, so the flush cadence does not change the result.
		for r := 0; r < bh; {
			group := min(r+32, bh)
			var sum16 archsimd.Int16x8
			for ; r < group; r++ {
				po := (r+1)*refStride + x
				p := (*[17]uint8)(ref[po : po+17])
				var out archsimd.Uint8x16
				switch yoff {
				case 0:
					out = prev
					prev = svtBilRowH(p, xoff, fx0, fx1, bias, sh7)
				case 4:
					cur := svtBilRowH(p, xoff, fx0, fx1, bias, sh7)
					out = prev.Average(cur)
					prev = cur
				default:
					cur := svtBilRowH(p, xoff, fx0, fx1, bias, sh7)
					out = svtBil16(prev, cur, fy0, fy1, bias, sh7)
					prev = cur
				}
				so := r*srcStride + x
				sv := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16]))
				dl := out.ExtendLo8ToUint16().BitsToInt16().Sub(sv.ExtendLo8ToUint16().BitsToInt16())
				if wide {
					dh := widenHi8(out).BitsToInt16().Sub(widenHi8(sv).BitsToInt16())
					sum16 = sum16.Add(dl).Add(dh)
					sse0 = sse0.Add(dl.MulWidenLo(dl)).Add(dh.MulWidenLo(dh))
					dl, dh = dl.HiToLo(), dh.HiToLo()
					sse1 = sse1.Add(dl.MulWidenLo(dl)).Add(dh.MulWidenLo(dh))
				} else {
					sum16 = sum16.Add(dl)
					sse0 = sse0.Add(dl.MulWidenLo(dl))
					dl = dl.HiToLo()
					sse1 = sse1.Add(dl.MulWidenLo(dl))
				}
			}
			sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
		}
	}
	total, signed := reducePixelAcc(sse0.Add(sse1), sum32)
	return varianceFromStats(uint32(total), int32(signed), uint(bits.TrailingZeros(uint(bw*bh))))
}
