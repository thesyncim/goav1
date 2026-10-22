//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// svtBilinearPredictArch is svtBilinearPredict on NEON, shaped like SVT's
// Source/Lib/ASM_NEON/variance_neon.c SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON:
// a zero phase skips its pass, the half-pel phase (taps 64/64) is a rounding
// byte average (vrhadd: (a+b+1)>>1 == (64a+64b+64)>>7), and the other phases
// run var_filter_block2d_bil_w16 (widening multiply-accumulate, round by 7).
// Both passes keep 8-bit intermediates as the C reference's values do. The
// passes are fused per 16-column strip, carrying the previous row's
// horizontal result in a register; an 8-wide block stores half a strip.
func svtBilinearPredictArch(scratch *svtSubpelScratch, ref []byte, refStride, bw, bh, xoff, yoff int) {
	if bw&7 != 0 || bw > 64 || len(ref) < bh*refStride+bw+16 {
		svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
		return
	}
	fx0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][0]))
	fx1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][1]))
	fy0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[yoff][0]))
	fy1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[yoff][1]))
	bias := archsimd.BroadcastUint16x8(64)
	base := unsafe.Pointer(&ref[0])
	dst := unsafe.Pointer(&scratch.pred[0])
	for x := 0; x < bw; x += 16 {
		p := unsafe.Add(base, x)
		prev := svtBilRowH(p, xoff, fx0, fx1, bias)
		for r := 0; r < bh; r++ {
			var out archsimd.Uint8x16
			if yoff == 0 {
				out = prev
			} else {
				p = unsafe.Add(p, refStride)
				cur := svtBilRowH(p, xoff, fx0, fx1, bias)
				if yoff == 4 {
					out = prev.Average(cur)
				} else {
					out = svtBil16(prev, cur, fy0, fy1, bias)
				}
				prev = cur
			}
			if yoff == 0 {
				p = unsafe.Add(p, refStride)
				prev = svtBilRowH(p, xoff, fx0, fx1, bias)
			}
			d := unsafe.Add(dst, r*bw+x)
			if x+16 <= bw {
				out.StoreArray((*[16]uint8)(d))
			} else {
				*(*uint64)(d) = out.ReshapeToUint64s().GetElem(0)
			}
		}
	}
}

// svtBilRowH is the first (horizontal) pass over the 16 samples at p.
func svtBilRowH(p unsafe.Pointer, xoff int, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8) archsimd.Uint8x16 {
	a := load16(p)
	switch xoff {
	case 0:
		return a
	case 4:
		return a.Average(load16(unsafe.Add(p, 1)))
	}
	return svtBil16(a, load16(unsafe.Add(p, 1)), f0, f1, bias)
}

// svtBil16 is var_filter_block2d_bil_w16's arithmetic on 16 lanes:
// (a*f0 + b*f1 + 64) >> 7.
func svtBil16(a, b, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8) archsimd.Uint8x16 {
	lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).ShiftAllRight(7)
	hi := a.HiToLo().MulWidenLo(f0).Add(b.HiToLo().MulWidenLo(f1)).Add(bias).ShiftAllRight(7)
	return lo.SaturateToUint8().ReshapeToUint64s().InterleaveLo(hi.SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
}

// svtSubpelVarianceArch is vfp->svf: the bilinear prediction of
// svtBilinearPredictArch folded straight into the residual statistics of
// svt_aom_variance (variance_neon.c variance_large_neon's accumulation:
// widened differences, int16 sums flushed to int32, squares in int32 lanes)
// instead of being stored and re-read.
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
	base := unsafe.Pointer(&ref[0])
	sbase := unsafe.Pointer(&src[0])
	var sse0, sse1, sum32 archsimd.Int32x4
	var sum16 archsimd.Int16x8
	steps := 0
	for x := 0; x < bw; x += 16 {
		p := unsafe.Add(base, x)
		s := unsafe.Add(sbase, x)
		last := bh - 1
		wide := x+16 <= bw
		prev := svtBilRowH(p, xoff, fx0, fx1, bias)
		for r := 0; r < bh; r++ {
			p = unsafe.Add(p, refStride)
			var out archsimd.Uint8x16
			switch yoff {
			case 0:
				out = prev
				prev = svtBilRowH(p, xoff, fx0, fx1, bias)
			case 4:
				cur := svtBilRowH(p, xoff, fx0, fx1, bias)
				out = prev.Average(cur)
				prev = cur
			default:
				cur := svtBilRowH(p, xoff, fx0, fx1, bias)
				out = svtBil16(prev, cur, fy0, fy1, bias)
				prev = cur
			}
			sv := load16(s)
			if r < last {
				// Stepping past the last row would leave src under checkptr.
				s = unsafe.Add(s, srcStride)
			}
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
			if steps++; steps == 32 {
				sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
				sum16 = archsimd.Int16x8{}
				steps = 0
			}
		}
	}
	sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
	total, signed := reducePixelAcc(sse0.Add(sse1), sum32)
	return varianceFromStats(uint32(total), int32(signed), uint(bits.TrailingZeros(uint(bw*bh))))
}
