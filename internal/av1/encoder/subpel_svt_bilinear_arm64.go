//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtBilinearPredictArch is svtBilinearPredict on NEON: the two bilinear
// passes fused per 16- (or 8-) column strip, carrying the previous row's
// horizontal result in registers. Both passes round exactly as
// aom_var_filter_block2d_bil_{first,second}_pass_c: (a*f0 + b*f1 + 64) >> 7,
// whose sums stay below 2^16 (taps sum to 128, samples <= 255).
func svtBilinearPredictArch(scratch *svtSubpelScratch, ref []byte, refStride, bw, bh, xoff, yoff int) {
	if bw&7 != 0 || bw > 64 || len(ref) < bh*refStride+bw+16 {
		svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
		return
	}
	fx0 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][0]))
	fx1 := archsimd.BroadcastUint8x16(uint8(svtBilinearFilters2T[xoff][1]))
	fy0 := archsimd.BroadcastUint16x8(uint16(svtBilinearFilters2T[yoff][0]))
	fy1 := archsimd.BroadcastUint16x8(uint16(svtBilinearFilters2T[yoff][1]))
	bias := archsimd.BroadcastUint16x8(64)
	base := unsafe.Pointer(&ref[0])
	dst := unsafe.Pointer(&scratch.pred[0])
	for x := 0; x < bw; x += 16 {
		p := unsafe.Add(base, x)
		if x+16 <= bw {
			plo, phi := svtBilH16(p, fx0, fx1, bias)
			for r := 0; r < bh; r++ {
				p = unsafe.Add(p, refStride)
				clo, chi := svtBilH16(p, fx0, fx1, bias)
				olo := plo.MulAdd(fy0, clo.MulAdd(fy1, bias)).ShiftAllRight(7)
				ohi := phi.MulAdd(fy0, chi.MulAdd(fy1, bias)).ShiftAllRight(7)
				out := olo.SaturateToUint8().ReshapeToUint64s().InterleaveLo(ohi.SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
				out.StoreArray((*[16]uint8)(unsafe.Add(dst, r*bw+x)))
				plo, phi = clo, chi
			}
			continue
		}
		// Final 8-column strip.
		plo := svtBilH8(p, fx0, fx1, bias)
		for r := 0; r < bh; r++ {
			p = unsafe.Add(p, refStride)
			clo := svtBilH8(p, fx0, fx1, bias)
			olo := plo.MulAdd(fy0, clo.MulAdd(fy1, bias)).ShiftAllRight(7)
			*(*uint64)(unsafe.Add(dst, r*bw+x)) = olo.SaturateToUint8().ReshapeToUint64s().GetElem(0)
			plo = clo
		}
	}
}

// svtBilH16 is the first (horizontal) pass over 16 samples at p.
func svtBilH16(p unsafe.Pointer, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8) (archsimd.Uint16x8, archsimd.Uint16x8) {
	a := load16(p)
	b := load16(unsafe.Add(p, 1))
	lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).ShiftAllRight(7)
	hi := a.HiToLo().MulWidenLo(f0).Add(b.HiToLo().MulWidenLo(f1)).Add(bias).ShiftAllRight(7)
	return lo, hi
}

// svtBilH8 is the first pass over 8 samples at p (16-byte loads; the caller
// guarantees the over-read is in bounds).
func svtBilH8(p unsafe.Pointer, f0, f1 archsimd.Uint8x16, bias archsimd.Uint16x8) archsimd.Uint16x8 {
	a := load16(p)
	b := load16(unsafe.Add(p, 1))
	return a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).ShiftAllRight(7)
}
