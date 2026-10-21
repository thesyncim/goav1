//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// bilinearSubpelSADImpl is the NEON form of bilinearSubpelSADScalar. Widths
// that are a multiple of 16 run as 16-column strips whose horizontal rows
// ride in registers between the two passes (UMULL widening: the bilinear
// taps fit a byte); 8-wide blocks use one 8-lane chunk. The caller
// guarantees the 16-byte over-reads.
func bilinearSubpelSADImpl(src []byte, srcStride int, ref []byte, refStride, n, fx, fy int) int {
	hf := bilinearFilters2T[fx]
	vf := bilinearFilters2T[fy]
	rp := unsafe.Pointer(&ref[0])
	sp := unsafe.Pointer(&src[0])
	switch {
	case n&15 == 0:
		total := 0
		for c := 0; c < n; c += 16 {
			total += bilinearStrip16(unsafe.Add(sp, c), srcStride, unsafe.Add(rp, c), refStride, n, hf, vf)
		}
		return total
	case n == 8:
		return bilinearStrip8(sp, srcStride, rp, refStride, hf, vf)
	}
	return bilinearSubpelSADScalar(src, srcStride, ref, refStride, n, fx, fy)
}

// bilinearStrip16 scores one 16-column strip of rows rows.
func bilinearStrip16(sp unsafe.Pointer, srcStride int, rp unsafe.Pointer, refStride, rows int, hf, vf [2]uint16) int {
	h0 := archsimd.BroadcastUint8x16(uint8(hf[0]))
	h1 := archsimd.BroadcastUint8x16(uint8(hf[1]))
	v0 := archsimd.BroadcastUint16x8(vf[0])
	v1 := archsimd.BroadcastUint16x8(vf[1])
	round := archsimd.BroadcastUint16x8(64)
	a := load16(rp)
	b := load16(unsafe.Add(rp, 1))
	prevLo := a.MulWidenLo(h0).Add(b.MulWidenLo(h1)).Add(round).ShiftAllRight(7)
	prevHi := a.HiToLo().MulWidenLo(h0).Add(b.HiToLo().MulWidenLo(h1)).Add(round).ShiftAllRight(7)
	acc := archsimd.BroadcastUint16x8(0)
	total := 0
	for r := 0; r < rows; r++ {
		rp = unsafe.Add(rp, refStride)
		a = load16(rp)
		b = load16(unsafe.Add(rp, 1))
		curLo := a.MulWidenLo(h0).Add(b.MulWidenLo(h1)).Add(round).ShiftAllRight(7)
		curHi := a.HiToLo().MulWidenLo(h0).Add(b.HiToLo().MulWidenLo(h1)).Add(round).ShiftAllRight(7)
		pLo := prevLo.Mul(v0).Add(curLo.Mul(v1)).Add(round).ShiftAllRight(7)
		pHi := prevHi.Mul(v0).Add(curHi.Mul(v1)).Add(round).ShiftAllRight(7)
		prevLo, prevHi = curLo, curHi
		s := load16(unsafe.Add(sp, r*srcStride))
		sLo := s.ExtendLo8ToUint16()
		sHi := s.HiToLo().ExtendLo8ToUint16()
		acc = acc.Add(pLo.Max(sLo).Sub(pLo.Min(sLo))).Add(pHi.Max(sHi).Sub(pHi.Min(sHi)))
		// Each lane gains at most 2*255 per row; flush every 64 rows.
		if r&63 == 63 {
			total += sumU16x8(acc)
			acc = archsimd.BroadcastUint16x8(0)
		}
	}
	return total + sumU16x8(acc)
}

// bilinearStrip8 scores an 8 x 8 block.
func bilinearStrip8(sp unsafe.Pointer, srcStride int, rp unsafe.Pointer, refStride int, hf, vf [2]uint16) int {
	h0 := archsimd.BroadcastUint8x16(uint8(hf[0]))
	h1 := archsimd.BroadcastUint8x16(uint8(hf[1]))
	v0 := archsimd.BroadcastUint16x8(vf[0])
	v1 := archsimd.BroadcastUint16x8(vf[1])
	round := archsimd.BroadcastUint16x8(64)
	prev := load16(rp).MulWidenLo(h0).Add(load16(unsafe.Add(rp, 1)).MulWidenLo(h1)).Add(round).ShiftAllRight(7)
	acc := archsimd.BroadcastUint16x8(0)
	for r := 0; r < 8; r++ {
		rp = unsafe.Add(rp, refStride)
		cur := load16(rp).MulWidenLo(h0).Add(load16(unsafe.Add(rp, 1)).MulWidenLo(h1)).Add(round).ShiftAllRight(7)
		p := prev.Mul(v0).Add(cur.Mul(v1)).Add(round).ShiftAllRight(7)
		prev = cur
		s := load16(unsafe.Add(sp, r*srcStride)).ExtendLo8ToUint16()
		acc = acc.Add(p.Max(s).Sub(p.Min(s)))
	}
	return sumU16x8(acc)
}

// sumU16x8 returns the exact sum of the eight lanes.
func sumU16x8(acc archsimd.Uint16x8) int {
	return int(acc.ExtendLo4ToUint32().Add(acc.HiToLo().ExtendLo4ToUint32()).ReduceSum())
}
