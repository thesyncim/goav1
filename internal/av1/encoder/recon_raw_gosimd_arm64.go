//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

func init() {
	addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockSIMD
}

func addRawResidualToPredReconBlockSIMD(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	if h <= 0 || (w != 4 && w&7 != 0) {
		addRawResidualToPredReconBlockPureGo(dst, stride, pred, predStride, raw, w, h)
		return
	}
	if rawPredReconNeedsCopy(dst, stride, pred, predStride, w, h) {
		addRawResidualToPredReconBlockCopied(dst, stride, pred, predStride, raw, w, h)
		return
	}
	_ = dst[(h-1)*stride+w-1]
	_ = pred[(h-1)*predStride+w-1]
	_ = raw[h*w-1]
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	rp := unsafe.Pointer(unsafe.SliceData(raw))
	switch w {
	case 4:
		rawPredRecon4(dp, stride, pp, predStride, rp, h)
	case 8:
		rawPredRecon8(dp, stride, pp, predStride, rp, h)
	default:
		rawPredReconWide(dp, stride, pp, predStride, rp, w, h)
	}
}

// Saturating v+8 differs from the int64 sum only near MaxInt32, where both
// rounded values saturate to 32767 on narrowing. Saturating pred+residual to
// int16 before the byte clamp likewise preserves the final [0, 255] result.
func rawPredReconRoundPair(a, b unsafe.Pointer, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(a)).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	hi := archsimd.LoadInt32x4Array((*[4]int32)(b)).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	return lo.ToBits().ReshapeToUint64s().InterleaveLo(hi.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

func rawPredReconRound(p unsafe.Pointer, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	return rawPredReconRoundPair(p, unsafe.Add(p, 16), c8, sh4)
}

func rawPredReconAdd(d archsimd.Uint8x16, r archsimd.Int16x8) archsimd.Uint64x2 {
	return d.ReshapeToUint16s().BitsToInt16().AddSaturated(r).SaturateToUint8().ReshapeToUint64s()
}

func rawPredRecon4(dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, rp unsafe.Pointer, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	y := 0
	for ; y+1 < h; y += 2 {
		p := unsafe.Add(pp, y*ps)
		r := unsafe.Add(rp, y*16)
		rv := rawPredReconRoundPair(r, unsafe.Add(r, 16), c8, sh4)
		pv := archsimd.BroadcastUint64x2(uint64(*(*uint32)(p)) | uint64(*(*uint32)(unsafe.Add(p, ps)))<<32).ReshapeToUint8s()
		out := rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0)
		d := unsafe.Add(dp, y*ds)
		*(*uint32)(d) = uint32(out)
		*(*uint32)(unsafe.Add(d, ds)) = uint32(out >> 32)
	}
	if y < h {
		r := unsafe.Add(rp, y*16)
		rv := rawPredReconRoundPair(r, r, c8, sh4)
		pv := archsimd.BroadcastUint64x2(uint64(*(*uint32)(unsafe.Add(pp, y*ps)))).ReshapeToUint8s()
		*(*uint32)(unsafe.Add(dp, y*ds)) = uint32(rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0))
	}
}

func rawPredRecon8(dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, rp unsafe.Pointer, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	y := 0
	for ; y+1 < h; y += 2 {
		p := unsafe.Add(pp, y*ps)
		r := unsafe.Add(rp, y*32)
		ra := rawPredReconRound(r, c8, sh4)
		rb := rawPredReconRound(unsafe.Add(r, 32), c8, sh4)
		pair := [2]uint64{*(*uint64)(p), *(*uint64)(unsafe.Add(p, ps))}
		pv := archsimd.LoadUint64x2Array(&pair).ReshapeToUint8s()
		d := unsafe.Add(dp, y*ds)
		*(*uint64)(d) = rawPredReconAdd(pv.InterleaveLo(zero), ra).GetElem(0)
		*(*uint64)(unsafe.Add(d, ds)) = rawPredReconAdd(pv.InterleaveHi(zero), rb).GetElem(0)
	}
	if y < h {
		r := rawPredReconRound(unsafe.Add(rp, y*32), c8, sh4)
		pv := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(pp, y*ps))).ReshapeToUint8s()
		*(*uint64)(unsafe.Add(dp, y*ds)) = rawPredReconAdd(pv.InterleaveLo(zero), r).GetElem(0)
	}
}

func rawPredReconWide(dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, rp unsafe.Pointer, w, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		p := unsafe.Add(pp, y*ps)
		r := unsafe.Add(rp, 4*y*w)
		d := unsafe.Add(dp, y*ds)
		x := 0
		for ; x+16 <= w; x += 16 {
			s := unsafe.Add(r, 4*x)
			lo := rawPredReconRound(s, c8, sh4)
			hi := rawPredReconRound(unsafe.Add(s, 32), c8, sh4)
			pv := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, x)))
			a := rawPredReconAdd(pv.InterleaveLo(zero), lo)
			b := rawPredReconAdd(pv.InterleaveHi(zero), hi)
			a.InterleaveLo(b).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(d, x)))
		}
		if x < w {
			rv := rawPredReconRound(unsafe.Add(r, 4*x), c8, sh4)
			pv := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(p, x))).ReshapeToUint8s()
			*(*uint64)(unsafe.Add(d, x)) = rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0)
		}
	}
}
