//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

func init() {
	addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockSIMD
}

// addRawResidualToPredReconBlockSIMD reads and writes through bounds-checked
// windows: one per row pair and array for the 4- and 8-wide blocks (row 2i
// at w[:n], row 2i+1 at w[stride:]), one per row for the wider ones.
func addRawResidualToPredReconBlockSIMD(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	if h <= 0 || (w != 4 && w&7 != 0) {
		addRawResidualToPredReconBlockPureGo(dst, stride, pred, predStride, raw, w, h)
		return
	}
	if uint(stride) > rowStrideMax || uint(predStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	switch w {
	case 4:
		rawPredRecon4(dst, stride, pred, predStride, raw, h)
	case 8:
		rawPredRecon8(dst, stride, pred, predStride, raw, h)
	default:
		rawPredReconWide(dst, stride, pred, predStride, raw, w, h)
	}
}

// Saturating v+8 differs from the int64 sum only near MaxInt32, where both
// rounded values saturate to 32767 on narrowing. Saturating pred+residual to
// int16 before the byte clamp likewise preserves the final [0, 255] result.
func rawPredReconRoundPair(a, b *[4]int32, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array(a).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	hi := archsimd.LoadInt32x4Array(b).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	return lo.ToBits().ReshapeToUint64s().InterleaveLo(hi.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

func rawPredReconRound(r *[8]int32, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	return rawPredReconRoundPair((*[4]int32)(r[0:4]), (*[4]int32)(r[4:8]), c8, sh4)
}

func rawPredReconAdd(d archsimd.Uint8x16, r archsimd.Int16x8) archsimd.Uint64x2 {
	return d.ReshapeToUint16s().BitsToInt16().AddSaturated(r).SaturateToUint8().ReshapeToUint64s()
}

// le32 is the little-endian uint32 of b (one 4-byte load).
func le32(b *[4]byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// put32 stores v little-endian into b (one 4-byte store).
func put32(b *[4]byte, v uint32) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func rawPredRecon4(dst []byte, ds int, pred []byte, ps int, raw []int32, h int) {
	if uint(ds) > rowStrideMax || uint(ps) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	dn, pn := ds+4, ps+4
	for i := 0; i < h/2; i++ {
		po, do := 2*i*ps, 2*i*ds
		pw := pred[po : po+pn : po+pn]
		r := (*[8]int32)(raw[8*i : 8*i+8])
		rv := rawPredReconRound(r, c8, sh4)
		pv := archsimd.BroadcastUint64x2(uint64(le32((*[4]byte)(pw))) | uint64(le32((*[4]byte)(pw[ps:])))<<32).ReshapeToUint8s()
		out := rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0)
		dw := dst[do : do+dn : do+dn]
		put32((*[4]byte)(dw), uint32(out))
		put32((*[4]byte)(dw[ds:]), uint32(out>>32))
	}
	if h&1 != 0 {
		y := h - 1
		r := (*[4]int32)(raw[4*y : 4*y+4])
		rv := rawPredReconRoundPair(r, r, c8, sh4)
		pv := archsimd.BroadcastUint64x2(uint64(le32((*[4]byte)(pred[y*ps : y*ps+4])))).ReshapeToUint8s()
		put32((*[4]byte)(dst[y*ds:y*ds+4]), uint32(rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0)))
	}
}

func rawPredRecon8(dst []byte, ds int, pred []byte, ps int, raw []int32, h int) {
	if uint(ds) > rowStrideMax || uint(ps) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	dn, pn := ds+8, ps+8
	for i := 0; i < h/2; i++ {
		po, do := 2*i*ps, 2*i*ds
		pw := pred[po : po+pn : po+pn]
		r := (*[16]int32)(raw[16*i : 16*i+16])
		ra := rawPredReconRound((*[8]int32)(r[0:8]), c8, sh4)
		rb := rawPredReconRound((*[8]int32)(r[8:16]), c8, sh4)
		pair := [2]uint64{le64((*[8]byte)(pw)), le64((*[8]byte)(pw[ps:]))}
		pv := archsimd.LoadUint64x2Array(&pair).ReshapeToUint8s()
		dw := dst[do : do+dn : do+dn]
		put64((*[8]byte)(dw), rawPredReconAdd(pv.InterleaveLo(zero), ra).GetElem(0))
		put64((*[8]byte)(dw[ds:]), rawPredReconAdd(pv.InterleaveHi(zero), rb).GetElem(0))
	}
	if h&1 != 0 {
		y := h - 1
		r := rawPredReconRound((*[8]int32)(raw[8*y:8*y+8]), c8, sh4)
		pv := archsimd.BroadcastUint64x2(le64((*[8]byte)(pred[y*ps : y*ps+8]))).ReshapeToUint8s()
		put64((*[8]byte)(dst[y*ds:y*ds+8]), rawPredReconAdd(pv.InterleaveLo(zero), r).GetElem(0))
	}
}

func rawPredReconWide(dst []byte, ds int, pred []byte, ps int, raw []int32, w, h int) {
	if uint(ds) > rowStrideMax || uint(ps) > rowStrideMax || uint(h) > rowsMax || uint(w) > rowsMax {
		panic(errKernelArgs)
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	switch w {
	case 16:
		// One window per row pair and array; the raw rows are consecutive
		// 16-entry arrays.
		dn, pn := ds+16, ps+16
		for i := 0; i < h/2; i++ {
			po, do := 2*i*ps, 2*i*ds
			pw, dw := pred[po:po+pn:po+pn], dst[do:do+dn:do+dn]
			r := (*[32]int32)(raw[32*i : 32*i+32])
			rawPredRecon16((*[16]uint8)(dw), (*[16]uint8)(pw), (*[16]int32)(r[0:16]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(dw[ds:]), (*[16]uint8)(pw[ps:]), (*[16]int32)(r[16:32]), zero, c8, sh4)
		}
		if h&1 != 0 {
			y := h - 1
			rawPredRecon16((*[16]uint8)(dst[y*ds:y*ds+16]), (*[16]uint8)(pred[y*ps:y*ps+16]), (*[16]int32)(raw[16*y:16*y+16]), zero, c8, sh4)
		}
		return
	case 32:
		for y := 0; y < h; y++ {
			p := (*[32]uint8)(pred[y*ps : y*ps+32])
			d := (*[32]uint8)(dst[y*ds : y*ds+32])
			r := (*[32]int32)(raw[32*y : 32*y+32])
			rawPredRecon16((*[16]uint8)(d[0:16]), (*[16]uint8)(p[0:16]), (*[16]int32)(r[0:16]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[16:32]), (*[16]uint8)(p[16:32]), (*[16]int32)(r[16:32]), zero, c8, sh4)
		}
		return
	case 64:
		for y := 0; y < h; y++ {
			p := (*[64]uint8)(pred[y*ps : y*ps+64])
			d := (*[64]uint8)(dst[y*ds : y*ds+64])
			r := (*[64]int32)(raw[64*y : 64*y+64])
			rawPredRecon16((*[16]uint8)(d[0:16]), (*[16]uint8)(p[0:16]), (*[16]int32)(r[0:16]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[16:32]), (*[16]uint8)(p[16:32]), (*[16]int32)(r[16:32]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[32:48]), (*[16]uint8)(p[32:48]), (*[16]int32)(r[32:48]), zero, c8, sh4)
			rawPredRecon16((*[16]uint8)(d[48:64]), (*[16]uint8)(p[48:64]), (*[16]int32)(r[48:64]), zero, c8, sh4)
		}
		return
	}
	for y := 0; y < h; y++ {
		p := pred[y*ps : y*ps+w]
		d := dst[y*ds : y*ds+w]
		r := raw[y*w : y*w+w]
		x := 0
		for ; x+16 <= w; x += 16 {
			rawPredRecon16((*[16]uint8)(d[x:x+16]), (*[16]uint8)(p[x:x+16]), (*[16]int32)(r[x:x+16]), zero, c8, sh4)
		}
		if x+8 <= w {
			rv := rawPredReconRound((*[8]int32)(r[x:x+8]), c8, sh4)
			pv := archsimd.BroadcastUint64x2(le64((*[8]byte)(p[x : x+8]))).ReshapeToUint8s()
			put64((*[8]byte)(d[x:x+8]), rawPredReconAdd(pv.InterleaveLo(zero), rv).GetElem(0))
		}
	}
}

// rawPredRecon16 writes 16 samples: d = clip(p + round(r)).
func rawPredRecon16(d, p *[16]uint8, r *[16]int32, zero archsimd.Uint8x16, c8, sh4 archsimd.Int32x4) {
	lo := rawPredReconRound((*[8]int32)(r[0:8]), c8, sh4)
	hi := rawPredReconRound((*[8]int32)(r[8:16]), c8, sh4)
	pv := archsimd.LoadUint8x16Array(p)
	a := rawPredReconAdd(pv.InterleaveLo(zero), lo)
	b := rawPredReconAdd(pv.InterleaveHi(zero), hi)
	a.InterleaveLo(b).ReshapeToUint8s().StoreArray(d)
}
