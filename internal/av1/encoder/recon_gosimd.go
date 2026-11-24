//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import "simd/archsimd"

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	copyPredToReconBlockImpl = copyPredToReconBlockSIMD
	addConstantResidualToPredReconBlockImpl = addConstantResidualToPredReconBlockSIMD
}

// copyPredToReconBlockSIMD is copyPredToReconBlockPureGo with fixed-size row
// copies (vector moves) instead of a memmove call per row.
func copyPredToReconBlockSIMD(reconPlane, pred []byte, predStride, stride, px, py, w, h int) {
	if h <= 0 || w&7 != 0 {
		copyPredToReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h)
		return
	}
	copyBlockRows(reconPlane[py*stride+px:], stride, pred, predStride, w, h)
}

// copyBlockRows copies the w x h block src (stride ss) to dst (stride ds), w a
// multiple of 8: one loop per block width, each taking one bounds-checked
// window per row pair and array (row 2i at w[:n], row 2i+1 at w[stride:]).
func copyBlockRows(dst []byte, ds int, src []byte, ss, w, h int) {
	if uint(ds) > rowStrideMax || uint(ss) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	switch w {
	case 8:
		dn, sn := ds+8, ss+8
		for i := 0; i < h/2; i++ {
			do, so := 2*i*ds, 2*i*ss
			dw, sw := dst[do:do+dn:do+dn], src[so:so+sn:so+sn]
			*(*[8]byte)(dw) = *(*[8]byte)(sw)
			*(*[8]byte)(dw[ds:]) = *(*[8]byte)(sw[ss:])
		}
	case 16:
		dn, sn := ds+16, ss+16
		for i := 0; i < h/2; i++ {
			do, so := 2*i*ds, 2*i*ss
			dw, sw := dst[do:do+dn:do+dn], src[so:so+sn:so+sn]
			*(*[16]byte)(dw) = *(*[16]byte)(sw)
			*(*[16]byte)(dw[ds:]) = *(*[16]byte)(sw[ss:])
		}
	case 32:
		dn, sn := ds+32, ss+32
		for i := 0; i < h/2; i++ {
			do, so := 2*i*ds, 2*i*ss
			dw, sw := dst[do:do+dn:do+dn], src[so:so+sn:so+sn]
			*(*[32]byte)(dw) = *(*[32]byte)(sw)
			*(*[32]byte)(dw[ds:]) = *(*[32]byte)(sw[ss:])
		}
	case 64:
		dn, sn := ds+64, ss+64
		for i := 0; i < h/2; i++ {
			do, so := 2*i*ds, 2*i*ss
			dw, sw := dst[do:do+dn:do+dn], src[so:so+sn:so+sn]
			*(*[64]byte)(dw) = *(*[64]byte)(sw)
			*(*[64]byte)(dw[ds:]) = *(*[64]byte)(sw[ss:])
		}
	default:
		for y := range h {
			d, s := dst[y*ds:y*ds+w], src[y*ss:y*ss+w]
			d = d[:len(s)]
			x := 0
			for ; x+16 <= len(s); x += 16 {
				*(*[16]byte)(d[x : x+16]) = *(*[16]byte)(s[x : x+16])
			}
			if x+8 <= len(s) {
				*(*[8]byte)(d[x : x+8]) = *(*[8]byte)(s[x : x+8])
			}
		}
		return
	}
	if h&1 != 0 {
		y := h - 1
		copy(dst[y*ds:y*ds+w], src[y*ss:y*ss+w])
	}
}

// addConstantResidualToPredReconBlockSIMD is
// addConstantResidualToPredReconBlockPureGo on 16-byte vectors: with the
// residual clamped to [-255, 255] (beyond which the clamp already pins every
// sample), pred + r clamped to [0, 255] is a saturating byte add of r or a
// saturating byte subtract of -r. Rows are bounds-checked windows: one per
// row pair for 8-wide blocks, one per row otherwise.
func addConstantResidualToPredReconBlockSIMD(reconPlane, pred []byte, predStride, stride, px, py, w, h int, residual int16) {
	if h <= 0 || (w&15 != 0 && (w != 8 || h&1 != 0)) {
		addConstantResidualToPredReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h, residual)
		return
	}
	if uint(stride) > rowStrideMax || uint(predStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	dst := reconPlane[py*stride+px:]
	r := max(min(int(residual), 255), -255)
	add := archsimd.BroadcastUint8x16(uint8(max(r, 0)))
	sub := archsimd.BroadcastUint8x16(uint8(max(-r, 0)))
	if w == 8 {
		// Two 8-byte rows per vector, staged through memory (cheaper than
		// lane inserts on Apple cores).
		var pair [16]uint8
		dn, sn := stride+8, predStride+8
		for i := 0; i < h/2; i++ {
			do, so := 2*i*stride, 2*i*predStride
			dw, sw := dst[do:do+dn:do+dn], pred[so:so+sn:so+sn]
			*(*[8]byte)(pair[0:8]) = *(*[8]byte)(sw)
			*(*[8]byte)(pair[8:16]) = *(*[8]byte)(sw[predStride:])
			archsimd.LoadUint8x16Array(&pair).AddSaturated(add).SubSaturated(sub).StoreArray(&pair)
			*(*[8]byte)(dw) = *(*[8]byte)(pair[0:8])
			*(*[8]byte)(dw[stride:]) = *(*[8]byte)(pair[8:16])
		}
		return
	}
	switch w {
	case 16:
		dn, sn := stride+16, predStride+16
		for i := 0; i < h/2; i++ {
			do, so := 2*i*stride, 2*i*predStride
			dw, sw := dst[do:do+dn:do+dn], pred[so:so+sn:so+sn]
			addConst16((*[16]uint8)(dw), (*[16]uint8)(sw), add, sub, r >= 0)
			addConst16((*[16]uint8)(dw[stride:]), (*[16]uint8)(sw[predStride:]), add, sub, r >= 0)
		}
		if h&1 != 0 {
			y := h - 1
			addConst16((*[16]uint8)(dst[y*stride:y*stride+16]), (*[16]uint8)(pred[y*predStride:y*predStride+16]), add, sub, r >= 0)
		}
		return
	case 32:
		for y := 0; y < h; y++ {
			d, s := (*[32]uint8)(dst[y*stride:y*stride+32]), (*[32]uint8)(pred[y*predStride:y*predStride+32])
			addConst16((*[16]uint8)(d[0:16]), (*[16]uint8)(s[0:16]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[16:32]), (*[16]uint8)(s[16:32]), add, sub, r >= 0)
		}
		return
	case 64:
		for y := 0; y < h; y++ {
			d, s := (*[64]uint8)(dst[y*stride:y*stride+64]), (*[64]uint8)(pred[y*predStride:y*predStride+64])
			addConst16((*[16]uint8)(d[0:16]), (*[16]uint8)(s[0:16]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[16:32]), (*[16]uint8)(s[16:32]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[32:48]), (*[16]uint8)(s[32:48]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[48:64]), (*[16]uint8)(s[48:64]), add, sub, r >= 0)
		}
		return
	}
	for y := 0; y < h; y++ {
		d, s := dst[y*stride:y*stride+w], pred[y*predStride:y*predStride+w]
		for c := 0; c+16 <= w; c += 16 {
			addConst16((*[16]uint8)(d[c:c+16]), (*[16]uint8)(s[c:c+16]), add, sub, r >= 0)
		}
	}
}

// addConst16 stores the 16 samples s plus the clamped residual into d: a
// saturating add of add when pos, else a saturating subtract of sub.
func addConst16(d, s *[16]uint8, add, sub archsimd.Uint8x16, pos bool) {
	v := archsimd.LoadUint8x16Array(s)
	if pos {
		v = v.AddSaturated(add)
	} else {
		v = v.SubSaturated(sub)
	}
	v.StoreArray(d)
}
