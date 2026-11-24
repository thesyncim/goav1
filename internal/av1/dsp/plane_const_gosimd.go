// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import "simd/archsimd"

// addConstantPlaneBlock8SIMD is the 8-bit DC-only reconstruction of dav1d's
// inv_txfm_add_dct_dct eob == 0 path: one constant residual added to every
// sample of the block with unsigned saturation. clamp(d + r, 0, 255) for a
// byte d equals satsub(satadd(d, p), n) with p = min(max(r, 0), 255) and
// n = min(max(-r, 0), 255): one of them is zero, and the other saturates
// exactly where d + r leaves [0, 255].
// Exact-width 4- and 8-byte row accesses, merged by the compiler into single
// loads and stores.

func planeLE32(b *[4]byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func planeLE64(b *[8]byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func planePut32(b *[4]byte, v uint32) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func planePut64(b *[8]byte, v uint64) {
	b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

func addConstantPlaneBlock8SIMD(dst []byte, ds int, width int, height int, residual int16) {
	if width != 4 && width%8 != 0 {
		addConstantPlaneBlock8PureGo(dst, ds, width, height, residual)
		return
	}
	if height <= 0 || height > 1<<16 || uint(ds) > 1<<30 {
		addConstantPlaneBlock8PureGo(dst, ds, width, height, residual)
		return
	}
	r := int(residual)
	pv := archsimd.BroadcastUint8x16(uint8(min(max(r, 0), 0xff)))
	nv := archsimd.BroadcastUint8x16(uint8(min(max(-r, 0), 0xff)))
	switch width {
	case 4:
		for i := 0; i < height>>1; i++ {
			y := 2 * i
			pa, pb := (*[4]byte)(dst[y*ds:y*ds+4]), (*[4]byte)(dst[(y+1)*ds:(y+1)*ds+4])
			d := archsimd.BroadcastUint64x2(uint64(planeLE32(pa)) | uint64(planeLE32(pb))<<32).ReshapeToUint8s()
			w := d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0)
			planePut32(pa, uint32(w))
			planePut32(pb, uint32(w>>32))
		}
		if y := height &^ 1; y < height {
			pa := (*[4]byte)(dst[y*ds : y*ds+4])
			d := archsimd.BroadcastUint64x2(uint64(planeLE32(pa))).ReshapeToUint8s()
			planePut32(pa, uint32(d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0)))
		}
	case 8:
		for i := 0; i < height>>1; i++ {
			y := 2 * i
			pa, pb := (*[8]byte)(dst[y*ds:y*ds+8]), (*[8]byte)(dst[(y+1)*ds:(y+1)*ds+8])
			o := archsimd.LoadUint64x2Array(&[2]uint64{planeLE64(pa), planeLE64(pb)}).ReshapeToUint8s().AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s()
			planePut64(pa, o.GetElem(0))
			planePut64(pb, o.GetElem(1))
		}
		if y := height &^ 1; y < height {
			pa := (*[8]byte)(dst[y*ds : y*ds+8])
			d := archsimd.BroadcastUint64x2(planeLE64(pa)).ReshapeToUint8s()
			planePut64(pa, d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0))
		}
	case 16:
		for y := 0; y < height; y++ {
			p := (*[16]uint8)(dst[y*ds : y*ds+16])
			archsimd.LoadUint8x16Array(p).AddSaturated(pv).SubSaturated(nv).StoreArray(p)
		}
	case 32:
		for y := 0; y < height; y++ {
			pa := (*[32]byte)(dst[y*ds : y*ds+32])
			for k := 0; k < 2; k++ {
				p := (*[16]uint8)(pa[16*k : 16*k+16])
				archsimd.LoadUint8x16Array(p).AddSaturated(pv).SubSaturated(nv).StoreArray(p)
			}
		}
	case 64:
		for y := 0; y < height; y++ {
			pa := (*[64]byte)(dst[y*ds : y*ds+64])
			for k := 0; k < 4; k++ {
				p := (*[16]uint8)(pa[16*k : 16*k+16])
				archsimd.LoadUint8x16Array(p).AddSaturated(pv).SubSaturated(nv).StoreArray(p)
			}
		}
	default:
		for y := 0; y < height; y++ {
			pa := dst[y*ds : y*ds+width]
			x := 0
			for ; x+16 <= len(pa); x += 16 {
				p := (*[16]uint8)(pa[x : x+16])
				archsimd.LoadUint8x16Array(p).AddSaturated(pv).SubSaturated(nv).StoreArray(p)
			}
			if x+8 <= len(pa) {
				p := (*[8]byte)(pa[x : x+8])
				d := archsimd.BroadcastUint64x2(planeLE64(p)).ReshapeToUint8s()
				planePut64(p, d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0))
			}
		}
	}
}

// addConstantPlaneBlock16SIMD is the 16bpc DC-only reconstruction (dav1d
// itx16 inv_txfm_add_dct_dct eob == 0): clamp(d + r, 0, max) for 2-byte
// samples, as min(satsub(satadd(d, p), n), max) in uint16 lanes with
// p = max(r, 0) and n = max(-r, 0), exact for every sample value.
func addConstantPlaneBlock16SIMD(dst []byte, ds int, width int, height int, maxV uint16, residual int16) {
	if width != 4 && width%8 != 0 {
		addConstantPlaneBlock16PureGo(dst, ds, width, height, maxV, residual)
		return
	}
	if height <= 0 || height > 1<<16 || uint(ds) > 1<<30 {
		addConstantPlaneBlock16PureGo(dst, ds, width, height, maxV, residual)
		return
	}
	r := int(residual)
	pv := archsimd.BroadcastUint16x8(uint16(max(r, 0)))
	nv := archsimd.BroadcastUint16x8(uint16(max(-r, 0)))
	mv := archsimd.BroadcastUint16x8(maxV)
	if width == 4 {
		for i := 0; i < height>>1; i++ {
			y := 2 * i
			pa, pb := (*[8]byte)(dst[y*ds:y*ds+8]), (*[8]byte)(dst[(y+1)*ds:(y+1)*ds+8])
			o := archsimd.LoadUint64x2Array(&[2]uint64{planeLE64(pa), planeLE64(pb)}).ReshapeToUint16s().
				AddSaturated(pv).SubSaturated(nv).Min(mv).ReshapeToUint64s()
			planePut64(pa, o.GetElem(0))
			planePut64(pb, o.GetElem(1))
		}
		if y := height &^ 1; y < height {
			pa := (*[8]byte)(dst[y*ds : y*ds+8])
			o := archsimd.BroadcastUint64x2(planeLE64(pa)).ReshapeToUint16s().
				AddSaturated(pv).SubSaturated(nv).Min(mv).ReshapeToUint64s()
			planePut64(pa, o.GetElem(0))
		}
		return
	}
	if width == 8 || width == 16 || width == 32 {
		// Width-specialized: a fixed-size row window, constant chunk offsets.
		for y := 0; y < height; y++ {
			o := y * ds
			switch width {
			case 8:
				p := (*[16]uint8)(dst[o : o+16])
				archsimd.LoadUint8x16Array(p).ReshapeToUint16s().AddSaturated(pv).SubSaturated(nv).Min(mv).ReshapeToUint8s().StoreArray(p)
			case 16:
				row := (*[32]byte)(dst[o : o+32])
				for k := 0; k < 2; k++ {
					p := (*[16]uint8)(row[16*k : 16*k+16])
					archsimd.LoadUint8x16Array(p).ReshapeToUint16s().AddSaturated(pv).SubSaturated(nv).Min(mv).ReshapeToUint8s().StoreArray(p)
				}
			default:
				row := (*[64]byte)(dst[o : o+64])
				for k := 0; k < 4; k++ {
					p := (*[16]uint8)(row[16*k : 16*k+16])
					archsimd.LoadUint8x16Array(p).ReshapeToUint16s().AddSaturated(pv).SubSaturated(nv).Min(mv).ReshapeToUint8s().StoreArray(p)
				}
			}
		}
		return
	}
	rowBytes := 2 * width
	for y := 0; y < height; y++ {
		row := dst[y*ds : y*ds+rowBytes]
		for c := 0; c+16 <= len(row); c += 16 {
			p := (*[16]uint8)(row[c : c+16])
			archsimd.LoadUint8x16Array(p).ReshapeToUint16s().AddSaturated(pv).SubSaturated(nv).Min(mv).ReshapeToUint8s().StoreArray(p)
		}
	}
}
