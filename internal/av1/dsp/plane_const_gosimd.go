// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import (
	"simd/archsimd"
	"unsafe"
)

// addConstantPlaneBlock8SIMD is the 8-bit DC-only reconstruction of dav1d's
// inv_txfm_add_dct_dct eob == 0 path: one constant residual added to every
// sample of the block with unsigned saturation. clamp(d + r, 0, 255) for a
// byte d equals satsub(satadd(d, p), n) with p = min(max(r, 0), 255) and
// n = min(max(-r, 0), 255): one of them is zero, and the other saturates
// exactly where d + r leaves [0, 255].
func addConstantPlaneBlock8SIMD(dst []byte, ds int, width int, height int, residual int16) {
	if width != 4 && width%8 != 0 {
		addConstantPlaneBlock8PureGo(dst, ds, width, height, residual)
		return
	}
	_ = dst[(height-1)*ds+width-1]
	r := int(residual)
	pv := archsimd.BroadcastUint8x16(uint8(min(max(r, 0), 0xff)))
	nv := archsimd.BroadcastUint8x16(uint8(min(max(-r, 0), 0xff)))
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	switch width {
	case 4:
		y := 0
		for ; y+1 < height; y += 2 {
			pa := unsafe.Add(dp, y*ds)
			pb := unsafe.Add(pa, ds)
			d := archsimd.BroadcastUint64x2(uint64(*(*uint32)(pa)) | uint64(*(*uint32)(pb))<<32).ReshapeToUint8s()
			w := d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0)
			*(*uint32)(pa) = uint32(w)
			*(*uint32)(pb) = uint32(w >> 32)
		}
		if y < height {
			pa := unsafe.Add(dp, y*ds)
			d := archsimd.BroadcastUint64x2(uint64(*(*uint32)(pa))).ReshapeToUint8s()
			*(*uint32)(pa) = uint32(d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0))
		}
	case 8:
		y := 0
		for ; y+1 < height; y += 2 {
			pa := unsafe.Add(dp, y*ds)
			pb := unsafe.Add(pa, ds)
			o := archsimd.LoadUint64x2Array(&[2]uint64{*(*uint64)(pa), *(*uint64)(pb)}).ReshapeToUint8s().AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s()
			*(*uint64)(pa) = o.GetElem(0)
			*(*uint64)(pb) = o.GetElem(1)
		}
		if y < height {
			pa := unsafe.Add(dp, y*ds)
			d := archsimd.BroadcastUint64x2(*(*uint64)(pa)).ReshapeToUint8s()
			*(*uint64)(pa) = d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0)
		}
	default:
		for y := 0; y < height; y++ {
			pa := unsafe.Add(dp, y*ds)
			x := 0
			for ; x+16 <= width; x += 16 {
				p := (*[16]uint8)(unsafe.Add(pa, x))
				archsimd.LoadUint8x16Array(p).AddSaturated(pv).SubSaturated(nv).StoreArray(p)
			}
			if x < width {
				p := unsafe.Add(pa, x)
				d := archsimd.BroadcastUint64x2(*(*uint64)(p)).ReshapeToUint8s()
				*(*uint64)(p) = d.AddSaturated(pv).SubSaturated(nv).ReshapeToUint64s().GetElem(0)
			}
		}
	}
}
