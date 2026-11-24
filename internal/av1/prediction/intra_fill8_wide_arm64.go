// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "simd/archsimd"

// fillWide8, verticalWide8 and horizontalWide8 store 16, 32 or 64 wide rows
// sixteen bytes per vector store, as dav1d's st1 {v0.16b, ...} row stores.

func fillWide8(block planeBlock, value byte) {
	v := archsimd.BroadcastUint8x16(value)
	storeRows16(block.pix, block.stride, block.width, block.height, v, v, v, v)
}

func verticalWide8(block planeBlock, top []byte) {
	w := block.width
	top = top[:w]
	v0 := archsimd.LoadUint8x16Array((*[16]uint8)(top))
	v1, v2, v3 := v0, v0, v0
	if w >= 32 {
		v1 = archsimd.LoadUint8x16Array((*[16]uint8)(top[16:]))
	}
	if w == 64 {
		v2 = archsimd.LoadUint8x16Array((*[16]uint8)(top[32:]))
		v3 = archsimd.LoadUint8x16Array((*[16]uint8)(top[48:]))
	}
	storeRows16(block.pix, block.stride, w, block.height, v0, v1, v2, v3)
}

// storeRows16 stores h rows of w bytes (16, 32 or 64), v0..v3 per sixteen
// bytes of each row.
func storeRows16(pix []byte, stride int, w int, h int, v0, v1, v2, v3 archsimd.Uint8x16) {
	switch w {
	case 16:
		for o := 0; h > 0; h-- {
			v0.StoreArray((*[16]uint8)(pix[o : o+16]))
			o += stride
		}
	case 32:
		for o := 0; h > 0; h-- {
			line := pix[o : o+32]
			v0.StoreArray((*[16]uint8)(line[:16]))
			v1.StoreArray((*[16]uint8)(line[16:]))
			o += stride
		}
	default:
		for o := 0; h > 0; h-- {
			line := pix[o : o+64]
			v0.StoreArray((*[16]uint8)(line[:16]))
			v1.StoreArray((*[16]uint8)(line[16:32]))
			v2.StoreArray((*[16]uint8)(line[32:48]))
			v3.StoreArray((*[16]uint8)(line[48:]))
			o += stride
		}
	}
}

func horizontalWide8(block planeBlock, left []byte) {
	pix, stride, w := block.pix, block.stride, block.width
	o := 0
	for _, l := range left {
		v := archsimd.BroadcastUint8x16(l)
		switch w {
		case 16:
			v.StoreArray((*[16]uint8)(pix[o : o+16]))
		case 32:
			line := pix[o : o+32]
			v.StoreArray((*[16]uint8)(line[:16]))
			v.StoreArray((*[16]uint8)(line[16:]))
		default:
			line := pix[o : o+64]
			v.StoreArray((*[16]uint8)(line[:16]))
			v.StoreArray((*[16]uint8)(line[16:32]))
			v.StoreArray((*[16]uint8)(line[32:48]))
			v.StoreArray((*[16]uint8)(line[48:]))
		}
		o += stride
	}
}
