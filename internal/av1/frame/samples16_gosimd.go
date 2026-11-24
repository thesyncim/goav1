// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package frame

import "simd/archsimd"

// loadSampleRows16 copies width little-endian uint16 samples per row from the
// byte plane src into the uint16 staging plane dst. On these little-endian
// arches a 16-byte load reinterpreted as eight uint16 lanes is the decode, so
// each row moves 32 samples per step through one checked window of each
// buffer, then 8 at a time; the last few samples take the scalar rows.
func loadSampleRows16(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if !sampleRows16UseSIMD() {
		loadSampleRows16LE(dst, dstStride, src, srcStride, width, height)
		return
	}
	if width <= 0 || height <= 0 {
		return
	}
	rowBytes := width * 2
	srcOff := 0
	dstOff := 0
	for y := 0; y < height; y++ {
		d := dst[dstOff : dstOff+width : dstOff+width]
		s := src[srcOff : srcOff+rowBytes : srcOff+rowBytes]
		for len(d) >= 32 && len(s) >= 64 {
			dw, sw := (*[32]uint16)(d), (*[64]uint8)(s)
			archsimd.LoadUint8x16Array((*[16]uint8)(sw[0:16])).ReshapeToUint16s().StoreArray((*[8]uint16)(dw[0:8]))
			archsimd.LoadUint8x16Array((*[16]uint8)(sw[16:32])).ReshapeToUint16s().StoreArray((*[8]uint16)(dw[8:16]))
			archsimd.LoadUint8x16Array((*[16]uint8)(sw[32:48])).ReshapeToUint16s().StoreArray((*[8]uint16)(dw[16:24]))
			archsimd.LoadUint8x16Array((*[16]uint8)(sw[48:64])).ReshapeToUint16s().StoreArray((*[8]uint16)(dw[24:32]))
			d, s = d[32:], s[64:]
		}
		for len(d) >= 8 && len(s) >= 16 {
			archsimd.LoadUint8x16Array((*[16]uint8)(s)).ReshapeToUint16s().StoreArray((*[8]uint16)(d))
			d, s = d[8:], s[16:]
		}
		if len(d) > 0 {
			loadSampleRows16LE(d, 0, s, 0, len(d), 1)
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}

// storeSampleRows16 is loadSampleRows16 in reverse: eight uint16 lanes
// reinterpreted as sixteen bytes are their little-endian encoding.
func storeSampleRows16(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	if !sampleRows16UseSIMD() {
		storeSampleRows16LE(dst, dstStride, src, srcStride, width, height)
		return
	}
	if width <= 0 || height <= 0 {
		return
	}
	rowBytes := width * 2
	dstOff := 0
	srcOff := 0
	for y := 0; y < height; y++ {
		d := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
		s := src[srcOff : srcOff+width : srcOff+width]
		for len(s) >= 32 && len(d) >= 64 {
			sw, dw := (*[32]uint16)(s), (*[64]uint8)(d)
			archsimd.LoadUint16x8Array((*[8]uint16)(sw[0:8])).ReshapeToUint8s().StoreArray((*[16]uint8)(dw[0:16]))
			archsimd.LoadUint16x8Array((*[8]uint16)(sw[8:16])).ReshapeToUint8s().StoreArray((*[16]uint8)(dw[16:32]))
			archsimd.LoadUint16x8Array((*[8]uint16)(sw[16:24])).ReshapeToUint8s().StoreArray((*[16]uint8)(dw[32:48]))
			archsimd.LoadUint16x8Array((*[8]uint16)(sw[24:32])).ReshapeToUint8s().StoreArray((*[16]uint8)(dw[48:64]))
			d, s = d[64:], s[32:]
		}
		for len(s) >= 8 && len(d) >= 16 {
			archsimd.LoadUint16x8Array((*[8]uint16)(s)).ReshapeToUint8s().StoreArray((*[16]uint8)(d))
			d, s = d[16:], s[8:]
		}
		if len(s) > 0 {
			storeSampleRows16LE(d, 0, s, 0, len(s), 1)
		}
		dstOff += dstStride
		srcOff += srcStride
	}
}
