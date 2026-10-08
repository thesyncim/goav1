// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package frame

import "unsafe"

// loadSampleRows16 and storeSampleRows16 stage little-endian 16-bit sample
// rows. On a little-endian host the in-memory bytes of a []uint16 are
// byte-identical to that slice's little-endian serialization, so the per-sample
// OR/shift (load) and byte-split (store) loops reduce to a plain per-row copy of
// width*2 bytes. copy() lowers to runtime.memmove (SIMD-fast) and, unlike the
// scalar loops, moves whole cache lines at a time. The result is bit-identical
// to the samples_generic.go scalar reference on every little-endian arch listed
// in the build constraint; big-endian and unknown arches keep that reference.
//
// Both stores share this helper because 16-bit samples always fit the
// destination depth, so the trusted and untrusted store paths are identical.

// loadSampleRows16 copies width little-endian uint16 samples per row from the
// byte plane src into the uint16 staging plane dst, respecting per-row strides.
func loadSampleRows16(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	rowBytes := width * 2
	srcOff := 0
	dstOff := 0
	for y := 0; y < height; y++ {
		dstLine := dst[dstOff : dstOff+width : dstOff+width]
		// View exactly the row's width*2 sample bytes; the pointer and length
		// stay within dstLine, so the unsafe view never escapes its row.
		dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dstLine[0])), rowBytes)
		copy(dstBytes, src[srcOff:srcOff+rowBytes:srcOff+rowBytes])
		srcOff += srcStride
		dstOff += dstStride
	}
}

// storeSampleRows16 copies width uint16 samples per row from the staging plane
// src into the little-endian byte plane dst, respecting per-row strides.
func storeSampleRows16(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	rowBytes := width * 2
	dstOff := 0
	srcOff := 0
	for y := 0; y < height; y++ {
		srcLine := src[srcOff : srcOff+width : srcOff+width]
		srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&srcLine[0])), rowBytes)
		copy(dst[dstOff:dstOff+rowBytes:dstOff+rowBytes], srcBytes)
		dstOff += dstStride
		srcOff += srcStride
	}
}
