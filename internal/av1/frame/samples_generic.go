// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm)

package frame

// loadSampleRows16 and storeSampleRows16 are the portable scalar reference for
// 16-bit sample staging on big-endian and unknown arches, where the in-memory
// bytes of a []uint16 do not match its little-endian serialization and the
// per-row byte copy in samples_le.go would produce wrong bytes. Every arch not
// explicitly listed as little-endian in samples_le.go's build constraint takes
// this path, so a newly added arch defaults to the correct scalar behavior.

// loadSampleRows16 decodes width little-endian uint16 samples per row from the
// byte plane src into the uint16 staging plane dst, respecting per-row strides.
func loadSampleRows16(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	srcOff := 0
	dstOff := 0
	rowBytes := width * 2
	for y := 0; y < height; y++ {
		srcLine := src[srcOff : srcOff+rowBytes : srcOff+rowBytes]
		dstLine := dst[dstOff : dstOff+width : dstOff+width]
		for x := 0; x < width; x++ {
			off := x * 2
			dstLine[x] = uint16(srcLine[off]) | uint16(srcLine[off+1])<<8
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}

// storeSampleRows16 encodes width uint16 samples per row from the staging plane
// src into little-endian bytes in the byte plane dst, respecting per-row
// strides.
func storeSampleRows16(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	dstOff := 0
	srcOff := 0
	rowBytes := width * 2
	for y := 0; y < height; y++ {
		dstLine := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
		srcLine := src[srcOff : srcOff+width : srcOff+width]
		for x, sample := range srcLine {
			off := x * 2
			dstLine[off] = byte(sample)
			dstLine[off+1] = byte(sample >> 8)
		}
		dstOff += dstStride
		srcOff += srcStride
	}
}
