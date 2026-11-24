// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package frame

import "encoding/binary"

// loadSampleRows16LE and storeSampleRows16LE stage little-endian 16-bit
// sample rows four samples at a time: one 8-byte little-endian load and four uint16
// stores (or four uint16 loads and one 8-byte store), which the compiler
// merges into single 64-bit accesses on these little-endian arches. The
// result is bit-identical to the samples_generic.go scalar reference; the
// big-endian and unknown arches keep that reference.
//
// Both stores share this helper because 16-bit samples always fit the
// destination depth, so the trusted and untrusted store paths are identical.
// Go SIMD builds bind loadSampleRows16/storeSampleRows16 to the vector rows
// in samples16_gosimd.go and keep these for the remainders.

// loadSampleRows16LE copies width little-endian uint16 samples per row from the
// byte plane src into the uint16 staging plane dst, respecting per-row strides.
func loadSampleRows16LE(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	rowBytes := width * 2
	srcOff := 0
	dstOff := 0
	for y := 0; y < height; y++ {
		d := dst[dstOff : dstOff+width : dstOff+width]
		s := src[srcOff : srcOff+rowBytes : srcOff+rowBytes]
		for len(d) >= 4 && len(s) >= 8 {
			v := binary.LittleEndian.Uint64(s)
			d[0], d[1], d[2], d[3] = uint16(v), uint16(v>>16), uint16(v>>32), uint16(v>>48)
			d, s = d[4:], s[8:]
		}
		for i := range d {
			d[i] = binary.LittleEndian.Uint16(s[2*i:])
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}

// storeSampleRows16LE copies width uint16 samples per row from the staging plane
// src into the little-endian byte plane dst, respecting per-row strides.
func storeSampleRows16LE(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	rowBytes := width * 2
	dstOff := 0
	srcOff := 0
	for y := 0; y < height; y++ {
		d := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
		s := src[srcOff : srcOff+width : srcOff+width]
		for len(s) >= 4 && len(d) >= 8 {
			binary.LittleEndian.PutUint64(d, uint64(s[0])|uint64(s[1])<<16|uint64(s[2])<<32|uint64(s[3])<<48)
			d, s = d[8:], s[4:]
		}
		for i, v := range s {
			binary.LittleEndian.PutUint16(d[2*i:], v)
		}
		dstOff += dstStride
		srcOff += srcStride
	}
}
