// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm) && (purego || !goexperiment.simd || (!arm64 && !amd64))

package frame

// loadSampleRows16 binds the scalar little-endian staging rows on builds
// without the Go SIMD rows (samples16_gosimd.go).
func loadSampleRows16(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	loadSampleRows16LE(dst, dstStride, src, srcStride, width, height)
}

// storeSampleRows16 binds the scalar little-endian staging rows on builds
// without the Go SIMD rows (samples16_gosimd.go).
func storeSampleRows16(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	storeSampleRows16LE(dst, dstStride, src, srcStride, width, height)
}
