// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// init binds the Go-native SIMD four-column DCT8 kernel under GOEXPERIMENT=simd
// (Go 1.27+). It is byte-exact with the scalar column pass and faster than the
// hand NEON asm (int32 4-wide round-once vs int64 2-wide). See SIMD_PORT.md.
func init() {
	inverseDCT8Col4Impl = inverseDCT8Col4SIMD
	inverseDCT16Col4Impl = inverseDCT16Col4SIMD
	// inverseDCT32Col4SIMD (colpass_gosimd32_arm64.go) measured ~2.5% behind
	// inverseDCT32Col4NEON after every optimize-go-simd pattern (raw pointers,
	// rodata constants via an opaque base, bias-seeded MLA chains, full even-
	// part textual inline; 0 spills, 0 calls, exact multiply/clip parity). The
	// asm is clang -O2 output of the identical algebra, so the residual is the
	// documented LLVM scheduling gap; the binding stays on the asm.
}
