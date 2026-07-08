// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// Bind the int16 8-wide SIMD DCT8 column kernel under GOEXPERIMENT=simd. It is
// byte-exact and ~3.8x faster than the NEON asm (no boundary narrowing).
func init() {
	inverseDCT8Col8Impl16 = inverseDCT8Col8SIMD16
	inverseDCT16Col8Impl16 = inverseDCT16Col8SIMD16
	inverseDCT32Col8Impl16 = inverseDCT32Col8SIMD16
	inverseDCT64Col8Impl16 = inverseDCT64Col8SIMD16
	int16ColumnFast = true
}
