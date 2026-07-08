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
}
