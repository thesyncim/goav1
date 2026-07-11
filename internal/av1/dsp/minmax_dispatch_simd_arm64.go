// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD MinMaxAbsDiff8x8 kernel under GOEXPERIMENT=simd,
// replacing the NEON asm binding (minmax_dispatch_arm64.go is excluded by
// !goexperiment.simd). See SIMD_PORT.md.
func init() {
	if cpu.Detected.NEON {
		minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8SIMD
		return
	}
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
