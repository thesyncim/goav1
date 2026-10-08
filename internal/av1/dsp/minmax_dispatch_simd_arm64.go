// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init keeps MinMaxAbsDiff8x8 on the measured NEON implementation in SIMD builds.
func init() {
	if cpu.Detected.NEON {
		minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8NEON
		return
	}
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
