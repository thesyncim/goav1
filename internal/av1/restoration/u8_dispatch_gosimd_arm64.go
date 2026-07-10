// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

func init() {
	if cpu.Detected.NEON {
		wienerVerticalU8Impl = wienerVerticalU8SIMD
	}
}
