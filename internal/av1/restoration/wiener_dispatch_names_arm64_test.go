// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// wienerSIMDDetected and sgrSIMDDetected report whether init bound the SIMD
// Wiener and self-guided kernels (NEON is mandatory on arm64).
func wienerSIMDDetected() bool { return cpu.Detected.NEON }

func sgrSIMDDetected() bool { return cpu.Detected.NEON }

// sgrSIMDBoxsumKernel is the box-sum kernel bound alongside the SIMD
// self-guided drivers: the separable running sum on arm64.
var sgrSIMDBoxsumKernel = boxsumSeparable

// arm64 routes five-tap filters to dedicated kernels through these wrappers;
// the seven-tap SIMD kernels sit behind them.
var (
	wantWienerHorizontalU8Impl any = wienerHorizontalU8SIMDTaps
	wantWienerVerticalU8Impl   any = wienerVerticalU8SIMDTaps
)
