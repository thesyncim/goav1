// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

// arm64 routes five-tap filters to dedicated kernels through these wrappers;
// the seven-tap SIMD kernels sit behind them.
var (
	wantWienerHorizontalU8Impl any = wienerHorizontalU8SIMDTaps
	wantWienerVerticalU8Impl   any = wienerVerticalU8SIMDTaps
)
