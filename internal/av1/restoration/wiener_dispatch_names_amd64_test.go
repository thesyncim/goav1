// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

var (
	wantWienerHorizontalU8Impl any = wienerHorizontalU8SIMDChecked
	wantWienerVerticalU8Impl   any = wienerVerticalU8SIMD
)
