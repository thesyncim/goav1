// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || purego

package dsp

// init binds the pure-Go MinMaxAbsDiff8x8 wherever no Go SIMD variant is
// selected: the default build (no GOEXPERIMENT=simd), purego builds, and
// architectures without archsimd support. This file's only purpose is to make
// the dispatch wiring symmetric across build configurations.
func init() {
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
