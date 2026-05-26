// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !amd64 && !arm64

package dsp

// init binds the pure-Go MinMaxAbsDiff8x8 variant on architectures that the
// DSP dispatcher does not yet special-case. This file's only purpose is to
// make the dispatch wiring symmetric across architectures.
func init() {
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
