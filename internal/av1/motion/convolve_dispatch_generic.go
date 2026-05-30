// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !amd64 && !arm64

package motion

// init binds the pure-Go convolve variants on architectures the dispatcher does
// not special-case. This file only keeps the dispatch wiring symmetric across
// architectures.
func init() {
	convolveX8Impl = convolveX8PureGo
	convolveY8Impl = convolveY8PureGo
	convolve2D8Impl = convolve2D8PureGo
}
