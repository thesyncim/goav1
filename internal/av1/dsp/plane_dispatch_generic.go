// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || (arm64 && purego)

package dsp

// init binds the pure-Go AddResidualPlaneBlock inner loop on every target that
// does not ship a tuned variant: all non-arm64 architectures, plus arm64 built
// with the purego tag (which excludes all assembly). This keeps the dispatch
// wiring symmetric across build configurations.
func init() {
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
}
