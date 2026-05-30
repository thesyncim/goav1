// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!arm64 && !amd64) || ((arm64 || amd64) && purego)

package dsp

// init binds the pure-Go AddResidualPlaneBlock inner loop on every target that
// does not ship a tuned variant: all architectures without hand-written
// assembly, plus arm64/amd64 built with the purego tag (which excludes all
// assembly). This keeps the dispatch wiring symmetric across build
// configurations.
func init() {
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
}
