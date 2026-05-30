// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package transform

// init binds the pure-Go batched column kernels on architectures without a
// NEON variant (and on the purego build). It keeps the dispatch wiring
// symmetric across architectures.
func init() {
	inverseDCT8Col2Impl = inverseDCT8Col2PureGo
	inverseDCT16Col2Impl = inverseDCT16Col2PureGo
}
