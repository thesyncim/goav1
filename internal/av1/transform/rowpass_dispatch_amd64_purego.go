// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && purego

package transform

// init binds the pure-Go batched row kernels on the amd64 purego build (which
// excludes the AVX2 assembly). It keeps the dispatch wiring symmetric across
// build configurations.
func init() {
	inverseDCT4Row2Impl = inverseDCT4Row2PureGo
	inverseDCT8Row2Impl = inverseDCT8Row2PureGo
	inverseDCT16Row2Impl = inverseDCT16Row2PureGo
	inverseADST4Row2Impl = inverseADST4Row2PureGo
	inverseADST8Row2Impl = inverseADST8Row2PureGo
}
