// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !amd64 && !arm64

package transform

// init binds the pure-Go batched row kernels on architectures the dispatcher
// does not special-case. Its only purpose is to keep the dispatch wiring
// symmetric across architectures.
func init() {
	inverseDCT4Row2Impl = inverseDCT4Row2PureGo
	inverseDCT8Row2Impl = inverseDCT8Row2PureGo
	inverseDCT16Row2Impl = inverseDCT16Row2PureGo
	inverseADST4Row2Impl = inverseADST4Row2PureGo
	inverseADST8Row2Impl = inverseADST8Row2PureGo
}
