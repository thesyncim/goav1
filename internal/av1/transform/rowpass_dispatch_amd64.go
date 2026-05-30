// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the batched row kernels on amd64.
//
// No tuned amd64 variant ships yet, so this binds the pure-Go references. The
// file exists to keep the dispatch wiring symmetric across architectures and
// to provide a single place to add SSE/AVX variants later (gated on the
// matching cpu.Detected flag).
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	inverseDCT4Row2Impl = inverseDCT4Row2PureGo
	inverseDCT8Row2Impl = inverseDCT8Row2PureGo
	inverseDCT16Row2Impl = inverseDCT16Row2PureGo
	inverseADST4Row2Impl = inverseADST4Row2PureGo
	inverseADST8Row2Impl = inverseADST8Row2PureGo
}
