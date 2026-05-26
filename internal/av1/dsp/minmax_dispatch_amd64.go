// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best MinMaxAbsDiff8x8 variant on amd64.
//
// The function-pointer assignment happens exactly once, before any decoder
// goroutine starts work, so the steady-state cost of the dispatch is a
// single indirect call.
//
// TODO: add SSE4.1 (PSADBW + PMINUB/PMAXUB) and AVX2 (VPSADBW + VPMINUB /
// VPMAXUB) variants. Each would be a Go-assembly stub gated by the
// matching cpu.Detected flag below.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
