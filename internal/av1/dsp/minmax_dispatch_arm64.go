// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best MinMaxAbsDiff8x8 variant on arm64.
//
// The function-pointer assignment happens exactly once, before any decoder
// goroutine starts work, so the steady-state cost of the dispatch is a
// single indirect call.
//
// TODO: add an ARM64 NEON variant (minMaxAbsDiff8x8NEON) that loads each
// 8-byte row as a vector, computes the absolute byte difference with
// `uabd`, and reduces with `uminv`/`umaxv`. Once added, gate it on
// cpu.Detected.NEON. SVE (cpu.Detected.SVE) would be a separate variant.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
