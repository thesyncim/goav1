// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best MinMaxAbsDiff8x8 variant on arm64.
//
// The function-pointer assignment happens exactly once, before any decoder
// goroutine starts work, so the steady-state cost of the dispatch is a
// single indirect call.
//
// The NEON variant loads each row as a vector, computes the absolute sample
// difference with `uabd`, folds a running per-lane min/max, and reduces with
// `uminv`/`umaxv`. It is gated on cpu.Detected.NEON (mandatory on every arm64
// target Go supports); the pure-Go reference is the fallback. SVE
// (cpu.Detected.SVE) would be a separate variant.
func init() {
	if cpu.Detected.NEON {
		minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8NEON
		return
	}
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
