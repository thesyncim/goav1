// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best MinMaxAbsDiff8x8 variant on amd64.
//
// The function-pointer assignment happens exactly once, before any decoder
// goroutine starts work, so the steady-state cost of the dispatch is a
// single indirect call. The AVX2 variant (VPMINUB/VPMAXUB byte lanes, and
// VPMINUW/VPMAXUW word lanes) is selected when the CPU advertises AVX2; the
// pure-Go reference is the fallback.
//
// TODO: add an SSE4.1 (PSADBW + PMINUB/PMAXUB) intermediate tier between AVX2
// and pure Go for hosts without AVX2.
func init() {
	if cpu.Detected.AVX2 {
		minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8AVX2
		return
	}
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
