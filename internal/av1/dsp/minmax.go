// Ported from libaom: aom_dsp/sad.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package dsp

// minMaxAbsDiff8x8Impl is the dispatch slot for the MinMaxAbsDiff8x8 kernel.
// It is resolved exactly once, at package init (see minmax_dispatch_*.go),
// so the per-call cost is a single indirect call with no feature-detection
// branches. Tests and benchmarks must not mutate this variable concurrently
// with live decoding.
var minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo

// MinMaxAbsDiff8x8 returns the minimum and maximum absolute sample difference
// over an 8x8 block. Samples are 8-bit when bytesPerSample is 1 and little-
// endian unsigned 16-bit when bytesPerSample is 2.
//
// The function is the DSP-layer entry point: it dispatches to the
// architecture-appropriate variant selected at package init. The pure-Go
// implementation is the bit-exact reference and is used on every target
// that does not yet ship a tuned variant.
func MinMaxAbsDiff8x8(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	return minMaxAbsDiff8x8Impl(a, aStride, b, bStride, bytesPerSample)
}
