// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package quantize

// init binds the pure-Go dequant column variant on amd64. No tuned amd64
// variant ships yet; this keeps the dispatch wiring symmetric across
// architectures.
//
// TODO: add SSE2/AVX2 variants (widening signed multiply of int16 coeffs to
// int32, 24-bit mask, arithmetic shift, clamp) gated on the matching
// cpu.Detected flag.
func init() {
	dequantColumnImpl = dequantColumnPureGo
}
