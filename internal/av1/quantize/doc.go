// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package quantize implements AV1 coefficient dequantization (spec §7.12.2).
//
// It maps a quantizer index to the DC and AC dequant factors for the active
// bit depth, applies the optional quantizer matrix (spec §7.12.3), and scales
// decoded transform coefficients back to the spatial-residual domain ahead of
// the inverse transform. The dequant tables and rounding mirror libaom's
// av1/common/quant_common.c and av1/decoder/decodetxb.c.
//
// Quantizer-matrix data is generated into qmatrix_tables_gen.go; the runtime
// code in qmatrix.go selects the matrix for a given qm level, plane, and
// transform size.
package quantize
