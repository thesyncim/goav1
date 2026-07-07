// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego && !goav1_trace_rng

package entropy

import "unsafe"

// HasMVResidual reports that the arm64 motion-vector residual kernel is built
// in (M-D3 D3-e; see internal/av1/tile/coeff_asm_arm64_spec.go §7). The
// Cursor and CDF field offsets it hardcodes are pinned at compile time in
// reader_coeff_base_arm64.go; the tile.MVCDFs offsets it hardcodes are pinned
// in internal/av1/tile/mv_asm.go.
const HasMVResidual = true

//go:noescape
func mvResidualARM64(c *Cursor, mvCDFs unsafe.Pointer, precision int64, update bool) (joint, comp0, comp1 int64)

// MVResidual decodes one motion-vector residual — the 4-symbol joint read
// plus, per coded component, the sign/class/integer/fraction/high-precision
// chain — entirely in scalar arm64 assembly, adapting every CDF row in place
// per the cursor's update mode. mvCDFs points at a tile.MVCDFs (layout pinned
// by the tile package); precision is the raw MVSubpelPrecision (-1 none,
// 0 low, 1 high). Component results come back packed (0 when the joint skips
// the component): diff int16 in bits 0-15, integerPart in 16-25, class in
// 26-29, fraction in 30-31, highPrecision in bit 32 and sign in bit 33. The
// chain cannot fail; the caller pre-validates every row shape.
func (c *Cursor) MVResidual(mvCDFs unsafe.Pointer, precision int64) (joint, comp0, comp1 int64) {
	return mvResidualARM64(c, mvCDFs, precision, c.allowCDFUpdate)
}
