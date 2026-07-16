// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego && !goav1_trace_rng

package entropy

import "unsafe"

// HasCoeffBaseLevels reports that the arm64 TXB base-levels kernel is built in
// (see internal/av1/tile/coeff_asm_arm64_spec.go for the M-D3 plan).
const HasCoeffBaseLevels = true

// The kernel hardcodes Cursor and CDF field offsets; pin them at compile time
// so any layout change breaks the build instead of the decode.
var (
	_ [unsafe.Offsetof(Cursor{}.src)]byte
	_ [unsafe.Offsetof(Cursor{}.dif) - 24]byte
	_ [24 - unsafe.Offsetof(Cursor{}.dif)]byte
	_ [unsafe.Offsetof(Cursor{}.pos) - 32]byte
	_ [32 - unsafe.Offsetof(Cursor{}.pos)]byte
	_ [unsafe.Offsetof(Cursor{}.rng) - 36]byte
	_ [36 - unsafe.Offsetof(Cursor{}.rng)]byte
	_ [unsafe.Offsetof(Cursor{}.cnt) - 38]byte
	_ [38 - unsafe.Offsetof(Cursor{}.cnt)]byte
	_ [unsafe.Offsetof(Cursor{}.tellOffs) - 40]byte
	_ [40 - unsafe.Offsetof(Cursor{}.tellOffs)]byte
	_ [unsafe.Sizeof(CDF{}) - 36]byte
	_ [36 - unsafe.Sizeof(CDF{})]byte
	_ [unsafe.Offsetof(CDF{}.values) - 2]byte
	_ [2 - unsafe.Offsetof(CDF{}.values)]byte
)

//go:noescape
func coeffBaseLevelsARM64(c *Cursor, scanHot unsafe.Pointer, cHi, cLo int, levels *uint8, stride int, base *CDF, br *CDF, dirty *int16, levelDirty *int16, class int, update bool) int

// CoeffBaseLevels runs the AV1 coefficient base-levels walk (base 4-symbol
// reads with the BR high-token chain inlined) for scan indices cHi down to cLo
// over a scan-hot table, entirely in scalar arm64 assembly. It is the M-D3
// D3-b kernel: the range-decoder window stays in registers for the whole loop
// instead of round-tripping through the Cursor per symbol. class is the AV1
// transform class: 0=2D, 1=horizontal, 2=vertical.
//
// scanHot points at entry 0 of the caller's packed scan table; each entry is
// 8 bytes: pos u16 | padded u16 | lower2DOffset i8 | br2DOffset i8 | 2 bytes
// ignored (the tile package pins this layout with compile-time assertions).
// base and br are the first rows of the CoeffBase/CoeffBR context arrays;
// levels is the padded level scratch with the given column stride. Decoded
// nonzero levels are written to levels[padded] and appended in lockstep as
// packed level<<10|pos to dirty and padded to levelDirty; the appended pair
// count is returned. The caller must guarantee list capacity (eob-bounded)
// and levels capacity (padded+2*stride+2 in range), exactly as the pure-Go
// loops it replaces prove via bounds-check hints. update selects the exact
// updateCDFWindow adaptation after every read (the no-update variant leaves
// CDF memory untouched).
func (c *Cursor) CoeffBaseLevels(scanHot unsafe.Pointer, cHi, cLo int, levels *uint8, stride int, base *CDF, br *CDF, dirty *int16, levelDirty *int16, class int, update bool) int {
	return coeffBaseLevelsARM64(c, scanHot, cHi, cLo, levels, stride, base, br, dirty, levelDirty, class, update)
}
