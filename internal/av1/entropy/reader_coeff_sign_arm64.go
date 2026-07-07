// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego && !goav1_trace_rng

package entropy

// HasCoeffSignGolomb reports that the arm64 TXB sign/golomb replay kernel is
// built in (M-D3 D3-c; see internal/av1/tile/coeff_asm_arm64_spec.go). The
// Cursor and CDF field offsets it hardcodes are pinned at compile time in
// reader_coeff_base_arm64.go (same layout contract as the D3-b kernel).
const HasCoeffSignGolomb = true

//go:noescape
func coeffSignGolombARM64(c *Cursor, dirty *int16, n int, coeffs *int16, dcSign *CDF, update bool) (culLevel, dcValue, maxScanLine int)

// CoeffSignGolomb replays a TXB dirty-scan list of n packed (level<<10 | pos)
// entries from index n-1 down to 0, entirely in scalar arm64 assembly: one
// DC-sign binary-CDF read when the last-appended entry is the DC position
// (adapting the row per update, the exact updateCDF2 shape), one equiprobable
// sign bit for every other entry, and the Exp-Golomb tail (maxLength 20)
// whenever the replayed level saturated at MaxBaseBRRange. It writes the
// signed coefficients to coeffs[pos] and unpacks each dirty entry back to its
// bare position, exactly like the pure-Go replay loops it replaces.
//
// Returns the accumulated culLevel (unclamped level sum), the signed DC value
// (0 when the DC coefficient is absent) and the maximum replayed raster
// position. A negative culLevel signals the pure-Go loop's validation errors
// (golomb prefix beyond maxLength, or a level above math.MaxInt16); the
// cursor then holds the exact post-failure reader state and the failing
// entry's coefficient/dirty writes were not performed, matching the Go error
// path byte-for-byte. The caller must guarantee every packed pos indexes
// coeffs in bounds (the decode loops validate positions before packing).
func (c *Cursor) CoeffSignGolomb(dirty *int16, n int, coeffs *int16, dcSign *CDF, update bool) (culLevel, dcValue, maxScanLine int) {
	return coeffSignGolombARM64(c, dirty, n, coeffs, dcSign, update)
}
