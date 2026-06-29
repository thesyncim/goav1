// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego

package transform

// PrepTXB8x8Levels2D prepares padded levels and scan-ordered absolute levels
// for a default 8x8/Class2D TXB. levels must be zeroed by the caller; this
// mirrors the trusted TXB writer stack scratch.
func PrepTXB8x8Levels2D(coeffs *[64]int16, levels *[256]uint8, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	return txbPrep8x8Levels2DPureGo(coeffs, levels, absLevels, eob)
}
