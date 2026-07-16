// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego || goav1_trace_rng

package entropy

import "unsafe"

// HasCoeffBaseLevels reports that the arm64 TXB base-levels kernel is built in.
// Callers must keep the pure-Go loop as the dispatch fallback.
const HasCoeffBaseLevels = false

// CoeffBaseLevels is the no-kernel stub; it must never be called (dispatch
// is gated on HasCoeffBaseLevels).
func (c *Cursor) CoeffBaseLevels(scanHot unsafe.Pointer, cHi, cLo int, levels *uint8, stride int, base *CDF, br *CDF, dirty *int16, levelDirty *int16, class int, update bool) int {
	return 0
}
