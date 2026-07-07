// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego || goav1_trace_rng

package entropy

// HasCoeffSignGolomb reports that the arm64 TXB sign/golomb replay kernel is
// built in. Callers must keep the pure-Go replay as the dispatch fallback.
const HasCoeffSignGolomb = false

// CoeffSignGolomb is the no-kernel stub; it must never be called (dispatch is
// gated on HasCoeffSignGolomb).
func (c *Cursor) CoeffSignGolomb(dirty *int16, n int, coeffs *int16, dcSign *CDF, update bool) (culLevel, dcValue, maxScanLine int) {
	return 0, 0, 0
}
