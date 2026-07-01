// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego || goav1_trace_rng

package entropy

const cursorCDF4HighTokenUpdateArch = false

func readCDF4HighTokenUpdateArch(_ *Cursor, _ *[MaxSymbols + 1]uint16) uint8 {
	return 0
}
