// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goav1_ec32asm_retired

package entropy

const cursorCDF4HighTokenUpdateArch = true

//go:noescape
func readCDF4HighTokenUpdateArch(c *Cursor, values *[MaxSymbols + 1]uint16) uint8
