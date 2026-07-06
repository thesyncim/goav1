// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goav1_ec32asm_retired

package entropy

// ReadBitTrusted decodes one equiprobable bit from cursor state.
//
//go:nosplit
func (c *Cursor) ReadBitTrusted() uint8 {
	return readBitTrustedARM64(c)
}

//go:noescape
func readBitTrustedARM64(c *Cursor) uint8
