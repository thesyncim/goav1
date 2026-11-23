// Ported from libaom: aom_ports/bitops.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package common

import "math/bits"

// MSB32 returns the index of the most significant set bit, or -1 for zero.
func MSB32(v uint32) int {
	if v == 0 {
		return -1
	}
	return bits.Len32(v) - 1
}
