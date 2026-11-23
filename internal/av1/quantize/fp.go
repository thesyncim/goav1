// Ported from libaom: av1/encoder/av1_quantize.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package quantize

func roundPowerOfTwo(v int32, bits uint8) int32 {
	if bits == 0 {
		return v
	}
	return (v + (1 << (bits - 1))) >> bits
}

const (
	minInt16 = -1 << 15
	maxInt16 = 1<<15 - 1
)
