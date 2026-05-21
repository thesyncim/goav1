package common

import "math/bits"

// MSB32 returns the index of the most significant set bit, or -1 for zero.
func MSB32(v uint32) int {
	if v == 0 {
		return -1
	}
	return bits.Len32(v) - 1
}

// CeilLog2 returns ceil(log2(v)); it returns 0 for v == 0.
func CeilLog2(v uint32) int {
	if v <= 1 {
		return 0
	}
	return bits.Len32(v - 1)
}
