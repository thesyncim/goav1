// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package tile

func coeffCulLevel(coeffs []int16, scan []int16, eob int) uint8 {
	// The standalone SIMD shape mirrors SVT's reducer, but goav1's scalar
	// reducer can stop once the context saturates. Keep this dispatch scalar
	// unless an active caller proves the split SIMD pass wins.
	return coeffCulLevelScalar(coeffs, scan, eob)
}

func coeffCulLevelScalar(coeffs []int16, scan []int16, eob int) uint8 {
	if eob <= 0 {
		return 0
	}
	culLevel := 0
	for c := range eob {
		pos := int(scan[c])
		culLevel += absInt(int(coeffs[pos]))
		if culLevel >= CoeffContextMask {
			break
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if coeffs[0] < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if coeffs[0] > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return uint8(culLevel)
}
