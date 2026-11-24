// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package prediction

// widenEdge8 widens the bytes src[:len(dst)] into dst; src may extend past
// len(dst).
func widenEdge8(dst []uint16, src []byte) {
	for i, v := range src[:len(dst)] {
		dst[i] = uint16(v)
	}
}
