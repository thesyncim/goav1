// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package frame

func fillBytes(dst []byte, v byte) {
	fillBytesScalar(dst, v)
}
