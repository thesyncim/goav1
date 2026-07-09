// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build arm64 && !purego

package cdef

//go:noescape
func cdefFindDirectionNEONAsm(img *uint16, stride uintptr, coeffShift uintptr, dir *int32, variance *int32)

// The dispatch binding lives in direction_dispatch_arm64.go so it is excluded
// under goexperiment.simd (where the Go-native SIMD kernel binds instead); the
// NEON wrapper below stays compiled in both builds for the differential test and
// the head-to-head benchmark.

func findDirectionNEON(img []uint16, stride int, coeffShift int) (int, int32) {
	var dir, variance int32
	cdefFindDirectionNEONAsm(&img[0], uintptr(stride), uintptr(coeffShift), &dir, &variance)
	return int(dir), variance
}

func findDirectionDualNEON(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	dir1, var1 := findDirectionNEON(img1, stride, coeffShift)
	dir2, var2 := findDirectionNEON(img2, stride, coeffShift)
	return dir1, var1, dir2, var2
}
