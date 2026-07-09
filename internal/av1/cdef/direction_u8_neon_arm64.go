// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build arm64 && !purego

package cdef

//go:noescape
func cdefFindDirectionU8NEONAsm(img *byte, stride uintptr, dir *int32, variance *int32)

// The dispatch binding lives in direction_dispatch_arm64.go (excluded under
// goexperiment.simd); the wrapper below stays compiled in both builds.

func findDirectionU8NEON(img []byte, stride int) (int, int32) {
	var dir, variance int32
	cdefFindDirectionU8NEONAsm(&img[0], uintptr(stride), &dir, &variance)
	return int(dir), variance
}

func findDirectionDualU8NEON(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8NEON(img1, stride)
	dir2, var2 := findDirectionU8NEON(img2, stride)
	return dir1, var1, dir2, var2
}
