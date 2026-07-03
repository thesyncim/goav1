// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build amd64 && !purego

package cdef

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

//go:noescape
func cdefFindDirectionAVX2Asm(img *uint16, stride uintptr, coeffShift uintptr, dir *int32, variance *int32)

// init binds the AVX2 CDEF direction search when the CPU advertises AVX2.
// Otherwise the scalar reference (findDirectionScalar) stays in place. The
// assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.AVX2 {
		findDirectionImpl = findDirectionAVX2
		findDirectionDualImpl = findDirectionDualAVX2
	}
}

func findDirectionAVX2(img []uint16, stride int, coeffShift int) (int, int32) {
	var dir, variance int32
	cdefFindDirectionAVX2Asm(&img[0], uintptr(stride), uintptr(coeffShift), &dir, &variance)
	return int(dir), variance
}

func findDirectionDualAVX2(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	dir1, var1 := findDirectionAVX2(img1, stride, coeffShift)
	dir2, var2 := findDirectionAVX2(img2, stride, coeffShift)
	return dir1, var1, dir2, var2
}
