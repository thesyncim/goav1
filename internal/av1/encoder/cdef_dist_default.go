// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

package encoder

import "github.com/thesyncim/goav1/internal/av1/cdef"

// cdefDist8 is the scalar svt_aom_compute_cdef_dist_8bit_c on targets
// without the Go SIMD kernel.
func cdefDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	return computeCDEFDist8(src, sstride, filtered, fstride, dlist, bwLog2, bhLog2, subsampling)
}
