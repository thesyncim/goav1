// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package frame

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

//go:noescape
func loadSampleRows8AVX2Asm(dst *uint16, src *byte, dstStride uintptr, srcStride uintptr, width uintptr, height uintptr)

// loadSampleRows8 binds the AVX2 widen on amd64 when the CPU advertises AVX2,
// falling back to the pure-Go reference otherwise. Unlike most kernels this one
// dispatches by an inline runtime branch rather than a package-level func
// variable: an indirect call would make the compiler leak dst/src at every
// Load*SamplePlane entry point and break the decoder's zero-alloc steady-state
// budget (see the dispatch note on loadSampleRows8PureGo). The concrete AVX2
// call only takes &dst[0]/&src[0] on the vector path, so escape analysis keeps
// the slices stack-resident for the common case.
func loadSampleRows8(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if cpu.Detected.AVX2 && width >= 8 && height > 0 {
		// Bounds contract identical to the pure-Go reference slices: the last
		// row's [0, width) region must be resident in both buffers.
		_ = src[(height-1)*srcStride+width-1]
		_ = dst[(height-1)*dstStride+width-1]
		loadSampleRows8AVX2Asm(
			&dst[0], &src[0],
			uintptr(dstStride), uintptr(srcStride),
			uintptr(width), uintptr(height),
		)
		return
	}
	loadSampleRows8PureGo(dst, dstStride, src, srcStride, width, height)
}
