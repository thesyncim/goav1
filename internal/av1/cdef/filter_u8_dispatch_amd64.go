// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2-backed 8-bit-dst CDEF block filter on amd64. The
// wrapper runs the proven uint16 AVX2 kernel into a small stack tmp and
// narrows the block to bytes: the tap math (the dominant cost) stays in the
// AVX2 asm while the uint8 dst contract of filter_u8.go is preserved. A
// native uint8-store AVX2 epilogue can replace the wrapper later without
// changing any caller.
func init() {
	_ = cpu.Detected
	if cpu.Detected.AVX2 {
		filterBlockU8Impl = filterBlockU8AVX2
		return
	}
	filterBlockU8Impl = filterBlockU8PureGo
}

func filterBlockU8AVX2(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	var tmp [8 * 8]uint16
	filterBlockAVX2(tmp[:], 8, 0, input, inputOrigin, params)
	width := int(params.Width)
	for row := 0; row < int(params.Height); row++ {
		dstRow := dst[dstOrigin+row*dstStride : dstOrigin+row*dstStride+width]
		for col, v := range tmp[row*8 : row*8+width] {
			dstRow[col] = byte(v)
		}
	}
}
