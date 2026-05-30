// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2 batched column kernels on amd64 when the CPU supports
// AVX2. As with the row pass, when AVX2 is unavailable the slots keep their
// pure-Go defaults. The kernels are proven bit-exact by the column dispatch
// differential test.
func init() {
	if cpu.Detected.AVX2 {
		inverseDCT8Col2Impl = inverseDCT8Col2AVX2Adapter
		inverseDCT16Col2Impl = inverseDCT16Col2AVX2Adapter
	}
}

// The AVX2 column kernels take a base element pointer, the row stride in bytes
// and int64 clamp bounds. The two adjacent columns occupy lanes 0 and 1. These
// adapters present the dispatch-slot signature (slice + element rowStride +
// int32 bounds) and verify the buffer holds both columns across all rows before
// handing off, so the assembly can index with a fixed stride without bounds
// checks. Short buffers fall back to the pure-Go reference.

func inverseDCT8Col2AVX2Adapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT8Col2AVX2(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

func inverseDCT16Col2AVX2Adapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col2AVX2(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}
