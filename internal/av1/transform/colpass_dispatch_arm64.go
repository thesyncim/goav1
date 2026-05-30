// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON batched column kernels on arm64.
//
// As with the row pass, every arm64 chip Go runs on supports NEON, but we gate
// on cpu.Detected.NEON so a test can force the pure-Go path. The DCT8 and
// DCT16 column-pair kernels both beat the scalar pure-Go column path on this
// host (their wide butterflies amortise the assembly-call overhead), so both
// are bound. The kernels are proven bit-exact by the column dispatch
// differential test.
func init() {
	if cpu.Detected.NEON {
		inverseDCT8Col2Impl = inverseDCT8Col2NEONAdapter
		inverseDCT16Col2Impl = inverseDCT16Col2NEONAdapter
	}
}

// The NEON column kernels take a base element pointer, the row stride in bytes
// and int64 clamp bounds. The two adjacent columns occupy lanes 0 and 1 of
// each loaded vector. These adapters present the dispatch-slot signature
// (slice + element rowStride + int32 bounds) and verify the buffer holds both
// columns across all rows before handing off, so the assembly can index with a
// fixed post-indexed stride without bounds checks.

func inverseDCT8Col2NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT8Col2NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

func inverseDCT16Col2NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col2NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}
