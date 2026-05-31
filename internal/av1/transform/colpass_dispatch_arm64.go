// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON batched column kernels on arm64.
//
// As with the row pass, every arm64 chip Go runs on supports NEON, but we gate
// on cpu.Detected.NEON so a test can force the pure-Go path. The DCT8, DCT16,
// DCT32 and DCT64 column-pair kernels all beat the scalar pure-Go column path
// on this host (their wide butterflies amortise the assembly-call overhead), so
// the 8/16/32-point variants are bound. DCT64 stays on the pure-Go path for
// now: a 64x64 DCT fuzz seed can corrupt memory through the paired arm64
// assembly dispatch even though the scalar path is stable.
func init() {
	if cpu.Detected.NEON {
		inverseDCT8Col2Impl = inverseDCT8Col2NEONAdapter
		inverseDCT16Col2Impl = inverseDCT16Col2NEONAdapter
		inverseDCT32Col2Impl = inverseDCT32Col2NEONAdapter
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

func inverseDCT32Col2NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct32Size-1)*rowStride+2 {
		inverseDCT32Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col2NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

func inverseDCT64Col2NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct64Size-1)*rowStride+2 {
		inverseDCT64Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT64Col2NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}
