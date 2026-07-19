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
		inverseDCT16Col4Impl = inverseDCT16Col4AVX2Adapter
		inverseDCT32Col4Impl = inverseDCT32Col4AVX2Adapter
		inverseDCT64Col4Impl = inverseDCT64Col4AVX2Adapter
	}
}

// colClampBoundAVX2 is the stage-range envelope the four-lane AVX2 column
// kernels are proven overflow-free for: every VPMULDQ multiplicand (a raw input
// or a clipRange'd stage value) satisfies |v| <= 1<<19, so the signed 32x32->64
// products are exact. Every supported bit depth's row/column stage bounds are
// within it (stageRangeBounds caps at bitDepth 12: rowBits 20). Wider bounds or
// short buffers fall back to the pure-Go four-column reference.
const colClampBoundAVX2 = 1 << 19

// inverseDCT16Col4AVX2Adapter transforms four adjacent DCT16 columns as two
// AVX2 two-column passes. There is no dedicated four-lane AVX2 DCT16 kernel, so
// this preserves the pre-existing amd64 behaviour (before the four-column
// dct16Size dispatch case, DCT16 column groups reached the AVX2 two-column
// kernel via the col2 fallback) while the arm64 build gets a native four-lane
// kernel. Bit-identical to inverseDCT16Col4PureGo.
func inverseDCT16Col4AVX2Adapter(buf []int32, rowStride int, min, max int32) {
	inverseDCT16Col2AVX2Adapter(buf, rowStride, min, max)
	inverseDCT16Col2AVX2Adapter(buf[2:], rowStride, min, max)
}

func inverseDCT32Col4AVX2Adapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 ||
		min < -colClampBoundAVX2 || max >= colClampBoundAVX2 {
		inverseDCT32Col4PureGo(buf, rowStride, min, max)
		return
	}
	var scratch [avx2Scratch4Ints]int32
	inverseDCT32Col4AVX2(&buf[0], int64(rowStride)*4, int64(min), int64(max), &scratch[0])
}

func inverseDCT64Col4AVX2Adapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct64Size-1)*rowStride+4 ||
		min < -colClampBoundAVX2 || max >= colClampBoundAVX2 {
		inverseDCT64Col4PureGo(buf, rowStride, min, max)
		return
	}
	var scratch [avx2Scratch4Ints]int32
	inverseDCT64Col4AVX2(&buf[0], int64(rowStride)*4, int64(min), int64(max), &scratch[0])
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
