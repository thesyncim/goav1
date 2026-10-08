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
