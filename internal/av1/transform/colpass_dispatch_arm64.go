// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON batched DCT64 column kernel that still comes from
// assembly on arm64. The DCT4/8/16/32 and ADST16 column kernels are Go SIMD
// (itx_dispatch_gosimd_arm64.go).
//
// The DCT64 four-column kernel keeps its stage buffer in Go-provided scratch
// and stays inside the nosplit stack budget; the old two-column DCT64 kernels
// remain unbound because their manual frames overflow that budget.
func init() {
	if cpu.Detected.NEON {
		inverseDCT64Col4Impl = inverseDCT64Col4NEONAdapter
	}
}

// colClampBoundNEON is the stage-range envelope the four-column int32-lane
// kernels are proven overflow-free for: every supported bit depth's row and
// column stage bounds satisfy |bound| <= 1<<19 (stageRangeBounds caps at
// bitDepth 12: rowBits 20). Wider bounds fall back to pure Go.
const colClampBoundNEON = 1 << 19

// The DCT64 four-column kernel takes a base element pointer, the row stride in
// bytes and int64 clamp bounds, and runs the whole butterfly in int32 lanes;
// it is exact only inside the +/-2^19 envelope (see the .s header for the
// range proof). The column pass pre-clamps every input to [min, max]
// (hybrid.go clampRoundImpl). Out-of-envelope bounds or short buffers use the
// pure-Go reference.

func inverseDCT64Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct64Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT64Col4PureGo(buf, rowStride, min, max)
		return
	}
	var scratch [4 * dct64Size]int32
	inverseDCT64Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max), &scratch[0])
}
