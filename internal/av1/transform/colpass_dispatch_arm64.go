// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON batched column kernels that still come from assembly on
// arm64: the DCT32 column pair and quad and the DCT64 four-column kernel. The
// DCT4/DCT8/DCT16 and ADST16 column kernels are Go SIMD (itx_dispatch_gosimd_arm64.go).
//
// The DCT64 four-column kernel keeps its stage buffer in Go-provided scratch
// and stays inside the nosplit stack budget; the old two-column DCT64 kernels
// remain unbound because their manual frames overflow that budget.
func init() {
	if cpu.Detected.NEON {
		inverseDCT32Col2Impl = inverseDCT32Col2NEONAdapter
		inverseDCT32Col4Impl = inverseDCT32Col4NEONAdapter
		inverseDCT64Col4Impl = inverseDCT64Col4NEONAdapter
	}
}

// colClampBoundNEON is the stage-range envelope the four-column int32-lane
// kernels are proven overflow-free for: every supported bit depth's row and
// column stage bounds satisfy |bound| <= 1<<19 (stageRangeBounds caps at
// bitDepth 12: rowBits 20). Wider bounds fall back to pure Go.
const colClampBoundNEON = 1 << 19

// The NEON column kernels take a base element pointer, the row stride in bytes
// and int64 clamp bounds. The two adjacent columns occupy lanes 0 and 1 of
// each loaded vector. These adapters present the dispatch-slot signature
// (slice + element rowStride + int32 bounds) and verify the buffer holds both
// columns across all rows before handing off, so the assembly can index with a
// fixed post-indexed stride without bounds checks.

func inverseDCT32Col2NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct32Size-1)*rowStride+2 {
		inverseDCT32Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col2NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

// The four-column kernels run the whole butterfly in int32 lanes, which is
// exact only while the stage clamp bounds stay inside the +/-2^19 envelope
// (see the .s headers for the range proof); the column pass pre-clamps every
// input to [min, max] before invoking them (hybrid.go clampRoundImpl), and
// the differential tests stage inputs the same way. Out-of-envelope bounds or
// short buffers fall back to the column-pair path.

func inverseDCT32Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT32Col2NEONAdapter(buf, rowStride, min, max)
		inverseDCT32Col2NEONAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT32Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

func inverseDCT64Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct64Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT64Col4PureGo(buf, rowStride, min, max)
		return
	}
	var scratch [4 * dct64Size]int32
	inverseDCT64Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max), &scratch[0])
}
