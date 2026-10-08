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
// DCT16 column-pair kernels beat the scalar pure-Go column path on this host
// (their wide butterflies amortise the assembly-call overhead). DCT32 and
// DCT64 use the four-column int32-lane kernels (dav1d's hbd itx shape,
// src/arm/64/itx16.S), which beat both the scalar path and the older
// two-column int64-lane DCT32 kernel. The DCT32 column-pair kernel stays
// bound as the narrow-buffer tail. The old two-column DCT64 kernels remain
// unbound: their manual ~2KB frames below a NOSPLIT $0 Go frame overflow the
// nosplit stack headroom guarantee and can corrupt memory on shallow
// goroutine stacks (the historical "64x64 DCT fuzz seed" corruption); the
// four-column DCT64 kernel keeps its stage buffer in Go-provided scratch and
// stays inside the budget.
func init() {
	if cpu.Detected.NEON {
		inverseDCT8Col2Impl = inverseDCT8Col2NEONAdapter
		inverseDCT16Col2Impl = inverseDCT16Col2NEONAdapter
		inverseDCT32Col2Impl = inverseDCT32Col2NEONAdapter
		inverseDCT4Col4Impl = inverseDCT4Col4NEONAdapter
		inverseDCT8Col4Impl = inverseDCT8Col4NEONAdapter
		inverseDCT16Col4Impl = inverseDCT16Col4NEONAdapter
		inverseDCT32Col4Impl = inverseDCT32Col4NEONAdapter
		inverseDCT64Col4Impl = inverseDCT64Col4NEONAdapter
		inverseADST16Col4Impl = inverseADST16Col4NEONAdapter
		inverseADST16Col4FlipImpl = inverseADST16Col4FlipNEONAdapter
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

// The four-column kernels run the whole butterfly in int32 lanes, which is
// exact only while the stage clamp bounds stay inside the +/-2^19 envelope
// (see the .s headers for the range proof); the column pass pre-clamps every
// input to [min, max] before invoking them (hybrid.go clampRoundImpl), and
// the differential tests stage inputs the same way. Out-of-envelope bounds or
// short buffers fall back to the column-pair path.

func inverseDCT4Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct4Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT4Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT4Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

func inverseDCT8Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct8Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT8Col2NEONAdapter(buf, rowStride, min, max)
		inverseDCT8Col2NEONAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT8Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

func inverseDCT16Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct16Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT16Col2NEONAdapter(buf, rowStride, min, max)
		inverseDCT16Col2NEONAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT16Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

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

func inverseADST16Col4NEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseADST16Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
}

// inverseADST16Col4FlipNEONAdapter runs the natural ADST16 kernel then reverses
// the 16 rows of each of the four columns in place: the flip transform is
// exactly the vertical reversal of the natural ADST16 output (inverseFlipADST1D
// writes natural[i] to position 15-i), so this is bit-for-bit identical to
// inverseFlipADST1D run on each column. The natural kernel reads all 16 inputs
// before storing, so the in-place reversal is safe.
func inverseADST16Col4FlipNEONAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseADST16Col4FlipPureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4NEON(&buf[0], int64(rowStride)*4, int64(min), int64(max))
	for i := 0; i < adst16Size/2; i++ {
		top := buf[i*rowStride : i*rowStride+4 : i*rowStride+4]
		j := (adst16Size - 1 - i) * rowStride
		bot := buf[j : j+4 : j+4]
		top[0], bot[0] = bot[0], top[0]
		top[1], bot[1] = bot[1], top[1]
		top[2], bot[2] = bot[2], top[2]
		top[3], bot[3] = bot[3], top[3]
	}
}
