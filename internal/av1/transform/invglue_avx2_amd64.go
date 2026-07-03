// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

//go:noescape
func clampRoundAVX2Asm(ptr *int32, n uintptr, shift uint64, round uint32, min uint32, max uint32)

//go:noescape
func narrowStoreAVX2Asm(dst *int16, dstStride uintptr, src *int32, width uintptr, height uintptr)

// clampRoundImpl rounds-and-shifts then clamps scratch in place.
var clampRoundImpl = clampRoundPureGo

// narrowStoreImpl writes the final residual rows.
var narrowStoreImpl = narrowStorePureGo

// init binds the AVX2 inverse-transform glue kernels on amd64 when the CPU
// advertises AVX2; otherwise the slots keep their pure-Go defaults. The kernels
// are proven bit-exact by the invglue dispatch differential test.
func init() {
	if cpu.Detected.AVX2 {
		clampRoundImpl = clampRoundAVX2
		narrowStoreImpl = narrowStoreAVX2
	}
}

// clampRoundAVX2 processes the scratch buffer eight int32 lanes at a time when
// its length is a multiple of eight; other lengths fall back to the pure-Go
// reference. shift is always non-negative in the inverse-transform column pass
// (matching the pure-Go shift>0 round and the shift==0 clamp-only path).
func clampRoundAVX2(scratch []int32, shift int, min, max int32) {
	if len(scratch) == 0 || len(scratch)%8 != 0 || shift < 0 {
		clampRoundPureGo(scratch, shift, min, max)
		return
	}
	var round int32
	if shift > 0 {
		round = int32(1) << (shift - 1)
	}
	clampRoundAVX2Asm(&scratch[0], uintptr(len(scratch)), uint64(shift), uint32(round), uint32(min), uint32(max))
}

// narrowStoreAVX2 handles width four or any multiple of eight; other widths
// fall back to the pure-Go reference.
func narrowStoreAVX2(dst []int16, dstStride int, scratch []int32, width, height int) {
	if height <= 0 || (width != 4 && width%8 != 0) {
		narrowStorePureGo(dst, dstStride, scratch, width, height)
		return
	}
	narrowStoreAVX2Asm(&dst[0], uintptr(dstStride), &scratch[0], uintptr(width), uintptr(height))
}
