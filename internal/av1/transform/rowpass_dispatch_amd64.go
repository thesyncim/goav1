// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2 batched row kernels on amd64 when the CPU supports AVX2.
//
// The function-pointer assignment happens once, before any decoder goroutine
// starts, so steady-state dispatch is a single indirect call with no
// per-call feature-detection branch. When AVX2 is unavailable (the cpu package
// reports it via CPUID; notably Rosetta 2 does not advertise AVX2) the slots
// keep their pure-Go defaults, so the decoder stays bit-exact and correct.
//
// AVX2 kernels exist (and are proven bit-exact by the dispatch differential
// test) for DCT8 and DCT16; their wide butterflies amortise the call overhead.
func init() {
	if cpu.Detected.AVX2 {
		inverseDCT8Row2Impl = inverseDCT8Row2AVX2Adapter
		inverseDCT16Row2Impl = inverseDCT16Row2AVX2Adapter
		inverseDCT32Row4Impl = inverseDCT32Row4AVX2Adapter
		inverseDCT64Row4Impl = inverseDCT64Row4AVX2Adapter
	}
}

// The four-row AVX2 kernels transpose four contiguous rows into the four int64
// lanes of each YMM register; they are bit-exact to the pure-Go references while
// every input is clamped to [min, max] and [min, max] lies within the +/-2^19
// stage-range envelope (see colClampBoundAVX2). Out-of-envelope bounds or short
// rows fall back to the pure-Go four-row reference.

func inverseDCT32Row4AVX2Adapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || len(r2) < dct32Size || len(r3) < dct32Size ||
		min < -colClampBoundAVX2 || max >= colClampBoundAVX2 {
		inverseDCT32Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0, r1, r2, r3 = r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size]
	var scratch [avx2Scratch4Ints]int32
	inverseDCT32Row4AVX2(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max), &scratch[0])
}

func inverseDCT64Row4AVX2Adapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct64Size || len(r1) < dct64Size || len(r2) < dct64Size || len(r3) < dct64Size ||
		min < -colClampBoundAVX2 || max >= colClampBoundAVX2 {
		inverseDCT64Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0, r1, r2, r3 = r0[:dct64Size], r1[:dct64Size], r2[:dct64Size], r3[:dct64Size]
	var scratch [avx2Scratch4Ints]int32
	inverseDCT64Row4AVX2(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max), &scratch[0])
}

// The AVX2 kernels take element pointers and int64 clamp bounds; these adapters
// present the dispatch-slot signature (slices + int32 bounds) and reslice to
// the exact transform length so the assembly can index without re-checking
// bounds. Short inputs fall back to the pure-Go reference.

func inverseDCT8Row2AVX2Adapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct8Size]
	r1 = r1[:dct8Size]
	inverseDCT8Row2AVX2(&r0[0], &r1[0], int64(min), int64(max))
}

func inverseDCT16Row2AVX2Adapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct16Size]
	r1 = r1[:dct16Size]
	inverseDCT16Row2AVX2(&r0[0], &r1[0], int64(min), int64(max))
}
