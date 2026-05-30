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
	}
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
