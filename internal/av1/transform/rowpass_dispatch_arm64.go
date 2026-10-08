// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the arm64 NEON row kernels that still come from assembly: the
// DCT32 row pair and quad and the DCT64 four-row kernel. The DCT8, DCT16 and
// ADST16 row kernels are Go SIMD (itx_dispatch_gosimd_arm64.go).
func init() {
	if cpu.Detected.NEON {
		inverseDCT32Row2Impl = inverseDCT32Row2NEONAdapter
		inverseDCT32Row4Impl = inverseDCT32Row4NEONAdapter
		inverseDCT64Row4Impl = inverseDCT64Row4NEONAdapter
	}
}

// The NEON kernels take element pointers and int64 clamp bounds; these
// adapters present the dispatch-slot signature (slices + int32 bounds) and
// reslice to the exact transform length so the assembly can index without
// re-checking bounds.

func inverseDCT32Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size {
		inverseDCT32Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct32Size]
	r1 = r1[:dct32Size]
	inverseDCT32Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}

// The four-row kernels run the whole butterfly in int32 lanes, which is exact
// only while the stage clamp bounds stay inside the +/-2^19 envelope (see the
// .s headers for the range proof); the row pass pre-clamps every staged input
// to [min, max] before invoking them (hybrid.go staging loops), and the
// differential tests stage inputs the same way. Out-of-envelope bounds or
// short rows fall back to the row-pair path.

func inverseDCT32Row4NEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || len(r2) < dct32Size || len(r3) < dct32Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT32Row2NEONAdapter(r0, r1, min, max)
		inverseDCT32Row2NEONAdapter(r2, r3, min, max)
		return
	}
	r0 = r0[:dct32Size]
	r1 = r1[:dct32Size]
	r2 = r2[:dct32Size]
	r3 = r3[:dct32Size]
	inverseDCT32Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
}

func inverseDCT64Row4NEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct64Size || len(r1) < dct64Size || len(r2) < dct64Size || len(r3) < dct64Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT64Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0 = r0[:dct64Size]
	r1 = r1[:dct64Size]
	r2 = r2[:dct64Size]
	r3 = r3[:dct64Size]
	var scratch [8 * dct64Size]int32
	inverseDCT64Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max), &scratch[0])
}
