// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON batched DCT64 four-row kernel that still comes from
// assembly on arm64. The DCT8/16/32 and ADST16 row kernels are Go SIMD
// (itx_dispatch_gosimd_arm64.go).
//
// The four-row DCT64 kernel keeps its buffers in Go-provided scratch and stays
// inside the nosplit stack budget, so odd DCT64 row tails take the scalar path.
// The old two-row DCT64 kernel remains unbound: its manual frame overflows the
// nosplit stack headroom guarantee (the historical "64x64 DCT fuzz seed"
// corruption).
func init() {
	if cpu.Detected.NEON {
		inverseDCT64Row4Impl = inverseDCT64Row4NEONAdapter
	}
}

// The NEON kernel takes element pointers and int64 clamp bounds; the adapter
// presents the dispatch-slot signature (slices + int32 bounds) and reslices to
// the exact transform length so the assembly can index without re-checking
// bounds.

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
