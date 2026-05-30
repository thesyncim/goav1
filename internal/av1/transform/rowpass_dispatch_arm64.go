// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best batched row kernels on arm64.
//
// Every arm64 chip Go runs on supports NEON (it is mandatory in the ARMv8-A
// baseline Go targets), but we still gate on cpu.Detected.NEON so a test can
// force the pure-Go path. The function-pointer assignment happens once, before
// any decoder goroutine starts, so steady-state dispatch is a single indirect
// call.
//
// NEON kernels exist (and are proven bit-exact by the dispatch differential
// test) for DCT4, DCT8, DCT16, DCT32, DCT64, ADST4 and ADST8. DCT8, DCT16,
// DCT32 and DCT64 are bound here: their butterflies are large enough that
// two-row vectorisation beats the scalar pure-Go kernels on this host. The
// DCT4/ADST4/ADST8 NEON bodies do not amortise the assembly-call and
// slice-to-pointer overhead over their smaller / clamp-heavy work and measured
// slower than pure-Go (see the Row2 benchmarks), so those three stay on
// pure-Go. The unbound kernels remain available for re-evaluation if the call
// overhead or kernel shape changes.
func init() {
	if cpu.Detected.NEON {
		inverseDCT8Row2Impl = inverseDCT8Row2NEONAdapter
		inverseDCT16Row2Impl = inverseDCT16Row2NEONAdapter
		inverseDCT32Row2Impl = inverseDCT32Row2NEONAdapter
		inverseDCT64Row2Impl = inverseDCT64Row2NEONAdapter
	}
}

// The NEON kernels take element pointers and int64 clamp bounds; these
// adapters present the dispatch-slot signature (slices + int32 bounds) and
// reslice to the exact transform length so the assembly can index without
// re-checking bounds.

func inverseDCT4Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct4Size || len(r1) < dct4Size {
		inverseDCT4Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct4Size]
	r1 = r1[:dct4Size]
	inverseDCT4Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}

func inverseDCT8Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct8Size]
	r1 = r1[:dct8Size]
	inverseDCT8Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}

func inverseDCT16Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct16Size]
	r1 = r1[:dct16Size]
	inverseDCT16Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}

func inverseDCT32Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size {
		inverseDCT32Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct32Size]
	r1 = r1[:dct32Size]
	inverseDCT32Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}

func inverseDCT64Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct64Size || len(r1) < dct64Size {
		inverseDCT64Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:dct64Size]
	r1 = r1[:dct64Size]
	inverseDCT64Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}

func inverseADST4Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < adst4Size || len(r1) < adst4Size {
		inverseADST4Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:adst4Size]
	r1 = r1[:adst4Size]
	inverseADST4Row2NEON(&r0[0], &r1[0])
}

func inverseADST8Row2NEONAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < adst8Size || len(r1) < adst8Size {
		inverseADST8Row2PureGo(r0, r1, min, max)
		return
	}
	r0 = r0[:adst8Size]
	r1 = r1[:adst8Size]
	inverseADST8Row2NEON(&r0[0], &r1[0], int64(min), int64(max))
}
