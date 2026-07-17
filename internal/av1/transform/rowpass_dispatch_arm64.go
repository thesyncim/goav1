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
// test) for DCT4, DCT8, DCT16, DCT32, DCT64, ADST4 and ADST8. DCT8, DCT16 and
// DCT32 pairs are bound here: their butterflies are large enough that two-row
// vectorisation beats the scalar pure-Go kernels on this host. The
// DCT4/ADST4/ADST8 NEON bodies do not amortise the assembly-call and
// slice-to-pointer overhead over their smaller / clamp-heavy work and measured
// slower than pure-Go (see the Row2 benchmarks), so those three stay on
// pure-Go. DCT32 and DCT64 row quads use the transposed four-row int32-lane
// kernels (dav1d's hbd itx shape, src/arm/64/itx16.S). The old two-row DCT64
// kernel remains unbound: its manual ~1.7KB frame below a NOSPLIT $0 Go frame
// overflows the nosplit stack headroom guarantee and can corrupt memory on
// shallow goroutine stacks (the historical "64x64 DCT fuzz seed" corruption);
// the four-row DCT64 kernel keeps its buffers in Go-provided scratch and
// stays inside the budget, so odd DCT64 row tails take the scalar path.
func init() {
	if cpu.Detected.NEON {
		inverseDCT8Row2Impl = inverseDCT8Row2NEONAdapter
		inverseDCT16Row2Impl = inverseDCT16Row2NEONAdapter
		inverseDCT32Row2Impl = inverseDCT32Row2NEONAdapter
		inverseDCT8Row4Impl = inverseDCT8Row4NEONAdapter
		inverseDCT16Row4Impl = inverseDCT16Row4NEONAdapter
		inverseDCT32Row4Impl = inverseDCT32Row4NEONAdapter
		inverseDCT64Row4Impl = inverseDCT64Row4NEONAdapter
		inverseADST8Row4Impl = inverseADST8Row4NEONAdapter
		inverseADST8Row4FlipImpl = inverseADST8Row4FlipNEONAdapter
		inverseADST16Row4Impl = inverseADST16Row4NEONAdapter
		inverseADST16Row4FlipImpl = inverseADST16Row4FlipNEONAdapter
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

// The four-row kernels run the whole butterfly in int32 lanes, which is exact
// only while the stage clamp bounds stay inside the +/-2^19 envelope (see the
// .s headers for the range proof); the row pass pre-clamps every staged input
// to [min, max] before invoking them (hybrid.go staging loops), and the
// differential tests stage inputs the same way. Out-of-envelope bounds or
// short rows fall back to the row-pair path.

func inverseDCT8Row4NEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size || len(r2) < dct8Size || len(r3) < dct8Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT8Row2NEONAdapter(r0, r1, min, max)
		inverseDCT8Row2NEONAdapter(r2, r3, min, max)
		return
	}
	r0 = r0[:dct8Size]
	r1 = r1[:dct8Size]
	r2 = r2[:dct8Size]
	r3 = r3[:dct8Size]
	inverseDCT8Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
}

func inverseDCT16Row4NEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size || len(r2) < dct16Size || len(r3) < dct16Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseDCT16Row2NEONAdapter(r0, r1, min, max)
		inverseDCT16Row2NEONAdapter(r2, r3, min, max)
		return
	}
	r0 = r0[:dct16Size]
	r1 = r1[:dct16Size]
	r2 = r2[:dct16Size]
	r3 = r3[:dct16Size]
	inverseDCT16Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
}

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

func inverseADST16Row4NEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseADST16Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0 = r0[:adst16Size]
	r1 = r1[:adst16Size]
	r2 = r2[:adst16Size]
	r3 = r3[:adst16Size]
	inverseADST16Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
}

func inverseADST8Row4NEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst8Size || len(r1) < adst8Size || len(r2) < adst8Size || len(r3) < adst8Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseADST8Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0 = r0[:adst8Size]
	r1 = r1[:adst8Size]
	r2 = r2[:adst8Size]
	r3 = r3[:adst8Size]
	inverseADST8Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
}

func inverseADST8Row4FlipNEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst8Size || len(r1) < adst8Size || len(r2) < adst8Size || len(r3) < adst8Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseADST8Row4FlipPureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0 = r0[:adst8Size]
	r1 = r1[:adst8Size]
	r2 = r2[:adst8Size]
	r3 = r3[:adst8Size]
	inverseADST8Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
	reverseADST8Row(r0)
	reverseADST8Row(r1)
	reverseADST8Row(r2)
	reverseADST8Row(r3)
}

func reverseADST8Row(r []int32) {
	for i, j := 0, adst8Size-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
}

// inverseADST16Row4FlipNEONAdapter runs the natural ADST16 row kernel then
// reverses each row's 16 elements in place: the flip transform is exactly the
// horizontal reversal of the natural ADST16 output (inverseFlipADST1D writes
// natural[i] to position 15-i), so this is bit-for-bit identical to
// inverseFlipADST1D run on each row. The natural kernel reads every input
// before storing, so the in-place reversal is safe.
func inverseADST16Row4FlipNEONAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		min < -colClampBoundNEON || max >= colClampBoundNEON {
		inverseADST16Row4FlipPureGo(r0, r1, r2, r3, min, max)
		return
	}
	r0 = r0[:adst16Size]
	r1 = r1[:adst16Size]
	r2 = r2[:adst16Size]
	r3 = r3[:adst16Size]
	inverseADST16Row4NEON(&r0[0], &r1[0], &r2[0], &r3[0], int64(min), int64(max))
	reverseADST16Row(r0)
	reverseADST16Row(r1)
	reverseADST16Row(r2)
	reverseADST16Row(r3)
}

func reverseADST16Row(r []int32) {
	for i, j := 0, adst16Size-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
}
