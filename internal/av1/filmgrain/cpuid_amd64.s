// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// CPUID / XGETBV helpers for runtime AVX2 detection. The cpu package reports an
// SSE2-only baseline and does not run CPUID itself, so the filmgrain dispatcher
// probes AVX2 here (see apply_segment_avx2_amd64.go: detectAVX2).

//go:build amd64 && !purego

#include "textflag.h"

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv0() uint32
// Reads the low 32 bits of XCR0 (XGETBV with ECX=0).
TEXT ·xgetbv0(SB), NOSPLIT, $0-4
	MOVL    $0, CX
	WORD $0x010f; BYTE $0xd0 // XGETBV -> EDX:EAX
	MOVL AX, ret+0(FP)
	RET
