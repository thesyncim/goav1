// AVX2 CDEF direction-search kernel, a faithful port of dav1d's
// cdef_dir_8bpc_avx2 (third_party/upstream/dav1d/src/x86/cdef_avx2.asm,
// cglobal cdef_dir_8bpc). dav1d's kernel is 8bpc-only: it loads bytes,
// zero-extends to words and biases by -128. goav1's CDEF buffer is uint16 for
// every bit depth, so the ONLY adaptation here is the load: each row is loaded
// as eight 16-bit samples, shifted right by coeffShift (VPSRLW) and biased by
// -128, exactly reproducing the scalar `int32(img>>shift) - 128`. After the
// bias every value is back in [-128, 127] so the remainder of the routine is a
// byte-for-byte transcription of dav1d and is bit-exact with findDirectionScalar
// (which matches libaom's cdef_find_dir_c). The eight AV1 directional partial
// sums are reduced with the libaom div-table costs, the argmax is taken with
// PHMINPOSUW (lowest index wins ties, matching the scalar `>` scan) and the
// variance is (bestCost - oppositeCost) >> 10.
//
//go:build amd64 && !purego

#include "textflag.h"

// pd_47130256 (dav1d): permute indices that gather the eight direction costs.
DATA cdefDir47<>+0(SB)/4, $4
DATA cdefDir47<>+4(SB)/4, $7
DATA cdefDir47<>+8(SB)/4, $1
DATA cdefDir47<>+12(SB)/4, $3
DATA cdefDir47<>+16(SB)/4, $0
DATA cdefDir47<>+20(SB)/4, $2
DATA cdefDir47<>+24(SB)/4, $5
DATA cdefDir47<>+28(SB)/4, $6
GLOBL cdefDir47<>(SB), RODATA|NOPTR, $32

// div_table (dav1d): 840,420,280,210,168,140,120,105 | 420,210,140,105.
DATA cdefDirDiv<>+0(SB)/4, $840
DATA cdefDirDiv<>+4(SB)/4, $420
DATA cdefDirDiv<>+8(SB)/4, $280
DATA cdefDirDiv<>+12(SB)/4, $210
DATA cdefDirDiv<>+16(SB)/4, $168
DATA cdefDirDiv<>+20(SB)/4, $140
DATA cdefDirDiv<>+24(SB)/4, $120
DATA cdefDirDiv<>+28(SB)/4, $105
DATA cdefDirDiv<>+32(SB)/4, $420
DATA cdefDirDiv<>+36(SB)/4, $210
DATA cdefDirDiv<>+40(SB)/4, $140
DATA cdefDirDiv<>+44(SB)/4, $105
GLOBL cdefDirDiv<>(SB), RODATA|NOPTR, $48

// shufw_6543210x (dav1d): db 12,13,10,11,8,9,6,7,4,5,2,3,0,1,14,15.
DATA cdefDirShuf<>+0(SB)/1, $12
DATA cdefDirShuf<>+1(SB)/1, $13
DATA cdefDirShuf<>+2(SB)/1, $10
DATA cdefDirShuf<>+3(SB)/1, $11
DATA cdefDirShuf<>+4(SB)/1, $8
DATA cdefDirShuf<>+5(SB)/1, $9
DATA cdefDirShuf<>+6(SB)/1, $6
DATA cdefDirShuf<>+7(SB)/1, $7
DATA cdefDirShuf<>+8(SB)/1, $4
DATA cdefDirShuf<>+9(SB)/1, $5
DATA cdefDirShuf<>+10(SB)/1, $2
DATA cdefDirShuf<>+11(SB)/1, $3
DATA cdefDirShuf<>+12(SB)/1, $0
DATA cdefDirShuf<>+13(SB)/1, $1
DATA cdefDirShuf<>+14(SB)/1, $14
DATA cdefDirShuf<>+15(SB)/1, $15
GLOBL cdefDirShuf<>(SB), RODATA|NOPTR, $16

// pw_128 broadcast source (two words of 128 packed into one dword).
DATA cdefDir128<>+0(SB)/4, $0x00800080
GLOBL cdefDir128<>(SB), RODATA|NOPTR, $4

// func cdefFindDirectionAVX2Asm(img *uint16, stride uintptr, coeffShift uintptr, dir *int32, variance *int32)
TEXT ·cdefFindDirectionAVX2Asm(SB), NOSPLIT, $0-40
	MOVQ img+0(FP), AX
	MOVQ stride+8(FP), BX
	SHLQ $1, BX          // stride bytes
	MOVQ coeffShift+16(FP), CX

	// Load the eight 8x16-bit rows into the low 128 bits of X0..X7.
	VMOVDQU (AX), X0     // row0
	LEAQ    (AX)(BX*1), DX
	VMOVDQU (DX), X1     // row1
	LEAQ    (DX)(BX*1), DX
	VMOVDQU (DX), X2     // row2
	LEAQ    (DX)(BX*1), DX
	VMOVDQU (DX), X3     // row3
	LEAQ    (DX)(BX*1), DX
	VMOVDQU (DX), X4     // row4
	LEAQ    (DX)(BX*1), DX
	VMOVDQU (DX), X5     // row5
	LEAQ    (DX)(BX*1), DX
	VMOVDQU (DX), X6     // row6
	LEAQ    (DX)(BX*1), DX
	VMOVDQU (DX), X7     // row7

	// Pack rows two-per-register: Y0=row0|row7, Y1=row1|row6,
	// Y2=row2|row5, Y3=row3|row4 (dav1d's vpblendd layout).
	VINSERTI128 $1, X7, Y0, Y0
	VINSERTI128 $1, X6, Y1, Y1
	VINSERTI128 $1, X5, Y2, Y2
	VINSERTI128 $1, X4, Y3, Y3

	// x = (img >> coeffShift) - 128, all in int16 lanes.
	MOVQ    CX, X8
	VPSRLW  X8, Y0, Y0
	VPSRLW  X8, Y1, Y1
	VPSRLW  X8, Y2, Y2
	VPSRLW  X8, Y3, Y3
	VPBROADCASTD cdefDir128<>(SB), Y8
	VPSUBW  Y8, Y0, Y0
	VPSUBW  Y8, Y1, Y1
	VPSUBW  Y8, Y2, Y2
	VPSUBW  Y8, Y3, Y3

	// ---- dav1d cdef_dir .main (verbatim transcription) ----

	// shuffle registers to generate partial_sum_diag[0-1] together
	VPERM2I128 $0x01, Y0, Y0, Y7
	VPERM2I128 $0x01, Y1, Y1, Y6
	VPERM2I128 $0x01, Y2, Y2, Y5
	VPERM2I128 $0x01, Y3, Y3, Y4

	// start with partial_sum_hv[0-1]
	VPADDW  Y1, Y0, Y8
	VPADDW  Y3, Y2, Y9
	VPHADDW Y1, Y0, Y10
	VPHADDW Y3, Y2, Y11
	VPADDW  Y9, Y8, Y8
	VPHADDW Y11, Y10, Y10
	VEXTRACTI128 $1, Y8, X9
	VEXTRACTI128 $1, Y10, X11
	VPADDW  X9, X8, X8            // partial_sum_hv[1]
	VPHADDW X11, X10, X10         // partial_sum_hv[0]
	VINSERTI128 $1, X10, Y8, Y8
	VPBROADCASTD cdefDirDiv<>+44(SB), Y9
	VPMADDWD Y8, Y8, Y8
	VPMULLD Y9, Y8, Y8           // cost6[2a-d] | cost2[a-d]

	// partial_sum_diag[0/1]
	VPSLLDQ $2, Y1, Y9
	VPSRLDQ $14, Y1, Y10
	VPSLLDQ $4, Y2, Y11
	VPSRLDQ $12, Y2, Y12
	VPSLLDQ $6, Y3, Y13
	VPSRLDQ $10, Y3, Y14
	VPADDW  Y11, Y9, Y9
	VPADDW  Y12, Y10, Y10
	VPADDW  Y13, Y9, Y9
	VPADDW  Y14, Y10, Y10
	VPSLLDQ $8, Y4, Y11
	VPSRLDQ $8, Y4, Y12
	VPSLLDQ $10, Y5, Y13
	VPSRLDQ $6, Y5, Y14
	VPADDW  Y11, Y9, Y9
	VPADDW  Y12, Y10, Y10
	VPADDW  Y13, Y9, Y9
	VPADDW  Y14, Y10, Y10
	VPSLLDQ $12, Y6, Y11
	VPSRLDQ $4, Y6, Y12
	VPSLLDQ $14, Y7, Y13
	VPSRLDQ $2, Y7, Y14
	VPADDW  Y11, Y9, Y9
	VPADDW  Y12, Y10, Y10
	VPADDW  Y13, Y9, Y9
	VPADDW  Y14, Y10, Y10        // partial_sum_diag[0/1][8-14,zero]
	VBROADCASTI128 cdefDirShuf<>(SB), Y14
	VBROADCASTI128 cdefDirDiv<>+16(SB), Y13
	VBROADCASTI128 cdefDirDiv<>+0(SB), Y12
	VPADDW  Y0, Y9, Y9           // partial_sum_diag[0/1][0-7]
	VPSHUFB Y14, Y10, Y10
	VPUNPCKHWD Y10, Y9, Y11
	VPUNPCKLWD Y10, Y9, Y9
	VPMADDWD Y11, Y11, Y11
	VPMADDWD Y9, Y9, Y9
	VPMULLD Y13, Y11, Y11
	VPMULLD Y12, Y9, Y9
	VPADDD  Y11, Y9, Y9          // cost0[a-d] | cost4[a-d]

	// merge horizontally and vertically for partial_sum_alt[0-3]
	VPADDW  Y1, Y0, Y10
	VPADDW  Y3, Y2, Y11
	VPADDW  Y5, Y4, Y12
	VPADDW  Y7, Y6, Y13
	VPHADDW Y4, Y0, Y0
	VPHADDW Y5, Y1, Y1
	VPHADDW Y6, Y2, Y2
	VPHADDW Y7, Y3, Y3

	VPSLLDQ $2, Y11, Y4
	VPSRLDQ $14, Y11, Y11
	VPSLLDQ $4, Y12, Y5
	VPSRLDQ $12, Y12, Y12
	VPSLLDQ $6, Y13, Y6
	VPSRLDQ $10, Y13, Y13
	VPADDW  Y10, Y4, Y4
	VPADDW  Y12, Y11, Y11
	VPBROADCASTD cdefDirDiv<>+44(SB), Y12
	VPADDW  Y6, Y5, Y5
	VPADDW  Y13, Y11, Y11        // partial_sum_alt[3/2] right
	VBROADCASTI128 cdefDirDiv<>+32(SB), Y13
	VPADDW  Y5, Y4, Y4          // partial_sum_alt[3/2] left
	VPSHUFLW $0xC6, Y11, Y5
	VPUNPCKHWD Y4, Y11, Y6
	VPUNPCKLWD Y5, Y4, Y4
	VPMADDWD Y6, Y6, Y6
	VPMADDWD Y4, Y4, Y4
	VPMULLD Y12, Y6, Y6
	VPMULLD Y13, Y4, Y4
	VPADDD  Y6, Y4, Y4          // cost7[a-d] | cost5[a-d]

	VPSLLDQ $2, Y1, Y5
	VPSRLDQ $14, Y1, Y1
	VPSLLDQ $4, Y2, Y6
	VPSRLDQ $12, Y2, Y2
	VPSLLDQ $6, Y3, Y7
	VPSRLDQ $10, Y3, Y3
	VPADDW  Y0, Y5, Y5
	VPADDW  Y2, Y1, Y1
	VPADDW  Y7, Y6, Y6
	VPADDW  Y3, Y1, Y1          // partial_sum_alt[0/1] right
	VPADDW  Y6, Y5, Y5          // partial_sum_alt[0/1] left
	VPSHUFLW $0xC6, Y1, Y0
	VPUNPCKHWD Y5, Y1, Y1
	VPUNPCKLWD Y0, Y5, Y5
	VPMADDWD Y1, Y1, Y1
	VPMADDWD Y5, Y5, Y5
	VPMULLD Y12, Y1, Y1
	VPMULLD Y13, Y5, Y5
	VPADDD  Y1, Y5, Y5          // cost1[a-d] | cost3[a-d]

	VMOVDQU cdefDir47<>+16(SB), X0
	VMOVDQU cdefDir47<>+0(SB), Y1
	VPHADDD Y8, Y9, Y9
	VPHADDD Y4, Y5, Y5
	VPHADDD Y5, Y9, Y9
	VPERMD  Y9, Y0, Y0          // cost[0-3]
	VPERMD  Y9, Y1, Y1          // cost[4-7] | cost[0-3]

	// find the best cost
	VPMAXSD X1, X0, X2
	VPSHUFD $0x4E, X2, X3
	VPMAXSD X3, X2, X2
	VPSHUFD $0xB1, X2, X3
	VPMAXSD X3, X2, X2          // best cost (all lanes)

	// find idx via minpos (best cost differences become 0, rest negative)
	VPSUBD  X2, X1, X4
	VPSUBD  X2, X0, X3
	VPACKSSDW X4, X3, X3
	VPHMINPOSUW X3, X3
	VPSRLD  $16, X3, X3         // idx in low dword
	MOVD    X3, R8              // dir

	// variance = (best - opposite_cost) >> 10
	VPERMD  Y1, Y3, Y3
	VPSUBD  X3, X2, X2
	VPSRLD  $10, X2, X2
	MOVD    X2, R9

	MOVQ    dir+24(FP), DX
	MOVL    R8, (DX)
	MOVQ    variance+32(FP), DX
	MOVL    R9, (DX)
	VZEROUPPER
	RET
