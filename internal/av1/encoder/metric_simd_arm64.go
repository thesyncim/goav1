//go:build goexperiment.simd && arm64 && !purego

// metric_simd_arm64.go hosts Go-native SIMD (simd/archsimd NEON) ports of the
// encoder SATD metric kernels. These mirror the hand-written NEON assembly in
// metric_neon_arm64.s bit-for-bit in numeric result, but are expressed with the
// standard-library archsimd intrinsics so the compiler owns register allocation
// and scheduling.
//
// SATD = apply a Hadamard transform to the residual block, then sum |coeff|.
// The reducer (satdCoeffs) is the pure abs+add+reduce tail; the Hadamard
// producers (hadamard4x4/8x8) are the all-butterfly Add/Sub transform with a
// single in-register transpose between the two 1-D passes.
//
// All hot paths use array-pointer loads/stores (no slices, no bounds checks in
// the vector body) and are zero-alloc.

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// init binds the SATD/Hadamard encoder metric kernels to their Go-native SIMD
// (archsimd NEON) ports. The pixelStats kernels stay on the hand-written NEON
// assembly via bindPixelStatsNEON. This replaces metric_neon_init_arm64.go's
// binding, which is gated off under goexperiment.simd.
func init() {
	bindPixelStatsNEON()
	satdCoeffsImpl = satdCoeffsSIMD
	hadamard4x4Impl = hadamard4x4SIMD
	hadamard8x8Impl = hadamard8x8SIMD
	hadamard16x16Impl = hadamard16x16SIMD
	hadamard32x32Impl = hadamard32x32SIMD
}

// satdCoeffsSIMD mirrors satdCoeffsPureGo: sum of abs(coeff[i]) for i in
// [0,count). It follows svt_aom_satd_neon's shape — process 16 int32 lanes per
// iteration and abs into four int32 vectors — but accumulates into four
// independent int32x4 lane-accumulators to break the serial add dependency
// (the big win over the NEON asm at large counts), then widen-reduces the
// lanes to int64 (matching NEON's SADDLV and the pure-Go int64 accumulator; no
// int32-lane overflow occurs for the coefficient ranges these kernels consume).
// count is always a multiple of 16 for the AV1 TX sizes {16,64,256,1024}; whole
// int32x4 and scalar tails handle any residual for safety.
func satdCoeffsSIMD(coeff []int32, count int) int {
	if count <= 0 {
		return 0
	}
	_ = coeff[count-1] // single bounds check; vector body is check-free
	base := unsafe.Pointer(&coeff[0])

	// Four independent lane accumulators break the abs+add dependency chain so
	// the four 16-wide loads/abs/adds pipeline; they are summed after the loop.
	acc0 := archsimd.BroadcastInt32x4(0)
	acc1 := archsimd.BroadcastInt32x4(0)
	acc2 := archsimd.BroadcastInt32x4(0)
	acc3 := archsimd.BroadcastInt32x4(0)
	i := 0
	for ; i+16 <= count; i += 16 {
		p0 := (*[4]int32)(unsafe.Add(base, uintptr(i)*4))
		p1 := (*[4]int32)(unsafe.Add(base, uintptr(i+4)*4))
		p2 := (*[4]int32)(unsafe.Add(base, uintptr(i+8)*4))
		p3 := (*[4]int32)(unsafe.Add(base, uintptr(i+12)*4))
		acc0 = acc0.Add(archsimd.LoadInt32x4Array(p0).Abs())
		acc1 = acc1.Add(archsimd.LoadInt32x4Array(p1).Abs())
		acc2 = acc2.Add(archsimd.LoadInt32x4Array(p2).Abs())
		acc3 = acc3.Add(archsimd.LoadInt32x4Array(p3).Abs())
	}
	acc := acc0.Add(acc1).Add(acc2.Add(acc3))
	// Tail: whole int32x4 groups, then scalar remainder.
	for ; i+4 <= count; i += 4 {
		p := (*[4]int32)(unsafe.Add(base, uintptr(i)*4))
		acc = acc.Add(archsimd.LoadInt32x4Array(p).Abs())
	}

	// Widen the four int32 lanes to int64 and sum, matching NEON's SADDLV and
	// the pure-Go 64-bit accumulator (no int32-lane overflow across the loop
	// for the coefficient ranges these kernels consume, mirroring the asm).
	lo := acc.ExtendLo2ToInt64()
	hi := acc.HiToLo().ExtendLo2ToInt64()
	wide := lo.Add(hi)
	total := int(wide.GetElem(0)) + int(wide.GetElem(1))

	for ; i < count; i++ {
		v := coeff[i]
		if v < 0 {
			v = -v
		}
		total += int(v)
	}
	return total
}

// as32 / as64 / as16From32 are free (no-op) bit reinterpretations between the
// int16/int32/int64 lane views of the same 128-bit register. NEON's transpose
// instructions operate on 8h/4s/2d element shapes over identical bit patterns;
// these helpers switch the archsimd lane type without moving any data.
func as32(v archsimd.Int16x8) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}

func as64(v archsimd.Int32x4) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}

func as16From64(v archsimd.Int64x2) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

// hadamardButterfly8 runs SVT's 8-point Hadamard butterfly across eight row
// vectors and returns the eight outputs in svt_aom_hadamard_8x8_neon's
// permuted order (the exact register assignment of metric_neon_arm64.s). The
// same butterfly is applied to columns (pass 1) and, after the transpose, to
// rows (pass 2).
func hadamardButterfly8(v0, v1, v2, v3, v4, v5, v6, v7 archsimd.Int16x8) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Int16x8) {
	a0 := v0.Add(v1)
	a1 := v0.Sub(v1)
	a2 := v2.Add(v3)
	a3 := v2.Sub(v3)
	a4 := v4.Add(v5)
	a5 := v4.Sub(v5)
	a6 := v6.Add(v7)
	a7 := v6.Sub(v7)

	b0 := a0.Add(a2)
	b1 := a1.Add(a3)
	b2 := a0.Sub(a2)
	b3 := a1.Sub(a3)
	b4 := a4.Add(a6)
	b5 := a5.Add(a7)
	b6 := a4.Sub(a6)
	b7 := a5.Sub(a7)

	o0 = b0.Add(b4)
	o1 = b2.Sub(b6)
	o2 = b0.Sub(b4)
	o3 = b2.Add(b6)
	o4 = b3.Add(b7)
	o5 = b3.Sub(b7)
	o6 = b1.Sub(b5)
	o7 = b1.Add(b5)
	return
}

// transpose8x8 transposes the 8x8 int16 matrix held in eight row vectors,
// mirroring the trn1/trn2 .8h → trn1/trn2 .4s → zip1/zip2 .2d ladder of
// svt_aom_hadamard_8x8_neon.
func transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7 archsimd.Int16x8) (r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Int16x8) {
	// 16-bit interleave (TRN1/TRN2 .8h).
	t0 := v0.InterleaveEven(v1)
	t1 := v0.InterleaveOdd(v1)
	t2 := v2.InterleaveEven(v3)
	t3 := v2.InterleaveOdd(v3)
	t4 := v4.InterleaveEven(v5)
	t5 := v4.InterleaveOdd(v5)
	t6 := v6.InterleaveEven(v7)
	t7 := v6.InterleaveOdd(v7)

	// 32-bit interleave (TRN1/TRN2 .4s).
	u0 := as32(t0).InterleaveEven(as32(t2))
	u1 := as32(t0).InterleaveOdd(as32(t2))
	u2 := as32(t1).InterleaveEven(as32(t3))
	u3 := as32(t1).InterleaveOdd(as32(t3))
	u4 := as32(t4).InterleaveEven(as32(t6))
	u5 := as32(t4).InterleaveOdd(as32(t6))
	u6 := as32(t5).InterleaveEven(as32(t7))
	u7 := as32(t5).InterleaveOdd(as32(t7))

	// 64-bit interleave (ZIP1/ZIP2 .2d).
	r0 = as16From64(as64(u0).InterleaveLo(as64(u4)))
	r1 = as16From64(as64(u2).InterleaveLo(as64(u6)))
	r2 = as16From64(as64(u1).InterleaveLo(as64(u5)))
	r3 = as16From64(as64(u3).InterleaveLo(as64(u7)))
	r4 = as16From64(as64(u0).InterleaveHi(as64(u4)))
	r5 = as16From64(as64(u2).InterleaveHi(as64(u6)))
	r6 = as16From64(as64(u1).InterleaveHi(as64(u5)))
	r7 = as16From64(as64(u3).InterleaveHi(as64(u7)))
	return
}

// storeCoeff8 sign-extends an Int16x8 row of Hadamard outputs into eight int32
// coefficients (NEON's sshll/sshll2 pair) at coeff[base:base+8].
func storeCoeff8(base unsafe.Pointer, row archsimd.Int16x8) {
	lo := row.ExtendLo4ToInt32()
	hi := row.HiToLo().ExtendLo4ToInt32()
	lo.StoreArray((*[4]int32)(base))
	hi.StoreArray((*[4]int32)(unsafe.Add(base, 4*4)))
}

// hadamard8x8SIMD mirrors hadamard8x8NEONAsm: load eight int16 residual rows,
// run the 8-point butterfly on columns, transpose, run it again on rows, then
// sign-extend to int32 coefficients in SVT's NEON order.
func hadamard8x8SIMD(src []int16, srcStride int, coeff []int32) {
	_ = src[7*srcStride+7]
	_ = coeff[63]
	sp := unsafe.Pointer(&src[0])
	stride := uintptr(srcStride) * 2 // int16 elements

	v0 := archsimd.LoadInt16x8Array((*[8]int16)(sp))
	v1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, stride)))
	v2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, 2*stride)))
	v3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, 3*stride)))
	v4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, 4*stride)))
	v5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, 5*stride)))
	v6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, 6*stride)))
	v7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(sp, 7*stride)))

	v0, v1, v2, v3, v4, v5, v6, v7 = hadamardButterfly8(v0, v1, v2, v3, v4, v5, v6, v7)
	v0, v1, v2, v3, v4, v5, v6, v7 = transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7)
	v0, v1, v2, v3, v4, v5, v6, v7 = hadamardButterfly8(v0, v1, v2, v3, v4, v5, v6, v7)

	cp := unsafe.Pointer(&coeff[0])
	storeCoeff8(cp, v0)
	storeCoeff8(unsafe.Add(cp, 8*4), v1)
	storeCoeff8(unsafe.Add(cp, 16*4), v2)
	storeCoeff8(unsafe.Add(cp, 24*4), v3)
	storeCoeff8(unsafe.Add(cp, 32*4), v4)
	storeCoeff8(unsafe.Add(cp, 40*4), v5)
	storeCoeff8(unsafe.Add(cp, 48*4), v6)
	storeCoeff8(unsafe.Add(cp, 56*4), v7)
}

// hadamard4x4SIMD mirrors hadamard4x4NEONAsm: load four int16 residual rows in
// the low 4 lanes of int16 vectors, run the 4-point signed-halving butterfly,
// transpose the 4x4, rerun the butterfly, and sign-extend the low 4 outputs to
// int32 coefficients. The signed-halving (a+b)>>1 is done as int16 Add then
// arithmetic ShiftAllRight(1), matching hadamardCol4's int16 arithmetic (the
// residual range keeps a+b within int16, so this equals NEON's SHADD).
func hadamard4x4SIMD(src []int16, srcStride int, coeff []int32) {
	_ = src[3*srcStride+3]
	_ = coeff[15]
	sp := unsafe.Pointer(&src[0])
	stride := uintptr(srcStride) * 2

	// Only the low 4 int16 lanes of each row carry data. Read each row's four
	// int16 as one 8-byte word (exactly the bounds-checked span) into lane 0 of
	// a vector via SetElem — the high 4 int16 lanes are don't-care (they stay in
	// the high half through the butterfly/transpose and are dropped at
	// ExtendLo4ToInt32). archsimd exposes no 64-bit partial vector load, so this
	// GPR-load + SetElem is the least-overhead safe construction.
	v0 := load4Row(sp)
	v1 := load4Row(unsafe.Add(sp, stride))
	v2 := load4Row(unsafe.Add(sp, 2*stride))
	v3 := load4Row(unsafe.Add(sp, 3*stride))

	// Precompute the per-lane shift-amount vector once: archsimd's constant
	// ShiftAllRight(1) expands to build-vector + variable-shift (5 insns) every
	// call, whereas x.Shift(rsh1) reuses one hoisted VSSHL amount (1 insn), so
	// the signed-halving (a+b)>>1 costs just VADD+VSSHL. rsh1 = all -1: a
	// negative amount is an arithmetic right shift in VSSHL.
	rsh1 := archsimd.BroadcastInt16x8(-1)
	v0, v1, v2, v3 = hadamardButterfly4(v0, v1, v2, v3, rsh1)
	v0, v1, v2, v3 = transpose4x4(v0, v1, v2, v3)
	v0, v1, v2, v3 = hadamardButterfly4(v0, v1, v2, v3, rsh1)

	cp := unsafe.Pointer(&coeff[0])
	v0.ExtendLo4ToInt32().StoreArray((*[4]int32)(cp))
	v1.ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(cp, 4*4)))
	v2.ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(cp, 8*4)))
	v3.ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(cp, 12*4)))
}

// load4Row loads four contiguous int16 (one 8-byte word) at p into lane 0 of a
// vector, reinterpreted as Int16x8. The 8-byte read is exactly the span
// guaranteed by hadamard4x4SIMD's bounds check; the high 4 int16 lanes are
// don't-care.
func load4Row(p unsafe.Pointer) archsimd.Int16x8 {
	word := *(*int64)(p)
	return (archsimd.Int64x2{}).SetElem(0, word).ToBits().ReshapeToUint16s().BitsToInt16()
}

// hadamardButterfly4 runs SVT's 4-point signed-halving Hadamard butterfly over
// the low 4 lanes of four int16 row vectors, matching hadamardCol4. rsh1 is a
// broadcast -1 used as the arithmetic-shift-right-by-1 amount for VSSHL.
func hadamardButterfly4(v0, v1, v2, v3, rsh1 archsimd.Int16x8) (o0, o1, o2, o3 archsimd.Int16x8) {
	b0 := v0.Add(v1).Shift(rsh1)
	b1 := v0.Sub(v1).Shift(rsh1)
	b2 := v2.Add(v3).Shift(rsh1)
	b3 := v2.Sub(v3).Shift(rsh1)
	o0 = b0.Add(b2)
	o1 = b1.Add(b3)
	o2 = b0.Sub(b2)
	o3 = b1.Sub(b3)
	return
}

// transpose4x4 transposes the 4x4 int16 matrix held in the low 4 lanes of four
// row vectors (trn1/trn2 .4h → trn1/trn2 .2s), matching the 4x4 NEON ladder.
func transpose4x4(v0, v1, v2, v3 archsimd.Int16x8) (r0, r1, r2, r3 archsimd.Int16x8) {
	t0 := v0.InterleaveEven(v1) // trn1 .4h
	t1 := v0.InterleaveOdd(v1)  // trn2 .4h
	t2 := v2.InterleaveEven(v3)
	t3 := v2.InterleaveOdd(v3)
	// trn1/trn2 .2s over the low 2 int32 lanes.
	r0 = as16From32(as32(t0).InterleaveEven(as32(t2)))
	r2 = as16From32(as32(t0).InterleaveOdd(as32(t2)))
	r1 = as16From32(as32(t1).InterleaveEven(as32(t3)))
	r3 = as16From32(as32(t1).InterleaveOdd(as32(t3)))
	return
}

// as16From32 reinterprets an Int32x4 lane view back to Int16x8 (no-op bitcast).
func as16From32(v archsimd.Int32x4) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

// hadamard16x16SIMD mirrors hadamard16x16NEON: four 8x8 SIMD producers (in
// NEON coefficient order) followed by the NEON quadrant combine, whose store
// order scatters the four butterfly groups so the result matches SVT's NEON
// coefficient layout (which the SATD consumer treats as order-invariant).
func hadamard16x16SIMD(src []int16, srcStride int, coeff []int32) {
	_ = src[15*srcStride+15]
	_ = coeff[255]
	hadamard8x8SIMD(src, srcStride, coeff)
	hadamard8x8SIMD(src[8:], srcStride, coeff[64:])
	hadamard8x8SIMD(src[8*srcStride:], srcStride, coeff[128:])
	hadamard8x8SIMD(src[8*srcStride+8:], srcStride, coeff[192:])
	hadamard16x16CombineSIMD(coeff)
}

// hadamard32x32SIMD mirrors hadamard32x32NEON: four 16x16 SIMD producers
// followed by the contiguous >>2 quadrant combine.
func hadamard32x32SIMD(src []int16, srcStride int, coeff []int32) {
	_ = src[31*srcStride+31]
	_ = coeff[1023]
	hadamard16x16SIMD(src, srcStride, coeff)
	hadamard16x16SIMD(src[16:], srcStride, coeff[256:])
	hadamard16x16SIMD(src[16*srcStride:], srcStride, coeff[512:])
	hadamard16x16SIMD(src[16*srcStride+16:], srcStride, coeff[768:])
	hadamard32x32CombineSIMD(coeff)
}

// hadamard16x16CombineSIMD ports hadamard16x16CombineNEONAsm exactly. Each of
// the four 16-element bases holds four lane-groups read at offsets 0,4,8,12.
// The NEON store scatters them: the group read at offset {0,4,8,12} is written
// at offset {0,8,4,12} within every quadrant row — SVT's NEON coefficient
// order. All groups are computed before any store, so the offset swap between
// groups 1 and 2 is hazard-free (matching the asm's register staging).
func hadamard16x16CombineSIMD(coeff []int32) {
	_ = coeff[255]
	base := unsafe.Pointer(&coeff[0])
	// Hoist the >>1 shift amount: see combine4. -1 = arithmetic shift right 1.
	rsh1 := archsimd.BroadcastInt32x4(-1)
	for b := 0; b < 64; b += 16 {
		// Groups 0 and 3 (offsets 0 and 12) store in place. Groups 1 and 2
		// (offsets 4 and 8) swap store positions, so both are computed before
		// either is stored to avoid the read/write hazard — no scratch arrays.
		combineGroup16(base, b+0, b+0, rsh1)   // group 0 -> pos 0
		combineGroup16(base, b+12, b+12, rsh1) // group 3 -> pos 12
		// group 1 (read off 4) -> pos 8; group 2 (read off 8) -> pos 4.
		g1p0 := (*[4]int32)(unsafe.Add(base, uintptr(b+4)*4))
		g1p1 := (*[4]int32)(unsafe.Add(base, uintptr(64+b+4)*4))
		g1p2 := (*[4]int32)(unsafe.Add(base, uintptr(128+b+4)*4))
		g1p3 := (*[4]int32)(unsafe.Add(base, uintptr(192+b+4)*4))
		g2p0 := (*[4]int32)(unsafe.Add(base, uintptr(b+8)*4))
		g2p1 := (*[4]int32)(unsafe.Add(base, uintptr(64+b+8)*4))
		g2p2 := (*[4]int32)(unsafe.Add(base, uintptr(128+b+8)*4))
		g2p3 := (*[4]int32)(unsafe.Add(base, uintptr(192+b+8)*4))
		o1p0, o1p1, o1p2, o1p3 := combine4(g1p0, g1p1, g1p2, g1p3, rsh1)
		o2p0, o2p1, o2p2, o2p3 := combine4(g2p0, g2p1, g2p2, g2p3, rsh1)
		// group 1 outputs -> pos 8 (g2's read positions).
		o1p0.StoreArray(g2p0)
		o1p1.StoreArray(g2p1)
		o1p2.StoreArray(g2p2)
		o1p3.StoreArray(g2p3)
		// group 2 outputs -> pos 4 (g1's read positions).
		o2p0.StoreArray(g1p0)
		o2p1.StoreArray(g1p1)
		o2p2.StoreArray(g1p2)
		o2p3.StoreArray(g1p3)
	}
}

// combine4 loads one lane-group's four quadrant vectors, runs the signed-
// halving Hadamard combine, and returns the four output vectors. rsh is the
// hoisted per-lane arithmetic-shift-right amount (-1 for 16x16 >>1, -2 for
// 32x32 >>2): x.Shift(rsh) is a single VSSHL, whereas constant ShiftAllRight
// re-materialises the shift-amount vector (MOVD+VMOV+VDUP+VSSHL) on every call.
func combine4(p0, p1, p2, p3 *[4]int32, rsh archsimd.Int32x4) (o0, o1, o2, o3 archsimd.Int32x4) {
	a0 := archsimd.LoadInt32x4Array(p0)
	a1 := archsimd.LoadInt32x4Array(p1)
	a2 := archsimd.LoadInt32x4Array(p2)
	a3 := archsimd.LoadInt32x4Array(p3)
	b0 := a0.Add(a1).Shift(rsh)
	b1 := a0.Sub(a1).Shift(rsh)
	b2 := a2.Add(a3).Shift(rsh)
	b3 := a2.Sub(a3).Shift(rsh)
	return b0.Add(b2), b1.Add(b3), b0.Sub(b2), b1.Sub(b3)
}

// combineGroup16 runs combine4 for the lane-group at read offset src and stores
// its outputs at destination offset dst (both within a 16x16 base block).
func combineGroup16(base unsafe.Pointer, src, dst int, rsh archsimd.Int32x4) {
	o0, o1, o2, o3 := combine4(
		(*[4]int32)(unsafe.Add(base, uintptr(src)*4)),
		(*[4]int32)(unsafe.Add(base, uintptr(64+src)*4)),
		(*[4]int32)(unsafe.Add(base, uintptr(128+src)*4)),
		(*[4]int32)(unsafe.Add(base, uintptr(192+src)*4)),
		rsh,
	)
	o0.StoreArray((*[4]int32)(unsafe.Add(base, uintptr(dst)*4)))
	o1.StoreArray((*[4]int32)(unsafe.Add(base, uintptr(64+dst)*4)))
	o2.StoreArray((*[4]int32)(unsafe.Add(base, uintptr(128+dst)*4)))
	o3.StoreArray((*[4]int32)(unsafe.Add(base, uintptr(192+dst)*4)))
}

// hadamard32x32CombineSIMD ports hadamard32x32CombineNEONAsm: a contiguous
// >>2 four-quadrant combine over the 256-element span (quadrants at 0,256,512,
// 768). The shift is applied to the sums before the second add/sub, matching
// both the NEON sshr #2 and hadamard32x32PureGo's (a+b)>>2. The >>2 amount is
// hoisted (rsh2 = -2) so each shift is a single VSSHL inside the loop.
func hadamard32x32CombineSIMD(coeff []int32) {
	_ = coeff[1023]
	base := unsafe.Pointer(&coeff[0])
	rsh2 := archsimd.BroadcastInt32x4(-2)
	for idx := 0; idx < 256; idx += 4 {
		p0 := (*[4]int32)(unsafe.Add(base, uintptr(idx)*4))
		p1 := (*[4]int32)(unsafe.Add(base, uintptr(256+idx)*4))
		p2 := (*[4]int32)(unsafe.Add(base, uintptr(512+idx)*4))
		p3 := (*[4]int32)(unsafe.Add(base, uintptr(768+idx)*4))
		a0 := archsimd.LoadInt32x4Array(p0)
		a1 := archsimd.LoadInt32x4Array(p1)
		a2 := archsimd.LoadInt32x4Array(p2)
		a3 := archsimd.LoadInt32x4Array(p3)
		b0 := a0.Add(a1).Shift(rsh2)
		b1 := a0.Sub(a1).Shift(rsh2)
		b2 := a2.Add(a3).Shift(rsh2)
		b3 := a2.Sub(a3).Shift(rsh2)
		b0.Add(b2).StoreArray(p0)
		b1.Add(b3).StoreArray(p1)
		b0.Sub(b2).StoreArray(p2)
		b1.Sub(b3).StoreArray(p3)
	}
}
