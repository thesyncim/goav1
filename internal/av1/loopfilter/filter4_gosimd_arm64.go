// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit and 10/12-bit narrow (four-sample) deblocking kernels,
// int16 8-wide (simd/archsimd, Go 1.27+ with GOEXPERIMENT=simd). Eight edge
// positions are processed per Int16x8, one lane per position.
//
// Data layout (assessed from filter4SampleOffset / edgeOuterStride):
//   - HORIZONTAL edges: the per-position stride (outer) is one sample, so the
//     four taps (p1,p0,q0,q1) of eight adjacent positions are each a CONTIGUOUS
//     load. This is the clean SIMD orientation and the only one these kernels
//     take.
//   - VERTICAL edges: the taps are one sample apart within a row while positions
//     advance by the row stride, so the same tap across eight positions is
//     STRIDED (a transpose/gather). Vertical edges and any tail shorter than
//     eight positions route back through the hand-written NEON asm (still
//     compiled in this build), which owns the ld4/st4 transpose; the SIMD path
//     stays a single contiguous-load code path.
//
// Byte-exactness with filter4EdgePureGo / filter4Edge16PureGo:
//   - all arithmetic runs in signed int16 lanes; 8-bit samples are 0..255 and
//     the centred ps/qs values -128..127, all inside int16. For 10/12-bit the
//     samples are 0..4095 and the centred values -2048..2047, and the widest
//     intermediate (3*(qs0-ps0), |qs0-ps0|<=4095) is 3*4095 = 12285, inside int16.
//   - needsFilter4 is reproduced lane-wise (AbsDiff, the *2 and /2 terms via
//     shifts, compared LessEqual against the broadcast limit/blimit) into a mask.
//   - hev is reproduced lane-wise as a second mask (AbsDiff Greater hev).
//   - the clamps to [min,max] use Max/Min against broadcast bounds and the
//     >>3 / >>1 steps use arithmetic ShiftAllRight, matching signedClamp and the
//     integer shifts in filter4Samples exactly.
//   - results are blended back with the originals through IfElse(mask) (arm64
//     BSL) so positions failing needsFilter4, and the p1/q1 taps of hev
//     positions, keep their original samples — identical to the reference's
//     conditional writes.

package loopfilter

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// init binds the Go-native SIMD narrow deblocking kernels as the dispatch slots
// under the goexperiment.simd build (the NEON binding in
// filter4_dispatch_arm64.go is excluded there via !goexperiment.simd). The wide
// kernels are re-bound to NEON in loopfilter_gosimd_dispatch_arm64.go so they do
// not regress. Both bindings are byte-identical to the pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
		return
	}
	filter4EdgeImpl = filter4EdgePureGo
	filter4Edge16Impl = filter4Edge16PureGo
}

// lf8LoadP widens 8 contiguous bytes at the raw pointer p to an Int16x8 (values
// 0..255). The caller derives p by unsafe.Add on a base taken once per edge, so
// there is no slice bounds check in the hot loop (the caller has validated the
// full tap extent — see filter4EdgeSIMD).
func lf8LoadP(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).
		ExtendLo8ToUint16().ConvertToInt16()
}

// lf8StoreP narrows the low 8 int16 lanes of v (each a filtered 0..255 sample)
// to bytes and writes them contiguously at p. Only the low 8 bytes are valid;
// the caller writes non-overlapping tap rows so the high 8 are unused. There is
// no 8-byte narrow store in archsimd, so the narrow lands in a stack array and
// its low half is copied out (the copy is 8 bytes, stays in registers).
func lf8StoreP(p unsafe.Pointer, v archsimd.Int16x8) {
	*(*float64)(p) = v.SaturateToUint8().ReshapeToFloat64x2().GetElem(0)
}

// lf16LoadP loads 8 contiguous uint16 samples at raw pointer p as an Int16x8.
// Samples are <= 4095, so the bit pattern is a non-negative int16.
func lf16LoadP(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// lf16StoreP writes the 8 int16 lanes of v as uint16 at raw pointer p. Filtered
// samples are in [0,4095], so the reinterpret is exact.
func lf16StoreP(p unsafe.Pointer, v archsimd.Int16x8) {
	v.StoreArray((*[8]int16)(p))
}

// filter4SIMDConst holds the broadcast constants for the narrow kernel.
type filter4SIMDConst struct {
	limit   archsimd.Int16x8
	blimit  archsimd.Int16x8
	hev     archsimd.Int16x8
	center  archsimd.Int16x8
	min     archsimd.Int16x8
	max     archsimd.Int16x8
	one     archsimd.Int16x8
	three16 archsimd.Int16x8
	four    archsimd.Int16x8
}

// Fixed 8-bit filter4 constants held in rodata: one VLD1 loads a whole Int16x8,
// vs MOVH+VMOV+VDUP per scalar Broadcast. center/min/max are the scale-1 values.
var (
	lf4Center8 = [8]int16{128, 128, 128, 128, 128, 128, 128, 128}
	lf4Min8    = [8]int16{-128, -128, -128, -128, -128, -128, -128, -128}
	lf4Max8    = [8]int16{127, 127, 127, 127, 127, 127, 127, 127}
	lf4One     = [8]int16{1, 1, 1, 1, 1, 1, 1, 1}
	lf4Three   = [8]int16{3, 3, 3, 3, 3, 3, 3, 3}
	lf4Four    = [8]int16{4, 4, 4, 4, 4, 4, 4, 4}
)

// filter4Chain is the narrow four-tap filter for one 8-lane group. All constants
// are passed in (the caller hoists them once) so the two chains of a 16-wide
// iteration share them register-resident; 13 vector args / 4 results ride in
// V-registers with no stack round-trip. Byte-identical to filter4TapVecs gated.
func filter4Chain(p1, p0, q0, q1, limit, blimit, hevT, center, minV, maxV, one, three16, four archsimd.Int16x8) (np1, np0, nq0, nq1 archsimd.Int16x8) {
	hev := p1.AbsDiff(p0).Greater(hevT).Or(q1.AbsDiff(q0).Greater(hevT))
	mask := p1.AbsDiff(p0).LessEqual(limit).
		And(q1.AbsDiff(q0).LessEqual(limit)).
		And(p0.AbsDiff(q0).ShiftAllLeftConst(1).Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).LessEqual(blimit))
	ps1 := p1.Sub(center)
	ps0 := p0.Sub(center)
	qs0 := q0.Sub(center)
	qs1 := q1.Sub(center)
	f := ps1.Sub(qs1).Max(minV).Min(maxV).Masked(hev)
	f = f.Add(qs0.Sub(ps0).Mul(three16)).Max(minV).Min(maxV)
	filter1 := f.Add(four).Max(minV).Min(maxV).ShiftAllRightConst(3)
	filter2 := f.Add(three16).Max(minV).Min(maxV).ShiftAllRightConst(3)
	np0 = ps0.Add(filter2).Max(minV).Min(maxV).Add(center)
	nq0 = qs0.Sub(filter1).Max(minV).Min(maxV).Add(center)
	ov := filter1.Add(one).ShiftAllRightConst(1)
	np1 = ps1.Add(ov).Max(minV).Min(maxV).Add(center)
	nq1 = qs1.Sub(ov).Max(minV).Min(maxV).Add(center)
	np1 = p1.IfElse(hev, np1).IfElse(mask, p1)
	nq1 = q1.IfElse(hev, nq1).IfElse(mask, q1)
	np0 = np0.IfElse(mask, p0)
	nq0 = nq0.IfElse(mask, q0)
	return
}

// lf16StoreP2 packs two 8-lane results (low half a, high half b) to 16 bytes
// with one SQXTUN+SQXTUN2 and a single 16-byte store.
func lf16StoreP2(p unsafe.Pointer, a, b archsimd.Int16x8) {
	a.SaturateToUint8().SaturateToUint8Hi(b).StoreArray((*[16]uint8)(p))
}

func makeFilter4SIMDConst(params filter4Params) filter4SIMDConst {
	return filter4SIMDConst{
		limit:   archsimd.BroadcastInt16x8(params.limit),
		blimit:  archsimd.BroadcastInt16x8(params.blimit),
		hev:     archsimd.BroadcastInt16x8(params.hev),
		center:  archsimd.BroadcastInt16x8(params.center),
		min:     archsimd.BroadcastInt16x8(params.min),
		max:     archsimd.BroadcastInt16x8(params.max),
		one:     archsimd.BroadcastInt16x8(1),
		three16: archsimd.BroadcastInt16x8(3),
		four:    archsimd.BroadcastInt16x8(4),
	}
}

// clampSIMD reproduces signedClamp(v, min, max) lane-wise.
func clampSIMD(v archsimd.Int16x8, cst filter4SIMDConst) archsimd.Int16x8 {
	return v.Max(cst.min).Min(cst.max)
}

// filter4TapVecs computes the four narrow-filtered taps (p1,p0,q0,q1) for one
// 8-position group, with the needsFilter4 and hev blends applied, byte-identical
// to filter4Samples gated by needsFilter4. It is written as one expression block
// so the whole thing inlines into the edge loops (a helper *call* would push the
// eight Int16x8 args/returns through the stack ABI, spilling the tap vectors and
// roughly halving throughput — see the file header). This is also the wide
// kernel's not-flat branch (filter8Samples calls filter4Samples unconditionally),
// exposed via filter4TapNoGate for filter8's inlined loop.
//
// The compiler cannot auto-inline this (its Int16x8/IfElse expansion is over the
// inline budget), so it stays a package-level func used only where an outlined
// call is acceptable; the hot edge loops paste the same arithmetic inline.
func filter4TapVecs(p1, p0, q0, q1 archsimd.Int16x8, cst filter4SIMDConst) (archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8) {
	np1, np0, nq0, nq1 := filter4TapNoGate(p1, p0, q0, q1, cst)
	mask := needsFilter4Mask(p1, p0, q0, q1, cst)
	np1 = np1.IfElse(mask, p1)
	np0 = np0.IfElse(mask, p0)
	nq0 = nq0.IfElse(mask, q0)
	nq1 = nq1.IfElse(mask, q1)
	return np1, np0, nq0, nq1
}

// filter4TapNoGate is the unconditional narrow tap update (hev blend applied to
// p1/q1, no needsFilter4 gate) — filter4Samples lane-wise.
func filter4TapNoGate(p1, p0, q0, q1 archsimd.Int16x8, cst filter4SIMDConst) (archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8) {
	hev := p1.AbsDiff(p0).Greater(cst.hev).Or(q1.AbsDiff(q0).Greater(cst.hev))
	ps1 := p1.Sub(cst.center)
	ps0 := p0.Sub(cst.center)
	qs0 := q0.Sub(cst.center)
	qs1 := q1.Sub(cst.center)
	f := clampSIMD(ps1.Sub(qs1), cst).Masked(hev)
	f = clampSIMD(f.Add(qs0.Sub(ps0).Mul(cst.three16)), cst)
	filter1 := clampSIMD(f.Add(cst.four), cst).ShiftAllRightConst(3)
	filter2 := clampSIMD(f.Add(cst.three16), cst).ShiftAllRightConst(3)
	newP0 := clampSIMD(ps0.Add(filter2), cst).Add(cst.center)
	newQ0 := clampSIMD(qs0.Sub(filter1), cst).Add(cst.center)
	outer := filter1.Add(cst.one).ShiftAllRightConst(1)
	newP1 := clampSIMD(ps1.Add(outer), cst).Add(cst.center)
	newQ1 := clampSIMD(qs1.Sub(outer), cst).Add(cst.center)
	newP1 = p1.IfElse(hev, newP1)
	newQ1 = q1.IfElse(hev, newQ1)
	return newP1, newP0, newQ0, newQ1
}

// needsFilter4Mask reproduces needsFilter4 lane-wise.
func needsFilter4Mask(p1, p0, q0, q1 archsimd.Int16x8, cst filter4SIMDConst) archsimd.Mask16x8 {
	return p1.AbsDiff(p0).LessEqual(cst.limit).
		And(q1.AbsDiff(q0).LessEqual(cst.limit)).
		And(p0.AbsDiff(q0).ShiftAllLeftConst(1).
			Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).
			LessEqual(cst.blimit))
}

// filter4EdgeSIMD is the Go-native SIMD form of filter4EdgePureGo for 8-bit
// horizontal edges (outer == 1, contiguous taps). Vertical edges (step == 1,
// strided taps) route through the NEON asm, other layouts and short tails
// through the pure-Go reference (via filter4EdgeNEON's own fallbacks).
//
// The per-group arithmetic is pasted inline (not a helper call) so the four tap
// vectors stay in V-registers across the group; an outlined kernel call spills
// them through the stack ABI and costs ~1.8x (measured). filter4TapVecs holds
// the canonical form.
func filter4EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	groups := length / 8
	// center==128 (min/max==-128/127) is the 8-bit scale-1 invariant this kernel
	// assumes for its rodata constants; anything else routes to the asm.
	if outer != 1 || groups == 0 || params.center != 128 {
		filter4EdgeNEON(pix, q0Base, step, outer, length, params)
		return
	}
	// Runtime thresholds broadcast from params; the fixed 8-bit constants
	// (center/min/max are 128/-128/127 at scale 1, plus the 1/3/4 literals) load
	// from rodata as one VLD1 each instead of MOVH+VMOV+VDUP per scalar broadcast.
	limit := archsimd.BroadcastInt16x8(params.limit)
	blimit := archsimd.BroadcastInt16x8(params.blimit)
	hevT := archsimd.BroadcastInt16x8(params.hev)
	center := archsimd.LoadInt16x8Array(&lf4Center8)
	minV := archsimd.LoadInt16x8Array(&lf4Min8)
	maxV := archsimd.LoadInt16x8Array(&lf4Max8)
	one := archsimd.LoadInt16x8Array(&lf4One)
	three16 := archsimd.LoadInt16x8Array(&lf4Three)
	four := archsimd.LoadInt16x8Array(&lf4Four)
	q0p := unsafe.Pointer(&pix[q0Base])
	i := 0
	// 16 positions per iteration: one 128-bit load per tap drives two independent
	// filter chains (low half via ExtendLo8, high via ExtendHi8) that pipeline.
	// The 8-wide form loaded the same 16 bytes but used only the low half, so this
	// halves loads, stores and loop overhead.
	for ; i+16 <= length; i += 16 {
		base := unsafe.Add(q0p, i)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		rp1 := archsimd.LoadUint8x16Array((*[16]uint8)(pP1))
		rp0 := archsimd.LoadUint8x16Array((*[16]uint8)(pP0))
		rq0 := archsimd.LoadUint8x16Array((*[16]uint8)(base))
		rq1 := archsimd.LoadUint8x16Array((*[16]uint8)(pQ1))
		np1, np0, nq0, nq1 := filter4Chain(
			rp1.ExtendLo8ToUint16().ConvertToInt16(), rp0.ExtendLo8ToUint16().ConvertToInt16(),
			rq0.ExtendLo8ToUint16().ConvertToInt16(), rq1.ExtendLo8ToUint16().ConvertToInt16(),
			limit, blimit, hevT, center, minV, maxV, one, three16, four)
		mp1, mp0, mq0, mq1 := filter4Chain(
			rp1.ExtendHi8ToUint16().ConvertToInt16(), rp0.ExtendHi8ToUint16().ConvertToInt16(),
			rq0.ExtendHi8ToUint16().ConvertToInt16(), rq1.ExtendHi8ToUint16().ConvertToInt16(),
			limit, blimit, hevT, center, minV, maxV, one, three16, four)
		lf16StoreP2(pP1, np1, mp1)
		lf16StoreP2(pP0, np0, mp0)
		lf16StoreP2(base, nq0, mq0)
		lf16StoreP2(pQ1, nq1, mq1)
	}
	for ; i+8 <= length; i += 8 {
		base := unsafe.Add(q0p, i)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		np1, np0, nq0, nq1 := filter4Chain(
			lf8LoadP(pP1), lf8LoadP(pP0), lf8LoadP(base), lf8LoadP(pQ1),
			limit, blimit, hevT, center, minV, maxV, one, three16, four)
		lf8StoreP(pP1, np1)
		lf8StoreP(pP0, np0)
		lf8StoreP(base, nq0)
		lf8StoreP(pQ1, nq1)
	}
	if rem := length - i; rem > 0 {
		filter4EdgePureGo(pix, q0Base+i, step, outer, rem, params)
	}
}

// filter4Edge16SIMD is the Go-native SIMD form of filter4Edge16PureGo for
// 10/12-bit horizontal edges (outer == 2, contiguous two-byte taps). Vertical
// edges and short tails route through the NEON asm / pure-Go reference. The
// per-group math is pasted inline for the same register-residency reason as the
// 8-bit path.
func filter4Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	groups := length / 8
	if outer != 2 || groups == 0 {
		filter4Edge16NEON(pix, q0Base, step, outer, length, params)
		return
	}
	cst := makeFilter4SIMDConst(params)
	q0p := unsafe.Pointer(&pix[q0Base])
	for g := 0; g < groups; g++ {
		base := unsafe.Add(q0p, g*16) // 8 positions * 2 bytes
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		p1 := lf16LoadP(pP1)
		p0 := lf16LoadP(pP0)
		q0 := lf16LoadP(base)
		q1 := lf16LoadP(pQ1)

		hev := p1.AbsDiff(p0).Greater(cst.hev).Or(q1.AbsDiff(q0).Greater(cst.hev))
		mask := p1.AbsDiff(p0).LessEqual(cst.limit).
			And(q1.AbsDiff(q0).LessEqual(cst.limit)).
			And(p0.AbsDiff(q0).ShiftAllLeftConst(1).Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).LessEqual(cst.blimit))
		ps1 := p1.Sub(cst.center)
		ps0 := p0.Sub(cst.center)
		qs0 := q0.Sub(cst.center)
		qs1 := q1.Sub(cst.center)
		f := clampSIMD(ps1.Sub(qs1), cst).Masked(hev)
		f = clampSIMD(f.Add(qs0.Sub(ps0).Mul(cst.three16)), cst)
		filter1 := clampSIMD(f.Add(cst.four), cst).ShiftAllRightConst(3)
		filter2 := clampSIMD(f.Add(cst.three16), cst).ShiftAllRightConst(3)
		np0 := clampSIMD(ps0.Add(filter2), cst).Add(cst.center)
		nq0 := clampSIMD(qs0.Sub(filter1), cst).Add(cst.center)
		ov := filter1.Add(cst.one).ShiftAllRightConst(1)
		np1 := clampSIMD(ps1.Add(ov), cst).Add(cst.center)
		nq1 := clampSIMD(qs1.Sub(ov), cst).Add(cst.center)
		np1 = p1.IfElse(hev, np1).IfElse(mask, p1)
		nq1 = q1.IfElse(hev, nq1).IfElse(mask, q1)
		np0 = np0.IfElse(mask, p0)
		nq0 = nq0.IfElse(mask, q0)

		lf16StoreP(pP1, np1)
		lf16StoreP(pP0, np0)
		lf16StoreP(base, nq0)
		lf16StoreP(pQ1, nq1)
	}
	if rem := length - groups*8; rem > 0 {
		filter4Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, params)
	}
}
