// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// The fused 8-bit inverse transform + add kernels for amd64 (AVX2), in
// dav1d's itx_avx2 form: the int16 butterfly networks of the arm64 fused
// kernels (itx_avx2_nets*_amd64.go, copied from itx_fused*_gosimd_arm64.go)
// run sixteen lanes per ymm register, and every rotation is dav1d's
// ITX_MULSUB_2W: punpck{l,h}wd of the two inputs, vpmaddwd against the
// interleaved constant pair, the rounding bias, vpsrad and vpackssdw. The
// int32 sum is exact and the pack saturates, so under the shared per-lane
// L1 guard (itx16ColumnL1Bound) every rotation equals the scalar
// Round2(a*ca + b*cb, n), as the arm64 kernels' wrapping narrow does.
//
// Blocks wider than eight run sixteen rows (row pass) or sixteen columns
// (column pass) per strip, with one 16x16 transpose between the passes; the
// 8x8 blocks run eight lanes in xmm registers end to end.

// itxAVX2 gates every fused kernel; without AVX2 they all decline.
var itxAVX2 = archsimd.X86.AVX2()

// itx16ColumnL1Bound is the per-lane guard of the int16 networks: a lane
// whose inputs have L1 norm at most the bound has every rotation output in
// int16 (itx16_gain_test.go and its DCT32/DCT64/ADST companions).
const itx16ColumnL1Bound = 32767 - 64

// itxBroadcast16 broadcasts each constant to sixteen int16 lanes.
func itxBroadcast16(v []int16) [][16]int16 {
	out := make([][16]int16, len(v))
	for i, c := range v {
		for l := range out[i] {
			out[i][l] = c
		}
	}
	return out
}

// itx64K[i] broadcasts cospi(i) and itx64K[64+i] broadcasts -cospi(i).
var itx64K = func() (k [128][16]int16) {
	for i, c := range itx64Cos {
		for l := 0; l < 16; l++ {
			k[i][l] = c
			k[64+i][l] = -c
		}
	}
	return k
}()

// itx16Rot is Round2(a*ca + b*cb, sh) on sixteen lanes (ca, cb broadcast).
func itx16Rot(a, b, ca, cb archsimd.Int16x16, sh uint8) archsimd.Int16x16 {
	k := ca.InterleaveLoGrouped(cb)
	bias := archsimd.BroadcastInt32x8(1 << (sh - 1))
	lo := a.InterleaveLoGrouped(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHiGrouped(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itx16Rot1 is Round2(a*ca, sh) on sixteen lanes; the rounding bias rides
// in the second product of the vpmaddwd pair.
func itx16Rot1(a, ca archsimd.Int16x16, sh uint8) archsimd.Int16x16 {
	k := ca.InterleaveLoGrouped(archsimd.BroadcastInt16x16(1 << (sh - 1)))
	one := archsimd.BroadcastInt16x16(1)
	lo := a.InterleaveLoGrouped(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHiGrouped(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itx8Rot is itx16Rot on eight lanes.
func itx8Rot(a, b, ca, cb archsimd.Int16x8, sh uint8) archsimd.Int16x8 {
	k := ca.InterleaveLo(cb)
	bias := archsimd.BroadcastInt32x4(1 << (sh - 1))
	lo := a.InterleaveLo(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHi(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16Concat(hi)
}

// itx8Rot1 is itx16Rot1 on eight lanes.
func itx8Rot1(a, ca archsimd.Int16x8, sh uint8) archsimd.Int16x8 {
	k := ca.InterleaveLo(archsimd.BroadcastInt16x8(1 << (sh - 1)))
	one := archsimd.BroadcastInt16x8(1)
	lo := a.InterleaveLo(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHi(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16Concat(hi)
}

// itxLoad8 loads the first eight lanes of a sixteen-lane constant.
func itxLoad8(p *[16]int16) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p[:8]))
}

// itxRowPerm is the row of each lane after itxLoadNarrow16: vpackssdw packs
// within 128-bit lanes, so lanes 4..7 hold rows 8..11 and lanes 8..11 rows
// 4..7. The row pass is lane-wise, so the order is undone only where the
// transposed rows are stored.
var itxRowPerm = [16]int{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}

// itxLoadNarrow16 loads sixteen int32 values and narrows them to int16 with
// saturation, in itxRowPerm lane order.
func itxLoadNarrow16(p []int32) archsimd.Int16x16 {
	lo := archsimd.LoadInt32x8(p[:8])
	hi := archsimd.LoadInt32x8(p[8:16])
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itxRect2Pre is the pre-clamp that keeps c*181+128 inside int32 without
// changing the result (see stageTransposeClamp).
const itxRect2Pre = 1 << 22

// itxLoadNarrowRect2x16 is itxLoadNarrow16 with the rect2 scale applied
// first: lane i is clip16(Round2(p[i]*2896, 12)) = clip16((p*181+128)>>8).
func itxLoadNarrowRect2x16(p []int32) archsimd.Int16x16 {
	lo := archsimd.LoadInt32x8(p[:8])
	hi := archsimd.LoadInt32x8(p[8:16])
	preLo, preHi := archsimd.BroadcastInt32x8(-itxRect2Pre), archsimd.BroadcastInt32x8(itxRect2Pre)
	k181, r128 := archsimd.BroadcastInt32x8(181), archsimd.BroadcastInt32x8(128)
	lo = lo.Max(preLo).Min(preHi).Mul(k181).Add(r128).ShiftAllRight(8)
	hi = hi.Max(preLo).Min(preHi).Mul(k181).Add(r128).ShiftAllRight(8)
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itxLoadNarrow8 loads eight int32 values narrowed to int16 with saturation.
func itxLoadNarrow8(p []int32) archsimd.Int16x8 {
	return archsimd.LoadInt32x4(p[:4]).SaturateToInt16Concat(archsimd.LoadInt32x4(p[4:8]))
}

// itxL1Ok16 reports whether every lane of l1 is within the guard.
func itxL1Ok16(l1 archsimd.Uint16x16) bool {
	return l1.SubSaturated(archsimd.BroadcastUint16x16(itx16ColumnL1Bound)).IsZero()
}

// itxL1Ok8 is itxL1Ok16 on eight lanes.
func itxL1Ok8(l1 archsimd.Uint16x8) bool {
	return l1.SubSaturated(archsimd.BroadcastUint16x8(itx16ColumnL1Bound)).IsZero()
}

// itxRound2x16 is Round2(x, n) for n in {1, 2, 4} on int16 lanes, formed
// without overflow: (x >> 1) + (x & 1) for n = 1, ((x >> (n-1)) + 1) >> 1
// otherwise.
func itxRound2x16(x archsimd.Int16x16, n uint8) archsimd.Int16x16 {
	one := archsimd.BroadcastInt16x16(1)
	if n == 1 {
		return x.ShiftAllRight(1).Add(x.And(one))
	}
	return x.ShiftAllRight(uint64(n - 1)).Add(one).ShiftAllRight(1)
}

// itxRound2x8 is itxRound2x16 on eight lanes.
func itxRound2x8(x archsimd.Int16x8, n uint8) archsimd.Int16x8 {
	one := archsimd.BroadcastInt16x8(1)
	if n == 1 {
		return x.ShiftAllRight(1).Add(x.And(one))
	}
	return x.ShiftAllRight(uint64(n - 1)).Add(one).ShiftAllRight(1)
}

// itxTranspose8x8Lanes transposes the 8x8 int16 matrices held in each
// 128-bit lane of v[0..7] (v[r] = row r): three interleave stages at 16, 32
// and 64 bits, all within lanes.
func itxTranspose8x8Lanes(v []archsimd.Int16x16) {
	_ = v[7]
	a0, a1 := v[0].InterleaveLoGrouped(v[1]).AsInt32x8(), v[0].InterleaveHiGrouped(v[1]).AsInt32x8()
	a2, a3 := v[2].InterleaveLoGrouped(v[3]).AsInt32x8(), v[2].InterleaveHiGrouped(v[3]).AsInt32x8()
	a4, a5 := v[4].InterleaveLoGrouped(v[5]).AsInt32x8(), v[4].InterleaveHiGrouped(v[5]).AsInt32x8()
	a6, a7 := v[6].InterleaveLoGrouped(v[7]).AsInt32x8(), v[6].InterleaveHiGrouped(v[7]).AsInt32x8()
	b0, b1 := a0.InterleaveLoGrouped(a2).AsInt64x4(), a0.InterleaveHiGrouped(a2).AsInt64x4()
	b2, b3 := a1.InterleaveLoGrouped(a3).AsInt64x4(), a1.InterleaveHiGrouped(a3).AsInt64x4()
	b4, b5 := a4.InterleaveLoGrouped(a6).AsInt64x4(), a4.InterleaveHiGrouped(a6).AsInt64x4()
	b6, b7 := a5.InterleaveLoGrouped(a7).AsInt64x4(), a5.InterleaveHiGrouped(a7).AsInt64x4()
	v[0] = b0.InterleaveLoGrouped(b4).AsInt16x16()
	v[1] = b0.InterleaveHiGrouped(b4).AsInt16x16()
	v[2] = b1.InterleaveLoGrouped(b5).AsInt16x16()
	v[3] = b1.InterleaveHiGrouped(b5).AsInt16x16()
	v[4] = b2.InterleaveLoGrouped(b6).AsInt16x16()
	v[5] = b2.InterleaveHiGrouped(b6).AsInt16x16()
	v[6] = b3.InterleaveLoGrouped(b7).AsInt16x16()
	v[7] = b3.InterleaveHiGrouped(b7).AsInt16x16()
}

// itxTranspose16x16 transposes the 16x16 int16 matrix v (v[i] = row i): the
// two 8x8 quadrant pairs transpose within lanes, then vperm2i128 swaps the
// off-diagonal quadrants.
func itxTranspose16x16(v *[16]archsimd.Int16x16) {
	itxTranspose8x8Lanes(v[:8])
	itxTranspose8x8Lanes(v[8:])
	for i := 0; i < 8; i++ {
		w, u := v[i], v[8+i]
		v[i] = w.ConcatPermute128Scalars(0, 2, u)
		v[8+i] = w.ConcatPermute128Scalars(1, 3, u)
	}
}

// itxTranspose8x8 transposes the 8x8 int16 matrix v (v[r] = row r).
func itxTranspose8x8(v *[8]archsimd.Int16x8) {
	a0, a1 := v[0].InterleaveLo(v[1]).AsInt32x4(), v[0].InterleaveHi(v[1]).AsInt32x4()
	a2, a3 := v[2].InterleaveLo(v[3]).AsInt32x4(), v[2].InterleaveHi(v[3]).AsInt32x4()
	a4, a5 := v[4].InterleaveLo(v[5]).AsInt32x4(), v[4].InterleaveHi(v[5]).AsInt32x4()
	a6, a7 := v[6].InterleaveLo(v[7]).AsInt32x4(), v[6].InterleaveHi(v[7]).AsInt32x4()
	b0, b1 := a0.InterleaveLo(a2).AsInt64x2(), a0.InterleaveHi(a2).AsInt64x2()
	b2, b3 := a1.InterleaveLo(a3).AsInt64x2(), a1.InterleaveHi(a3).AsInt64x2()
	b4, b5 := a4.InterleaveLo(a6).AsInt64x2(), a4.InterleaveHi(a6).AsInt64x2()
	b6, b7 := a5.InterleaveLo(a7).AsInt64x2(), a5.InterleaveHi(a7).AsInt64x2()
	v[0] = b0.InterleaveLo(b4).AsInt16x8()
	v[1] = b0.InterleaveHi(b4).AsInt16x8()
	v[2] = b1.InterleaveLo(b5).AsInt16x8()
	v[3] = b1.InterleaveHi(b5).AsInt16x8()
	v[4] = b2.InterleaveLo(b6).AsInt16x8()
	v[5] = b2.InterleaveHi(b6).AsInt16x8()
	v[6] = b3.InterleaveLo(b7).AsInt16x8()
	v[7] = b3.InterleaveHi(b7).AsInt16x8()
}

// itxPackLoBytes gathers the low byte of every int16 lane into the low
// eight bytes of its 128-bit lane (vpshufb; index -1 zeroes).
var itxPackLoBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

// itxAddStore16 stores clip(pred[0..15] + res) to dst[0..15]. vpackuswb is
// not in archsimd, so the clamp is explicit and vpshufb plus a 64-bit
// interleave pack the bytes.
func itxAddStore16(dst, pred []byte, res archsimd.Int16x16) {
	p := archsimd.LoadUint8x16(pred[:16]).ExtendToUint16().AsInt16x16()
	v := p.Add(res).Max(archsimd.Int16x16{}).Min(archsimd.BroadcastInt16x16(255))
	b := v.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&itxPackLoBytes))
	b.GetLo().AsUint64x2().InterleaveLo(b.GetHi().AsUint64x2()).AsUint8x16().Store(dst[:16])
}

// itxAddStore8 stores clip(pred[0..7] + res) to dst[0..7].
func itxAddStore8(dst, pred []byte, res archsimd.Int16x8) {
	var pv archsimd.Uint64x2
	pv = pv.SetElem(0, binary.LittleEndian.Uint64(pred[:8]))
	p := pv.AsUint8x16().ExtendLo8ToUint16().AsInt16x8()
	v := p.Add(res).Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(255))
	b := v.AsUint8x16().PermuteOrZero(archsimd.LoadInt8x16Array((*[16]int8)(itxPackLoBytes[:16])))
	binary.LittleEndian.PutUint64(dst[:8], b.AsUint64x2().GetElem(0))
}

// itxPix is the reconstruction target of a fused kernel: dst = clip(pred +
// residual) with 1-byte samples when maxV is 0, else 2-byte samples clipped
// to [0, maxV] (strides in bytes).
type itxPix struct {
	dst        []byte
	dstStride  int
	pred       []byte
	predStride int
	maxV       int16
}

// itxAddStore16HBD stores clip(pred[0..15] + res, 0, maxV) for sixteen
// 2-byte samples. The saturating add equals the scalar clip for any int16
// residual and pred <= 4095.
//
//go:nocheckptr
func itxAddStore16HBD(dst, pred []byte, res archsimd.Int16x16, maxV int16) {
	_, _ = dst[31], pred[31]
	p := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Pointer(unsafe.SliceData(pred))))
	v := p.AddSaturated(res).Max(archsimd.Int16x16{}).Min(archsimd.BroadcastInt16x16(maxV))
	v.StoreArray((*[16]int16)(unsafe.Pointer(unsafe.SliceData(dst))))
}

// itxAddStore8HBD is itxAddStore16HBD for eight samples.
//
//go:nocheckptr
func itxAddStore8HBD(dst, pred []byte, res archsimd.Int16x8, maxV int16) {
	_, _ = dst[15], pred[15]
	p := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(unsafe.SliceData(pred))))
	v := p.AddSaturated(res).Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(maxV))
	v.StoreArray((*[8]int16)(unsafe.Pointer(unsafe.SliceData(dst))))
}

// itxStoreRowGroup rounds the sixteen row-pass outputs v (v[j] = output j
// of every lane, lanes in itxRowPerm order of rows r0..r0+15) by n,
// transposes them and stores row r0+itxRowPerm[e] of tmp (width w) at
// columns c0..c0+15, adding the stored magnitudes to the column guard.
func itxStoreRowGroup(v *[16]archsimd.Int16x16, n uint8, tmp []int16, w, r0, c0 int, colL1 *archsimd.Uint16x16) {
	// Immediate shift counts only: a variable ymm shift count is
	// rematerialized with a legacy-SSE movq, which stalls on dirty upper
	// state (Zen 3).
	if n == 1 {
		for j := range v {
			v[j] = itxRound2x16(v[j], 1)
		}
	} else {
		for j := range v {
			v[j] = itxRound2x16(v[j], 2)
		}
	}
	itxTranspose16x16(v)
	l1 := *colL1
	for e := 0; e < 16; e++ {
		v[e].Store(tmp[(r0+itxRowPerm[e])*w+c0:])
		l1 = l1.AddSaturated(v[e].Abs().AsUint16x16())
	}
	*colL1 = l1
	archsimd.ClearAVXUpperBits()
}

// itxRowLoad loads inCols coefficient columns of rows r0..r0+15 (16 lanes)
// into in, with the rect2 scale when rect2 is set, and reports whether
// every lane passes the L1 guard.
func itxRowLoad(coeff []int32, stride, r0, inCols int, rect2 bool, in []archsimd.Int16x16) bool {
	_ = coeff[(inCols-1)*stride+r0+15]
	var l1 archsimd.Uint16x16
	for k := 0; k < inCols; k++ {
		var x archsimd.Int16x16
		if rect2 {
			x = itxLoadNarrowRect2x16(coeff[k*stride+r0 : k*stride+r0+16])
		} else {
			x = itxLoadNarrow16(coeff[k*stride+r0 : k*stride+r0+16])
		}
		l1 = l1.AddSaturated(x.Abs().AsUint16x16())
		in[k] = x
	}
	return itxL1Ok16(l1)
}

// itxRowStrip16 runs the 16-point row pass (network t) for rows r0..r0+15
// into tmp (width 16), rounding by n.
func itxRowStrip16(coeff []int32, stride, r0, inCols int, t tx1DType, rect2 bool, n uint8, tmp []int16, colL1 *archsimd.Uint16x16) bool {
	var in [16]archsimd.Int16x16
	if !itxRowLoad(coeff, stride, r0, inCols, rect2, in[:]) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	var out [16]archsimd.Int16x16
	itx16Net1D(t, inCols, &in, &out)
	itxStoreRowGroup(&out, n, tmp, 16, r0, 0, colL1)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxRowStrip32 runs the 32-point DCT row pass for rows r0..r0+15 into tmp
// (width 32), rounding by n.
func itxRowStrip32(coeff []int32, stride, r0, inCols int, rect2 bool, n uint8, tmp []int16, colL1 *[2]archsimd.Uint16x16) bool {
	var in [32]archsimd.Int16x16
	if !itxRowLoad(coeff, stride, r0, inCols, rect2, in[:]) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	var out [32]archsimd.Int16x16
	itx32Net(inCols, &in, &out)
	for g := 0; g < 2; g++ {
		itxStoreRowGroup((*[16]archsimd.Int16x16)(out[16*g:]), n, tmp, 32, r0, 16*g, &colL1[g])
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxRowStrip64 runs the 64-point DCT row pass for rows r0..r0+15 into tmp
// (width 64), rounding by 2.
func itxRowStrip64(coeff []int32, stride, r0, inCols int, tmp []int16, colL1 *[4]archsimd.Uint16x16) bool {
	var in [32]archsimd.Int16x16
	if !itxRowLoad(coeff, stride, r0, inCols, false, in[:]) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	var out [64]archsimd.Int16x16
	itx64Net(inCols, &in, &out)
	for g := 0; g < 4; g++ {
		itxStoreRowGroup((*[16]archsimd.Int16x16)(out[16*g:]), 2, tmp, 64, r0, 16*g, &colL1[g])
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxColAdd adds Round2(out[r], 4) to rows 0..len(out)-1 of the target at
// columns c0..c0+15.
func itxColAdd(out []archsimd.Int16x16, c0 int, px *itxPix) {
	if px.maxV == 0 {
		for r := range out {
			itxAddStore16(px.dst[r*px.dstStride+c0:], px.pred[r*px.predStride+c0:], itxRound2x16(out[r], 4))
		}
		archsimd.ClearAVXUpperBits()
		return
	}
	for r := range out {
		itxAddStore16HBD(px.dst[r*px.dstStride+2*c0:], px.pred[r*px.predStride+2*c0:], itxRound2x16(out[r], 4), px.maxV)
	}
	archsimd.ClearAVXUpperBits()
}

// itxColStripAdd16 runs the 16-point column pass (network t) for columns
// c0..c0+15 from inRows tmp rows (width w) and adds the result.
func itxColStripAdd16(tmp []int16, w, c0, inRows int, t tx1DType, px *itxPix) {
	var in [16]archsimd.Int16x16
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x16(tmp[r*w+c0:])
	}
	var out [16]archsimd.Int16x16
	itx16Net1D(t, inRows, &in, &out)
	itxColAdd(out[:], c0, px)
	archsimd.ClearAVXUpperBits()
}

// itxColStripAdd32 runs the 32-point DCT column pass for columns c0..c0+15.
func itxColStripAdd32(tmp []int16, w, c0, inRows int, px *itxPix) {
	var in [32]archsimd.Int16x16
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x16(tmp[r*w+c0:])
	}
	var out [32]archsimd.Int16x16
	itx32Net(inRows, &in, &out)
	itxColAdd(out[:], c0, px)
	archsimd.ClearAVXUpperBits()
}

// itxColStripAdd64 runs the 64-point DCT column pass for columns c0..c0+15.
func itxColStripAdd64(tmp []int16, w, c0, inRows int, px *itxPix) {
	var in [32]archsimd.Int16x16
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x16(tmp[r*w+c0:])
	}
	var out [64]archsimd.Int16x16
	itx64Net(inRows, &in, &out)
	itxColAdd(out[:], c0, px)
	archsimd.ClearAVXUpperBits()
}

// itx16Form rounds a coded extent up to the 16-point network form covering
// it (4, 8 or 16).
func itx16Form(n int) int {
	switch {
	case n <= 4:
		return 4
	case n <= 8:
		return 8
	default:
		return 16
	}
}

// itx64Form rounds a coded extent up to the 32/64-point network form
// covering it (8, 16 or 32).
func itx64Form(n int) int {
	switch {
	case n <= 8:
		return 8
	case n <= 16:
		return 16
	default:
		return 32
	}
}

// itxBlockOK checks the common contract of the fused kernels for a w x h
// block of bytesPerSample-byte samples whose coefficients are maxCols
// columns of maxRows at coeffStride, activeRows x activeCols of them coded.
func itxBlockOK(w, h, bytesPerSample, maxRows, maxCols int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	rowBytes := w * bytesPerSample
	return itxAVX2 && activeRows > 0 && activeRows <= maxRows && activeCols > 0 && activeCols <= maxCols &&
		coeffStride >= maxRows && len(coeff) >= (maxCols-1)*coeffStride+maxRows &&
		dstStride >= rowBytes && predStride >= rowBytes &&
		len(dst) >= (h-1)*dstStride+rowBytes && len(pred) >= (h-1)*predStride+rowBytes
}

// itxHBDMax returns the pixel maximum of a 10- or 12-bit block, false for
// any other depth.
func itxHBDMax(bitDepth uint8) (int16, bool) {
	if bitDepth != 10 && bitDepth != 12 {
		return 0, false
	}
	return int16(1)<<bitDepth - 1, true
}

// itxSquare16 is the 16x16 kernel for a horizontal (row) network th and a
// vertical (column) network tv.
func itxSquare16(px *itxPix, coeff []int32, coeffStride int, tv, th tx1DType, activeRows, activeCols int) bool {
	inCols := itx16Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 16]int16
	var colL1 archsimd.Uint16x16
	if !itxRowStrip16(coeff, coeffStride, 0, inCols, th, false, 2, tmp[:], &colL1) || !itxL1Ok16(colL1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd16(tmp[:], 16, 0, inRows, tv, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxSquare8 is the 8x8 kernel: the whole block stays in xmm registers.
func itxSquare8(px *itxPix, coeff []int32, coeffStride int, tv, th tx1DType, activeRows, activeCols int) bool {
	var in [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < activeCols; k++ {
		x := itxLoadNarrow8(coeff[k*coeffStride : k*coeffStride+8])
		l1 = l1.AddSaturated(x.Abs().AsUint16x8())
		in[k] = x
	}
	if !itxL1Ok8(l1) {
		return false
	}
	var mid [8]archsimd.Int16x8
	itx8Net1D(th, activeCols, &in, &mid)
	for j := range mid {
		mid[j] = itxRound2x8(mid[j], 1)
	}
	itxTranspose8x8(&mid)
	var colL1 archsimd.Uint16x8
	for r := range mid {
		colL1 = colL1.AddSaturated(mid[r].Abs().AsUint16x8())
	}
	if !itxL1Ok8(colL1) {
		return false
	}
	var out [8]archsimd.Int16x8
	itx8Net1D(tv, activeRows, &mid, &out)
	if px.maxV == 0 {
		for r := 0; r < 8; r++ {
			itxAddStore8(px.dst[r*px.dstStride:], px.pred[r*px.predStride:], itxRound2x8(out[r], 4))
		}
		return true
	}
	for r := 0; r < 8; r++ {
		itxAddStore8HBD(px.dst[r*px.dstStride:], px.pred[r*px.predStride:], itxRound2x8(out[r], 4), px.maxV)
	}
	return true
}

// itxDCT32 is the 32x32 DCT_DCT kernel for any coded extent.
func itxDCT32(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 32]int16
	var colL1 [2]archsimd.Uint16x16
	written := 0
	for ; written < activeRows; written += 16 {
		if !itxRowStrip32(coeff, coeffStride, written, inCols, false, 2, tmp[:], &colL1) {
			archsimd.ClearAVXUpperBits()
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those
	// rows are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*32 : inRows*32])
	}
	if !itxL1Ok16(colL1[0].Max(colL1[1])) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd32(tmp[:], 32, 0, inRows, px)
	itxColStripAdd32(tmp[:], 32, 16, inRows, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxDCT64 is the 64x64 DCT_DCT kernel (coded coefficients in the 32x32
// corner).
func itxDCT64(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 64]int16
	var colL1 [4]archsimd.Uint16x16
	written := 0
	for ; written < activeRows; written += 16 {
		if !itxRowStrip64(coeff, coeffStride, written, inCols, tmp[:], &colL1) {
			archsimd.ClearAVXUpperBits()
			return false
		}
	}
	if written < inRows {
		clear(tmp[written*64 : inRows*64])
	}
	if !itxL1Ok16(colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3]))) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	for c0 := 0; c0 < 64; c0 += 16 {
		itxColStripAdd64(tmp[:], 64, c0, inRows, px)
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxDCT16x32 is the 16 wide, 32 high DCT_DCT kernel: rect2-scaled DCT16
// rows with the inter-pass Round2(., 1), then DCT32 columns.
func itxDCT16x32(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx16Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 16]int16
	var colL1 archsimd.Uint16x16
	written := 0
	for ; written < activeRows; written += 16 {
		if !itxRowStrip16(coeff, coeffStride, written, inCols, tx1DDCT, true, 1, tmp[:], &colL1) {
			archsimd.ClearAVXUpperBits()
			return false
		}
	}
	if written < inRows {
		clear(tmp[written*16 : inRows*16])
	}
	if !itxL1Ok16(colL1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd32(tmp[:], 16, 0, inRows, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxDCT32x16 is itxDCT16x32 for the 32 wide, 16 high block.
func itxDCT32x16(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx64Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 32]int16
	var colL1 [2]archsimd.Uint16x16
	if !itxRowStrip32(coeff, coeffStride, 0, inCols, true, 1, tmp[:], &colL1) || !itxL1Ok16(colL1[0].Max(colL1[1])) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd16(tmp[:], 32, 0, inRows, tx1DDCT, px)
	itxColStripAdd16(tmp[:], 32, 16, inRows, tx1DDCT, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxHybridTypes returns the per-pass networks of the DCT/ADST/flipped-ADST
// pairings the fused hybrid kernels cover; ok is false for every other
// transform type (including DCT_DCT, which has its own kernels).
func itxHybridTypes(txType Type) (vertical, horizontal tx1DType, ok bool) {
	switch txType {
	case TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST:
		return txType.tx1DTypes()
	}
	return 0, 0, false
}

// InverseAddDCT8x8 reconstructs one 8-bit 8x8 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride. pred may be dst. It reports false, with dst
// untouched, when the block is outside the kernel's contract or a guard
// declines; the caller then runs the staged path, which produces the same
// bytes.
func InverseAddDCT8x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(8, 8, 1, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare8(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd8x8Hybrid is InverseAddDCT8x8 for the DCT/ADST/flipped-ADST
// pairings (see itxHybridTypes).
func InverseAdd8x8Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok || !itxBlockOK(8, 8, 1, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare8(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT16x16 is InverseAddDCT8x8 for the 16x16 block.
func InverseAddDCT16x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(16, 16, 1, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare16(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd16x16Hybrid is InverseAdd8x8Hybrid for the 16x16 block.
func InverseAdd16x16Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok || !itxBlockOK(16, 16, 1, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare16(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT32x32Low8 is InverseAddDCT8x8 for the 32x32 block whose coded
// coefficients lie in rows and columns 0..7.
func InverseAddDCT32x32Low8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int) bool {
	if !itxBlockOK(32, 32, 1, 32, 8, dst, dstStride, pred, predStride, coeff, coeffStride, 8, 8) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32(&px, coeff, coeffStride, 8, 8)
}

// InverseAddDCT32x32Low16 is InverseAddDCT32x32Low8 for the 16x16 corner.
func InverseAddDCT32x32Low16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows int) bool {
	if !itxBlockOK(32, 32, 1, 32, 16, dst, dstStride, pred, predStride, coeff, coeffStride, 16, 16) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32(&px, coeff, coeffStride, 16, 16)
}

// InverseAddDCT32x32Full is InverseAddDCT8x8 for the 32x32 block with any
// coded extent.
func InverseAddDCT32x32Full(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(32, 32, 1, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT64x64 is InverseAddDCT8x8 for the 64x64 block, whose coded
// coefficients lie in the 32x32 corner (the 32x32 scan at coeffStride).
func InverseAddDCT64x64(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(64, 64, 1, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT64(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT16x32 is InverseAddDCT8x8 for the 16 wide, 32 high block.
func InverseAddDCT16x32(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(16, 32, 1, 32, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT16x32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT32x16 is InverseAddDCT8x8 for the 32 wide, 16 high block.
func InverseAddDCT32x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(32, 16, 1, 16, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32x16(&px, coeff, coeffStride, activeRows, activeCols)
}

// The 10- and 12-bit forms (2-byte samples, strides in bytes). dav1d's
// itx16 reuses its 8bpc int16 column functions at 10 bits; here both passes
// are the 8-bit int16 networks under the same per-lane L1 guards, so none
// of the staged path's bd + 8 row or max(bd + 6, 16) column clamps can bite
// (see itx_fused_hbd_gosimd_arm64.go), and the add saturates and clips to
// the bit depth.

// InverseAddDCT8x8HBD is InverseAddDCT8x8 for a 10- or 12-bit block.
func InverseAddDCT8x8HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(8, 8, 2, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare8(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd8x8HybridHBD is InverseAdd8x8Hybrid for a 10- or 12-bit block.
func InverseAdd8x8HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	maxV, okBD := itxHBDMax(bitDepth)
	if !ok || !okBD || !itxBlockOK(8, 8, 2, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare8(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT16x16HBD is InverseAddDCT16x16 for a 10- or 12-bit block.
func InverseAddDCT16x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(16, 16, 2, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare16(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd16x16HybridHBD is InverseAdd16x16Hybrid for a 10- or 12-bit
// block.
func InverseAdd16x16HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	maxV, okBD := itxHBDMax(bitDepth)
	if !ok || !okBD || !itxBlockOK(16, 16, 2, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare16(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT32x32HBD is InverseAddDCT32x32Full for a 10- or 12-bit block.
func InverseAddDCT32x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(32, 32, 2, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT64x64HBD is InverseAddDCT64x64 for a 10- or 12-bit block.
func InverseAddDCT64x64HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(64, 64, 2, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT64(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT16x32HBD is InverseAddDCT16x32 for a 10- or 12-bit block.
func InverseAddDCT16x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(16, 32, 2, 32, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT16x32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT32x16HBD is InverseAddDCT32x16 for a 10- or 12-bit block.
func InverseAddDCT32x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(32, 16, 2, 16, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT32x16(&px, coeff, coeffStride, activeRows, activeCols)
}
