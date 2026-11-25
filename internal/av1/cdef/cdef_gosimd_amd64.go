// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import (
	"simd/archsimd"
)

func init() {
	if archsimd.X86.AVX2() {
		findDirectionImpl = findDirectionSIMD
		findDirectionDualImpl = findDirectionDualSIMD
		findDirectionU8Impl = findDirectionU8SIMD
		findDirectionDualU8Impl = findDirectionDualU8SIMD
	}
}

// cdefDirRowPair16 packs the 8 samples at pi and pj as [row i | row j],
// shifted right by coeffShift and biased by -128. The word shift is a dword
// VPSRLVD by a count vector masked back to words (wmask = 0xffff >> shift),
// which keeps a legacy-SSE count move out of the ymm code.
func cdefDirRowPair16(pi, pj *[8]uint16, sh archsimd.Uint32x8, wmask archsimd.Uint16x16, bias archsimd.Int16x16) archsimd.Int16x16 {
	var z archsimd.Uint16x16
	v := z.SetLo(archsimd.LoadUint16x8Array(pi)).SetHi(archsimd.LoadUint16x8Array(pj))
	return v.AsUint32x8().ShiftRight(sh).AsUint16x16().And(wmask).AsInt16x16().Sub(bias)
}

func findDirectionSIMD(img []uint16, stride int, coeffShift int) (int, int32) {
	bias := archsimd.BroadcastInt16x16(128)
	sh := archsimd.BroadcastUint32x8(uint32(coeffShift))
	wmask := archsimd.BroadcastUint16x16(uint16(0xffff >> coeffShift))
	if uint(stride) > 1<<20 {
		panic("cdef: direction stride out of range")
	}
	s := stride
	r := func(i int) *[8]uint16 { return (*[8]uint16)(img[i*s : i*s+8]) }
	m0 := cdefDirRowPair16(r(0), r(7), sh, wmask, bias)
	m1 := cdefDirRowPair16(r(1), r(6), sh, wmask, bias)
	m2 := cdefDirRowPair16(r(2), r(5), sh, wmask, bias)
	m3 := cdefDirRowPair16(r(3), r(4), sh, wmask, bias)
	return cdefDirSearchAVX2(m0, m1, m2, m3)
}

func findDirectionDualSIMD(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	dir1, var1 := findDirectionSIMD(img1, stride, coeffShift)
	dir2, var2 := findDirectionSIMD(img2, stride, coeffShift)
	return dir1, var1, dir2, var2
}

var (
	// cdefDirShufW6543210x reverses words 0..6 of each lane and keeps word 7
	// (dav1d shufw_6543210x).
	cdefDirShufW6543210x = [32]int8{
		12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15,
		12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15,
	}
	// cdefDirShufW3012 is pshuflw q3012 on each lane: words 2,1,0,3 then 4..7.
	cdefDirShufW3012 = [32]int8{
		4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	}
	// div_table weights: diagonal pairs (k, 14-k) for k = 0..3 and 4..7, the
	// odd pairs (k, 10-k) for k = 0..2 plus the lone center term.
	cdefDirDivDiagLo = [8]int32{840, 420, 280, 210, 840, 420, 280, 210}
	cdefDirDivDiagHi = [8]int32{168, 140, 120, 105, 168, 140, 120, 105}
	cdefDirDivAlt    = [8]int32{420, 210, 140, 105, 420, 210, 140, 105}
	// cdefDirCostOrder maps the reduced [c0 c6 c1 c7 | c4 c2 c3 c5] to c0..c7.
	cdefDirCostOrder = [8]uint32{0, 2, 5, 6, 4, 7, 1, 3}
)

// cdefDirRowPair loads row i (low 8 bytes of a 16-byte load) and the 8 bytes
// of row j, and widens them to [row i | row j] biased by -128. Only rows
// 0..3 take the 16-byte load, so nothing is read past row 7's last byte.
func cdefDirRowPair(pi *[16]byte, pj *[8]byte, bias archsimd.Int16x16) archsimd.Int16x16 {
	j := uint64(pj[0]) | uint64(pj[1])<<8 | uint64(pj[2])<<16 | uint64(pj[3])<<24 |
		uint64(pj[4])<<32 | uint64(pj[5])<<40 | uint64(pj[6])<<48 | uint64(pj[7])<<56
	v := archsimd.LoadUint8x16Array(pi).AsUint64x2().SetElem(1, j)
	return v.AsUint8x16().ExtendToUint16().AsInt16x16().Sub(bias)
}

// cdefDirShl shifts each 128-bit lane left by n words (pslldq 2n).
func cdefDirShl(x, zero archsimd.Int16x16, n uint64) archsimd.Int16x16 {
	return x.AsUint8x32().ConcatShiftBytesRightGrouped(zero.AsUint8x32(), 16-2*n).AsInt16x16()
}

// cdefDirShr shifts each 128-bit lane right by n words (psrldq 2n).
func cdefDirShr(x, zero archsimd.Int16x16, n uint64) archsimd.Int16x16 {
	return zero.AsUint8x32().ConcatShiftBytesRightGrouped(x.AsUint8x32(), 2*n).AsInt16x16()
}

// cdefDirOddCost turns an odd family's left (k = 0..7) and right (k = 8..10
// in words 0..2, zeros above) partials into four int32 cost terms per lane.
func cdefDirOddCost(left, right archsimd.Int16x16, shuf archsimd.Int8x32, divAlt, div105 archsimd.Int32x8) archsimd.Int32x8 {
	rs := right.AsUint8x32().PermuteOrZeroGrouped(shuf).AsInt16x16()
	lo := left.InterleaveLoGrouped(rs)
	hi := right.InterleaveHiGrouped(left)
	return lo.DotProductPairs(lo).Mul(divAlt).Add(hi.DotProductPairs(hi).Mul(div105))
}

func findDirectionU8SIMD(img []byte, stride int) (int, int32) {
	bias := archsimd.BroadcastInt16x16(128)
	if uint(stride) > 1<<20 {
		panic("cdef: direction stride out of range")
	}
	s := stride
	r16 := func(i int) *[16]byte { return (*[16]byte)(img[i*s : i*s+16]) }
	r8 := func(i int) *[8]byte { return (*[8]byte)(img[i*s : i*s+8]) }
	m0 := cdefDirRowPair(r16(0), r8(7), bias)
	m1 := cdefDirRowPair(r16(1), r8(6), bias)
	m2 := cdefDirRowPair(r16(2), r8(5), bias)
	m3 := cdefDirRowPair(r16(3), r8(4), bias)
	return cdefDirSearchAVX2(m0, m1, m2, m3)
}

// cdefDirSearchAVX2 is cdef_dir's shared body (dav1d's .main label, which
// the 16bpc entry jumps into): m_i = [row i | row 7-i] for i < 4, already
// biased by -128.
func cdefDirSearchAVX2(m0, m1, m2, m3 archsimd.Int16x16) (int, int32) {
	var zero archsimd.Int16x16
	m7 := m0.ConcatPermute128Scalars(1, 0, m0)
	m6 := m1.ConcatPermute128Scalars(1, 0, m1)
	m5 := m2.ConcatPermute128Scalars(1, 0, m2)
	m4 := m3.ConcatPermute128Scalars(1, 0, m3)

	div105 := archsimd.BroadcastInt32x8(105)

	// partial_sum_hv: column sums (cost6) | row sums (cost2).
	s := m0.Add(m1).Add(m2.Add(m3))
	h := m0.ConcatAddPairsGrouped(m1).ConcatAddPairsGrouped(m2.ConcatAddPairsGrouped(m3))
	hv := s.SetHi(h.GetLo().ConcatAddPairs(h.GetHi())).SetLo(s.GetLo().Add(s.GetHi()))
	c62 := hv.DotProductPairs(hv).Mul(div105)

	// partial_sum_diag[0/1]: row i (lane 0) / row 7-i (lane 1) at word shift i.
	dl := m0.Add(cdefDirShl(m1, zero, 1)).Add(cdefDirShl(m2, zero, 2)).Add(cdefDirShl(m3, zero, 3)).
		Add(cdefDirShl(m4, zero, 4)).Add(cdefDirShl(m5, zero, 5)).Add(cdefDirShl(m6, zero, 6)).Add(cdefDirShl(m7, zero, 7))
	dr := cdefDirShr(m1, zero, 7).Add(cdefDirShr(m2, zero, 6)).Add(cdefDirShr(m3, zero, 5)).
		Add(cdefDirShr(m4, zero, 4)).Add(cdefDirShr(m5, zero, 3)).Add(cdefDirShr(m6, zero, 2)).Add(cdefDirShr(m7, zero, 1))
	dr = dr.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&cdefDirShufW6543210x)).AsInt16x16()
	dlo := dl.InterleaveLoGrouped(dr)
	dhi := dl.InterleaveHiGrouped(dr)
	c04 := dlo.DotProductPairs(dlo).Mul(archsimd.LoadInt32x8Array(&cdefDirDivDiagLo)).
		Add(dhi.DotProductPairs(dhi).Mul(archsimd.LoadInt32x8Array(&cdefDirDivDiagHi)))

	shuf := archsimd.LoadInt8x32Array(&cdefDirShufW3012)
	divAlt := archsimd.LoadInt32x8Array(&cdefDirDivAlt)

	// partial_sum_alt[3/2] (cost7 | cost5): row pairs at word shift 0..3.
	a0, a1, a2, a3 := m0.Add(m1), m2.Add(m3), m4.Add(m5), m6.Add(m7)
	al := a0.Add(cdefDirShl(a1, zero, 1)).Add(cdefDirShl(a2, zero, 2)).Add(cdefDirShl(a3, zero, 3))
	ar := cdefDirShr(a1, zero, 7).Add(cdefDirShr(a2, zero, 6)).Add(cdefDirShr(a3, zero, 5))
	c75 := cdefDirOddCost(al, ar, shuf, divAlt, div105)

	// partial_sum_alt[0/1] (cost1 | cost3): pairwise row sums at word shift i.
	q0 := m0.ConcatAddPairsGrouped(m4)
	q1 := m1.ConcatAddPairsGrouped(m5)
	q2 := m2.ConcatAddPairsGrouped(m6)
	q3 := m3.ConcatAddPairsGrouped(m7)
	ql := q0.Add(cdefDirShl(q1, zero, 1)).Add(cdefDirShl(q2, zero, 2)).Add(cdefDirShl(q3, zero, 3))
	qr := cdefDirShr(q1, zero, 7).Add(cdefDirShr(q2, zero, 6)).Add(cdefDirShr(q3, zero, 5))
	c13 := cdefDirOddCost(ql, qr, shuf, divAlt, div105)

	// [c0 c6 c1 c7 | c4 c2 c3 c5] -> c0..c7.
	t := c04.ConcatAddPairsGrouped(c62).ConcatAddPairsGrouped(c13.ConcatAddPairsGrouped(c75))
	var cost [8]int32
	t.Permute(archsimd.LoadUint32x8Array(&cdefDirCostOrder)).StoreArray(&cost)
	archsimd.ClearAVXUpperBits()

	bestDir, bestCost := 0, cost[0]
	for d := 1; d < 8; d++ {
		if cost[d] > bestCost {
			bestDir, bestCost = d, cost[d]
		}
	}
	return bestDir, (bestCost - cost[bestDir^4]) >> 10
}

func findDirectionDualU8SIMD(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8SIMD(img1, stride)
	dir2, var2 := findDirectionU8SIMD(img2, stride)
	return dir1, var1, dir2, var2
}

// init binds the AVX2 Go SIMD CDEF block filters when the CPU advertises AVX2.
// Otherwise the pure-Go references stay in place. The 256-bit word kernels
// (filter_word_amd64.go) take every shape they cover and hand the rest to the
// 128-bit kernels. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
func init() {
	if archsimd.X86.AVX2() {
		filterBlockImpl = filterBlockAVX2
		filterBlockU8Impl = filterBlockU8AVX2
	}
}

// cdefShiftCount is the per-block constrain() shift in the form AVX2 takes
// for a uniform per-block count: a scalar that VPSRLW reads from an xmm.
type cdefShiftCount = uint64

func cdefShiftCountOf(n int) cdefShiftCount {
	return uint64(n)
}

// cdefShrU16 is a logical right shift of each 16-bit lane by the per-block
// count. AVX2 has no per-lane 16-bit variable shift, so the uniform count
// rides in the scalar form.
func cdefShrU16(v archsimd.Uint16x8, sh cdefShiftCount) archsimd.Uint16x8 {
	return v.ShiftAllRight(sh)
}

// cdefLowByteIdx gathers the low byte of each 16-bit lane into bytes 0..7
// (PSHUFB, AVX2-era; the lanes 8..15 index with the zeroing high bit).
var cdefLowByteIdx = archsimd.LoadInt8x16Array(&[16]int8{0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128})

// cdefNarrowU8 keeps the low byte of each lane, the same bits as byte(y) in
// the pure-Go reference. AVX2 has no 16-to-8 truncation without AVX-512, so
// the bytes are gathered with PermuteOrZero (VPSHUFB) instead.
func cdefNarrowU8(y archsimd.Int16x8) archsimd.Uint8x16 {
	return y.ToBits().ReshapeToUint8s().PermuteOrZero(cdefLowByteIdx)
}

// cdefSarI16 is an arithmetic right shift of each 16-bit lane by a count from
// cdefShiftCountOf.
func cdefSarI16(v archsimd.Int16x8, sh cdefShiftCount) archsimd.Int16x8 {
	return v.ShiftAllRight(sh)
}

// cdefHasAVX2 gates the amd64 byte route at run time.
var cdefHasAVX2 = archsimd.X86.AVX2()

func u8ByteInteriorAvailable() bool { return cdefHasAVX2 }

// cdefByteAVX2Ctx is the per-block state of the byte kernels. Offsets are
// byte offsets into the BStride-spaced byte tap buffer.
type cdefByteAVX2Ctx struct {
	input  []byte
	dst    []byte
	srcOff int // the block's first tap-buffer byte
	dstOff int // the block's first output byte
	dstStr int
	height int

	// Tap window indices cdefByteCenter +- the direction's tap offsets
	// (cdefTapIdx's layout: p0+, p0-, p1+, p1-, s0+, s0-, ..., s3-).
	taps *[12]uint

	priStr, secStr uint8
	priSh, secSh   uint64
	// priW is the primary weight applied after the int8 pre-sum (tap1), and
	// priDbl is 0xff when tap0 == 2*tap1 (strength even: taps 4, 2).
	priW   uint8
	priDbl int8
}

func (c *cdefByteAVX2Ctx) setPrimary(s, damping int) {
	taps := cdefPrimaryTaps[s&1]
	c.priW = taps[1]
	c.priDbl = 0
	if taps[0] == 2*taps[1] {
		c.priDbl = -1
	}
	c.priStr = uint8(s)
	c.priSh = uint64(constrainShift(s, damping))
}

// filterUnitBlocksU8Byte is filterUnitBlocksU8PureGo on a byte tap buffer:
// per-block luma strength adjust, blocks with both strengths zero skipped
// (dav1d cdef_apply_tmpl.c), and the pri / sec / pri+sec kernel picked per
// block as dav1d's cdef_filter_%1x%2_8bpc does.
func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	ctx := cdefByteAVX2Ctx{dstStr: dstStride, height: u.blockHeight}
	sec := u.secondaryStrength
	ctx.secStr = uint8(sec)
	ctx.secSh = uint64(constrainShift(sec, u.damping))
	strength := u.primaryStrength
	if !u.lumaAdjust {
		ctx.setPrimary(strength, u.damping)
	}
	ctx.input, ctx.dst = input, dst
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			ctx.setPrimary(strength, u.damping)
		}
		if strength == 0 && sec == 0 {
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.taps = &cdefByteTapIdx[dir&7]
		ctx.srcOff = inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		ctx.dstOff = (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		w8 := u.blockWidth == 8
		switch {
		case strength != 0 && sec != 0:
			if w8 {
				cdefByteAVX2Filter8PriSec(&ctx)
			} else {
				cdefByteAVX2Filter4PriSec(&ctx)
			}
		case strength != 0:
			if w8 {
				cdefByteAVX2Filter8Pri(&ctx)
			} else {
				cdefByteAVX2Filter4Pri(&ctx)
			}
		default:
			if w8 {
				cdefByteAVX2Filter8Sec(&ctx)
			} else {
				cdefByteAVX2Filter4Sec(&ctx)
			}
		}
	}
}

// The kernels take a four-row group's taps from one window of the byte tap
// buffer starting cdefByteCenter bytes (two rows and two columns, the
// farthest tap) before the group's first pixel; each row's taps are a
// constant-offset sub-array of it, and a tap at offset t sits at
// cdefByteCenter + t there. One bounds check per group.
const (
	cdefByteCenter = 2*BStride + 2
	cdefByteMaxOff = 2 * cdefByteCenter
	cdefByteWinRow = cdefByteMaxOff + 16
	cdefByteWin4   = cdefByteWinRow + 3*BStride
)

// cdefByteTapIdx is cdefTapIdx for the byte tap buffer.
var cdefByteTapIdx = func() (t [8][12]uint) {
	for d := range t {
		offs := [6]int{
			int(cdefDirections[d+2][0]), int(cdefDirections[d+2][1]),
			int(cdefDirections[d+4][0]), int(cdefDirections[d][0]),
			int(cdefDirections[d+4][1]), int(cdefDirections[d][1]),
		}
		for k, o := range offs {
			t[d][2*k], t[d][2*k+1] = uint(cdefByteCenter+o), uint(cdefByteCenter-o)
		}
	}
	return
}()

const errCDEFByteWindow = "cdef: byte block outside the tap buffer or destination"

// cdefByteRows4 returns the four rows' tap windows of the group whose window
// starts at o.
func cdefByteRows4(in []byte, o int) (*[cdefByteWinRow]byte, *[cdefByteWinRow]byte, *[cdefByteWinRow]byte, *[cdefByteWinRow]byte) {
	g := (*[cdefByteWin4]byte)(in[o : o+cdefByteWin4])
	return (*[cdefByteWinRow]byte)(g[0:cdefByteWinRow]), (*[cdefByteWinRow]byte)(g[BStride : BStride+cdefByteWinRow]),
		(*[cdefByteWinRow]byte)(g[2*BStride : 2*BStride+cdefByteWinRow]), (*[cdefByteWinRow]byte)(g[3*BStride : 3*BStride+cdefByteWinRow])
}

// rows checks the block origin, height and output layout, so each group
// window and output row below is one compare, and returns the first group's
// window start.
func (c *cdefByteAVX2Ctx) rows() int {
	if c.srcOff < cdefByteCenter || c.srcOff > 1<<24 || uint(c.height) > 128 ||
		uint(c.dstOff) > 1<<30 || uint(c.dstStr) > 1<<20 {
		panic(errCDEFByteWindow)
	}
	return c.srcOff - cdefByteCenter
}

// priTaps / secTaps0 / secTaps1 return tap window indices checked against
// the row window, so the tap loads need no bounds checks.
func (c *cdefByteAVX2Ctx) priTaps() (uint, uint, uint, uint) {
	t := c.taps
	a, b, d, e := t[0], t[1], t[2], t[3]
	if a > cdefByteMaxOff || b > cdefByteMaxOff || d > cdefByteMaxOff || e > cdefByteMaxOff {
		panic(errCDEFByteWindow)
	}
	return a, b, d, e
}

func (c *cdefByteAVX2Ctx) secTaps0() (uint, uint, uint, uint) {
	t := c.taps
	a, b, d, e := t[4], t[5], t[6], t[7]
	if a > cdefByteMaxOff || b > cdefByteMaxOff || d > cdefByteMaxOff || e > cdefByteMaxOff {
		panic(errCDEFByteWindow)
	}
	return a, b, d, e
}

func (c *cdefByteAVX2Ctx) secTaps1() (uint, uint, uint, uint) {
	t := c.taps
	a, b, d, e := t[8], t[9], t[10], t[11]
	if a > cdefByteMaxOff || b > cdefByteMaxOff || d > cdefByteMaxOff || e > cdefByteMaxOff {
		panic(errCDEFByteWindow)
	}
	return a, b, d, e
}

func cdefByteLd16(r *[cdefByteWinRow]byte, off uint) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(r[off : off+16]))
}

// cdefByteAVX2Load8 packs the four rows' 8 pixels at off as [r0 r1 | r2 r3]
// (LOAD_BLOCK w=8: movq + vinserti128 + punpcklqdq).
func cdefByteAVX2Load8(r0, r1, r2, r3 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	ac := z.SetLo(cdefByteLd16(r0, off)).SetHi(cdefByteLd16(r2, off)).AsUint64x4()
	bd := z.SetLo(cdefByteLd16(r1, off)).SetHi(cdefByteLd16(r3, off)).AsUint64x4()
	return ac.InterleaveLoGrouped(bd).AsUint8x32()
}

// cdefByteAVX2Load4 packs the four rows' 4 pixels at off as
// [r0 r1 . . | r2 r3 . .] (LOAD_BLOCK w=4 h=4: movd + vinserti128 +
// punpckldq). Bytes 8..15 of each lane are don't-care.
func cdefByteAVX2Load4(r0, r1, r2, r3 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	ac := z.SetLo(cdefByteLd16(r0, off)).SetHi(cdefByteLd16(r2, off)).AsUint32x8()
	bd := z.SetLo(cdefByteLd16(r1, off)).SetHi(cdefByteLd16(r3, off)).AsUint32x8()
	return ac.InterleaveLoGrouped(bd).AsUint8x32()
}

func cdefBytePut8(b []byte, o int, v uint64) {
	w := (*[8]byte)(b[o : o+8])
	w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

func cdefBytePut4(b []byte, o int, v uint32) {
	w := (*[4]byte)(b[o : o+4])
	w[0], w[1], w[2], w[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func cdefByteAVX2Store8(dst []byte, o, dstStr int, v archsimd.Uint8x32) {
	lo := v.GetLo().AsUint64x2()
	hi := v.GetHi().AsUint64x2()
	cdefBytePut8(dst, o, lo.GetElem(0))
	cdefBytePut8(dst, o+dstStr, lo.GetElem(1))
	cdefBytePut8(dst, o+2*dstStr, hi.GetElem(0))
	cdefBytePut8(dst, o+3*dstStr, hi.GetElem(1))
}

func cdefByteAVX2Store4(dst []byte, o, dstStr int, v archsimd.Uint8x32) {
	lo := v.GetLo().AsUint32x4()
	hi := v.GetHi().AsUint32x4()
	cdefBytePut4(dst, o, lo.GetElem(0))
	cdefBytePut4(dst, o+dstStr, lo.GetElem(1))
	cdefBytePut4(dst, o+2*dstStr, hi.GetElem(0))
	cdefBytePut4(dst, o+3*dstStr, hi.GetElem(1))
}

// cdefByteAVX2Constrain is ACCUMULATE_TAP_BYTE's constrain(): the signed
// byte min(|t-x|, max(0, str - (|t-x| >> sh))) carrying the sign of t-x.
// The byte shift is a dword shift masked to the surviving bits (mask is
// 0xff >> sh broadcast). It is VPSRLVD by a hoisted count vector rather than
// VPSRLW by an xmm count: the compiler rematerializes that count with a
// legacy-SSE MOVQ inside the row loop, an SSE/AVX transition while the ymm
// upper halves are live.
func cdefByteAVX2Constrain(t, x, str archsimd.Uint8x32, sh archsimd.Uint32x8, mask archsimd.Uint8x32) archsimd.Int8x32 {
	pos := t.SubSaturated(x)
	neg := x.SubSaturated(t)
	ad := pos.Or(neg)
	lim := str.SubSaturated(ad.AsUint32x8().ShiftRight(sh).AsUint8x32().And(mask))
	return pos.Min(lim).AsInt8x32().Sub(neg.Min(lim).AsInt8x32())
}

// cdefByteAVX2Delta is ((8 + sum - (sum < 0)) >> 4) on one parity's int16
// sums.
func cdefByteAVX2Delta(sum, eight archsimd.Int16x16) archsimd.Int16x16 {
	return sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4)
}

// cdefByteAVX2Join recombines the even- and odd-pixel int16 deltas into
// one int8 delta per byte lane.
func cdefByteAVX2Join(dE, dO archsimd.Int16x16, lowMask archsimd.Uint16x16) archsimd.Uint8x32 {
	return dE.AsUint16x16().And(lowMask).Or(dO.AsUint16x16().ShiftAllLeft(8)).AsUint8x32()
}

// cdefByteAVX2Weights returns the even- and odd-pixel pmaddubsw weight
// patterns [w,0,w,0..] and [0,w,0,w..].
func cdefByteAVX2Weights(w uint8) (archsimd.Uint8x32, archsimd.Uint8x32) {
	return archsimd.BroadcastUint16x16(uint16(w)).AsUint8x32(), archsimd.BroadcastUint16x16(uint16(w) << 8).AsUint8x32()
}

func cdefByteAVX2Filter8PriSec(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(c.priSh)), archsimd.BroadcastUint32x8(uint32(c.secSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	bias := archsimd.BroadcastUint8x32(0x80)
	p0p, p0m, p1p, p1m := c.priTaps()
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load8(r0, r1, r2, r3, cdefByteCenter)
		p, q := cdefByteAVX2Load8(r0, r1, r2, r3, p0p), cdefByteAVX2Load8(r0, r1, r2, r3, p0m)
		mn, mx := x.Min(p).Min(q), x.Max(p).Max(q)
		p0 := cdefByteAVX2Constrain(p, x, priStr, priSh, priMask).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, p1p), cdefByteAVX2Load8(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		w := p0.Add(p0.And(dbl)).Add(cdefByteAVX2Constrain(p, x, priStr, priSh, priMask)).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s0p), cdefByteAVX2Load8(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 := cdefByteAVX2Constrain(p, x, secStr, secSh, secMask).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s1p), cdefByteAVX2Load8(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 = s0.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s2p), cdefByteAVX2Load8(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s := s0.Add(s0).Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load8(r0, r1, r2, r3, s3p), cdefByteAVX2Load8(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s = s.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		sumE := pE.DotProductPairsSaturated(w).Add(sE.DotProductPairsSaturated(s))
		sumO := pO.DotProductPairsSaturated(w).Add(sO.DotProductPairsSaturated(s))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sumE, eight), cdefByteAVX2Delta(sumO, eight), lowMask)
		y := x.Xor(bias).AsInt8x32().AddSaturated(d.AsInt8x32()).AsUint8x32().Xor(bias).Min(mx).Max(mn)
		cdefByteAVX2Store8(dst, do, dstStr, y)
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter8Pri(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	priSh := archsimd.BroadcastUint32x8(uint32(c.priSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	p0p, p0m, p1p, p1m := c.priTaps()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load8(r0, r1, r2, r3, cdefByteCenter)
		p0 := cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p0p), x, priStr, priSh, priMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p0m), x, priStr, priSh, priMask))
		w := p0.Add(p0.And(dbl)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p1p), x, priStr, priSh, priMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, p1m), x, priStr, priSh, priMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(pE.DotProductPairsSaturated(w), eight), cdefByteAVX2Delta(pO.DotProductPairsSaturated(w), eight), lowMask)
		cdefByteAVX2Store8(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter8Sec(c *cdefByteAVX2Ctx) {
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	secSh := archsimd.BroadcastUint32x8(uint32(c.secSh))
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load8(r0, r1, r2, r3, cdefByteCenter)
		s0 := cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s0p), x, secStr, secSh, secMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s0m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s1p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s1m), x, secStr, secSh, secMask))
		s := s0.Add(s0).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s2p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s2m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s3p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load8(r0, r1, r2, r3, s3m), x, secStr, secSh, secMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sE.DotProductPairsSaturated(s), eight), cdefByteAVX2Delta(sO.DotProductPairsSaturated(s), eight), lowMask)
		cdefByteAVX2Store8(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter4PriSec(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(c.priSh)), archsimd.BroadcastUint32x8(uint32(c.secSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	bias := archsimd.BroadcastUint8x32(0x80)
	p0p, p0m, p1p, p1m := c.priTaps()
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load4(r0, r1, r2, r3, cdefByteCenter)
		p, q := cdefByteAVX2Load4(r0, r1, r2, r3, p0p), cdefByteAVX2Load4(r0, r1, r2, r3, p0m)
		mn, mx := x.Min(p).Min(q), x.Max(p).Max(q)
		p0 := cdefByteAVX2Constrain(p, x, priStr, priSh, priMask).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, p1p), cdefByteAVX2Load4(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		w := p0.Add(p0.And(dbl)).Add(cdefByteAVX2Constrain(p, x, priStr, priSh, priMask)).Add(cdefByteAVX2Constrain(q, x, priStr, priSh, priMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s0p), cdefByteAVX2Load4(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 := cdefByteAVX2Constrain(p, x, secStr, secSh, secMask).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s1p), cdefByteAVX2Load4(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s0 = s0.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s2p), cdefByteAVX2Load4(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s := s0.Add(s0).Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		p, q = cdefByteAVX2Load4(r0, r1, r2, r3, s3p), cdefByteAVX2Load4(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		s = s.Add(cdefByteAVX2Constrain(p, x, secStr, secSh, secMask)).Add(cdefByteAVX2Constrain(q, x, secStr, secSh, secMask))
		sumE := pE.DotProductPairsSaturated(w).Add(sE.DotProductPairsSaturated(s))
		sumO := pO.DotProductPairsSaturated(w).Add(sO.DotProductPairsSaturated(s))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sumE, eight), cdefByteAVX2Delta(sumO, eight), lowMask)
		y := x.Xor(bias).AsInt8x32().AddSaturated(d.AsInt8x32()).AsUint8x32().Xor(bias).Min(mx).Max(mn)
		cdefByteAVX2Store4(dst, do, dstStr, y)
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter4Pri(c *cdefByteAVX2Ctx) {
	priStr := archsimd.BroadcastUint8x32(c.priStr)
	priMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.priSh))
	priSh := archsimd.BroadcastUint32x8(uint32(c.priSh))
	dbl := archsimd.BroadcastInt8x32(c.priDbl)
	pE, pO := cdefByteAVX2Weights(c.priW)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	p0p, p0m, p1p, p1m := c.priTaps()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load4(r0, r1, r2, r3, cdefByteCenter)
		p0 := cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p0p), x, priStr, priSh, priMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p0m), x, priStr, priSh, priMask))
		w := p0.Add(p0.And(dbl)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p1p), x, priStr, priSh, priMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, p1m), x, priStr, priSh, priMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(pE.DotProductPairsSaturated(w), eight), cdefByteAVX2Delta(pO.DotProductPairsSaturated(w), eight), lowMask)
		cdefByteAVX2Store4(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

func cdefByteAVX2Filter4Sec(c *cdefByteAVX2Ctx) {
	secStr := archsimd.BroadcastUint8x32(c.secStr)
	secMask := archsimd.BroadcastUint8x32(uint8(0xff >> c.secSh))
	secSh := archsimd.BroadcastUint32x8(uint32(c.secSh))
	sE, sO := cdefByteAVX2Weights(1)
	eight := archsimd.BroadcastInt16x16(8)
	lowMask := archsimd.BroadcastUint16x16(0xff)
	s0p, s0m, s1p, s1m := c.secTaps0()
	s2p, s2m, s3p, s3m := c.secTaps1()
	base := c.rows()
	input, dst, dstOff, dstStr := c.input, c.dst, c.dstOff, c.dstStr
	for i := 0; i < c.height>>2; i++ {
		r0, r1, r2, r3 := cdefByteRows4(input, base+4*i*BStride)
		do := dstOff + 4*i*dstStr
		x := cdefByteAVX2Load4(r0, r1, r2, r3, cdefByteCenter)
		s0 := cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s0p), x, secStr, secSh, secMask).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s0m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s1p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s1m), x, secStr, secSh, secMask))
		s := s0.Add(s0).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s2p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s2m), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s3p), x, secStr, secSh, secMask)).
			Add(cdefByteAVX2Constrain(cdefByteAVX2Load4(r0, r1, r2, r3, s3m), x, secStr, secSh, secMask))
		d := cdefByteAVX2Join(cdefByteAVX2Delta(sE.DotProductPairsSaturated(s), eight), cdefByteAVX2Delta(sO.DotProductPairsSaturated(s), eight), lowMask)
		cdefByteAVX2Store4(dst, do, dstStr, x.Add(d))
	}
	archsimd.ClearAVXUpperBits()
}

// The amd64 secondary-only byte route runs the AVX2 byte-lane sec-only
// kernels (dav1d cdef_avx2.asm CDEF_FILTER .sec_only, direction 0), the same
// kernels the general byte route picks for a primary-strength-zero unit, so
// the decoder takes the identical route on amd64 and arm64.
func u8SecondaryOnlyByteInteriorAvailable() bool { return cdefHasAVX2 }

func filterFrameBlocksU8SecondaryOnlyByteInterior(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, blockWidth int, blockHeight int, strength int, shift int) bool {
	if !cdefHasAVX2 || (blockWidth != 4 && blockWidth != 8) || (blockHeight != 4 && blockHeight != 8) {
		return false
	}
	// The AVX2 kernels load 16 bytes per tap row; the exported entry proves
	// only the 2-pixel halo. Decline before any write unless the whole
	// over-read window of the blocks' bounding box is inside input, the same
	// slack FilterFrameBlocksU8ByteInteriorTrusted requires.
	rowsEnd, colsEnd := 0, 0
	for _, block := range blocks {
		rowsEnd = max(rowsEnd, (int(block.BY)+1)*blockHeight)
		colsEnd = max(colsEnd, (int(block.BX)+1)*blockWidth)
	}
	if inputOrigin+(rowsEnd+1)*BStride+colsEnd+16 > len(input) {
		return false
	}
	ctx := cdefByteAVX2Ctx{dstStr: dstStride, height: blockHeight, secStr: uint8(strength), secSh: uint64(shift)}
	ctx.taps = &cdefByteTapIdx[0]
	ctx.input, ctx.dst = input, dst
	bwLog2, bhLog2 := 3, 3
	if blockWidth == 4 {
		bwLog2 = 2
	}
	if blockHeight == 4 {
		bhLog2 = 2
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		ctx.srcOff = inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		ctx.dstOff = (by<<bhLog2)*dstStride + (bx << bwLog2)
		if blockWidth == 8 {
			cdefByteAVX2Filter8Sec(&ctx)
		} else {
			cdefByteAVX2Filter4Sec(&ctx)
		}
	}
	return true
}

// cdefWordLd loads the 8 samples of w at i as Int16x8.
func cdefWordLd(w *[cdefWin1]uint16, i uint) archsimd.Int16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(w[i : i+8])).AsInt16x8()
}

// cdefWordLoad8W packs the 8 samples at window index i of two rows' windows
// as [r0 | r1].
func cdefWordLoad8W(w0, w1 *[cdefWin1]uint16, i uint) archsimd.Int16x16 {
	var z archsimd.Int16x16
	return z.SetLo(cdefWordLd(w0, i)).SetHi(cdefWordLd(w1, i))
}

// cdefWordLoad4W packs the 4 samples at window index i of four rows' windows
// as [r0 r1 | r2 r3] (BORDER_LOAD_BLOCK w=4: movq / movhps /
// vinserti128). The full-width loads read four halo samples past each row
// segment, inside the 8-column border.
func cdefWordLoad4W(w0, w1, w2, w3 *[cdefWin1]uint16, i uint) archsimd.Int16x16 {
	var z archsimd.Int16x16
	ac := z.SetLo(cdefWordLd(w0, i)).SetHi(cdefWordLd(w2, i)).AsUint64x4()
	bd := z.SetLo(cdefWordLd(w1, i)).SetHi(cdefWordLd(w3, i)).AsUint64x4()
	return ac.InterleaveLoGrouped(bd).AsInt16x16()
}

// cdefWordConstrain is constrain() on 16 word lanes.
//
// The shift is VPSRLVD by a hoisted count vector, masked back to words
// (wmask = 0xffff >> sh): VPSRLW takes its count from an xmm the compiler
// rematerializes with a legacy-SSE MOVQ inside the row loop, an SSE/AVX
// transition while the ymm upper halves are live.
func cdefWordConstrain(t, x archsimd.Int16x16, str archsimd.Uint16x16, sh archsimd.Uint32x8, wmask archsimd.Uint16x16) archsimd.Int16x16 {
	d := t.Sub(x)
	a := d.Abs()
	lim := str.SubSaturated(a.AsUint32x8().ShiftRight(sh).AsUint16x16().And(wmask)).AsInt16x16()
	return a.Min(lim).MulSign(d)
}

// cdefWordLowBytes gathers each 128-bit lane's low bytes of its eight words
// into that lane's bytes 0..7 (VPSHUFB).
var cdefWordLowBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
	0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
}

// cdefWordFilter8 filters an 8-wide block two rows per vector. Like the
// 128-bit kernel it selects the strength split with loop-invariant flags.
func cdefWordFilter8(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastUint16x16(uint16(ctx.priStrength))
	secStr := archsimd.BroadcastUint16x16(uint16(ctx.secStrength))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(ctx.priShift)), archsimd.BroadcastUint32x8(uint32(ctx.secShift))
	priM, secM := archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.priShift)), archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.secShift))
	priTap0 := archsimd.BroadcastInt16x16(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x16(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x16(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x16(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x16(8)
	noSentinel := archsimd.BroadcastInt16x16(int16(^VeryLarge))
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, out8, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRows2(in, base+r*BStride)
		x := cdefWordLoad8W(w0, w1, cdefWinReach)
		var sum archsimd.Int16x16
		mx, mn := x, x
		if priOn {
			t0 := cdefWordLoad8W(w0, w1, p0p)
			t1 := cdefWordLoad8W(w0, w1, p0m)
			t2 := cdefWordLoad8W(w0, w1, p1p)
			t3 := cdefWordLoad8W(w0, w1, p1m)
			sum = cdefWordConstrain(t0, x, priStr, priSh, priM).Add(cdefWordConstrain(t1, x, priStr, priSh, priM)).Mul(priTap0).
				Add(cdefWordConstrain(t2, x, priStr, priSh, priM).Add(cdefWordConstrain(t3, x, priStr, priSh, priM)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefWordLoad8W(w0, w1, s0p)
			s1 := cdefWordLoad8W(w0, w1, s0m)
			s2 := cdefWordLoad8W(w0, w1, s1p)
			s3 := cdefWordLoad8W(w0, w1, s1m)
			sum = sum.Add(cdefWordConstrain(s0, x, secStr, secSh, secM).Add(cdefWordConstrain(s1, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s2, x, secStr, secSh, secM)).Add(cdefWordConstrain(s3, x, secStr, secSh, secM)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefWordLoad8W(w0, w1, s2p)
			s5 := cdefWordLoad8W(w0, w1, s2m)
			s6 := cdefWordLoad8W(w0, w1, s3p)
			s7 := cdefWordLoad8W(w0, w1, s3m)
			sum = sum.Add(cdefWordConstrain(s4, x, secStr, secSh, secM).Add(cdefWordConstrain(s5, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s6, x, secStr, secSh, secM)).Add(cdefWordConstrain(s7, x, secStr, secSh, secM)).Mul(secTap1))
			if clip {
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		y := x.Add(sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			b := y.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&cdefWordLowBytes))
			cdefPut8(out8, do+r*ds, b.GetLo().AsUint64x2().GetElem(0))
			cdefPut8(out8, do+(r+1)*ds, b.GetHi().AsUint64x2().GetElem(0))
		} else {
			y.GetLo().AsUint16x8().StoreArray(cdefOut16(out16, do+r*ds))
			y.GetHi().AsUint16x8().StoreArray(cdefOut16(out16, do+(r+1)*ds))
		}
	}
	archsimd.ClearAVXUpperBits()
}

// cdefWordFilter4 filters a 4-wide block four rows per vector. The height is
// a multiple of 4 by dispatch contract.
func cdefWordFilter4(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastUint16x16(uint16(ctx.priStrength))
	secStr := archsimd.BroadcastUint16x16(uint16(ctx.secStrength))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(ctx.priShift)), archsimd.BroadcastUint32x8(uint32(ctx.secShift))
	priM, secM := archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.priShift)), archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.secShift))
	priTap0 := archsimd.BroadcastInt16x16(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x16(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x16(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x16(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x16(8)
	noSentinel := archsimd.BroadcastInt16x16(int16(^VeryLarge))
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, out8, do, ds := ctx.outRows()
	for i := 0; i < height>>2; i++ {
		r := 4 * i
		w0, w1, w2, w3 := cdefRows4(in, base+r*BStride)
		x := cdefWordLoad4W(w0, w1, w2, w3, cdefWinReach)
		var sum archsimd.Int16x16
		mx, mn := x, x
		if priOn {
			t0 := cdefWordLoad4W(w0, w1, w2, w3, p0p)
			t1 := cdefWordLoad4W(w0, w1, w2, w3, p0m)
			t2 := cdefWordLoad4W(w0, w1, w2, w3, p1p)
			t3 := cdefWordLoad4W(w0, w1, w2, w3, p1m)
			sum = cdefWordConstrain(t0, x, priStr, priSh, priM).Add(cdefWordConstrain(t1, x, priStr, priSh, priM)).Mul(priTap0).
				Add(cdefWordConstrain(t2, x, priStr, priSh, priM).Add(cdefWordConstrain(t3, x, priStr, priSh, priM)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefWordLoad4W(w0, w1, w2, w3, s0p)
			s1 := cdefWordLoad4W(w0, w1, w2, w3, s0m)
			s2 := cdefWordLoad4W(w0, w1, w2, w3, s1p)
			s3 := cdefWordLoad4W(w0, w1, w2, w3, s1m)
			sum = sum.Add(cdefWordConstrain(s0, x, secStr, secSh, secM).Add(cdefWordConstrain(s1, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s2, x, secStr, secSh, secM)).Add(cdefWordConstrain(s3, x, secStr, secSh, secM)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefWordLoad4W(w0, w1, w2, w3, s2p)
			s5 := cdefWordLoad4W(w0, w1, w2, w3, s2m)
			s6 := cdefWordLoad4W(w0, w1, w2, w3, s3p)
			s7 := cdefWordLoad4W(w0, w1, w2, w3, s3m)
			sum = sum.Add(cdefWordConstrain(s4, x, secStr, secSh, secM).Add(cdefWordConstrain(s5, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s6, x, secStr, secSh, secM)).Add(cdefWordConstrain(s7, x, secStr, secSh, secM)).Mul(secTap1))
			if clip {
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		y := x.Add(sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			b := y.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&cdefWordLowBytes))
			lo, hi := b.GetLo().AsUint32x4(), b.GetHi().AsUint32x4()
			cdefPut4(out8, do+r*ds, lo.GetElem(0))
			cdefPut4(out8, do+(r+1)*ds, lo.GetElem(1))
			cdefPut4(out8, do+(r+2)*ds, hi.GetElem(0))
			cdefPut4(out8, do+(r+3)*ds, hi.GetElem(1))
		} else {
			lo, hi := y.GetLo().AsUint64x2(), y.GetHi().AsUint64x2()
			cdefPutU16x4(out16, do+r*ds, lo.GetElem(0))
			cdefPutU16x4(out16, do+(r+1)*ds, lo.GetElem(1))
			cdefPutU16x4(out16, do+(r+2)*ds, hi.GetElem(0))
			cdefPutU16x4(out16, do+(r+3)*ds, hi.GetElem(1))
		}
	}
	archsimd.ClearAVXUpperBits()
}

// cdefWordShapeOK reports whether the 256-bit kernels cover a block shape:
// 8 wide with an even height, or 4 wide with a height that is a multiple of 4.
func cdefWordShapeOK(params BlockFilterParams) bool {
	switch params.Width {
	case 8:
		return params.Height&1 == 0
	case 4:
		return params.Height&3 == 0
	}
	return false
}

// filterBlockAVX2 is the uint16-output block filter on AVX2. Shapes the
// 256-bit kernels do not cover go to the 128-bit kernel.
func filterBlockAVX2(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if !cdefWordShapeOK(params) {
		filterBlockSIMD(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	var ctx cdefSIMDCtx
	ctx.setBlock(dst, nil, dstOrigin, dstStride, input, inputOrigin, params, false)
	if params.Width == 8 {
		cdefWordFilter8(&ctx)
	} else {
		cdefWordFilter4(&ctx)
	}
}

// filterBlockU8AVX2 is the 8-bit-dst block filter on AVX2, under the 8-bit
// contract of filter_u8.go.
func filterBlockU8AVX2(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if !cdefWordShapeOK(params) {
		filterBlockU8SIMD(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	var ctx cdefSIMDCtx
	ctx.setBlock(nil, dst, dstOrigin, dstStride, input, inputOrigin, params, true)
	if params.Width == 8 {
		cdefWordFilter8(&ctx)
	} else {
		cdefWordFilter4(&ctx)
	}
}
