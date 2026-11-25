// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/unsafeutil"
	"math/bits"
	"simd/archsimd"
)

// lfFilter4U8Wide follows the byte-domain sixteen-lane dataflow of the old
// NEON path. The block-limit gate excludes 255, so saturating the byte-domain
// threshold sum cannot turn a rejected lane into an accepted one.
func lfFilter4U8Wide(pix []byte, q0Base, step, length int, params filter4Params, changed *uint) {
	limit := archsimd.BroadcastUint8x16(uint8(params.limit))
	blimit := archsimd.BroadcastUint8x16(uint8(params.blimit))
	hevT := archsimd.BroadcastUint8x16(uint8(params.hev))
	bias := archsimd.BroadcastUint8x16(128)
	three16 := archsimd.BroadcastInt16x8(3)
	three8 := archsimd.BroadcastInt8x16(3)
	four8 := archsimd.BroadcastInt8x16(4)
	one8 := archsimd.BroadcastInt8x16(1)
	shift1 := archsimd.BroadcastInt8x16(-1)
	shift3 := archsimd.BroadcastInt8x16(-3)
	// The four tap rows are windows of pix taken once per call, all
	// len(p1Row) long; with the loop bound written x <= len-16 the
	// sixteen-lane groups index them with no bounds checks.
	p1Row := pix[q0Base-2*step:][:length:length]
	p0Row := pix[q0Base-step:][:len(p1Row)]
	q0Row := pix[q0Base:][:len(p1Row)]
	q1Row := pix[q0Base+step:][:len(p1Row)]
	for x := 0; x <= len(p1Row)-16; x += 16 {
		p1p := (*[16]uint8)(p1Row[x : x+16])
		p0p := (*[16]uint8)(p0Row[x : x+16])
		q0p := (*[16]uint8)(q0Row[x : x+16])
		q1p := (*[16]uint8)(q1Row[x : x+16])
		p1 := archsimd.LoadUint8x16Array(p1p)
		p0 := archsimd.LoadUint8x16Array(p0p)
		q0 := archsimd.LoadUint8x16Array(q0p)
		q1 := archsimd.LoadUint8x16Array(q1p)

		d1 := p1.Max(p0).Sub(p1.Min(p0))
		d2 := q1.Max(q0).Sub(q1.Min(q0))
		d0 := p0.Max(q0).Sub(p0.Min(q0))
		d3 := p1.Max(q1).Sub(p1.Min(q1))
		threshold := d0.AddSaturated(d0).AddSaturated(d3.Shift(shift1))
		need := limit.GreaterEqual(d1.Max(d2)).And(blimit.GreaterEqual(threshold))
		mask := need.ToInt8x16().ToBits().ReshapeToUint64s()
		lo, hi := mask.GetElem(0), mask.GetElem(1)
		if lo|hi == 0 {
			continue
		}
		if changed != nil {
			if lo != 0 {
				*changed |= 1 << uint(x/8)
			}
			if hi != 0 {
				*changed |= 1 << uint(x/8+1)
			}
		}
		hev := d1.Max(d2).Greater(hevT)

		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		outerF := p1c.SubSaturated(q1c).Masked(hev)
		deltaLo := q0.ExtendLo8ToUint16().BitsToInt16().Sub(p0.ExtendLo8ToUint16().BitsToInt16())
		deltaHi := q0.HiToLo().ExtendLo8ToUint16().BitsToInt16().Sub(p0.HiToLo().ExtendLo8ToUint16().BitsToInt16())
		fLo := deltaLo.Mul(three16).Add(outerF.ExtendLo8ToInt16()).SaturateToInt8()
		fHi := deltaHi.Mul(three16).Add(outerF.HiToLo().ExtendLo8ToInt16()).SaturateToInt8()
		fHiBits := fHi.ToBits()
		f := fLo.ToBits().Or(fHiBits.ConcatShiftBytesRight(fHiBits, 8)).BitsToInt8()
		filter1 := f.AddSaturated(four8).Shift(shift3)
		filter2 := f.AddSaturated(three8).Shift(shift3)
		outer := filter1.Add(one8).Shift(shift1)

		np0 := p0c.AddSaturated(filter2).ToBits().Xor(bias).IfElse(need, p0)
		nq0 := q0c.SubSaturated(filter1).ToBits().Xor(bias).IfElse(need, q0)
		np1 := p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(need.And(hev.Not()), p1)
		nq1 := q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(need.And(hev.Not()), q1)
		np1.StoreArray(p1p)
		np0.StoreArray(p0p)
		nq0.StoreArray(q0p)
		nq1.StoreArray(q1p)
	}
}

// init binds the Go-native SIMD deblocking kernels on arm64. NEON is mandatory
// on every arm64 chip Go runs on; the cpu check keeps the same gate as the
// pre-SIMD dispatch. Without it the slots keep the pure-Go reference. The
// assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
		filter6EdgeImpl = filter6EdgeSIMD
		filter8EdgeImpl = filter8EdgeSIMD
		filter14EdgeImpl = filter14EdgeSIMD
		filter6Edge16Impl = filter6Edge16SIMD
		filter8Edge16Impl = filter8Edge16SIMD
		filter14Edge16Impl = filter14Edge16SIMD
		return
	}
	filter4EdgeImpl = filter4EdgePureGo
	filter4Edge16Impl = filter4Edge16PureGo
	filter6EdgeImpl = filter6EdgePureGo
	filter8EdgeImpl = filter8EdgePureGo
	filter14EdgeImpl = filter14EdgePureGo
	filter6Edge16Impl = filter6Edge16PureGo
	filter8Edge16Impl = filter8Edge16PureGo
	filter14Edge16Impl = filter14Edge16PureGo
}

// lfStore8 narrows eight in-range lanes and writes exactly eight bytes. The
// narrow is a NEON SQXTUN and the low 64-bit lane becomes one store.
func lfStore8(pix []byte, off int, v archsimd.Int16x8) {
	binary.LittleEndian.PutUint64(pix[off:off+8], v.ToBits().SaturateToUint8().ReshapeToUint64s().GetElem(0))
}

// lfLoad8 reads exactly eight bytes and zero-extends them to signed lanes.
func lfLoad8(pix []byte, off int) archsimd.Int16x8 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(pix[off : off+8])).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
}

// lfAny reports whether any lane of the mask is set.
func lfAny(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() != 0
}

func lfAll(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() == 65528
}

// lfShift1 shifts by a reusable vector on arm64.
func lfShift1(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// lfShift3 shifts by a reusable vector on arm64.
func lfShift3(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// lfShift4 shifts by a reusable vector on arm64.
func lfShift4(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// The vertical-edge transposes between the plane and lfEdge's scratch. The
// plane side is one bounds-checked window per row: lfEdge sends planes past
// lfMaxPlane to the scalar reference, so offsets and strides fit 32 bits and
// int(uint32(x)) proves each row offset non-negative and overflow-free, one
// compare per row. The scratch side is a fixed array with a constant row
// stride: the tile is cast once to an array, so its rows index with constants.

// lfTile8 and lfTile16 are the scratch tile at so: eight sixteen-byte rows a
// constant stride apart, cast once so its rows index with constants.
type (
	lfTile8  = [7*lfScrRow8 + 16]byte
	lfTile16 = [7*lfScrRow16 + 16]byte
)

func lfScrTile8(s *lfScratch, so int) *lfTile8 {
	return (*lfTile8)(s[so : so+len(lfTile8{})])
}

func lfScrTile16(s *lfScratch, so int) *lfTile16 {
	return (*lfTile16)(s[so : so+len(lfTile16{})])
}

// lfGet64 and lfPut64 read and write the eight bytes of s at o.
func lfGet64(s []byte, o int) uint64 {
	return binary.LittleEndian.Uint64(s[o : o+8])
}

func lfPut64(s []byte, o int, v uint64) {
	binary.LittleEndian.PutUint64(s[o:o+8], v)
}

// lfRow16 is the sixteen-byte window of s at o.
func lfRow16(s []byte, o int) *[16]byte {
	return (*[16]byte)(s[o : o+16])
}

// lfTranspose8x8U8 transposes the 8x8 byte matrix in the low halves of a..h:
// every interleave keeps lanes inside their 64-bit half, so the high halves
// (whatever the loads put there) never reach the low halves of the results.
func lfTranspose8x8U8(a, b, c, d, e, f, g, h archsimd.Uint8x16) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Uint64x2) {
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint16s().InterleaveEven(cd0.ReshapeToUint16s()), ab0.ReshapeToUint16s().InterleaveOdd(cd0.ReshapeToUint16s())
	ac1, ac3 := ab1.ReshapeToUint16s().InterleaveEven(cd1.ReshapeToUint16s()), ab1.ReshapeToUint16s().InterleaveOdd(cd1.ReshapeToUint16s())
	eg0, eg2 := ef0.ReshapeToUint16s().InterleaveEven(gh0.ReshapeToUint16s()), ef0.ReshapeToUint16s().InterleaveOdd(gh0.ReshapeToUint16s())
	eg1, eg3 := ef1.ReshapeToUint16s().InterleaveEven(gh1.ReshapeToUint16s()), ef1.ReshapeToUint16s().InterleaveOdd(gh1.ReshapeToUint16s())
	o0 = ac0.ReshapeToUint32s().InterleaveEven(eg0.ReshapeToUint32s()).ReshapeToUint64s()
	o1 = ac1.ReshapeToUint32s().InterleaveEven(eg1.ReshapeToUint32s()).ReshapeToUint64s()
	o2 = ac2.ReshapeToUint32s().InterleaveEven(eg2.ReshapeToUint32s()).ReshapeToUint64s()
	o3 = ac3.ReshapeToUint32s().InterleaveEven(eg3.ReshapeToUint32s()).ReshapeToUint64s()
	o4 = ac0.ReshapeToUint32s().InterleaveOdd(eg0.ReshapeToUint32s()).ReshapeToUint64s()
	o5 = ac1.ReshapeToUint32s().InterleaveOdd(eg1.ReshapeToUint32s()).ReshapeToUint64s()
	o6 = ac2.ReshapeToUint32s().InterleaveOdd(eg2.ReshapeToUint32s()).ReshapeToUint64s()
	o7 = ac3.ReshapeToUint32s().InterleaveOdd(eg3.ReshapeToUint32s()).ReshapeToUint64s()
	return
}

// lfTranspose8x8U16 transposes the 8x8 matrix of uint16 samples a..h.
func lfTranspose8x8U16(a, b, c, d, e, f, g, h archsimd.Uint16x8) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Uint8x16) {
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint32s().InterleaveEven(cd0.ReshapeToUint32s()), ab0.ReshapeToUint32s().InterleaveOdd(cd0.ReshapeToUint32s())
	ac1, ac3 := ab1.ReshapeToUint32s().InterleaveEven(cd1.ReshapeToUint32s()), ab1.ReshapeToUint32s().InterleaveOdd(cd1.ReshapeToUint32s())
	eg0, eg2 := ef0.ReshapeToUint32s().InterleaveEven(gh0.ReshapeToUint32s()), ef0.ReshapeToUint32s().InterleaveOdd(gh0.ReshapeToUint32s())
	eg1, eg3 := ef1.ReshapeToUint32s().InterleaveEven(gh1.ReshapeToUint32s()), ef1.ReshapeToUint32s().InterleaveOdd(gh1.ReshapeToUint32s())
	o0 = ac0.ReshapeToUint64s().InterleaveEven(eg0.ReshapeToUint64s()).ReshapeToUint8s()
	o1 = ac1.ReshapeToUint64s().InterleaveEven(eg1.ReshapeToUint64s()).ReshapeToUint8s()
	o2 = ac2.ReshapeToUint64s().InterleaveEven(eg2.ReshapeToUint64s()).ReshapeToUint8s()
	o3 = ac3.ReshapeToUint64s().InterleaveEven(eg3.ReshapeToUint64s()).ReshapeToUint8s()
	o4 = ac0.ReshapeToUint64s().InterleaveOdd(eg0.ReshapeToUint64s()).ReshapeToUint8s()
	o5 = ac1.ReshapeToUint64s().InterleaveOdd(eg1.ReshapeToUint64s()).ReshapeToUint8s()
	o6 = ac2.ReshapeToUint64s().InterleaveOdd(eg2.ReshapeToUint64s()).ReshapeToUint8s()
	o7 = ac3.ReshapeToUint64s().InterleaveOdd(eg3.ReshapeToUint64s()).ReshapeToUint8s()
	return
}

// lfLoadShortU8 reads four, six or eight bytes of s at o into the low lanes.
func lfLoadShortU8(s []byte, o int, width int) archsimd.Uint8x16 {
	if width == 8 {
		return archsimd.BroadcastUint64x2(lfGet64(s, o)).ReshapeToUint8s()
	}
	v := uint64(binary.LittleEndian.Uint32(s[o : o+4]))
	if width == 6 {
		v |= uint64(binary.LittleEndian.Uint16(s[o+4:o+6])) << 32
	}
	return archsimd.BroadcastUint64x2(v).ReshapeToUint8s()
}

// lfStoreShortU8 writes the low four, six or eight bytes of v to s at o.
func lfStoreShortU8(s []byte, o int, width int, v uint64) {
	if width == 8 {
		lfPut64(s, o, v)
		return
	}
	binary.LittleEndian.PutUint32(s[o:o+4], uint32(v))
	if width == 6 {
		binary.LittleEndian.PutUint16(s[o+4:o+6], uint16(v>>32))
	}
}

// lfGatherU8 transposes the 8x8 byte tile at po (rows ps apart) of pix into
// the scratch tile at so. Plane rows are read sixteen bytes at a time when the
// last one has them inside pix, eight otherwise; readBytes < 8 (an exact-width
// plane) reads only that many bytes of each row.
func lfGatherU8(pix []byte, po, ps int, scr *lfScratch, so int, readBytes int) {
	t := lfScrTile8(scr, so)
	b, st := int(uint32(po)), int(uint32(ps))
	var r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Uint8x16
	switch {
	case readBytes < 8:
		r0 = lfLoadShortU8(pix, b, readBytes)
		r1 = lfLoadShortU8(pix, b+st, readBytes)
		r2 = lfLoadShortU8(pix, b+2*st, readBytes)
		r3 = lfLoadShortU8(pix, b+3*st, readBytes)
		r4 = lfLoadShortU8(pix, b+4*st, readBytes)
		r5 = lfLoadShortU8(pix, b+5*st, readBytes)
		r6 = lfLoadShortU8(pix, b+6*st, readBytes)
		r7 = lfLoadShortU8(pix, b+7*st, readBytes)
	case b+7*st+16 <= len(pix):
		r0 = archsimd.LoadUint8x16Array(lfRow16(pix, b))
		r1 = archsimd.LoadUint8x16Array(lfRow16(pix, b+st))
		r2 = archsimd.LoadUint8x16Array(lfRow16(pix, b+2*st))
		r3 = archsimd.LoadUint8x16Array(lfRow16(pix, b+3*st))
		r4 = archsimd.LoadUint8x16Array(lfRow16(pix, b+4*st))
		r5 = archsimd.LoadUint8x16Array(lfRow16(pix, b+5*st))
		r6 = archsimd.LoadUint8x16Array(lfRow16(pix, b+6*st))
		r7 = archsimd.LoadUint8x16Array(lfRow16(pix, b+7*st))
	default:
		r0 = archsimd.BroadcastUint64x2(lfGet64(pix, b)).ReshapeToUint8s()
		r1 = archsimd.BroadcastUint64x2(lfGet64(pix, b+st)).ReshapeToUint8s()
		r2 = archsimd.BroadcastUint64x2(lfGet64(pix, b+2*st)).ReshapeToUint8s()
		r3 = archsimd.BroadcastUint64x2(lfGet64(pix, b+3*st)).ReshapeToUint8s()
		r4 = archsimd.BroadcastUint64x2(lfGet64(pix, b+4*st)).ReshapeToUint8s()
		r5 = archsimd.BroadcastUint64x2(lfGet64(pix, b+5*st)).ReshapeToUint8s()
		r6 = archsimd.BroadcastUint64x2(lfGet64(pix, b+6*st)).ReshapeToUint8s()
		r7 = archsimd.BroadcastUint64x2(lfGet64(pix, b+7*st)).ReshapeToUint8s()
	}
	o0, o1, o2, o3, o4, o5, o6, o7 := lfTranspose8x8U8(r0, r1, r2, r3, r4, r5, r6, r7)
	binary.LittleEndian.PutUint64((*[16]byte)(t[0:16])[:8], o0.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[32:48])[:8], o1.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[64:80])[:8], o2.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[96:112])[:8], o3.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[128:144])[:8], o4.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[160:176])[:8], o5.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[192:208])[:8], o6.GetElem(0))
	binary.LittleEndian.PutUint64((*[16]byte)(t[224:240])[:8], o7.GetElem(0))
}

// lfScatterU8 is lfGatherU8 back: the scratch tile at so, transposed, to the
// plane tile at po, writeBytes (4, 6 or 8) bytes per plane row.
func lfScatterU8(scr *lfScratch, so int, pix []byte, po, ps int, writeBytes int) {
	t := lfScrTile8(scr, so)
	o0, o1, o2, o3, o4, o5, o6, o7 := lfTranspose8x8U8(
		archsimd.LoadUint8x16Array((*[16]byte)(t[0:16])), archsimd.LoadUint8x16Array((*[16]byte)(t[32:48])),
		archsimd.LoadUint8x16Array((*[16]byte)(t[64:80])), archsimd.LoadUint8x16Array((*[16]byte)(t[96:112])),
		archsimd.LoadUint8x16Array((*[16]byte)(t[128:144])), archsimd.LoadUint8x16Array((*[16]byte)(t[160:176])),
		archsimd.LoadUint8x16Array((*[16]byte)(t[192:208])), archsimd.LoadUint8x16Array((*[16]byte)(t[224:240])))
	b, st := int(uint32(po)), int(uint32(ps))
	if writeBytes == 8 {
		lfPut64(pix, b, o0.GetElem(0))
		lfPut64(pix, b+st, o1.GetElem(0))
		lfPut64(pix, b+2*st, o2.GetElem(0))
		lfPut64(pix, b+3*st, o3.GetElem(0))
		lfPut64(pix, b+4*st, o4.GetElem(0))
		lfPut64(pix, b+5*st, o5.GetElem(0))
		lfPut64(pix, b+6*st, o6.GetElem(0))
		lfPut64(pix, b+7*st, o7.GetElem(0))
		return
	}
	lfStoreShortU8(pix, b, writeBytes, o0.GetElem(0))
	lfStoreShortU8(pix, b+st, writeBytes, o1.GetElem(0))
	lfStoreShortU8(pix, b+2*st, writeBytes, o2.GetElem(0))
	lfStoreShortU8(pix, b+3*st, writeBytes, o3.GetElem(0))
	lfStoreShortU8(pix, b+4*st, writeBytes, o4.GetElem(0))
	lfStoreShortU8(pix, b+5*st, writeBytes, o5.GetElem(0))
	lfStoreShortU8(pix, b+6*st, writeBytes, o6.GetElem(0))
	lfStoreShortU8(pix, b+7*st, writeBytes, o7.GetElem(0))
}

func lfLoadShortU16(s []byte, o int, width int) archsimd.Uint16x8 {
	if width == 8 {
		return archsimd.LoadUint8x16Array(lfRow16(s, o)).ReshapeToUint16s()
	}
	var tmp [16]byte
	binary.LittleEndian.PutUint64(tmp[:], lfGet64(s, o))
	if width == 6 {
		binary.LittleEndian.PutUint32(tmp[8:], binary.LittleEndian.Uint32(s[o+8:o+12]))
	}
	return archsimd.LoadUint8x16Array(&tmp).ReshapeToUint16s()
}

func lfStoreShortU16(s []byte, o int, width int, v archsimd.Uint8x16) {
	if width == 8 {
		v.StoreArray(lfRow16(s, o))
		return
	}
	var tmp [16]byte
	v.StoreArray(&tmp)
	lfPut64(s, o, binary.LittleEndian.Uint64(tmp[:]))
	if width == 6 {
		binary.LittleEndian.PutUint32(s[o+8:o+12], binary.LittleEndian.Uint32(tmp[8:]))
	}
}

// lfGatherU16 is lfGatherU8 for two-byte samples: readWidth (4, 6 or 8)
// samples of each plane row.
func lfGatherU16(pix []byte, po, ps int, scr *lfScratch, so int, readWidth int) {
	t := lfScrTile16(scr, so)
	b, st := int(uint32(po)), int(uint32(ps))
	var r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Uint16x8
	if readWidth == 8 {
		r0 = archsimd.LoadUint8x16Array(lfRow16(pix, b)).ReshapeToUint16s()
		r1 = archsimd.LoadUint8x16Array(lfRow16(pix, b+st)).ReshapeToUint16s()
		r2 = archsimd.LoadUint8x16Array(lfRow16(pix, b+2*st)).ReshapeToUint16s()
		r3 = archsimd.LoadUint8x16Array(lfRow16(pix, b+3*st)).ReshapeToUint16s()
		r4 = archsimd.LoadUint8x16Array(lfRow16(pix, b+4*st)).ReshapeToUint16s()
		r5 = archsimd.LoadUint8x16Array(lfRow16(pix, b+5*st)).ReshapeToUint16s()
		r6 = archsimd.LoadUint8x16Array(lfRow16(pix, b+6*st)).ReshapeToUint16s()
		r7 = archsimd.LoadUint8x16Array(lfRow16(pix, b+7*st)).ReshapeToUint16s()
	} else {
		r0 = lfLoadShortU16(pix, b, readWidth)
		r1 = lfLoadShortU16(pix, b+st, readWidth)
		r2 = lfLoadShortU16(pix, b+2*st, readWidth)
		r3 = lfLoadShortU16(pix, b+3*st, readWidth)
		r4 = lfLoadShortU16(pix, b+4*st, readWidth)
		r5 = lfLoadShortU16(pix, b+5*st, readWidth)
		r6 = lfLoadShortU16(pix, b+6*st, readWidth)
		r7 = lfLoadShortU16(pix, b+7*st, readWidth)
	}
	o0, o1, o2, o3, o4, o5, o6, o7 := lfTranspose8x8U16(r0, r1, r2, r3, r4, r5, r6, r7)
	o0.StoreArray((*[16]byte)(t[0:16]))
	o1.StoreArray((*[16]byte)(t[64:80]))
	o2.StoreArray((*[16]byte)(t[128:144]))
	o3.StoreArray((*[16]byte)(t[192:208]))
	o4.StoreArray((*[16]byte)(t[256:272]))
	o5.StoreArray((*[16]byte)(t[320:336]))
	o6.StoreArray((*[16]byte)(t[384:400]))
	o7.StoreArray((*[16]byte)(t[448:464]))
}

// lfScatterU16 is lfGatherU16 back, writeWidth (4, 6 or 8) samples per row.
func lfScatterU16(scr *lfScratch, so int, pix []byte, po, ps int, writeWidth int) {
	t := lfScrTile16(scr, so)
	o0, o1, o2, o3, o4, o5, o6, o7 := lfTranspose8x8U16(
		archsimd.LoadUint8x16Array((*[16]byte)(t[0:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(t[64:80])).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array((*[16]byte)(t[128:144])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(t[192:208])).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array((*[16]byte)(t[256:272])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(t[320:336])).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array((*[16]byte)(t[384:400])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(t[448:464])).ReshapeToUint16s())
	b, st := int(uint32(po)), int(uint32(ps))
	if writeWidth == 8 {
		o0.StoreArray(lfRow16(pix, b))
		o1.StoreArray(lfRow16(pix, b+st))
		o2.StoreArray(lfRow16(pix, b+2*st))
		o3.StoreArray(lfRow16(pix, b+3*st))
		o4.StoreArray(lfRow16(pix, b+4*st))
		o5.StoreArray(lfRow16(pix, b+5*st))
		o6.StoreArray(lfRow16(pix, b+6*st))
		o7.StoreArray(lfRow16(pix, b+7*st))
		return
	}
	lfStoreShortU16(pix, b, writeWidth, o0)
	lfStoreShortU16(pix, b+st, writeWidth, o1)
	lfStoreShortU16(pix, b+2*st, writeWidth, o2)
	lfStoreShortU16(pix, b+3*st, writeWidth, o3)
	lfStoreShortU16(pix, b+4*st, writeWidth, o4)
	lfStoreShortU16(pix, b+5*st, writeWidth, o5)
	lfStoreShortU16(pix, b+6*st, writeWidth, o6)
	lfStoreShortU16(pix, b+7*st, writeWidth, o7)
}

// The 10/12-bit vertical-edge kernels, dav1d's lpf_h_{4,6,8,16}_8_neon
// (src/arm/64/loopfilter16.S): eight rows are loaded, transposed into one
// vector per tap in registers, filtered by the lpf_8_wd* body written out in
// the same function (a call here would spill every live vector: Go has no
// callee-saved vector registers), and only the columns the filter can have
// changed are transposed back and stored, through dav1d's exits: nothing
// when no lane passed the filter mask, p1..q1 (a 4x8 transpose and one
// 8-byte store per row) when only the wd4 filter ran, p3..q3 (8x8, sixteen
// bytes) when the flat8in filter ran but flat8out did not.
//
// The rows are unsafeutil.Blocks of the drivers' checked views, named by
// constant row numbers at each load and store.

// lpf16Consts holds the kernels' vector constants, loaded from memory (one
// LDR each) where an archsimd broadcast or constant shift costs a MOVD, an
// INS and a DUP every use: the bitdepth-scaled iclip_diff/iclip_pixel bounds
// and flatness threshold, the wd4 multipliers, the shift counts (USHL/SSHL by
// a negative count is a right shift) and the flat filters' rounding biases.
type lpf16Consts struct {
	dmin, dmax, maxPix [8]int16
	flat               [8]uint16
	three, four, one   [8]int16
	shr1, shr3, shr4   [8]int16
	rnd4, rnd8         [8]uint16
}

// lpf16Tab is indexed by bitdepth_min_8 / 2 (8, 10 and 12 bits).
var lpf16Tab = func() (t [3]lpf16Consts) {
	for i := range t {
		bd8 := 2 * i
		for j := 0; j < 8; j++ {
			c := &t[i]
			c.dmin[j] = int16(-128 << bd8)
			c.dmax[j] = int16(128<<bd8 - 1)
			c.maxPix[j] = int16(256<<bd8 - 1)
			c.flat[j] = uint16(1) << bd8
			c.three[j], c.four[j], c.one[j] = 3, 4, 1
			c.shr1[j], c.shr3[j], c.shr4[j] = -1, -3, -4
			c.rnd4[j], c.rnd8[j] = 4, 8
		}
	}
	return
}()

// lpf16Abd is |a - b| on 10/12-bit samples (dav1d uabd): the difference of
// two samples below 2^15 is exact in signed 16-bit lanes, so a sub and an abs
// replace the two saturating subtracts and the orr of the unsigned form.
func lpf16Abd(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	return a.BitsToInt16().Sub(b.BitsToInt16()).Abs().ToBits()
}

// lpf16Transpose4x8 is dav1d transpose_4x8h: the four tap vectors a..d of
// eight rows become four vectors whose 64-bit halves are rows (0, 4), (1, 5),
// (2, 6) and (3, 7), four samples a..d each.
func lpf16Transpose4x8(a, b, c, d archsimd.Uint16x8) (r04, r15, r26, r37 archsimd.Uint64x2) {
	ab0, ab1 := a.InterleaveEven(b).ReshapeToUint32s(), a.InterleaveOdd(b).ReshapeToUint32s()
	cd0, cd1 := c.InterleaveEven(d).ReshapeToUint32s(), c.InterleaveOdd(d).ReshapeToUint32s()
	r04 = ab0.InterleaveEven(cd0).ReshapeToUint64s()
	r26 = ab0.InterleaveOdd(cd0).ReshapeToUint64s()
	r15 = ab1.InterleaveEven(cd1).ReshapeToUint64s()
	r37 = ab1.InterleaveOdd(cd1).ReshapeToUint64s()
	return
}

// lpfCols16W is dav1d lpf_h_{4,6,8}_8_neon for one two-unit group: the eight
// rows of tv, sixteen bytes p3..q3 each.
func lpfCols16W[W lpfWidth](tv unsafeutil.Block[lpfR8], k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4 archsimd.Mask16x8) {
	var wdT W
	wd := len(wdT)
	p3, p2, p1, p0, q0, q1, q2, q3 := lpfTranspose8x8U16Regs(archsimd.LoadUint8x16Array(tv.Row16(0)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tv.Row16(1)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tv.Row16(2)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tv.Row16(3)).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array(tv.Row16(4)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tv.Row16(5)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tv.Row16(6)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tv.Row16(7)).ReshapeToUint16s())

	maxd := lpf16Abd(p1, p0).Max(lpf16Abd(q1, q0))
	t := maxd
	if wd >= 6 {
		tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1))
		if wd >= 8 {
			tt = tt.Max(lpf16Abd(p3, p2).Max(lpf16Abd(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpf16Abd(p0, q0)
	sum := d00.Add(d00).Add(lpf16Abd(p1, q1).Shift(archsimd.LoadInt16x8Array(&k.shr1)))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	mF4 := fm
	var mFlat archsimd.Mask16x8
	anyFlat := false
	if wd >= 6 {
		fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
		}
		mFlat = archsimd.LoadUint16x8Array(&k.flat).GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		anyFlat = lpf16Any(mFlat)
	}

	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	if wd == 4 || lpf16Any(mF4) {
		hev := maxd.Greater(hV)
		dmin := archsimd.LoadInt16x8Array(&k.dmin)
		dmax := archsimd.LoadInt16x8Array(&k.dmax)
		zero := archsimd.Int16x8{}
		maxPix := archsimd.LoadInt16x8Array(&k.maxPix)
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(dmin).Min(dmax).Masked(hev)
		f = q0s.Sub(p0s).Mul(archsimd.LoadInt16x8Array(&k.three)).Add(f).Max(dmin).Min(dmax)
		f1 := f.Add(archsimd.LoadInt16x8Array(&k.four)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		f2 := f.Add(archsimd.LoadInt16x8Array(&k.three)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		o0 = p0s.Add(f2).Max(zero).Min(maxPix).ToBits().IfElse(mF4, p0)
		n0 = q0s.Sub(f1).Max(zero).Min(maxPix).ToBits().IfElse(mF4, q0)
		f3 := f1.Add(archsimd.LoadInt16x8Array(&k.one)).Shift(archsimd.LoadInt16x8Array(&k.shr1))
		mOuter := mF4.And(hev.Not())
		o1 = p1s.Add(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, p1)
		n1 = q1s.Sub(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, q1)
	}
	if wd == 6 && anyFlat {
		s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd4))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
		s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
		s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
		s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
	}
	if wd == 8 && anyFlat {
		s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.LoadUint16x8Array(&k.rnd4)))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o2)
		s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
		s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
		s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
		s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
		s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n2)

		r0, r1, r2, r3, r4, r5, r6, r7 := lpfTranspose8x8U16Regs(p3, o2, o1, o0, n0, n1, n2, q3)
		r0.ReshapeToUint8s().StoreArray(tv.Row16(0))
		r1.ReshapeToUint8s().StoreArray(tv.Row16(1))
		r2.ReshapeToUint8s().StoreArray(tv.Row16(2))
		r3.ReshapeToUint8s().StoreArray(tv.Row16(3))
		r4.ReshapeToUint8s().StoreArray(tv.Row16(4))
		r5.ReshapeToUint8s().StoreArray(tv.Row16(5))
		r6.ReshapeToUint8s().StoreArray(tv.Row16(6))
		r7.ReshapeToUint8s().StoreArray(tv.Row16(7))
		return
	}
	r04, r15, r26, r37 := lpf16Transpose4x8(o1, o0, n0, n1)
	tv.PutF64(0, 4, r04.BitsToFloat64().GetElem(0))
	tv.PutF64(1, 4, r15.BitsToFloat64().GetElem(0))
	tv.PutF64(2, 4, r26.BitsToFloat64().GetElem(0))
	tv.PutF64(3, 4, r37.BitsToFloat64().GetElem(0))
	tv.PutF64(4, 4, r04.BitsToFloat64().GetElem(1))
	tv.PutF64(5, 4, r15.BitsToFloat64().GetElem(1))
	tv.PutF64(6, 4, r26.BitsToFloat64().GetElem(1))
	tv.PutF64(7, 4, r37.BitsToFloat64().GetElem(1))
}

// lpfCols16Wd16 is dav1d lpf_h_16_8_neon for one two-unit group: eight rows
// of p7..q7, sixteen bytes p7..p0 in tl, q0..q7 in tr, and p3..q3 in tm
// (all three Blocks of the same eight rows).
func lpfCols16Wd16(tl, tm, tr unsafeutil.Block[lpfR8], k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	p7, p6, p5, p4, p3, p2, p1, p0 := lpfTranspose8x8U16Regs(
		archsimd.LoadUint8x16Array(tl.Row16(0)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tl.Row16(1)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tl.Row16(2)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tl.Row16(3)).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array(tl.Row16(4)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tl.Row16(5)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tl.Row16(6)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tl.Row16(7)).ReshapeToUint16s())
	q0, q1, q2, q3, q4, q5, q6, q7 := lpfTranspose8x8U16Regs(
		archsimd.LoadUint8x16Array(tr.Row16(0)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tr.Row16(1)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tr.Row16(2)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tr.Row16(3)).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array(tr.Row16(4)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tr.Row16(5)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tr.Row16(6)).ReshapeToUint16s(), archsimd.LoadUint8x16Array(tr.Row16(7)).ReshapeToUint16s())

	maxd := lpf16Abd(p1, p0).Max(lpf16Abd(q1, q0))
	tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1)).Max(lpf16Abd(p3, p2).Max(lpf16Abd(q3, q2)))
	t := maxd.Max(tt.Masked(mGt4))
	d00 := lpf16Abd(p0, q0)
	sum := d00.Add(d00).Add(lpf16Abd(p1, q1).Shift(archsimd.LoadInt16x8Array(&k.shr1)))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	fV := archsimd.LoadUint16x8Array(&k.flat)
	fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0)).Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
	mFlat := fV.GreaterEqual(fl).And(mGt4).And(fm)
	mF4 := fm.And(mFlat.Not())
	fo := lpf16Abd(p6, p0).Max(lpf16Abd(p5, p0)).Max(lpf16Abd(p4, p0)).
		Max(lpf16Abd(q4, q0)).Max(lpf16Abd(q5, q0)).Max(lpf16Abd(q6, q0))
	mFlat16 := fV.GreaterEqual(fo).And(m16).And(mFlat)
	mFlat = mFlat.And(mFlat16.Not())
	anyFlat, anyFlat16 := lpf16Any(mFlat), lpf16Any(mFlat16)

	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	if lpf16Any(mF4) {
		hev := maxd.Greater(hV)
		dmin := archsimd.LoadInt16x8Array(&k.dmin)
		dmax := archsimd.LoadInt16x8Array(&k.dmax)
		zero := archsimd.Int16x8{}
		maxPix := archsimd.LoadInt16x8Array(&k.maxPix)
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(dmin).Min(dmax).Masked(hev)
		f = q0s.Sub(p0s).Mul(archsimd.LoadInt16x8Array(&k.three)).Add(f).Max(dmin).Min(dmax)
		f1 := f.Add(archsimd.LoadInt16x8Array(&k.four)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		f2 := f.Add(archsimd.LoadInt16x8Array(&k.three)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		o0 = p0s.Add(f2).Max(zero).Min(maxPix).ToBits().IfElse(mF4, p0)
		n0 = q0s.Sub(f1).Max(zero).Min(maxPix).ToBits().IfElse(mF4, q0)
		f3 := f1.Add(archsimd.LoadInt16x8Array(&k.one)).Shift(archsimd.LoadInt16x8Array(&k.shr1))
		mOuter := mF4.And(hev.Not())
		o1 = p1s.Add(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, p1)
		n1 = q1s.Sub(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, q1)
	}
	if anyFlat {
		s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.LoadUint16x8Array(&k.rnd4)))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o2)
		s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
		s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
		s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
		s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
		s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n2)
	}
	if anyFlat16 {
		m := mFlat16
		p6x2 := p6.Add(p6)
		s := p6.Add(p6x2).Add(p6x2.Add(p6x2))
		s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd8))
		o5 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p5)
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		o4 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p4)
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		o3 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p3)
		s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o2)
		s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o1)
		s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o0)
		s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n0)
		s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n1)
		s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n2)
		s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
		n3 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q3)
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		n4 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q4)
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		n5 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q5)

		a0, a1, a2, a3, a4, a5, a6, a7 := lpfTranspose8x8U16Regs(p7, p6, o5, o4, o3, o2, o1, o0)
		a0.ReshapeToUint8s().StoreArray(tl.Row16(0))
		a1.ReshapeToUint8s().StoreArray(tl.Row16(1))
		a2.ReshapeToUint8s().StoreArray(tl.Row16(2))
		a3.ReshapeToUint8s().StoreArray(tl.Row16(3))
		a4.ReshapeToUint8s().StoreArray(tl.Row16(4))
		a5.ReshapeToUint8s().StoreArray(tl.Row16(5))
		a6.ReshapeToUint8s().StoreArray(tl.Row16(6))
		a7.ReshapeToUint8s().StoreArray(tl.Row16(7))
		c0, c1, c2, c3, c4, c5, c6, c7 := lpfTranspose8x8U16Regs(n0, n1, n2, n3, n4, n5, q6, q7)
		c0.ReshapeToUint8s().StoreArray(tr.Row16(0))
		c1.ReshapeToUint8s().StoreArray(tr.Row16(1))
		c2.ReshapeToUint8s().StoreArray(tr.Row16(2))
		c3.ReshapeToUint8s().StoreArray(tr.Row16(3))
		c4.ReshapeToUint8s().StoreArray(tr.Row16(4))
		c5.ReshapeToUint8s().StoreArray(tr.Row16(5))
		c6.ReshapeToUint8s().StoreArray(tr.Row16(6))
		c7.ReshapeToUint8s().StoreArray(tr.Row16(7))
		return
	}
	if anyFlat {
		r0, r1, r2, r3, r4, r5, r6, r7 := lpfTranspose8x8U16Regs(p3, o2, o1, o0, n0, n1, n2, q3)
		r0.ReshapeToUint8s().StoreArray(tm.Row16(0))
		r1.ReshapeToUint8s().StoreArray(tm.Row16(1))
		r2.ReshapeToUint8s().StoreArray(tm.Row16(2))
		r3.ReshapeToUint8s().StoreArray(tm.Row16(3))
		r4.ReshapeToUint8s().StoreArray(tm.Row16(4))
		r5.ReshapeToUint8s().StoreArray(tm.Row16(5))
		r6.ReshapeToUint8s().StoreArray(tm.Row16(6))
		r7.ReshapeToUint8s().StoreArray(tm.Row16(7))
		return
	}
	r04, r15, r26, r37 := lpf16Transpose4x8(o1, o0, n0, n1)
	tm.PutF64(0, 4, r04.BitsToFloat64().GetElem(0))
	tm.PutF64(1, 4, r15.BitsToFloat64().GetElem(0))
	tm.PutF64(2, 4, r26.BitsToFloat64().GetElem(0))
	tm.PutF64(3, 4, r37.BitsToFloat64().GetElem(0))
	tm.PutF64(4, 4, r04.BitsToFloat64().GetElem(1))
	tm.PutF64(5, 4, r15.BitsToFloat64().GetElem(1))
	tm.PutF64(6, 4, r26.BitsToFloat64().GetElem(1))
	tm.PutF64(7, 4, r37.BitsToFloat64().GetElem(1))
}

// lpfCols16 runs the width-specialized vertical-edge kernel on the eight
// rows of v from row r whose q0 is column c (p7..q7, c-16..c+15, at wd 16;
// p3..q3, c-8..c+7, otherwise), all inside v.
func lpfCols16(v unsafeutil.Rows, r, c, wd int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	switch wd {
	case 16:
		lpfCols16Wd16(unsafeutil.BlockAt[lpfR8](v, r, c-16), unsafeutil.BlockAt[lpfR8](v, r, c-8), unsafeutil.BlockAt[lpfR8](v, r, c), k, eV, iV, hV, m4, mGt4, m16)
	case 8:
		lpfCols16W[lpfW8](unsafeutil.BlockAt[lpfR8](v, r, c-8), k, eV, iV, hV, m4, mGt4)
	case 6:
		lpfCols16W[lpfW6](unsafeutil.BlockAt[lpfR8](v, r, c-8), k, eV, iV, hV, m4, mGt4)
	default:
		lpfCols16W[lpfW4](unsafeutil.BlockAt[lpfR8](v, r, c-8), k, eV, iV, hV, m4, mGt4)
	}
}

// The 8-bit kernel arithmetic, shared by the horizontal (row) and transposed
// vertical (column) kernels. Each helper is an exact rewrite of a dav1d
// lpf_16_wd* step (src/arm/64/loopfilter.S) into the operations archsimd has
// on arm64; the comments name the dav1d instructions each replaces.

// lpfSpread returns max over x in {a, b} of |x - c|, dav1d's two uabd and a
// umax for the flat8in/flat8out tests. archsimd has no UABD (|x - c| costs
// two uqsub and an orr here), but against one centre the maximum distance is
// max(max(a, b) - c, c - min(a, b)) in saturating arithmetic: five
// operations where two emulated uabd and a umax take seven, and each extra
// tap adds a max and a min instead of four.
func lpfSpread(a, b, c archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).SubSaturated(c).Max(c.SubSaturated(a.Min(b)))
}

// lpfSpread3 is lpfSpread over three taps.
func lpfSpread3(a, b, d, c archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).Max(d).SubSaturated(c).Max(c.SubSaturated(a.Min(b).Min(d)))
}

// The flat filters run on widened lanes pre-scaled by 2^(8-shift) (32 for
// the >>3 filters, 16 for the >>4 one): the widening multiply that scales
// costs what a plain widen does (umull/umull2 against uxtl/uxtl2), the
// rounding bias rides in the accumulator, and the rounded, narrowed output
// byte is then the high byte of each 16-bit lane, one uzp2 per sixteen
// outputs where dav1d spends rshrn + rshrn2. Every intermediate is exact mod
// 2^16 and every output sum is below 2^16 (8*255+4 scaled by 32, 16*255+8 by
// 16), so the high byte is exactly (sum + bias) >> shift. The kernels write
// the scaled widen (x.MulWidenLo(k), x.HiToLo().MulWidenLo(k.HiToLo())) and
// the high-byte narrow (lo.ReshapeToUint8s().ConcatOdd(hi...)) out at each
// use: as inlined helpers, each call left an inline-mark NOP.

// lpfFlat6Half is the dav1d wd6 sliding sum (out p1, p0, q0, q1) on scaled
// widened lanes; s is the scaled, biased 3*p2 + 2*p1 + 2*p0 + q0 + 4.
func lpfFlat6Half(P2, P1, P0, Q0, Q1, Q2, s archsimd.Uint16x8) (a1, a0, b0, b1 archsimd.Uint16x8) {
	a1 = s
	s = s.Add(Q0.Add(Q1)).Sub(P2.Add(P2))
	a0 = s
	s = s.Add(Q1.Add(Q2)).Sub(P2.Add(P1))
	b0 = s
	b1 = s.Add(Q2.Add(Q2)).Sub(P1.Add(P0))
	return
}

// lpfFlat8Half is the dav1d wd8 sliding sum (out p2..q2) on scaled widened
// lanes; s is the scaled, biased 3*p3 + 2*p2 + p1 + p0 + q0 + 4.
func lpfFlat8Half(P3, P2, P1, P0, Q0, Q1, Q2, Q3, s archsimd.Uint16x8) (a2, a1, a0, b0, b1, b2 archsimd.Uint16x8) {
	a2 = s
	s = s.Add(P1.Add(Q1)).Sub(P3.Add(P2))
	a1 = s
	s = s.Add(P0.Add(Q2)).Sub(P3.Add(P1))
	a0 = s
	s = s.Add(Q0.Add(Q3)).Sub(P3.Add(P0))
	b0 = s
	s = s.Add(Q1.Add(Q3)).Sub(P2.Add(Q0))
	b1 = s
	b2 = s.Add(Q2.Add(Q3)).Sub(P1.Add(Q1))
	return
}

// The horizontal-edge drivers check each call's (or edge group's) sample
// and level footprints once, as unsafeutil.Rows views, and hand each group's
// tap rows to the row kernel as an unsafeutil.Block, exactly as the 8-bit
// drivers do (sb_u8_arm64.go). The vertical-edge kernels (lpfCols16) take
// checked row windows.

// sbPairLanes16[b] selects the four uint16 lanes of every unit of a two-unit
// half group whose bit is set in b (unit k owns lanes 4k..4k+3).
var sbPairLanes16 = func() (t [4][8]uint16) {
	for b := range t {
		for k := 0; k < 2; k++ {
			if b&(1<<k) != 0 {
				for i := 0; i < 4; i++ {
					t[b][4*k+i] = 0xffff
				}
			}
		}
	}
	return t
}()

func lpf16Any(m archsimd.Mask16x8) bool {
	b := m.ToInt16x8().ToBits().ReshapeToUint64s()
	return b.GetElem(0)|b.GetElem(1) != 0
}

// lpf8U16W is the high-bit-depth counterpart of lpf16U8Wd16/lpf8U8W on eight
// uint16 lanes (two 4-sample units): dav1d's loop_filter() with E/I/H and the
// flatness threshold scaled by bitdepth_min_8 (the bitdepth's lpf16Tab row
// k), filter4 in signed 16-bit lanes with the iclip_diff/iclip_pixel bounds,
// and the flat filters' sums held exactly in uint16 lanes (at most
// 16*4095+8). The tap rows are tv's: R's row count picks the width (4, 6 and
// 8 rows run wd 4, 6 and 8, fourteen rows p6..q6 run wd 16), q0 is row
// len(R)/2, and every row is named by a constant index at its load and its
// store (no checks, no inline marks). Vertical edges run lpfCols16.
func lpf8U16W[R unsafeutil.BlockRows](tv unsafeutil.Block[R], k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	var rows R
	n := len(rows)
	wd := n
	if n == 14 {
		wd = 16
	}
	q := n / 2 // row of q0
	p1 := archsimd.LoadUint8x16Array(tv.Row16(q - 2)).ReshapeToUint16s()
	p0 := archsimd.LoadUint8x16Array(tv.Row16(q - 1)).ReshapeToUint16s()
	q0 := archsimd.LoadUint8x16Array(tv.Row16(q)).ReshapeToUint16s()
	q1 := archsimd.LoadUint8x16Array(tv.Row16(q + 1)).ReshapeToUint16s()
	maxd := lpf16Abd(p1, p0).Max(lpf16Abd(q1, q0))
	t := maxd
	var p2, q2, p3, q3 archsimd.Uint16x8
	if wd >= 6 {
		p2 = archsimd.LoadUint8x16Array(tv.Row16(q - 3)).ReshapeToUint16s()
		q2 = archsimd.LoadUint8x16Array(tv.Row16(q + 2)).ReshapeToUint16s()
		tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1))
		if wd >= 8 {
			p3 = archsimd.LoadUint8x16Array(tv.Row16(q - 4)).ReshapeToUint16s()
			q3 = archsimd.LoadUint8x16Array(tv.Row16(q + 3)).ReshapeToUint16s()
			tt = tt.Max(lpf16Abd(p3, p2).Max(lpf16Abd(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpf16Abd(p0, q0)
	sum := d00.Add(d00).Add(lpf16Abd(p1, q1).Shift(archsimd.LoadInt16x8Array(&k.shr1)))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	fV := archsimd.LoadUint16x8Array(&k.flat)
	mF4 := fm
	var mFlat, mFlat16 archsimd.Mask16x8
	var p6, p5, p4, q4, q5, q6 archsimd.Uint16x8
	if wd >= 6 {
		fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
		}
		mFlat = fV.GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		if wd == 16 {
			p6 = archsimd.LoadUint8x16Array(tv.Row16(q - 7)).ReshapeToUint16s()
			p5 = archsimd.LoadUint8x16Array(tv.Row16(q - 6)).ReshapeToUint16s()
			p4 = archsimd.LoadUint8x16Array(tv.Row16(q - 5)).ReshapeToUint16s()
			q4 = archsimd.LoadUint8x16Array(tv.Row16(q + 4)).ReshapeToUint16s()
			q5 = archsimd.LoadUint8x16Array(tv.Row16(q + 5)).ReshapeToUint16s()
			q6 = archsimd.LoadUint8x16Array(tv.Row16(q + 6)).ReshapeToUint16s()
			fo := lpf16Abd(p6, p0).Max(lpf16Abd(p5, p0)).Max(lpf16Abd(p4, p0)).
				Max(lpf16Abd(q4, q0)).Max(lpf16Abd(q5, q0)).Max(lpf16Abd(q6, q0))
			mFlat16 = fV.GreaterEqual(fo).And(m16).And(mFlat)
			mFlat = mFlat.And(mFlat16.Not())
		}
	}

	o1, o0, n0, n1 := p1, p0, q0, q1
	if lpf16Any(mF4) {
		hev := maxd.Greater(hV)
		dmin := archsimd.LoadInt16x8Array(&k.dmin)
		dmax := archsimd.LoadInt16x8Array(&k.dmax)
		zero := archsimd.Int16x8{}
		maxPix := archsimd.LoadInt16x8Array(&k.maxPix)
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(dmin).Min(dmax).Masked(hev)
		f = q0s.Sub(p0s).Mul(archsimd.LoadInt16x8Array(&k.three)).Add(f).Max(dmin).Min(dmax)
		f1 := f.Add(archsimd.LoadInt16x8Array(&k.four)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		f2 := f.Add(archsimd.LoadInt16x8Array(&k.three)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		o0 = p0s.Add(f2).Max(zero).Min(maxPix).ToBits().IfElse(mF4, p0)
		n0 = q0s.Sub(f1).Max(zero).Min(maxPix).ToBits().IfElse(mF4, q0)
		f3 := f1.Add(archsimd.LoadInt16x8Array(&k.one)).Shift(archsimd.LoadInt16x8Array(&k.shr1))
		mOuter := mF4.And(hev.Not())
		o1 = p1s.Add(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, p1)
		n1 = q1s.Sub(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, q1)
	}

	if wd == 6 {
		if lpf16Any(mFlat) {
			s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd4))
			o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
			s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
			o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
			s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
			n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
			s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
			n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
		}
		o1.ReshapeToUint8s().StoreArray(tv.Row16(q - 2))
		o0.ReshapeToUint8s().StoreArray(tv.Row16(q - 1))
		n0.ReshapeToUint8s().StoreArray(tv.Row16(q))
		n1.ReshapeToUint8s().StoreArray(tv.Row16(q + 1))
		return
	}

	o2, n2 := p2, q2
	if wd >= 8 && lpf16Any(mFlat) {
		s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.LoadUint16x8Array(&k.rnd4)))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o2)
		s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
		s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
		s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
		s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
		s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n2)
	}

	if wd == 16 && lpf16Any(mFlat16) {
		m := mFlat16
		p6x2 := p6.Add(p6)
		s := p6.Add(p6x2).Add(p6x2.Add(p6x2))
		s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd8))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p5).ReshapeToUint8s().StoreArray(tv.Row16(q - 6))
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p4).ReshapeToUint8s().StoreArray(tv.Row16(q - 5))
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p3).ReshapeToUint8s().StoreArray(tv.Row16(q - 4))
		s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o2)
		s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o1)
		s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o0)
		s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n0)
		s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n1)
		s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n2)
		s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q3).ReshapeToUint8s().StoreArray(tv.Row16(q + 3))
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q4).ReshapeToUint8s().StoreArray(tv.Row16(q + 4))
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q5).ReshapeToUint8s().StoreArray(tv.Row16(q + 5))
	}
	if wd >= 8 {
		o2.ReshapeToUint8s().StoreArray(tv.Row16(q - 3))
		n2.ReshapeToUint8s().StoreArray(tv.Row16(q + 2))
	}
	o1.ReshapeToUint8s().StoreArray(tv.Row16(q - 2))
	o0.ReshapeToUint8s().StoreArray(tv.Row16(q - 1))
	n0.ReshapeToUint8s().StoreArray(tv.Row16(q))
	n1.ReshapeToUint8s().StoreArray(tv.Row16(q + 1))
}

// lpfRowsU16 runs the width-wd row kernel on the tap rows of gv (the wd/2
// rows each side of the edge, seven at wd 16): the edge driver's choice.
func lpfRowsU16(gv unsafeutil.Rows, wd int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	switch wd {
	case 16:
		lpf8U16W(unsafeutil.BlockAt[lpfR14](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	case 8:
		lpf8U16W(unsafeutil.BlockAt[lpfR8](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	case 6:
		lpf8U16W(unsafeutil.BlockAt[lpfR6](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	default:
		lpf8U16W(unsafeutil.BlockAt[lpfR4](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	}
}

// lpfTranspose8x8U16Regs transposes an 8x8 matrix of uint16 rows.
func lpfTranspose8x8U16Regs(a, b, c, d, e, f, g, h archsimd.Uint16x8) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Uint16x8) {
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint32s().InterleaveEven(cd0.ReshapeToUint32s()), ab0.ReshapeToUint32s().InterleaveOdd(cd0.ReshapeToUint32s())
	ac1, ac3 := ab1.ReshapeToUint32s().InterleaveEven(cd1.ReshapeToUint32s()), ab1.ReshapeToUint32s().InterleaveOdd(cd1.ReshapeToUint32s())
	eg0, eg2 := ef0.ReshapeToUint32s().InterleaveEven(gh0.ReshapeToUint32s()), ef0.ReshapeToUint32s().InterleaveOdd(gh0.ReshapeToUint32s())
	eg1, eg3 := ef1.ReshapeToUint32s().InterleaveEven(gh1.ReshapeToUint32s()), ef1.ReshapeToUint32s().InterleaveOdd(gh1.ReshapeToUint32s())
	o0 = ac0.ReshapeToUint64s().InterleaveLo(eg0.ReshapeToUint64s()).ReshapeToUint16s()
	o1 = ac1.ReshapeToUint64s().InterleaveLo(eg1.ReshapeToUint64s()).ReshapeToUint16s()
	o2 = ac2.ReshapeToUint64s().InterleaveLo(eg2.ReshapeToUint64s()).ReshapeToUint16s()
	o3 = ac3.ReshapeToUint64s().InterleaveLo(eg3.ReshapeToUint64s()).ReshapeToUint16s()
	o4 = ac0.ReshapeToUint64s().InterleaveHi(eg0.ReshapeToUint64s()).ReshapeToUint16s()
	o5 = ac1.ReshapeToUint64s().InterleaveHi(eg1.ReshapeToUint64s()).ReshapeToUint16s()
	o6 = ac2.ReshapeToUint64s().InterleaveHi(eg2.ReshapeToUint64s()).ReshapeToUint16s()
	o7 = ac3.ReshapeToUint64s().InterleaveHi(eg3.ReshapeToUint64s()).ReshapeToUint16s()
	return
}

// sbHalf16Limits is the per-group prologue of dav1d's 16bpc
// lpf_{v,h}_sb_{y,uv}_16bpc_neon drivers (src/arm/64/loopfilter16.S
// lpf_func) for one two-unit group: lv holds the units' own levels in lanes 0
// and 1 and their fallback (left or above) levels in lanes 2 and 3, component
// in the low byte. L = cur ? cur : prev; I, E and H follow from L and
// Av1FilterLUT.sharp on byte lanes, are widened to the units' four uint16
// lanes (uxtl) and shifted left by bitdepth_min_8; the "wd >= 4" lane mask is
// the vmask pair restricted to L != 0. ok is false when no selected unit has
// a level.
func sbHalf16Limits(lv archsimd.Uint32x4, pair uint32, sharp0 uint64, sharp1 uint8, bd8 uint64) (eV, iV, hV archsimd.Uint16x8, m4 archsimd.Mask16x8, ok bool) {
	zero32 := archsimd.Uint32x4{}
	prev := lv.ReshapeToUint64s().InterleaveHi(lv.ReshapeToUint64s()).ReshapeToUint32s()
	L := prev.IfElse(lv.Equal(zero32), lv)
	lr := L.Mul(archsimd.BroadcastUint32x4(0x01010101)).ReshapeToUint8s()
	lw := lr.ExtendLo8ToUint16()
	act := archsimd.LoadUint16x8Array(&sbPairLanes16[pair&3]).AndNot(lw.Equal(archsimd.Uint16x8{}).ToInt16x8().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hB := lr.ShiftAllRight(4)
	iB := lr.ShiftAllRight(sharp0).Min(archsimd.BroadcastUint8x16(sharp1)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eB := e.Add(e).Add(iB)
	hV = hB.ExtendLo8ToUint16().ShiftAllLeft(bd8)
	iV = iB.ExtendLo8ToUint16().ShiftAllLeft(bd8)
	eV = eB.ExtendLo8ToUint16().ShiftAllLeft(bd8)
	m4 = archsimd.Int16x8{}.Greater(act.BitsToInt16())
	ok = true
	return
}

func sbPairMask16(n uint32) archsimd.Mask16x8 {
	return archsimd.Int16x8{}.Greater(archsimd.LoadUint16x8Array(&sbPairLanes16[n&3]).BitsToInt16())
}

// sbLevels16Row loads two consecutive units' levels and their above
// fallbacks (dav1d 16bpc lpf_v_sb ld1 {v0.8b}/{v1.8b}).
func sbLevels16Row(lvl []uint8, u0, lvlStride, comp int, pair uint32) archsimd.Uint32x4 {
	if li := 4*u0 + comp; u0-lvlStride >= 0 && li+8 <= len(lvl) {
		return sbLevels16Pair(lvl, li, 4*lvlStride)
	}
	var v [2]uint64
	{
		for k := 0; k < 2; k++ {
			if pair&(1<<uint(k)) != 0 {
				v[0] |= uint64(lvl[4*(u0+k)+comp]) << (8*comp + 32*k)
				v[1] |= uint64(lvl[4*(u0+k-lvlStride)+comp]) << (8*comp + 32*k)
			}
		}
	}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().ShiftAllRight(uint64(8 * comp)).And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevels16Col loads two vertically stacked units' levels and their left
// fallbacks (dav1d 16bpc lpf_h_sb ld2 of the l[-1], l[0] pairs).
func sbLevels16Col(lvl []uint8, li, lvlStride, comp int, pair uint32) archsimd.Uint32x4 {
	if li-1 >= 0 && 4*(li+lvlStride)+comp+4 <= len(lvl) {
		return sbLevels16Pairs(lvl, 4*li+comp, 4*lvlStride)
	}
	var v [2]uint64
	{
		for k := 0; k < 2; k++ {
			if pair&(1<<uint(k)) != 0 {
				v[0] |= uint64(lvl[4*(li+k*lvlStride)+comp]) << (8*comp + 32*k)
				v[1] |= uint64(lvl[4*(li+k*lvlStride-1)+comp]) << (8*comp + 32*k)
			}
		}
	}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().ShiftAllRight(uint64(8 * comp)).And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevels16Pair is component comp of two consecutive cells from the byte
// cursor li (lanes 0-1) and of the two cells ls4 bytes above (lanes 2-3).
func sbLevels16Pair(lvl []uint8, li, ls4 int) archsimd.Uint32x4 {
	v := [2]uint64{binary.LittleEndian.Uint64(lvl[li : li+8]), binary.LittleEndian.Uint64(lvl[li-ls4 : li-ls4+8])}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevels16Pairs is component comp of the (l[-1], l[0]) cell pairs of two
// units ls4 bytes apart from the byte cursor li, the units' own levels in
// lanes 0-1 and their left fallbacks in lanes 2-3; one window holds both
// pairs (one compare from 32-bit offsets).
func sbLevels16Pairs(lvl []uint8, li, ls4 int) archsimd.Uint32x4 {
	o, l := int(uint32(li-4)), int(uint32(ls4))
	e := o + l + 8
	x0 := binary.LittleEndian.Uint64(lvl[o:e])
	x1 := binary.LittleEndian.Uint64(lvl[o+l : e])
	v := [2]uint64{x0>>32 | x1&^0xffffffff, x0&0xffffffff | x1<<32}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// filterSBHorizontal16Edge is the group-by-group driver for calls whose
// samples or levels reach the end of their buffers.
func filterSBHorizontal16Edge(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	bd8 := uint(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 16 && all>>(2*uint(g)) != 0; g++ {
		sh := 2 * uint(g)
		pair := (all >> sh) & 3
		if pair == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 3
		n16 := (vm[2] >> sh) & 3
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 2*g
		q0Off := off + 16*g
		// The group's tap rows: the wd/2 rows each side of the edge (seven
		// at wd 16), sixteen bytes each.
		rh := wd / 2
		if wd == 16 {
			rh = 7
		}
		if 16*g+16 > rowBytes || !unsafeutil.RowsFit(len(pix), q0Off-rh*stride, stride, 16, 2*rh) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 3
			}
			filterSBHorizontal16Scalar(pix, q0Off, stride, bitDepth, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		eV, iV, hV, m4, ok := sbHalf16Limits(sbLevels16Row(lvl, u0, lvlStride, comp, pair), pair, sharp0, sharp1, uint64(bd8))
		if !ok {
			continue
		}
		lpfRowsU16(unsafeutil.NewRows(pix, q0Off-rh*stride, stride, 16, 2*rh), wd, k, eV, iV, hV, m4, sbPairMask16(nGt4), sbPairMask16(n16))
	}
}

// filterSBVertical16Edge is filterSBHorizontal16Edge for vertical edges.
func filterSBVertical16Edge(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	bd8 := uint(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 16 && all>>(2*uint(g)) != 0; g++ {
		sh := 2 * uint(g)
		pair := (all >> sh) & 3
		if pair == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 3
		n16 := (vm[2] >> sh) & 3
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 2*g*lvlStride
		q0Off := off + 8*g*stride
		reach := 8 // bytes on each side: p3..q3
		if wd == 16 {
			reach = 16 // p7..q7
		}
		if rowBytes < reach || !unsafeutil.RowsFit(len(pix), q0Off-reach, stride, 2*reach, 8) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 3
			}
			filterSBVertical16Scalar(pix, q0Off, stride, bitDepth, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		eV, iV, hV, m4, ok := sbHalf16Limits(sbLevels16Col(lvl, u0, lvlStride, comp, pair), pair, sharp0, sharp1, uint64(bd8))
		if !ok {
			continue
		}
		lpfCols16(unsafeutil.NewRows(pix, q0Off-reach, stride, 2*reach, 8), 0, reach, wd, k, eV, iV, hV, m4, sbPairMask16(nGt4), sbPairMask16(n16))
	}
}

// sbLimits16 is sbHalf16Limits for the dav1d-shaped drivers: lv holds the two
// units' own levels in lanes 0 and 1 and their fallbacks in lanes 2 and 3,
// already in the low byte of each lane; I and E come from the LUT's sharp
// vectors and the widened limits shift left by bitdepth_min_8 through one
// count vector.
func sbLimits16(lv archsimd.Uint32x4, pair uint32, lut *SBLUT, bd8 int16) (eV, iV, hV archsimd.Uint16x8, m4 archsimd.Mask16x8, ok bool) {
	one := archsimd.BroadcastUint8x16(1)
	prev := lv.ReshapeToUint64s().InterleaveHi(lv.ReshapeToUint64s()).ReshapeToUint32s()
	L := prev.IfElse(lv.Equal(archsimd.Uint32x4{}), lv)
	lr := L.Mul(one.ReshapeToUint32s()).ReshapeToUint8s()
	lw := lr.ExtendLo8ToUint16()
	act := archsimd.LoadUint16x8Array(&sbPairLanes16[pair&3]).AndNot(lw.Equal(archsimd.Uint16x8{}).ToInt16x8().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hB := lr.ShiftAllRight(4)
	iB := lr.Shift(archsimd.LoadInt8x16Array(&lut.SharpShift)).Min(archsimd.LoadUint8x16Array(&lut.SharpCap)).Max(one)
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eB := e.Add(e).Add(iB)
	bdV := archsimd.BroadcastInt16x8(bd8)
	hV = hB.ExtendLo8ToUint16().Shift(bdV)
	iV = iB.ExtendLo8ToUint16().Shift(bdV)
	eV = eB.ExtendLo8ToUint16().Shift(bdV)
	m4 = archsimd.Int16x8{}.Greater(act.BitsToInt16())
	ok = true
	return
}

// sbWidth16 is the kernel width from a group's raw width-mask pairs.
func sbWidth16(vm1, vm2 uint32, luma bool) int {
	if vm2&3 != 0 {
		return 16
	}
	if vm1&3 != 0 {
		if luma {
			return 8
		}
		return 6
	}
	return 4
}

// filterSBHorizontal16 is dav1d lpf_v_sb_{y,uv}_16bpc_neon (two units per
// group): the width masks shift down one pair per group while the sample and
// level cursors advance, and each group's levels are one 8-byte load each of
// l[0] and l[-b4_stride] from the component-offset level cursor. The call's
// reach is checked once, as one view of its samples and one of its levels
// (sbRowsU16); calls touching a buffer end take the edge driver.
func filterSBHorizontal16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	end := 16 * ((31-bits.LeadingZeros32(vm0))>>1 + 1)
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBHorizontal16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if end > rowBytes {
		filterSBHorizontal16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	k := &lpf16Tab[uint(bitDepth-8)>>1]
	var ok bool
	if vm2 != 0 {
		ok = sbRowsU16[lpfW16](pix, off, stride, end, vm0, vm1, vm2, luma, lvl, li, ls4, lut, k, int16(bitDepth-8))
	} else {
		ok = sbRowsU16[lpfW8](pix, off, stride, end, vm0, vm1, 0, luma, lvl, li, ls4, lut, k, int16(bitDepth-8))
	}
	if !ok {
		filterSBHorizontal16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
}

// sbRowsU16 is filterSBHorizontal16's group loop for calls whose widest group
// is W (lpfW16, or lpfW8 for every narrower call): one view of the call's
// tap rows (the four rows each side of the edge at off, seven at wd 16, n
// bytes each) and one of its two level rows (l[0] at li and l[-b4_stride] ls4
// bytes before it, n/2 bytes) replace the reach checks; every group's
// sixteen-byte (two-unit) block is then a column of the sample view and an
// eight-byte column of the level view, and the narrower kernels of a wider
// call take its inner rows. It reports false, having filtered nothing, when
// either view leaves its slice.
func sbRowsU16[W lpfWidth](pix []byte, off, stride, n int, vm0, vm1, vm2 uint32, luma bool, lvl []uint8, li, ls4 int, lut *SBLUT, k *lpf16Consts, bd8 int16) bool {
	var wdT W
	rh := 4
	if len(wdT) == 16 {
		rh = 7
	}
	if !unsafeutil.RowsFit(len(pix), off-rh*stride, stride, n, 2*rh) || !unsafeutil.RowsFit(len(lvl), li-ls4, ls4, n/2, 2) {
		return false
	}
	v := unsafeutil.NewRows(pix, off-rh*stride, stride, n, 2*rh)
	lv := unsafeutil.NewRows(lvl, li-ls4, ls4, n/2, 2)
	for x := 0; vm0 != 0 && x <= v.Width()-16; vm0, vm1, vm2, x = vm0>>2, vm1>>2, vm2>>2, x+16 {
		if vm0&3 == 0 {
			continue
		}
		// The group's (l[0], l[-b4_stride]) cell pairs, as sbLevels16Pair.
		lp := [2]uint64{binary.LittleEndian.Uint64(lv.Row8(1, x/2)[:]), binary.LittleEndian.Uint64(lv.Row8(0, x/2)[:])}
		lw := archsimd.LoadUint64x2Array(&lp).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
		if eV, iV, hV, m4, ok := sbLimits16(lw, vm0, lut, bd8); !ok {
			continue
		} else if len(wdT) == 16 && vm2&3 != 0 {
			lpf8U16W(unsafeutil.BlockAt[lpfR14](v, 0, x), k, eV, iV, hV, m4, sbPairMask16(vm1), sbPairMask16(vm2))
		} else if vm1&3 == 0 {
			lpf8U16W(unsafeutil.BlockAt[lpfR4](v, rh-2, x), k, eV, iV, hV, m4, archsimd.Mask16x8{}, archsimd.Mask16x8{})
		} else if luma {
			lpf8U16W(unsafeutil.BlockAt[lpfR8](v, rh-4, x), k, eV, iV, hV, m4, sbPairMask16(vm1), archsimd.Mask16x8{})
		} else {
			lpf8U16W(unsafeutil.BlockAt[lpfR6](v, rh-3, x), k, eV, iV, hV, m4, sbPairMask16(vm1), archsimd.Mask16x8{})
		}
	}
	return true
}

// filterSBVertical16 is dav1d lpf_h_sb_{y,uv}_16bpc_neon in the same shape:
// each group reads the (l[-1], l[0]) level pairs of its two units.
func filterSBVertical16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	hiG := (31 - bits.LeadingZeros32(vm0)) >> 1
	reach := 8 // bytes on each side: p3..q3
	if vm2 != 0 {
		reach = 16 // p7..q7
	}
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBVertical16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	// One view of the call's samples (its 8*(hiG+1) rows, reach bytes each
	// side of the edge) and one of its level pairs (the (l[-1], l[0]) cell
	// pairs of its 2*(hiG+1) units) replace the reach checks; each group is
	// then an eight-row block of the sample view and two rows of the level
	// view.
	if rowBytes < reach || !unsafeutil.RowsFit(len(pix), off-reach, stride, 2*reach, 8*hiG+8) || !unsafeutil.RowsFit(len(lvl), li-4, ls4, 8, 2*hiG+2) {
		filterSBVertical16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	v := unsafeutil.NewRows(pix, off-reach, stride, 2*reach, 8*hiG+8)
	lv := unsafeutil.NewRows(lvl, li-4, ls4, 8, 2*hiG+2)
	bd8 := int16(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	for g := 0; vm0 != 0; vm0, vm1, vm2, g = vm0>>2, vm1>>2, vm2>>2, g+1 {
		if vm0&3 == 0 {
			continue
		}
		// sbLevels16Pairs on the group's two level rows.
		x0, x1 := binary.LittleEndian.Uint64(lv.Row8(2*g, 0)[:]), binary.LittleEndian.Uint64(lv.Row8(2*g+1, 0)[:])
		lp := [2]uint64{x0>>32 | x1&^0xffffffff, x0&0xffffffff | x1<<32}
		lw := archsimd.LoadUint64x2Array(&lp).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
		if eV, iV, hV, m4, ok := sbLimits16(lw, vm0, lut, bd8); ok {
			lpfCols16(v, 8*g, reach, sbWidth16(vm1, vm2, luma), k, eV, iV, hV, m4, sbPairMask16(vm1), sbPairMask16(vm2))
		}
	}
}

// The 8-bit drivers check each call's sample and level footprints once, as
// unsafeutil.Rows views, and hand every group's tap rows to the kernels as an
// unsafeutil.Block: the kernels name their rows by constant row numbers, so
// no tap access costs a check (dav1d trusts its caller the same way, with
// the check moved to the view). The transposes run through local arrays.
// The 16-bit drivers (sb_u16_arm64.go) work the same way.

// lpfMaxPlane bounds the planes the vector drivers take; larger planes
// (beyond any AV1 frame) take the scalar reference.
const lpfMaxPlane = 1 << 32

func lpfAbsDiff(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

func lpfAny(m archsimd.Mask8x16) bool {
	b := m.ToInt8x16().ToBits().ReshapeToUint64s()
	return b.GetElem(0)|b.GetElem(1) != 0
}

// The filter bodies load, store, widen (ExtendLo8ToUint16, and HiToLo +
// ExtendLo8ToUint16, which is UXTL2) and narrow through the intrinsics written
// out at each use: a helper call alone on its line leaves an inline-mark NOP,
// and ShiftAllRight with a constant rebuilds its count vector (MOVD + INS +
// DUP) every time, so the narrows shift by a count vector built once.

// lpf16U8Wd16 is dav1d's lpf_16_wd16_neon: it filters sixteen lanes whose
// taps are the fourteen rows p6..q6 of tv (q0 is row 7, the first row below
// the edge), and stores the changed rows back; the wd4, flat8in and flat8out
// lanes run in one body, each lane at its own width exactly as the C reference
// does. Widths up to 8 run lpf8U8W, which the transposed vertical path shares.
func lpf16U8Wd16(tv unsafeutil.Block[lpfR14], eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) bool {
	// Each row is named at its load and its store, on the line of the
	// access, so the accessor leaves no inline mark (CSE shares the address).
	p3 := archsimd.LoadUint8x16Array(tv.Row16(3))
	p2 := archsimd.LoadUint8x16Array(tv.Row16(4))
	p1 := archsimd.LoadUint8x16Array(tv.Row16(5))
	p0 := archsimd.LoadUint8x16Array(tv.Row16(6))
	q0 := archsimd.LoadUint8x16Array(tv.Row16(7))
	q1 := archsimd.LoadUint8x16Array(tv.Row16(8))
	q2 := archsimd.LoadUint8x16Array(tv.Row16(9))
	q3 := archsimd.LoadUint8x16Array(tv.Row16(10))
	maxd := lpfAbsDiff(p1, p0).Max(lpfAbsDiff(q1, q0))
	tt := lpfAbsDiff(p2, p1).Max(lpfAbsDiff(q2, q1)).Max(lpfAbsDiff(p3, p2).Max(lpfAbsDiff(q3, q2)))
	t := maxd.Max(tt.Masked(mGt4))
	d00 := lpfAbsDiff(p0, q0)
	sum := d00.AddSaturated(d00).AddSaturated(lpfAbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpfAny(fm) {
		return false
	}
	one := archsimd.BroadcastUint8x16(1)
	mFlat := one.GreaterEqual(lpfSpread(p2, p3, p0).Max(lpfSpread(q2, q3, q0)).Max(maxd)).And(mGt4).And(fm)
	mF4 := fm.And(mFlat.Not())
	var mFlat16 archsimd.Mask8x16
	hasFlat16 := false
	if m16Cand := m16.And(mFlat); lpfAny(m16Cand) {
		p6 := archsimd.LoadUint8x16Array(tv.Row16(0))
		p5 := archsimd.LoadUint8x16Array(tv.Row16(1))
		p4 := archsimd.LoadUint8x16Array(tv.Row16(2))
		q4 := archsimd.LoadUint8x16Array(tv.Row16(11))
		q5 := archsimd.LoadUint8x16Array(tv.Row16(12))
		q6 := archsimd.LoadUint8x16Array(tv.Row16(13))
		fo := lpfSpread3(p4, p5, p6, p0).Max(lpfSpread3(q4, q5, q6, q0))
		mFlat16 = one.GreaterEqual(fo).And(m16Cand)
		if lpfAny(mFlat16) {
			hasFlat16 = true
			mFlat = mFlat.And(mFlat16.Not())
		}
	}

	// Outputs; every filter reads the original taps, lanes are disjoint.
	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	if lpfAny(mF4) {
		// dav1d wd4 filter. archsimd has no USQADD, so the taps are biased to
		// signed as the C reference does; f = iclip_diff(3*(q0-p0) + (hev ?
		// iclip_diff(p1-q1) : 0)) is three saturating adds of the saturated
		// q0-p0 rather than dav1d's widened usubl/mul/saddw/sqxtn (exact: the
		// partial sums move monotonically toward the clamp, so saturating
		// early never changes the clamped total), and (f1+1)>>1 is a rounding
		// halving add with zero (srshr #1).
		bias := archsimd.BroadcastUint8x16(128)
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		d := q0c.SubSaturated(p0c)
		f := p1c.SubSaturated(q1c).Masked(hev).AddSaturated(d).AddSaturated(d).AddSaturated(d)
		sh3 := archsimd.BroadcastInt8x16(-3)
		f1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(sh3)
		f2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(sh3)
		outer := f1.Average(archsimd.Int8x16{})
		o0 = p0c.AddSaturated(f2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(f1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}
	if lpfAny(mFlat) {
		k, k2, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(64), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.LoadUint16x8Array(&lpfBias128)
		lP3, lP2, lP1, lP0 := p3.MulWidenLo(k), p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k)
		lQ0, lQ1, lQ2, lQ3 := q0.MulWidenLo(k), q1.MulWidenLo(k), q2.MulWidenLo(k), q3.MulWidenLo(k)
		la2, la1, la0, lb0, lb1, lb2 := lpfFlat8Half(lP3, lP2, lP1, lP0, lQ0, lQ1, lQ2, lQ3, p3.MulWidenLo(k3).Add(p2.MulWidenLo(k2)).Add(lP1.Add(lP0)).Add(lQ0.Add(bias8)))
		hP3, hP2, hP1, hP0 := p3.HiToLo().MulWidenLo(k), p2.HiToLo().MulWidenLo(k), p1.HiToLo().MulWidenLo(k), p0.HiToLo().MulWidenLo(k)
		hQ0, hQ1, hQ2, hQ3 := q0.HiToLo().MulWidenLo(k), q1.HiToLo().MulWidenLo(k), q2.HiToLo().MulWidenLo(k), q3.HiToLo().MulWidenLo(k)
		ha2, ha1, ha0, hb0, hb1, hb2 := lpfFlat8Half(hP3, hP2, hP1, hP0, hQ0, hQ1, hQ2, hQ3, p3.HiToLo().MulWidenLo(k3).Add(p2.HiToLo().MulWidenLo(k2)).Add(hP1.Add(hP0)).Add(hQ0.Add(bias8)))
		o2 = la2.ReshapeToUint8s().ConcatOdd(ha2.ReshapeToUint8s()).IfElse(mFlat, o2)
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
		n2 = lb2.ReshapeToUint8s().ConcatOdd(hb2.ReshapeToUint8s()).IfElse(mFlat, n2)
	}
	if hasFlat16 {
		// The flat16 lanes finish in their own frame, which reloads the
		// (still unmodified) taps and stores every output row, so nothing of
		// this frame stays live across the call.
		lpfFlat16U8Store(tv, mFlat16, o2, o1, o0, n0, n1, n2)
		return true
	}
	o2.StoreArray(tv.Row16(4))
	n2.StoreArray(tv.Row16(9))
	o1.StoreArray(tv.Row16(5))
	o0.StoreArray(tv.Row16(6))
	n0.StoreArray(tv.Row16(7))
	n1.StoreArray(tv.Row16(8))
	return true
}

// lpfFlat16U8Store is lpfFlat16U8 for lpf16U8Wd16: it loads the taps p6..q6
// from tv (rows 0..13, not yet written), blends the 13-tap outputs on the
// lanes in m into the caller's p2..q2 outputs (o2..n2) and stores the twelve
// rows p5..q5.
func lpfFlat16U8Store(tv unsafeutil.Block[lpfR14], m archsimd.Mask8x16, o2, o1, o0, n0, n1, n2 archsimd.Uint8x16) {
	p6 := archsimd.LoadUint8x16Array(tv.Row16(0))
	p5 := archsimd.LoadUint8x16Array(tv.Row16(1))
	p4 := archsimd.LoadUint8x16Array(tv.Row16(2))
	p3 := archsimd.LoadUint8x16Array(tv.Row16(3))
	p2 := archsimd.LoadUint8x16Array(tv.Row16(4))
	p1 := archsimd.LoadUint8x16Array(tv.Row16(5))
	p0 := archsimd.LoadUint8x16Array(tv.Row16(6))
	q0 := archsimd.LoadUint8x16Array(tv.Row16(7))
	q1 := archsimd.LoadUint8x16Array(tv.Row16(8))
	q2 := archsimd.LoadUint8x16Array(tv.Row16(9))
	q3 := archsimd.LoadUint8x16Array(tv.Row16(10))
	q4 := archsimd.LoadUint8x16Array(tv.Row16(11))
	q5 := archsimd.LoadUint8x16Array(tv.Row16(12))
	q6 := archsimd.LoadUint8x16Array(tv.Row16(13))
	r5, r4, r3, s3, s4, s5 := lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, m, &o2, &o1, &o0, &n0, &n1, &n2)
	r5.StoreArray(tv.Row16(1))
	r4.StoreArray(tv.Row16(2))
	r3.StoreArray(tv.Row16(3))
	o2.StoreArray(tv.Row16(4))
	o1.StoreArray(tv.Row16(5))
	o0.StoreArray(tv.Row16(6))
	n0.StoreArray(tv.Row16(7))
	n1.StoreArray(tv.Row16(8))
	n2.StoreArray(tv.Row16(9))
	s3.StoreArray(tv.Row16(10))
	s4.StoreArray(tv.Row16(11))
	s5.StoreArray(tv.Row16(12))
}

// lpfFlat16U8 computes the 13-tap flat filter (dav1d wd16, AV1 filter14) on
// the lanes in m and blends p2..q2 into the caller's outputs; it returns the
// outer p5..p3 / q3..q5 rows (unchanged outside m). The halves run on lanes
// widened and scaled by 16, so each output byte is the high byte of its
// biased sum.
func lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6 archsimd.Uint8x16, m archsimd.Mask8x16, o2, o1, o0, n0, n1, n2 *archsimd.Uint8x16) (r5, r4, r3, s3, s4, s5 archsimd.Uint8x16) {
	k := archsimd.BroadcastUint8x16(16)
	k7 := archsimd.BroadcastUint8x16(112)
	var lo, hi [12]archsimd.Uint16x8
	lpfFlat16Half(p6.MulWidenLo(k7), p6.MulWidenLo(k), p5.MulWidenLo(k), p4.MulWidenLo(k), p3.MulWidenLo(k),
		p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k), q0.MulWidenLo(k),
		q1.MulWidenLo(k), q2.MulWidenLo(k), q3.MulWidenLo(k), q4.MulWidenLo(k),
		q5.MulWidenLo(k), q6.MulWidenLo(k), &lo)
	lpfFlat16Half(p6.HiToLo().MulWidenLo(k7), p6.HiToLo().MulWidenLo(k), p5.HiToLo().MulWidenLo(k), p4.HiToLo().MulWidenLo(k), p3.HiToLo().MulWidenLo(k),
		p2.HiToLo().MulWidenLo(k), p1.HiToLo().MulWidenLo(k), p0.HiToLo().MulWidenLo(k), q0.HiToLo().MulWidenLo(k),
		q1.HiToLo().MulWidenLo(k), q2.HiToLo().MulWidenLo(k), q3.HiToLo().MulWidenLo(k), q4.HiToLo().MulWidenLo(k),
		q5.HiToLo().MulWidenLo(k), q6.HiToLo().MulWidenLo(k), &hi)
	r5 = lo[0].ReshapeToUint8s().ConcatOdd(hi[0].ReshapeToUint8s()).IfElse(m, p5)
	r4 = lo[1].ReshapeToUint8s().ConcatOdd(hi[1].ReshapeToUint8s()).IfElse(m, p4)
	r3 = lo[2].ReshapeToUint8s().ConcatOdd(hi[2].ReshapeToUint8s()).IfElse(m, p3)
	*o2 = lo[3].ReshapeToUint8s().ConcatOdd(hi[3].ReshapeToUint8s()).IfElse(m, *o2)
	*o1 = lo[4].ReshapeToUint8s().ConcatOdd(hi[4].ReshapeToUint8s()).IfElse(m, *o1)
	*o0 = lo[5].ReshapeToUint8s().ConcatOdd(hi[5].ReshapeToUint8s()).IfElse(m, *o0)
	*n0 = lo[6].ReshapeToUint8s().ConcatOdd(hi[6].ReshapeToUint8s()).IfElse(m, *n0)
	*n1 = lo[7].ReshapeToUint8s().ConcatOdd(hi[7].ReshapeToUint8s()).IfElse(m, *n1)
	*n2 = lo[8].ReshapeToUint8s().ConcatOdd(hi[8].ReshapeToUint8s()).IfElse(m, *n2)
	s3 = lo[9].ReshapeToUint8s().ConcatOdd(hi[9].ReshapeToUint8s()).IfElse(m, q3)
	s4 = lo[10].ReshapeToUint8s().ConcatOdd(hi[10].ReshapeToUint8s()).IfElse(m, q4)
	s5 = lo[11].ReshapeToUint8s().ConcatOdd(hi[11].ReshapeToUint8s()).IfElse(m, q5)
	return
}

// lpfFlat16Half runs the dav1d sliding-sum form of the 13-tap filter on eight
// widened lanes scaled by 16 (p6x7 is 7*p6 at that scale); out[k] holds the
// biased sum of output p5+k (p5..q5).
func lpfFlat16Half(p6x7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6 archsimd.Uint16x8, out *[12]archsimd.Uint16x8) {
	p6x2 := p6.Add(p6)
	s := p6x7.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.LoadUint16x8Array(&lpfBias128)))
	out[0] = s
	s = s.Add(p3.Add(q1)).Sub(p6x2)
	out[1] = s
	s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
	out[2] = s
	s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
	out[3] = s
	s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
	out[4] = s
	s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
	out[5] = s
	s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
	out[6] = s
	s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
	out[7] = s
	s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
	out[8] = s
	s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
	out[9] = s
	s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
	out[10] = s
	s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
	out[11] = s
}

// lpfBias128 is the uint16 rounding bias (128) for scaled flat filters:
// loaded from L1 rather than BroadcastUint16x8(128), which builds via GPR->SIMD.
var lpfBias128 = [8]uint16{128, 128, 128, 128, 128, 128, 128, 128}

// sbLowByteLanes keeps the low byte of each 32-bit lane: dav1d lpf_func's
// movi v2.4s, #0xff, as one table load instead of a GPR-to-vector broadcast.
var sbLowByteLanes = [4]uint32{0xff, 0xff, 0xff, 0xff}

// sbNibbleLanes[b] selects the four byte lanes of every unit in the four-unit
// group whose bit is set in b (unit k owns lanes 4k..4k+3): dav1d lpf_func's
// dup + cmtst of a vmask nibble against word_1248, as one table load.
var sbNibbleLanes = func() (t [16][16]uint8) {
	for b := range t {
		for k := 0; k < 4; k++ {
			if b&(1<<k) != 0 {
				for i := 0; i < 4; i++ {
					t[b][4*k+i] = 0xff
				}
			}
		}
	}
	return t
}()

func sbNibbleMask(n uint32) archsimd.Mask8x16 {
	return archsimd.Int8x16{}.Greater(archsimd.LoadUint8x16Array(&sbNibbleLanes[n&15]).BitsToInt8())
}

// sbGroupLimits is the per-group prologue of dav1d's lpf_{v,h}_sb_{y,uv}
// drivers (src/arm/64/loopfilter.S lpf_func) across lanes, one unit per
// 32-bit lane: cur and prev hold the units' own and fallback (left or above)
// levels in the low byte; L = cur ? cur : prev is replicated over the unit's
// four bytes, I = imax(imin(L >> sharp[0], sharp[1]), 1), E = 2*(L+2)+I and
// H = L>>4 follow without a table, and the "wd >= 4" lane mask is the vmask
// nibble restricted to L != 0. ok is false when no selected unit has a level
// (dav1d "if (!L) continue"). The wider-width masks need no level test: the
// kernel already ANDs them with fm, which includes m4.
func sbGroupLimits(cur, prev archsimd.Uint32x4, nAll uint32, sharp0 uint64, sharp1 uint8) (eV, iV, hV archsimd.Uint8x16, m4 archsimd.Mask8x16, ok bool) {
	zero32 := archsimd.Uint32x4{}
	L := prev.IfElse(cur.Equal(zero32), cur)
	lr := L.Mul(archsimd.BroadcastUint32x4(0x01010101)).ReshapeToUint8s()
	zero := archsimd.Uint8x16{}
	act := archsimd.LoadUint8x16Array(&sbNibbleLanes[nAll&15]).AndNot(lr.Equal(zero).ToInt8x16().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hV = lr.ShiftAllRight(4)
	iV = lr.ShiftAllRight(sharp0).Min(archsimd.BroadcastUint8x16(sharp1)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eV = e.Add(e).Add(iV)
	m4 = archsimd.Int8x16{}.Greater(act.BitsToInt8())
	ok = true
	return
}

// sbWidth is dav1d lpf_func's kernel choice from the raw vmask nibbles: the
// per-lane masks narrow every lane to its own width inside the kernel.
func sbWidth(nGt4, n16 uint32, luma bool) int {
	switch {
	case n16 != 0:
		return 16
	case nGt4 != 0:
		if luma {
			return 8
		}
		return 6
	default:
		return 4
	}
}

// The level map holds four bytes per 4x4 cell (lfmask.LevelCache), so a
// byte cursor li = 4*cell + comp reads component comp of a cell; loads of
// whole cells keep comp in the low byte of each 32-bit lane after an AND with
// sbLowByteLanes, as dav1d's lpf_func does.

// sbLevels4 is component comp of the four cells from the byte cursor li, one
// per 32-bit lane (dav1d's ld1 of l[0]): one sixteen-byte load.
func sbLevels4(lvl []uint8, li int) archsimd.Uint32x4 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(lvl[li : li+16])).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevelsPairs is component comp of the (l[-1], l[0]) cell pairs of four
// units (dav1d's ld2 lane loads): rows r0..r0+3 of lv, eight bytes each, a
// view whose row k starts at the cell before unit k's at the component offset.
// cur holds each unit's l[0], prev its l[-1]; a block outside lv reads zero
// levels (no unit filters).
func sbLevelsPairs(lv unsafeutil.Rows, r0 int) (cur, prev archsimd.Uint32x4) {
	tv := lv.Sub(r0, 0, 8, 4)
	return sbPairs(binary.LittleEndian.Uint64(tv.Row8(0, 0)[:]), binary.LittleEndian.Uint64(tv.Row8(1, 0)[:]),
		binary.LittleEndian.Uint64(tv.Row8(2, 0)[:]), binary.LittleEndian.Uint64(tv.Row8(3, 0)[:]))
}

// sbPairs splits four units' (l[-1], l[0]) pairs, one unit per uint64, into
// the l[0] lanes (cur) and the l[-1] lanes (prev), component in the low byte.
func sbPairs(x0, x1, y0, y1 uint64) (cur, prev archsimd.Uint32x4) {
	x := [2]uint64{x0, x1}
	y := [2]uint64{y0, y1}
	a := archsimd.LoadUint64x2Array(&x).ReshapeToUint32s()
	b := archsimd.LoadUint64x2Array(&y).ReshapeToUint32s()
	ff := archsimd.LoadUint32x4Array(&sbLowByteLanes)
	return a.ConcatOdd(b).And(ff), a.ConcatEven(b).And(ff)
}

// sbLevelsRow loads the levels of four consecutive units (horizontal edges:
// dav1d lpf_v_sb ld1 of l[0] and l[-b4_stride]) with component comp moved to
// the low byte of each lane.
func sbLevelsRow(lvl []uint8, u0, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	if li := 4*u0 + comp; u0-lvlStride >= 0 && li+16 <= len(lvl) {
		return sbLevels4(lvl, li), sbLevels4(lvl, li-4*lvlStride)
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = uint32(lvl[4*(u0+k)+comp])
			p[k] = uint32(lvl[4*(u0+k-lvlStride)+comp])
		}
	}
	return archsimd.LoadUint32x4Array(&c), archsimd.LoadUint32x4Array(&p)
}

// sbLevelsCol loads the levels of four units stacked lvlStride cells apart
// (vertical edges: dav1d lpf_h_sb ld2 of the l[-1], l[0] pairs) with
// component comp moved to the low byte of each lane.
func sbLevelsCol(lvl []uint8, li, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	if unsafeutil.RowsFit(len(lvl), 4*li+comp-4, 4*lvlStride, 8, 4) {
		return sbLevelsPairs(unsafeutil.NewRows(lvl, 4*li+comp-4, 4*lvlStride, 8, 4), 0)
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = uint32(lvl[4*(li+k*lvlStride)+comp])
			p[k] = uint32(lvl[4*(li+k*lvlStride-1)+comp])
		}
	}
	return archsimd.LoadUint32x4Array(&c), archsimd.LoadUint32x4Array(&p)
}

// filterSBHorizontal8Edge is the group-by-group driver for calls whose
// samples or levels reach the end of their buffers: each group checks its own
// reach and falls back to the unit-by-unit reference when it would leave pix.
func filterSBHorizontal8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 8 && all>>(4*uint(g)) != 0; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 4*g
		q0Off := off + 16*g
		// The group's tap rows: the wd/2 rows each side of the edge (seven
		// at wd 16), sixteen bytes each.
		rh := wd / 2
		if wd == 16 {
			rh = 7
		}
		if 16*g+16 > rowBytes || !unsafeutil.RowsFit(len(pix), q0Off-rh*stride, stride, 16, 2*rh) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 0xf
			}
			filterSBHorizontal8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := sbLevelsRow(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		lpfRowsU8(unsafeutil.NewRows(pix, q0Off-rh*stride, stride, 16, 2*rh), wd, eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
	}
}

// filterSBVertical8Edge is filterSBHorizontal8Edge for vertical edges.
func filterSBVertical8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 8 && all>>(4*uint(g)) != 0; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 4*g*lvlStride
		q0Off := off + 16*g*stride
		// The group's sixteen rows: p7..q7 (sixteen bytes) at wd 16, else
		// p3..q3 read sixteen bytes at a time when the row has twelve bytes
		// right of the edge, or eight bytes at a time.
		var gv unsafeutil.Rows
		inPix, wide := false, false
		switch {
		case wd == 16:
			if inPix = unsafeutil.RowsFit(len(pix), q0Off-8, stride, 16, 16); inPix {
				gv = unsafeutil.NewRows(pix, q0Off-8, stride, 16, 16)
			}
		case rowBytes >= 12 && unsafeutil.RowsFit(len(pix), q0Off-4, stride, 16, 16):
			gv, inPix, wide = unsafeutil.NewRows(pix, q0Off-4, stride, 16, 16), true, true
		default:
			if inPix = unsafeutil.RowsFit(len(pix), q0Off-4, stride, 8, 16); inPix {
				gv = unsafeutil.NewRows(pix, q0Off-4, stride, 8, 16)
			}
		}
		if !inPix {
			sbVerticalScalarGroup(pix, q0Off, stride, vm, sh, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := sbLevelsCol(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		if wd == 16 {
			lpfVertical16x16U8(unsafeutil.BlockAt[lpfR16](gv, 0, 0), eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
		} else if wide {
			mGt4 := sbNibbleMask(nGt4)
			gb := unsafeutil.BlockAt[lpfR16](gv, 0, 0)
			switch wd {
			case 8:
				lpfVertical16x8U8WideW[lpfW8](gb, eV, iV, hV, m4, mGt4)
			case 6:
				lpfVertical16x8U8WideW[lpfW6](gb, eV, iV, hV, m4, mGt4)
			default:
				lpfVertical16x8U8WideW[lpfW4](gb, eV, iV, hV, m4, mGt4)
			}
		} else {
			lpfVertical16x8U8(gv, 0, 0, wd, eV, iV, hV, m4, sbNibbleMask(nGt4))
		}
	}
}

// sbActive and sbLimitsOf are the per-group level prologue of dav1d lpf_func
// (src/arm/64/loopfilter.S), one unit per 32-bit lane: cur and prev hold each
// unit's own and fallback (left or above) level in the low byte of its lane
// with the other bytes zero. L = cur ? cur : prev is replicated over the
// unit's four bytes (lr) by a multiply with the bytes-of-one vector that also
// floors I; I = imax(imin(L >> sharp[0], sharp[1]), 1) and E = 2*(L+2)+I come
// from the LUT's replicated sharp vectors, H = L>>4.
//
// sbActive is the first half: lr and the active lanes act ("wd >= 4"
// restricted to L != 0), and whether any lane is active ("if (!L)
// continue"). The drivers test it before sbLimitsOf builds the limits, so a
// skipped group carries no zeroed limit vectors.
func sbActive(cur, prev archsimd.Uint32x4, nib uint32) (lr, act archsimd.Uint8x16, ok bool) {
	L := prev.IfElse(cur.Equal(archsimd.Uint32x4{}), cur)
	lr = L.Mul(archsimd.BroadcastUint8x16(1).ReshapeToUint32s()).ReshapeToUint8s()
	act = archsimd.LoadUint8x16Array(&sbNibbleLanes[nib&15]).AndNot(lr.Equal(archsimd.Uint8x16{}).ToInt8x16().ToBits())
	h := act.ReshapeToUint64s()
	return lr, act, h.GetElem(0)|h.GetElem(1) != 0
}

// sbLimitsOf is sbLimits' second half: E, I and H from lr, and the wd >= 4
// lane mask from act.
func sbLimitsOf(lr, act archsimd.Uint8x16, lut *SBLUT) (eV, iV, hV archsimd.Uint8x16, m4 archsimd.Mask8x16) {
	hV = lr.ShiftAllRight(4)
	iV = lr.Shift(archsimd.LoadInt8x16Array(&lut.SharpShift)).Min(archsimd.LoadUint8x16Array(&lut.SharpCap)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eV = e.Add(e).Add(iV)
	m4 = archsimd.Int8x16{}.Greater(act.BitsToInt8())
	return
}

// filterSBHorizontal8 is dav1d lpf_v_sb_{y,uv}_8bpc_neon: the three width
// masks shift down one four-unit nibble per group while the sample and level
// cursors advance, each group's levels are one load of l[0] and one of
// l[-b4_stride] from a level cursor already pointing at component comp (the
// C side's &lvl[x][comp]), and the widest set nibble picks the kernel. The
// call's reach is checked once up front, as dav1d trusts its caller; calls
// that touch the end of pix, the row or the level map take the edge driver.
func filterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	end := 16 * ((31-bits.LeadingZeros32(vm0))>>2 + 1)
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if end > rowBytes {
		filterSBHorizontal8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	var ok bool
	switch {
	case vm2 != 0:
		ok = sbRowsU8[lpfW16](pix, off, stride, end, vm0, vm1, vm2, luma, lvl, li, ls4, lut)
	case vm1 != 0 && luma:
		ok = sbRowsU8[lpfW8](pix, off, stride, end, vm0, vm1, 0, luma, lvl, li, ls4, lut)
	case vm1 != 0:
		ok = sbRowsU8[lpfW6](pix, off, stride, end, vm0, vm1, 0, luma, lvl, li, ls4, lut)
	default:
		ok = sbRowsU8[lpfW4](pix, off, stride, end, vm0, 0, 0, luma, lvl, li, ls4, lut)
	}
	if !ok {
		filterSBHorizontal8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
}

// sbRowsU8 is filterSBHorizontal8's group loop for calls whose widest group
// is W: one view of the call's tap rows (the W/2 rows each side of the edge at
// off, seven at wd 16, n bytes each) and one of its two level rows (l[0] at
// li and l[-b4_stride] ls4 bytes before it) replace the call's reach checks;
// every group's sixteen-byte (four-cell) block is then a column of those
// views, and the narrower kernels of a wider call take their inner rows. It
// reports false, having filtered nothing, when either view leaves its slice.
func sbRowsU8[W lpfWidth](pix []byte, off, stride, n int, vm0, vm1, vm2 uint32, luma bool, lvl []uint8, li, ls4 int, lut *SBLUT) bool {
	var wdT W
	wd := len(wdT)
	rh := wd / 2
	if wd == 16 {
		rh = 7
	}
	if !unsafeutil.RowsFit(len(pix), off-rh*stride, stride, n, 2*rh) || !unsafeutil.RowsFit(len(lvl), li-ls4, ls4, n, 2) {
		return false
	}
	v := unsafeutil.NewRows(pix, off-rh*stride, stride, n, 2*rh)
	lv := unsafeutil.NewRows(lvl, li-ls4, ls4, n, 2)
	ff := archsimd.LoadUint32x4Array(&sbLowByteLanes)
	maxX := v.Width() - 16
	for x := 0; vm0 != 0 && x <= maxX; {
		if vm0&15 == 0 {
			skip := bits.TrailingZeros32(vm0) >> 2
			sh := uint(skip) << 2
			vm0, vm1, vm2, x = vm0>>sh, vm1>>sh, vm2>>sh, x+skip<<4
			if x > maxX {
				break
			}
		}
		// The level block is named at each load (no inline mark of its own).
		cur := archsimd.LoadUint8x16Array(unsafeutil.BlockAt[lpfR2](lv, 0, x).Row16(1)).ReshapeToUint32s().And(ff)
		prev := archsimd.LoadUint8x16Array(unsafeutil.BlockAt[lpfR2](lv, 0, x).Row16(0)).ReshapeToUint32s().And(ff)
		// The helpers share their lines with the tests, so they leave no
		// inline marks.
		var lr, act archsimd.Uint8x16
		var any bool
		if lr, act, any = sbActive(cur, prev, vm0); !any {
			vm0, vm1, vm2, x = vm0>>4, vm1>>4, vm2>>4, x+16
			continue
		}
		if eV, iV, hV, m4 := sbLimitsOf(lr, act, lut); wd == 16 && vm2&15 != 0 {
			lpf16U8Wd16(unsafeutil.BlockAt[lpfR14](v, 0, x), eV, iV, hV, m4, sbNibbleMask(vm1), sbNibbleMask(vm2))
		} else if wd >= 6 && vm1&15 != 0 {
			if luma {
				lpf8U8Rows[lpfW8](unsafeutil.BlockAt[lpfR8](v, rh-4, x), eV, iV, hV, m4, sbNibbleMask(vm1))
			} else {
				lpf8U8Rows[lpfW6](unsafeutil.BlockAt[lpfR6](v, rh-3, x), eV, iV, hV, m4, sbNibbleMask(vm1))
			}
		} else {
			lpf8U8Rows[lpfW4](unsafeutil.BlockAt[lpfR4](v, rh-2, x), eV, iV, hV, m4, archsimd.Mask8x16{})
		}
		vm0, vm1, vm2, x = vm0>>4, vm1>>4, vm2>>4, x+16
	}
	return true
}

// lpfRowsU8 is the per-group kernel choice of filterSBHorizontal8 for the
// edge driver: the width-wd row kernel on the tap rows of gv (the wd/2 rows
// each side of the edge, seven at wd 16).
func lpfRowsU8(gv unsafeutil.Rows, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	switch wd {
	case 16:
		lpf16U8Wd16(unsafeutil.BlockAt[lpfR14](gv, 0, 0), eV, iV, hV, m4, mGt4, m16)
	case 8:
		lpf8U8Rows[lpfW8](unsafeutil.BlockAt[lpfR8](gv, 0, 0), eV, iV, hV, m4, mGt4)
	case 6:
		lpf8U8Rows[lpfW6](unsafeutil.BlockAt[lpfR6](gv, 0, 0), eV, iV, hV, m4, mGt4)
	default:
		lpf8U8Rows[lpfW4](unsafeutil.BlockAt[lpfR4](gv, 0, 0), eV, iV, hV, m4, mGt4)
	}
}

// filterSBVertical8 is dav1d lpf_h_sb_{y,uv}_8bpc_neon in the same shape:
// each group reads the (l[-1], l[0]) level pairs of its four units (the ld2
// lane loads) from the component-offset cursor.
func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	hiG := (31 - bits.LeadingZeros32(vm0)) >> 2
	reach := 4
	if vm2 != 0 {
		reach = 8
	}
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBVertical8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	// One view of the call's samples (its 16*(hiG+1) rows, reach bytes left
	// of the edge to twelve right of it) and one of its level pairs (the
	// (l[-1], l[0]) cell pairs of its 4*(hiG+1) units) replace the reach
	// checks; each group is then a sixteen-row block of the sample view and a
	// four-row block of the level view.
	if rowBytes < 12 || !unsafeutil.RowsFit(len(pix), off-reach, stride, reach+12, 16*hiG+16) || !unsafeutil.RowsFit(len(lvl), li-4, ls4, 8, 4*hiG+4) {
		filterSBVertical8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	v := unsafeutil.NewRows(pix, off-reach, stride, reach+12, 16*hiG+16)
	lv := unsafeutil.NewRows(lvl, li-4, ls4, 8, 4*hiG+4)
	ff := archsimd.LoadUint32x4Array(&sbLowByteLanes)
	c16, c8 := reach-8, reach-4 // p7 and p3 columns of the view
	for g := 0; vm0 != 0; {
		if vm0&15 == 0 {
			skip := bits.TrailingZeros32(vm0) >> 2
			sh := uint(skip) << 2
			vm0, vm1, vm2, g = vm0>>sh, vm1>>sh, vm2>>sh, g+skip
		}
		// sbLevelsPairs written out, each helper on the line of an
		// instruction of this function (no inline marks): the group's four
		// (l[-1], l[0]) pairs go to two vectors through the array stores.
		lb4 := unsafeutil.Block8At[lpfR4](lv, 4*g, 0)
		l0 := [2]uint64{binary.LittleEndian.Uint64(lb4.Row8(0)[:]), binary.LittleEndian.Uint64(lb4.Row8(1)[:])}
		l1 := [2]uint64{binary.LittleEndian.Uint64(lb4.Row8(2)[:]), binary.LittleEndian.Uint64(lb4.Row8(3)[:])}
		la := archsimd.LoadUint64x2Array(&l0).ReshapeToUint32s()
		lb := archsimd.LoadUint64x2Array(&l1).ReshapeToUint32s()
		var lr, act archsimd.Uint8x16
		var any bool
		if lr, act, any = sbActive(la.ConcatOdd(lb).And(ff), la.ConcatEven(lb).And(ff), vm0); !any {
			vm0, vm1, vm2, g = vm0>>4, vm1>>4, vm2>>4, g+1
			continue
		}
		if eV, iV, hV, m4 := sbLimitsOf(lr, act, lut); vm2&15 != 0 {
			lpfVertical16x16U8(unsafeutil.BlockAt[lpfR16](v, 16*g, c16), eV, iV, hV, m4, sbNibbleMask(vm1), sbNibbleMask(vm2))
		} else if vm1&15 == 0 {
			lpfVertical16x8U8WideW[lpfW4](unsafeutil.BlockAt[lpfR16](v, 16*g, c8), eV, iV, hV, m4, archsimd.Mask8x16{})
		} else if luma {
			lpfVertical16x8U8WideW[lpfW8](unsafeutil.BlockAt[lpfR16](v, 16*g, c8), eV, iV, hV, m4, sbNibbleMask(vm1))
		} else {
			lpfVertical16x8U8WideW[lpfW6](unsafeutil.BlockAt[lpfR16](v, 16*g, c8), eV, iV, hV, m4, sbNibbleMask(vm1))
		}
		vm0, vm1, vm2, g = vm0>>4, vm1>>4, vm2>>4, g+1
	}
}

func sbVerticalScalarGroup(pix []byte, q0Off, stride int, vm [3]uint32, sh uint, luma bool, lvl []uint8, u0, lvlStride, comp int, lut *SBLUT) {
	sub := vm
	for k := range sub {
		sub[k] = (sub[k] >> sh) & 0xf
	}
	filterSBVertical8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut)
}

// lpfVertical16x8U8 filters a vertical edge across sixteen rows using eight
// bytes per row (p3..q3): the sixteen rows of v from row r0, column c0 (p3).
// The rows are transposed into tap vectors, filtered, and transposed back. A
// block outside v filters nothing.
func lpfVertical16x8U8(v unsafeutil.Rows, r0, c0 int, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	tv := v.Sub(r0, c0, 8, 16)
	var buf [128]byte
	// Pair row r (low half) with row r+8 (high half), then an 8x8 byte
	// transpose of both halves at once yields one vector per tap.
	var pr [8][2]uint64
	for r := 0; r < 8; r++ {
		pr[r][0] = binary.LittleEndian.Uint64(tv.Row8(r, 0)[:])
		pr[r][1] = binary.LittleEndian.Uint64(tv.Row8(r+8, 0)[:])
	}
	a := archsimd.LoadUint64x2Array(&pr[0]).ReshapeToUint8s()
	b := archsimd.LoadUint64x2Array(&pr[1]).ReshapeToUint8s()
	c := archsimd.LoadUint64x2Array(&pr[2]).ReshapeToUint8s()
	d := archsimd.LoadUint64x2Array(&pr[3]).ReshapeToUint8s()
	e := archsimd.LoadUint64x2Array(&pr[4]).ReshapeToUint8s()
	f := archsimd.LoadUint64x2Array(&pr[5]).ReshapeToUint8s()
	g := archsimd.LoadUint64x2Array(&pr[6]).ReshapeToUint8s()
	h := archsimd.LoadUint64x2Array(&pr[7]).ReshapeToUint8s()
	t0, t1, t2, t3, t4, t5, t6, t7 := lpfTranspose8x8Pairs(a, b, c, d, e, f, g, h)
	// t_k now holds tap k (p3..q3) for rows 0..7 (low) and 8..15 (high), but
	// the pairing interleaves the row order: lane i<8 is row i, lane 8+i row 8+i.
	t0.StoreArray((*[16]byte)(buf[0:16]))
	t1.StoreArray((*[16]byte)(buf[16:32]))
	t2.StoreArray((*[16]byte)(buf[32:48]))
	t3.StoreArray((*[16]byte)(buf[48:64]))
	t4.StoreArray((*[16]byte)(buf[64:80]))
	t5.StoreArray((*[16]byte)(buf[80:96]))
	t6.StoreArray((*[16]byte)(buf[96:112]))
	t7.StoreArray((*[16]byte)(buf[112:128]))
	bv := unsafeutil.NewRows(buf[:], 0, 16, 16, 8)
	if !lpf8U8(wd, bv, 0, 0, eV, iV, hV, m4, mGt4) {
		return
	}
	a = archsimd.LoadUint8x16Array((*[16]byte)(buf[0:16]))
	b = archsimd.LoadUint8x16Array((*[16]byte)(buf[16:32]))
	c = archsimd.LoadUint8x16Array((*[16]byte)(buf[32:48]))
	d = archsimd.LoadUint8x16Array((*[16]byte)(buf[48:64]))
	e = archsimd.LoadUint8x16Array((*[16]byte)(buf[64:80]))
	f = archsimd.LoadUint8x16Array((*[16]byte)(buf[80:96]))
	g = archsimd.LoadUint8x16Array((*[16]byte)(buf[96:112]))
	h = archsimd.LoadUint8x16Array((*[16]byte)(buf[112:128]))
	t0, t1, t2, t3, t4, t5, t6, t7 = lpfTranspose8x8Pairs(a, b, c, d, e, f, g, h)
	t0.ReshapeToUint64s().StoreArray(&pr[0])
	t1.ReshapeToUint64s().StoreArray(&pr[1])
	t2.ReshapeToUint64s().StoreArray(&pr[2])
	t3.ReshapeToUint64s().StoreArray(&pr[3])
	t4.ReshapeToUint64s().StoreArray(&pr[4])
	t5.ReshapeToUint64s().StoreArray(&pr[5])
	t6.ReshapeToUint64s().StoreArray(&pr[6])
	t7.ReshapeToUint64s().StoreArray(&pr[7])
	for r := 0; r < 8; r++ {
		binary.LittleEndian.PutUint64(tv.Row8(r, 0)[:], pr[r][0])
		binary.LittleEndian.PutUint64(tv.Row8(r+8, 0)[:], pr[r][1])
	}
}

// lpfTranspose8x8Pairs transposes the 8x8 byte matrix held in the low halves
// and, independently, the one in the high halves of eight vectors.
func lpfTranspose8x8Pairs(a, b, c, d, e, f, g, h archsimd.Uint8x16) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Uint8x16) {
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint16s().InterleaveEven(cd0.ReshapeToUint16s()), ab0.ReshapeToUint16s().InterleaveOdd(cd0.ReshapeToUint16s())
	ac1, ac3 := ab1.ReshapeToUint16s().InterleaveEven(cd1.ReshapeToUint16s()), ab1.ReshapeToUint16s().InterleaveOdd(cd1.ReshapeToUint16s())
	eg0, eg2 := ef0.ReshapeToUint16s().InterleaveEven(gh0.ReshapeToUint16s()), ef0.ReshapeToUint16s().InterleaveOdd(gh0.ReshapeToUint16s())
	eg1, eg3 := ef1.ReshapeToUint16s().InterleaveEven(gh1.ReshapeToUint16s()), ef1.ReshapeToUint16s().InterleaveOdd(gh1.ReshapeToUint16s())
	o0 = ac0.ReshapeToUint32s().InterleaveEven(eg0.ReshapeToUint32s()).ReshapeToUint8s()
	o1 = ac1.ReshapeToUint32s().InterleaveEven(eg1.ReshapeToUint32s()).ReshapeToUint8s()
	o2 = ac2.ReshapeToUint32s().InterleaveEven(eg2.ReshapeToUint32s()).ReshapeToUint8s()
	o3 = ac3.ReshapeToUint32s().InterleaveEven(eg3.ReshapeToUint32s()).ReshapeToUint8s()
	o4 = ac0.ReshapeToUint32s().InterleaveOdd(eg0.ReshapeToUint32s()).ReshapeToUint8s()
	o5 = ac1.ReshapeToUint32s().InterleaveOdd(eg1.ReshapeToUint32s()).ReshapeToUint8s()
	o6 = ac2.ReshapeToUint32s().InterleaveOdd(eg2.ReshapeToUint32s()).ReshapeToUint8s()
	o7 = ac3.ReshapeToUint32s().InterleaveOdd(eg3.ReshapeToUint32s()).ReshapeToUint8s()
	return
}

// lpfVertical16x16U8 is the wd16 variant: sixteen bytes per row (p7..q7),
// the sixteen rows of tv, a full 16x16 transpose each way through buf, where
// the row kernel sees taps p6..q6 as buffer rows 1..14.
func lpfVertical16x16U8(tv unsafeutil.Block[lpfR16], eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	var buf [256]byte
	lpfTranspose16x16FromPix(tv, &buf)
	bv := unsafeutil.NewRows(buf[:], 0, 16, 16, 16)
	if !lpf16U8Wd16(unsafeutil.BlockAt[lpfR14](bv, 1, 0), eV, iV, hV, m4, mGt4, m16) {
		return
	}
	lpfTranspose16x16ToPix(&buf, tv)
}

// lpfR4..lpfR16 are the tap-row counts of the kernels' Blocks.
type (
	lpfR2  = [2]struct{}
	lpfR4  = [4]struct{}
	lpfR6  = [6]struct{}
	lpfR8  = [8]struct{}
	lpfR14 = [14]struct{}
	lpfR16 = [16]struct{}
)

// lpfTranspose16x16FromPix loads the sixteen 16-byte rows of tv, transposes
// the 16x16 byte matrix in registers and stores it as the rows of buf.
func lpfTranspose16x16FromPix(tv unsafeutil.Block[lpfR16], buf *[256]byte) {
	lo, hi := unsafeutil.BlockHalves(tv)
	r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 := lpfTranspose16x16(
		archsimd.LoadUint8x16Array(lo.Row16(0)), archsimd.LoadUint8x16Array(lo.Row16(1)), archsimd.LoadUint8x16Array(lo.Row16(2)), archsimd.LoadUint8x16Array(lo.Row16(3)), archsimd.LoadUint8x16Array(lo.Row16(4)), archsimd.LoadUint8x16Array(lo.Row16(5)), archsimd.LoadUint8x16Array(lo.Row16(6)), archsimd.LoadUint8x16Array(lo.Row16(7)), archsimd.LoadUint8x16Array(hi.Row16(0)), archsimd.LoadUint8x16Array(hi.Row16(1)), archsimd.LoadUint8x16Array(hi.Row16(2)), archsimd.LoadUint8x16Array(hi.Row16(3)), archsimd.LoadUint8x16Array(hi.Row16(4)), archsimd.LoadUint8x16Array(hi.Row16(5)), archsimd.LoadUint8x16Array(hi.Row16(6)), archsimd.LoadUint8x16Array(hi.Row16(7)))
	r0.StoreArray((*[16]byte)(buf[0:16]))
	r1.StoreArray((*[16]byte)(buf[16:32]))
	r2.StoreArray((*[16]byte)(buf[32:48]))
	r3.StoreArray((*[16]byte)(buf[48:64]))
	r4.StoreArray((*[16]byte)(buf[64:80]))
	r5.StoreArray((*[16]byte)(buf[80:96]))
	r6.StoreArray((*[16]byte)(buf[96:112]))
	r7.StoreArray((*[16]byte)(buf[112:128]))
	r8.StoreArray((*[16]byte)(buf[128:144]))
	r9.StoreArray((*[16]byte)(buf[144:160]))
	r10.StoreArray((*[16]byte)(buf[160:176]))
	r11.StoreArray((*[16]byte)(buf[176:192]))
	r12.StoreArray((*[16]byte)(buf[192:208]))
	r13.StoreArray((*[16]byte)(buf[208:224]))
	r14.StoreArray((*[16]byte)(buf[224:240]))
	r15.StoreArray((*[16]byte)(buf[240:256]))
}

// lpfTranspose16x16ToPix is lpfTranspose16x16FromPix back: the rows of buf,
// transposed, to the sixteen rows of tv.
func lpfTranspose16x16ToPix(buf *[256]byte, tv unsafeutil.Block[lpfR16]) {
	lo, hi := unsafeutil.BlockHalves(tv)
	r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 := lpfTranspose16x16(
		archsimd.LoadUint8x16Array((*[16]byte)(buf[0:16])), archsimd.LoadUint8x16Array((*[16]byte)(buf[16:32])), archsimd.LoadUint8x16Array((*[16]byte)(buf[32:48])), archsimd.LoadUint8x16Array((*[16]byte)(buf[48:64])), archsimd.LoadUint8x16Array((*[16]byte)(buf[64:80])), archsimd.LoadUint8x16Array((*[16]byte)(buf[80:96])), archsimd.LoadUint8x16Array((*[16]byte)(buf[96:112])), archsimd.LoadUint8x16Array((*[16]byte)(buf[112:128])), archsimd.LoadUint8x16Array((*[16]byte)(buf[128:144])), archsimd.LoadUint8x16Array((*[16]byte)(buf[144:160])), archsimd.LoadUint8x16Array((*[16]byte)(buf[160:176])), archsimd.LoadUint8x16Array((*[16]byte)(buf[176:192])), archsimd.LoadUint8x16Array((*[16]byte)(buf[192:208])), archsimd.LoadUint8x16Array((*[16]byte)(buf[208:224])), archsimd.LoadUint8x16Array((*[16]byte)(buf[224:240])), archsimd.LoadUint8x16Array((*[16]byte)(buf[240:256])))
	r0.StoreArray(lo.Row16(0))
	r1.StoreArray(lo.Row16(1))
	r2.StoreArray(lo.Row16(2))
	r3.StoreArray(lo.Row16(3))
	r4.StoreArray(lo.Row16(4))
	r5.StoreArray(lo.Row16(5))
	r6.StoreArray(lo.Row16(6))
	r7.StoreArray(lo.Row16(7))
	r8.StoreArray(hi.Row16(0))
	r9.StoreArray(hi.Row16(1))
	r10.StoreArray(hi.Row16(2))
	r11.StoreArray(hi.Row16(3))
	r12.StoreArray(hi.Row16(4))
	r13.StoreArray(hi.Row16(5))
	r14.StoreArray(hi.Row16(6))
	r15.StoreArray(hi.Row16(7))
}

// lpfTranspose16x16 transposes the 16x16 byte matrix whose rows are l0..l15.
func lpfTranspose16x16(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 archsimd.Uint8x16) (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 archsimd.Uint8x16) {
	r0, r1 = l0.InterleaveEven(l1), l0.InterleaveOdd(l1)
	r2, r3 = l2.InterleaveEven(l3), l2.InterleaveOdd(l3)
	r4, r5 = l4.InterleaveEven(l5), l4.InterleaveOdd(l5)
	r6, r7 = l6.InterleaveEven(l7), l6.InterleaveOdd(l7)
	r8, r9 = l8.InterleaveEven(l9), l8.InterleaveOdd(l9)
	r10, r11 = l10.InterleaveEven(l11), l10.InterleaveOdd(l11)
	r12, r13 = l12.InterleaveEven(l13), l12.InterleaveOdd(l13)
	r14, r15 = l14.InterleaveEven(l15), l14.InterleaveOdd(l15)

	r0, r2 = r0.ReshapeToUint16s().InterleaveEven(r2.ReshapeToUint16s()).ReshapeToUint8s(), r0.ReshapeToUint16s().InterleaveOdd(r2.ReshapeToUint16s()).ReshapeToUint8s()
	r1, r3 = r1.ReshapeToUint16s().InterleaveEven(r3.ReshapeToUint16s()).ReshapeToUint8s(), r1.ReshapeToUint16s().InterleaveOdd(r3.ReshapeToUint16s()).ReshapeToUint8s()
	r4, r6 = r4.ReshapeToUint16s().InterleaveEven(r6.ReshapeToUint16s()).ReshapeToUint8s(), r4.ReshapeToUint16s().InterleaveOdd(r6.ReshapeToUint16s()).ReshapeToUint8s()
	r5, r7 = r5.ReshapeToUint16s().InterleaveEven(r7.ReshapeToUint16s()).ReshapeToUint8s(), r5.ReshapeToUint16s().InterleaveOdd(r7.ReshapeToUint16s()).ReshapeToUint8s()
	r8, r10 = r8.ReshapeToUint16s().InterleaveEven(r10.ReshapeToUint16s()).ReshapeToUint8s(), r8.ReshapeToUint16s().InterleaveOdd(r10.ReshapeToUint16s()).ReshapeToUint8s()
	r9, r11 = r9.ReshapeToUint16s().InterleaveEven(r11.ReshapeToUint16s()).ReshapeToUint8s(), r9.ReshapeToUint16s().InterleaveOdd(r11.ReshapeToUint16s()).ReshapeToUint8s()
	r12, r14 = r12.ReshapeToUint16s().InterleaveEven(r14.ReshapeToUint16s()).ReshapeToUint8s(), r12.ReshapeToUint16s().InterleaveOdd(r14.ReshapeToUint16s()).ReshapeToUint8s()
	r13, r15 = r13.ReshapeToUint16s().InterleaveEven(r15.ReshapeToUint16s()).ReshapeToUint8s(), r13.ReshapeToUint16s().InterleaveOdd(r15.ReshapeToUint16s()).ReshapeToUint8s()

	r0, r4 = r0.ReshapeToUint32s().InterleaveEven(r4.ReshapeToUint32s()).ReshapeToUint8s(), r0.ReshapeToUint32s().InterleaveOdd(r4.ReshapeToUint32s()).ReshapeToUint8s()
	r1, r5 = r1.ReshapeToUint32s().InterleaveEven(r5.ReshapeToUint32s()).ReshapeToUint8s(), r1.ReshapeToUint32s().InterleaveOdd(r5.ReshapeToUint32s()).ReshapeToUint8s()
	r2, r6 = r2.ReshapeToUint32s().InterleaveEven(r6.ReshapeToUint32s()).ReshapeToUint8s(), r2.ReshapeToUint32s().InterleaveOdd(r6.ReshapeToUint32s()).ReshapeToUint8s()
	r3, r7 = r3.ReshapeToUint32s().InterleaveEven(r7.ReshapeToUint32s()).ReshapeToUint8s(), r3.ReshapeToUint32s().InterleaveOdd(r7.ReshapeToUint32s()).ReshapeToUint8s()
	r8, r12 = r8.ReshapeToUint32s().InterleaveEven(r12.ReshapeToUint32s()).ReshapeToUint8s(), r8.ReshapeToUint32s().InterleaveOdd(r12.ReshapeToUint32s()).ReshapeToUint8s()
	r9, r13 = r9.ReshapeToUint32s().InterleaveEven(r13.ReshapeToUint32s()).ReshapeToUint8s(), r9.ReshapeToUint32s().InterleaveOdd(r13.ReshapeToUint32s()).ReshapeToUint8s()
	r10, r14 = r10.ReshapeToUint32s().InterleaveEven(r14.ReshapeToUint32s()).ReshapeToUint8s(), r10.ReshapeToUint32s().InterleaveOdd(r14.ReshapeToUint32s()).ReshapeToUint8s()
	r11, r15 = r11.ReshapeToUint32s().InterleaveEven(r15.ReshapeToUint32s()).ReshapeToUint8s(), r11.ReshapeToUint32s().InterleaveOdd(r15.ReshapeToUint32s()).ReshapeToUint8s()

	r0, r8 = r0.ReshapeToUint64s().InterleaveLo(r8.ReshapeToUint64s()).ReshapeToUint8s(), r0.ReshapeToUint64s().InterleaveHi(r8.ReshapeToUint64s()).ReshapeToUint8s()
	r1, r9 = r1.ReshapeToUint64s().InterleaveLo(r9.ReshapeToUint64s()).ReshapeToUint8s(), r1.ReshapeToUint64s().InterleaveHi(r9.ReshapeToUint64s()).ReshapeToUint8s()
	r2, r10 = r2.ReshapeToUint64s().InterleaveLo(r10.ReshapeToUint64s()).ReshapeToUint8s(), r2.ReshapeToUint64s().InterleaveHi(r10.ReshapeToUint64s()).ReshapeToUint8s()
	r3, r11 = r3.ReshapeToUint64s().InterleaveLo(r11.ReshapeToUint64s()).ReshapeToUint8s(), r3.ReshapeToUint64s().InterleaveHi(r11.ReshapeToUint64s()).ReshapeToUint8s()
	r4, r12 = r4.ReshapeToUint64s().InterleaveLo(r12.ReshapeToUint64s()).ReshapeToUint8s(), r4.ReshapeToUint64s().InterleaveHi(r12.ReshapeToUint64s()).ReshapeToUint8s()
	r5, r13 = r5.ReshapeToUint64s().InterleaveLo(r13.ReshapeToUint64s()).ReshapeToUint8s(), r5.ReshapeToUint64s().InterleaveHi(r13.ReshapeToUint64s()).ReshapeToUint8s()
	r6, r14 = r6.ReshapeToUint64s().InterleaveLo(r14.ReshapeToUint64s()).ReshapeToUint8s(), r6.ReshapeToUint64s().InterleaveHi(r14.ReshapeToUint64s()).ReshapeToUint8s()
	r7, r15 = r7.ReshapeToUint64s().InterleaveLo(r15.ReshapeToUint64s()).ReshapeToUint8s(), r7.ReshapeToUint64s().InterleaveHi(r15.ReshapeToUint64s()).ReshapeToUint8s()
	return
}

// lpfVertical16x8U8WideW is lpfVertical16x8U8 for groups whose rows can be
// read sixteen bytes at a time from p3 of the edge (the sixteen rows of tv):
// lpf8U8Cols directly on tv without a generic layout type assertion.
func lpfVertical16x8U8WideW[W lpfWidth](tv unsafeutil.Block[lpfR16], eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	lpf8U8Cols[W](tv, eV, iV, hV, m4, mGt4)
}

// lpf8U8Cols is dav1d lpf_h_16_wd{4,6,8}_neon: it loads and transposes sixteen
// rows starting at p3, filters sixteen lanes at width W, and transposes/stores
// the modified taps back. Keeping tv concrete as Block[lpfR16] eliminates the
// runtime type assertion and hoists BlockHalves once, keeping the kernel a leaf.
func lpf8U8Cols[W lpfWidth](tv unsafeutil.Block[lpfR16], eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	var wdT W
	wd := len(wdT)
	lo, hi := unsafeutil.BlockHalves(tv)
	r0, r8 := archsimd.LoadUint8x16Array(lo.Row16(0)), archsimd.LoadUint8x16Array(hi.Row16(0))
	r1, r9 := archsimd.LoadUint8x16Array(lo.Row16(1)), archsimd.LoadUint8x16Array(hi.Row16(1))
	r2, r10 := archsimd.LoadUint8x16Array(lo.Row16(2)), archsimd.LoadUint8x16Array(hi.Row16(2))
	r3, r11 := archsimd.LoadUint8x16Array(lo.Row16(3)), archsimd.LoadUint8x16Array(hi.Row16(3))
	r4, r12 := archsimd.LoadUint8x16Array(lo.Row16(4)), archsimd.LoadUint8x16Array(hi.Row16(4))
	r5, r13 := archsimd.LoadUint8x16Array(lo.Row16(5)), archsimd.LoadUint8x16Array(hi.Row16(5))
	r6, r14 := archsimd.LoadUint8x16Array(lo.Row16(6)), archsimd.LoadUint8x16Array(hi.Row16(6))
	r7, r15 := archsimd.LoadUint8x16Array(lo.Row16(7)), archsimd.LoadUint8x16Array(hi.Row16(7))
	p3, p2, p1, p0, q0, q1, q2, q3 := lpfTranspose8x8Pairs(
		r0.ReshapeToUint64s().InterleaveLo(r8.ReshapeToUint64s()).ReshapeToUint8s(), r1.ReshapeToUint64s().InterleaveLo(r9.ReshapeToUint64s()).ReshapeToUint8s(), r2.ReshapeToUint64s().InterleaveLo(r10.ReshapeToUint64s()).ReshapeToUint8s(), r3.ReshapeToUint64s().InterleaveLo(r11.ReshapeToUint64s()).ReshapeToUint8s(),
		r4.ReshapeToUint64s().InterleaveLo(r12.ReshapeToUint64s()).ReshapeToUint8s(), r5.ReshapeToUint64s().InterleaveLo(r13.ReshapeToUint64s()).ReshapeToUint8s(), r6.ReshapeToUint64s().InterleaveLo(r14.ReshapeToUint64s()).ReshapeToUint8s(), r7.ReshapeToUint64s().InterleaveLo(r15.ReshapeToUint64s()).ReshapeToUint8s())
	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	maxd := lpfAbsDiff(p1, p0).Max(lpfAbsDiff(q1, q0))
	t := maxd
	if wd >= 6 {
		tt := lpfAbsDiff(p2, p1).Max(lpfAbsDiff(q2, q1))
		if wd >= 8 {
			tt = tt.Max(lpfAbsDiff(p3, p2).Max(lpfAbsDiff(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpfAbsDiff(p0, q0)
	sum := d00.AddSaturated(d00).AddSaturated(lpfAbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpfAny(fm) {
		return false
	}
	mF4 := fm
	var mFlat archsimd.Mask8x16
	if wd >= 6 {
		var fl archsimd.Uint8x16
		if wd >= 8 {
			fl = lpfSpread(p2, p3, p0).Max(lpfSpread(q2, q3, q0))
		} else {
			fl = lpfAbsDiff(p2, p0).Max(lpfAbsDiff(q2, q0))
		}
		mFlat = archsimd.BroadcastUint8x16(1).GreaterEqual(fl.Max(maxd)).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
	}
	if lpfAny(mF4) {
		bias := archsimd.BroadcastUint8x16(128)
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		d := q0c.SubSaturated(p0c)
		f := p1c.SubSaturated(q1c).Masked(hev).AddSaturated(d).AddSaturated(d).AddSaturated(d)
		sh3 := archsimd.BroadcastInt8x16(-3)
		f1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(sh3)
		f2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(sh3)
		outer := f1.Average(archsimd.Int8x16{})
		o0 = p0c.AddSaturated(f2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(f1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}
	if wd == 6 && lpfAny(mFlat) {
		k, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.LoadUint16x8Array(&lpfBias128)
		lP2, lP1, lP0 := p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k)
		lQ0, lQ1, lQ2 := q0.MulWidenLo(k), q1.MulWidenLo(k), q2.MulWidenLo(k)
		lt := lP1.Add(lP0)
		la1, la0, lb0, lb1 := lpfFlat6Half(lP2, lP1, lP0, lQ0, lQ1, lQ2, p2.MulWidenLo(k3).Add(lt.Add(lt)).Add(lQ0.Add(bias8)))
		hP2, hP1, hP0 := p2.HiToLo().MulWidenLo(k), p1.HiToLo().MulWidenLo(k), p0.HiToLo().MulWidenLo(k)
		hQ0, hQ1, hQ2 := q0.HiToLo().MulWidenLo(k), q1.HiToLo().MulWidenLo(k), q2.HiToLo().MulWidenLo(k)
		ht := hP1.Add(hP0)
		ha1, ha0, hb0, hb1 := lpfFlat6Half(hP2, hP1, hP0, hQ0, hQ1, hQ2, p2.HiToLo().MulWidenLo(k3).Add(ht.Add(ht)).Add(hQ0.Add(bias8)))
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
	} else if wd == 8 && lpfAny(mFlat) {
		k, k2, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(64), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.LoadUint16x8Array(&lpfBias128)
		lP3, lP2, lP1, lP0 := p3.MulWidenLo(k), p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k)
		lQ0, lQ1, lQ2, lQ3 := q0.MulWidenLo(k), q1.MulWidenLo(k), q2.MulWidenLo(k), q3.MulWidenLo(k)
		la2, la1, la0, lb0, lb1, lb2 := lpfFlat8Half(lP3, lP2, lP1, lP0, lQ0, lQ1, lQ2, lQ3, p3.MulWidenLo(k3).Add(p2.MulWidenLo(k2)).Add(lP1.Add(lP0)).Add(lQ0.Add(bias8)))
		hP3, hP2, hP1, hP0 := p3.HiToLo().MulWidenLo(k), p2.HiToLo().MulWidenLo(k), p1.HiToLo().MulWidenLo(k), p0.HiToLo().MulWidenLo(k)
		hQ0, hQ1, hQ2, hQ3 := q0.HiToLo().MulWidenLo(k), q1.HiToLo().MulWidenLo(k), q2.HiToLo().MulWidenLo(k), q3.HiToLo().MulWidenLo(k)
		ha2, ha1, ha0, hb0, hb1, hb2 := lpfFlat8Half(hP3, hP2, hP1, hP0, hQ0, hQ1, hQ2, hQ3, p3.HiToLo().MulWidenLo(k3).Add(p2.HiToLo().MulWidenLo(k2)).Add(hP1.Add(hP0)).Add(hQ0.Add(bias8)))
		o2 = la2.ReshapeToUint8s().ConcatOdd(ha2.ReshapeToUint8s()).IfElse(mFlat, o2)
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
		n2 = lb2.ReshapeToUint8s().ConcatOdd(hb2.ReshapeToUint8s()).IfElse(mFlat, n2)
	}
	if wd < 8 {
		// dav1d transpose_4x16b: byte zips pair p1/p0 and q0/q1, halfword
		// zips gather each row's four bytes; lane k of c_j is row 4j+k.
		a0, a1 := o1.InterleaveLo(o0), o1.InterleaveHi(o0)
		b0, b1 := n0.InterleaveLo(n1), n0.InterleaveHi(n1)
		c0 := a0.ReshapeToUint16s().InterleaveLo(b0.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
		c1 := a0.ReshapeToUint16s().InterleaveHi(b0.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
		c2 := a1.ReshapeToUint16s().InterleaveLo(b1.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
		c3 := a1.ReshapeToUint16s().InterleaveHi(b1.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
		lo.PutF32(0, 2, c0.GetElem(0))
		lo.PutF32(1, 2, c0.GetElem(1))
		lo.PutF32(2, 2, c0.GetElem(2))
		lo.PutF32(3, 2, c0.GetElem(3))
		lo.PutF32(4, 2, c1.GetElem(0))
		lo.PutF32(5, 2, c1.GetElem(1))
		lo.PutF32(6, 2, c1.GetElem(2))
		lo.PutF32(7, 2, c1.GetElem(3))
		hi.PutF32(0, 2, c2.GetElem(0))
		hi.PutF32(1, 2, c2.GetElem(1))
		hi.PutF32(2, 2, c2.GetElem(2))
		hi.PutF32(3, 2, c2.GetElem(3))
		hi.PutF32(4, 2, c3.GetElem(0))
		hi.PutF32(5, 2, c3.GetElem(1))
		hi.PutF32(6, 2, c3.GetElem(2))
		hi.PutF32(7, 2, c3.GetElem(3))
	} else {
		t0, t1, t2, t3, t4, t5, t6, t7 := lpfTranspose8x8Pairs(p3, o2, o1, o0, n0, n1, n2, q3)
		lo.PutF64(0, 0, t0.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(1, 0, t1.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(2, 0, t2.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(3, 0, t3.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(4, 0, t4.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(5, 0, t5.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(6, 0, t6.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		lo.PutF64(7, 0, t7.ReshapeToUint64s().BitsToFloat64().GetElem(0))
		hi.PutF64(0, 0, t0.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(1, 0, t1.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(2, 0, t2.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(3, 0, t3.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(4, 0, t4.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(5, 0, t5.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(6, 0, t6.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		hi.PutF64(7, 0, t7.ReshapeToUint64s().BitsToFloat64().GetElem(1))
	}
	return true
}

// lpf8U8Rows is dav1d lpf_v_16_wd{4,6,8}_neon: the taps are the wd/2 rows
// above and below the edge in tv (R holds 2*(wd/2) rows, sixteen bytes each).
func lpf8U8Rows[W lpfWidth, R unsafeutil.BlockRows](tv unsafeutil.Block[R], eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	var wdT W
	wd := len(wdT)
	rh := wd / 2
	var p3, p2, q2, q3 archsimd.Uint8x16
	p1 := archsimd.LoadUint8x16Array(tv.Row16(rh - 2))
	p0 := archsimd.LoadUint8x16Array(tv.Row16(rh - 1))
	q0 := archsimd.LoadUint8x16Array(tv.Row16(rh))
	q1 := archsimd.LoadUint8x16Array(tv.Row16(rh + 1))
	if wd >= 6 {
		p2 = archsimd.LoadUint8x16Array(tv.Row16(rh - 3))
		q2 = archsimd.LoadUint8x16Array(tv.Row16(rh + 2))
	}
	if wd >= 8 {
		p3 = archsimd.LoadUint8x16Array(tv.Row16(rh - 4))
		q3 = archsimd.LoadUint8x16Array(tv.Row16(rh + 3))
	}
	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	maxd := lpfAbsDiff(p1, p0).Max(lpfAbsDiff(q1, q0))
	t := maxd
	if wd >= 6 {
		tt := lpfAbsDiff(p2, p1).Max(lpfAbsDiff(q2, q1))
		if wd >= 8 {
			tt = tt.Max(lpfAbsDiff(p3, p2).Max(lpfAbsDiff(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpfAbsDiff(p0, q0)
	sum := d00.AddSaturated(d00).AddSaturated(lpfAbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpfAny(fm) {
		return false
	}
	mF4 := fm
	var mFlat archsimd.Mask8x16
	if wd >= 6 {
		var fl archsimd.Uint8x16
		if wd >= 8 {
			fl = lpfSpread(p2, p3, p0).Max(lpfSpread(q2, q3, q0))
		} else {
			fl = lpfAbsDiff(p2, p0).Max(lpfAbsDiff(q2, q0))
		}
		mFlat = archsimd.BroadcastUint8x16(1).GreaterEqual(fl.Max(maxd)).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
	}
	if lpfAny(mF4) {
		bias := archsimd.BroadcastUint8x16(128)
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		d := q0c.SubSaturated(p0c)
		f := p1c.SubSaturated(q1c).Masked(hev).AddSaturated(d).AddSaturated(d).AddSaturated(d)
		sh3 := archsimd.BroadcastInt8x16(-3)
		f1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(sh3)
		f2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(sh3)
		outer := f1.Average(archsimd.Int8x16{})
		o0 = p0c.AddSaturated(f2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(f1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}
	if wd == 6 && lpfAny(mFlat) {
		k, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.LoadUint16x8Array(&lpfBias128)
		lP2, lP1, lP0 := p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k)
		lQ0, lQ1, lQ2 := q0.MulWidenLo(k), q1.MulWidenLo(k), q2.MulWidenLo(k)
		lt := lP1.Add(lP0)
		la1, la0, lb0, lb1 := lpfFlat6Half(lP2, lP1, lP0, lQ0, lQ1, lQ2, p2.MulWidenLo(k3).Add(lt.Add(lt)).Add(lQ0.Add(bias8)))
		hP2, hP1, hP0 := p2.HiToLo().MulWidenLo(k), p1.HiToLo().MulWidenLo(k), p0.HiToLo().MulWidenLo(k)
		hQ0, hQ1, hQ2 := q0.HiToLo().MulWidenLo(k), q1.HiToLo().MulWidenLo(k), q2.HiToLo().MulWidenLo(k)
		ht := hP1.Add(hP0)
		ha1, ha0, hb0, hb1 := lpfFlat6Half(hP2, hP1, hP0, hQ0, hQ1, hQ2, p2.HiToLo().MulWidenLo(k3).Add(ht.Add(ht)).Add(hQ0.Add(bias8)))
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
	} else if wd == 8 && lpfAny(mFlat) {
		k, k2, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(64), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.LoadUint16x8Array(&lpfBias128)
		lP3, lP2, lP1, lP0 := p3.MulWidenLo(k), p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k)
		lQ0, lQ1, lQ2, lQ3 := q0.MulWidenLo(k), q1.MulWidenLo(k), q2.MulWidenLo(k), q3.MulWidenLo(k)
		la2, la1, la0, lb0, lb1, lb2 := lpfFlat8Half(lP3, lP2, lP1, lP0, lQ0, lQ1, lQ2, lQ3, p3.MulWidenLo(k3).Add(p2.MulWidenLo(k2)).Add(lP1.Add(lP0)).Add(lQ0.Add(bias8)))
		hP3, hP2, hP1, hP0 := p3.HiToLo().MulWidenLo(k), p2.HiToLo().MulWidenLo(k), p1.HiToLo().MulWidenLo(k), p0.HiToLo().MulWidenLo(k)
		hQ0, hQ1, hQ2, hQ3 := q0.HiToLo().MulWidenLo(k), q1.HiToLo().MulWidenLo(k), q2.HiToLo().MulWidenLo(k), q3.HiToLo().MulWidenLo(k)
		ha2, ha1, ha0, hb0, hb1, hb2 := lpfFlat8Half(hP3, hP2, hP1, hP0, hQ0, hQ1, hQ2, hQ3, p3.HiToLo().MulWidenLo(k3).Add(p2.HiToLo().MulWidenLo(k2)).Add(hP1.Add(hP0)).Add(hQ0.Add(bias8)))
		o2 = la2.ReshapeToUint8s().ConcatOdd(ha2.ReshapeToUint8s()).IfElse(mFlat, o2)
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
		n2 = lb2.ReshapeToUint8s().ConcatOdd(hb2.ReshapeToUint8s()).IfElse(mFlat, n2)
	}
	if wd >= 8 {
		o2.StoreArray(tv.Row16(rh - 3))
		n2.StoreArray(tv.Row16(rh + 2))
	}
	o1.StoreArray(tv.Row16(rh - 2))
	o0.StoreArray(tv.Row16(rh - 1))
	n0.StoreArray(tv.Row16(rh))
	n1.StoreArray(tv.Row16(rh + 1))
	return true
}

// lpfW4..lpfW16 select a width-specialized kernel instance, as dav1d's
// loop_filter wd macro emits lpf_16_wd{4,6,8,16}_neon from one source: each
// type's size is its filter width, so every stenciled body sees a constant
// width and drops the code of the other widths.
type (
	lpfW4  [4]byte
	lpfW6  [6]byte
	lpfW8  [8]byte
	lpfW16 [16]byte
)

type lpfWidth interface {
	lpfW4 | lpfW6 | lpfW8 | lpfW16
}

// lpf8U8 dispatches the eight row taps p3..q3 of v from row r0 (p3), column
// c0, to the width-specialized lpf8U8Rows instance (wd 4, 6 or 8; the narrower
// widths read only their inner rows).
func lpf8U8(wd int, v unsafeutil.Rows, r0, c0 int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	switch wd {
	case 8:
		return lpf8U8Rows[lpfW8](unsafeutil.BlockAt[lpfR8](v, r0, c0), eV, iV, hV, m4, mGt4)
	case 6:
		return lpf8U8Rows[lpfW6](unsafeutil.BlockAt[lpfR6](v, r0+1, c0), eV, iV, hV, m4, mGt4)
	default:
		return lpf8U8Rows[lpfW4](unsafeutil.BlockAt[lpfR4](v, r0+2, c0), eV, iV, hV, m4, mGt4)
	}
}
