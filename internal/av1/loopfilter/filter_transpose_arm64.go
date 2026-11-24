// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"simd/archsimd"
)

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
