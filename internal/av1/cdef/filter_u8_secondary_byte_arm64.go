// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

func u8SecondaryOnlyByteInteriorAvailable() bool { return true }

func filterFrameBlocksU8SecondaryOnlyByteInterior(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, blockWidth int, blockHeight int, strength int, shift int) bool {
	str := archsimd.BroadcastUint8x16(uint8(strength))
	sh := archsimd.BroadcastInt8x16(-int8(shift))
	for i := 0; i < len(blocks); {
		block := blocks[i]
		srcOffset := inputOrigin + (int(block.BY)*blockHeight)*BStride + int(block.BX)*blockWidth
		dstOffset := (int(block.BY)*blockHeight)*dstStride + int(block.BX)*blockWidth
		groupEntries := secondaryOnlyByteGroupEntries(blocks, i, blockWidth)
		if groupEntries > 1 {
			cdefFilterSecondaryOnly16xSIMD(dst[dstOffset:], dstStride, input[srcOffset:], blockHeight, str, sh)
			i += groupEntries
			continue
		}
		if blockWidth == 8 {
			cdefFilterSecondaryOnly8xSIMD(dst[dstOffset:], dstStride, input[srcOffset:], blockHeight, str, sh)
		} else {
			cdefFilterSecondaryOnly4xSIMD(dst[dstOffset:], dstStride, input[srcOffset:], blockHeight, str, sh)
		}
		i++
	}
	return true
}

// secondaryOnlyByteGroupEntries returns the exact number of consecutive list
// entries that cover one 16-pixel row strip. It never searches ahead, sorts,
// or fills a missing block; a failure leaves the current entry for the
// original single-block kernel.
func secondaryOnlyByteGroupEntries(blocks []BlockPosition, index int, blockWidth int) int {
	groupEntries := 0
	switch blockWidth {
	case 8:
		groupEntries = 2
	case 4:
		groupEntries = 4
	default:
		return 1
	}
	if index < 0 || index+groupEntries > len(blocks) {
		return 1
	}
	first := blocks[index]
	for offset := 1; offset < groupEntries; offset++ {
		next := blocks[index+offset]
		if next.BY != first.BY || int(next.BX) != int(first.BX)+offset {
			return 1
		}
	}
	return groupEntries
}

func cdefConstrainSecondaryByteSIMD(t, x, strength archsimd.Uint8x16, shift archsimd.Int8x16) archsimd.Int8x16 {
	diff := t.Max(x).Sub(t.Min(x))
	limit := strength.SubSaturated(diff.Shift(shift))
	mag := limit.Min(diff).BitsToInt8()
	return mag.Neg().IfElse(x.Greater(t), mag)
}

func cdefRoundSecondaryByteSIMD(sum archsimd.Int8x16) archsimd.Int8x16 {
	return sum.Add(sum.ShiftAllRight(7)).Add(archsimd.BroadcastInt8x16(8)).ShiftAllRight(4)
}

// cdefLoadSecondary8x2SIMD reads exactly two 8-byte rows and puts them in
// the low/high halves of a single vector.
func cdefLoadSecondary8x2SIMD(src unsafe.Pointer, rowStride int, dx int, dy int) archsimd.Uint8x16 {
	row0 := unsafe.Add(src, dy*BStride+dx)
	row1 := unsafe.Add(row0, rowStride)
	var packed archsimd.Uint64x2
	packed = packed.SetElem(0, binary.LittleEndian.Uint64(unsafe.Slice((*byte)(row0), 8)))
	packed = packed.SetElem(1, binary.LittleEndian.Uint64(unsafe.Slice((*byte)(row1), 8)))
	return packed.ReshapeToUint8s()
}

// cdefLoadSecondary4x4SIMD reads exactly four 4-byte rows and packs them in
// row order into one vector.
func cdefLoadSecondary4x4SIMD(src unsafe.Pointer, rowStride int, dx int, dy int) archsimd.Uint8x16 {
	row0 := unsafe.Add(src, dy*BStride+dx)
	var packed archsimd.Uint32x4
	packed = packed.SetElem(0, binary.LittleEndian.Uint32(unsafe.Slice((*byte)(row0), 4)))
	packed = packed.SetElem(1, binary.LittleEndian.Uint32(unsafe.Slice((*byte)(unsafe.Add(row0, rowStride)), 4)))
	packed = packed.SetElem(2, binary.LittleEndian.Uint32(unsafe.Slice((*byte)(unsafe.Add(row0, 2*rowStride)), 4)))
	packed = packed.SetElem(3, binary.LittleEndian.Uint32(unsafe.Slice((*byte)(unsafe.Add(row0, 3*rowStride)), 4)))
	return packed.ReshapeToUint8s()
}

func cdefStoreSecondary8x2SIMD(dst unsafe.Pointer, dstStride int, pixels archsimd.Uint8x16) {
	packed := pixels.ReshapeToUint64s()
	binary.LittleEndian.PutUint64(unsafe.Slice((*byte)(dst), 8), packed.GetElem(0))
	binary.LittleEndian.PutUint64(unsafe.Slice((*byte)(unsafe.Add(dst, dstStride)), 8), packed.GetElem(1))
}

func cdefStoreSecondary4x4SIMD(dst unsafe.Pointer, dstStride int, pixels archsimd.Uint8x16) {
	packed := pixels.ReshapeToUint32s()
	binary.LittleEndian.PutUint32(unsafe.Slice((*byte)(dst), 4), packed.GetElem(0))
	binary.LittleEndian.PutUint32(unsafe.Slice((*byte)(unsafe.Add(dst, dstStride)), 4), packed.GetElem(1))
	binary.LittleEndian.PutUint32(unsafe.Slice((*byte)(unsafe.Add(dst, 2*dstStride)), 4), packed.GetElem(2))
	binary.LittleEndian.PutUint32(unsafe.Slice((*byte)(unsafe.Add(dst, 3*dstStride)), 4), packed.GetElem(3))
}

func cdefLoadSecondary16SIMD(src unsafe.Pointer, dx int, dy int) archsimd.Uint8x16 {
	ptr := unsafe.Add(src, dy*BStride+dx)
	return archsimd.LoadUint8x16Array((*[16]uint8)(ptr))
}

func cdefFilterSecondaryOnly16xSIMD(dst []byte, dstStride int, input []byte, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	src := unsafe.Pointer(unsafe.SliceData(input))
	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	for y := 0; y < height; y++ {
		x := cdefLoadSecondary16SIMD(src, 0, y)
		var sum archsimd.Int8x16
		t := cdefLoadSecondary16SIMD(src, 1, y)
		d := cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary16SIMD(src, -1, y)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary16SIMD(src, 0, y+1)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary16SIMD(src, 0, y-1)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary16SIMD(src, 2, y)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary16SIMD(src, -2, y)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary16SIMD(src, 0, y+2)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary16SIMD(src, 0, y-2)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		pixels := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits())
		dstPtrRow := unsafe.Add(dstPtr, y*dstStride)
		pixels.StoreArray((*[16]uint8)(dstPtrRow))
	}
}

func cdefFilterSecondaryOnly8xSIMD(dst []byte, dstStride int, input []byte, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	src := unsafe.Pointer(unsafe.SliceData(input))
	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	for y := 0; y < height; y += 2 {
		x := cdefLoadSecondary8x2SIMD(src, BStride, 0, y)
		var sum archsimd.Int8x16
		t := cdefLoadSecondary8x2SIMD(src, BStride, 1, y)
		d := cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary8x2SIMD(src, BStride, -1, y)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary8x2SIMD(src, BStride, 0, y+1)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary8x2SIMD(src, BStride, 0, y-1)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary8x2SIMD(src, BStride, 2, y)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary8x2SIMD(src, BStride, -2, y)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary8x2SIMD(src, BStride, 0, y+2)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary8x2SIMD(src, BStride, 0, y-2)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		delta := cdefRoundSecondaryByteSIMD(sum)
		cdefStoreSecondary8x2SIMD(unsafe.Add(dstPtr, y*dstStride), dstStride, x.Add(delta.ToBits()))
	}
}

func cdefFilterSecondaryOnly4xSIMD(dst []byte, dstStride int, input []byte, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	src := unsafe.Pointer(unsafe.SliceData(input))
	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	for y := 0; y < height; y += 4 {
		x := cdefLoadSecondary4x4SIMD(src, BStride, 0, y)
		var sum archsimd.Int8x16
		t := cdefLoadSecondary4x4SIMD(src, BStride, 1, y)
		d := cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary4x4SIMD(src, BStride, -1, y)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary4x4SIMD(src, BStride, 0, y+1)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary4x4SIMD(src, BStride, 0, y-1)
		d = cdefConstrainSecondaryByteSIMD(t, x, strength, shift)
		sum = sum.Add(d.Add(d))
		t = cdefLoadSecondary4x4SIMD(src, BStride, 2, y)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary4x4SIMD(src, BStride, -2, y)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary4x4SIMD(src, BStride, 0, y+2)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		t = cdefLoadSecondary4x4SIMD(src, BStride, 0, y-2)
		sum = sum.Add(cdefConstrainSecondaryByteSIMD(t, x, strength, shift))
		delta := cdefRoundSecondaryByteSIMD(sum)
		cdefStoreSecondary4x4SIMD(unsafe.Add(dstPtr, y*dstStride), dstStride, x.Add(delta.ToBits()))
	}
}
