// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "simd/archsimd"

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
			cdefFilterSecondaryOnly16xSIMD(dst[dstOffset:], dstStride, input, srcOffset, blockHeight, str, sh)
			i += groupEntries
			continue
		}
		if blockWidth == 8 {
			cdefFilterSecondaryOnly8xSIMD(dst[dstOffset:], dstStride, input, srcOffset, blockHeight, str, sh)
		} else {
			cdefFilterSecondaryOnly4xSIMD(dst[dstOffset:], dstStride, input, srcOffset, blockHeight, str, sh)
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
	return cdefByteConstrain(t, x, strength, shift)
}

func cdefRoundSecondaryByteSIMD(sum archsimd.Int8x16) archsimd.Int8x16 {
	return sum.Add(sum.ShiftAllRight(7)).Add(archsimd.BroadcastInt8x16(8)).ShiftAllRight(4)
}

// The secondary-only taps sit at x+-1, x+-2 and y+-1, y+-2 of each output
// sample, so a row group's taps lie in one window of the tap buffer that
// starts two rows and two columns before the group: one bounds check per
// group, every tap a constant offset inside it. cdefSecWindowArgs bounds the
// origin, height and stride so that check is one compare.
const (
	cdefSecReach = 2*BStride + 2
	cdefSecWin16 = 4*BStride + 4 + 16 // one 16-wide row
	cdefSecWin8  = 5*BStride + 4 + 8  // two 8-wide rows
	cdefSecWin4  = 7*BStride + 4 + 4  // four 4-wide rows
)

// cdefSecWindowArgs panics unless the block origin leaves room for the taps
// before it and the height and destination stride are in range.
func cdefSecWindowArgs(so, height, dstStride int) {
	if so < cdefSecReach || so > 1<<24 || uint(height) > 1<<10 || uint(dstStride) > 1<<30 {
		panic("cdef: secondary-only block outside the tap buffer")
	}
}

// cdefSecLoad16 loads the 16 tap bytes at (dx, dy) relative to the window's
// output row.
func cdefSecLoad16(w *[cdefSecWin16]byte, dx, dy int) archsimd.Uint8x16 {
	o := (dy+2)*BStride + dx + 2
	return archsimd.LoadUint8x16Array((*[16]uint8)(w[o : o+16]))
}

// cdefSecLoad8x2 packs the 8 tap bytes at (dx, dy) and (dx, dy+1).
func cdefSecLoad8x2(w *[cdefSecWin8]byte, dx, dy int) archsimd.Uint8x16 {
	o := (dy+2)*BStride + dx + 2
	var packed archsimd.Uint64x2
	packed = packed.SetElem(0, cdefSecLE64(w[o:o+8]))
	packed = packed.SetElem(1, cdefSecLE64(w[o+BStride:o+BStride+8]))
	return packed.ReshapeToUint8s()
}

// cdefSecLoad4x4 packs the 4 tap bytes at (dx, dy) .. (dx, dy+3).
func cdefSecLoad4x4(w *[cdefSecWin4]byte, dx, dy int) archsimd.Uint8x16 {
	o := (dy+2)*BStride + dx + 2
	var packed archsimd.Uint32x4
	packed = packed.SetElem(0, cdefSecLE32(w[o:o+4]))
	packed = packed.SetElem(1, cdefSecLE32(w[o+BStride:o+BStride+4]))
	packed = packed.SetElem(2, cdefSecLE32(w[o+2*BStride:o+2*BStride+4]))
	packed = packed.SetElem(3, cdefSecLE32(w[o+3*BStride:o+3*BStride+4]))
	return packed.ReshapeToUint8s()
}

func cdefSecLE64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func cdefSecLE32(b []byte) uint32 {
	_ = b[3]
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func cdefSecPut64(b []byte, v uint64) {
	_ = b[7]
	b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

func cdefSecPut32(b []byte, v uint32) {
	_ = b[3]
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func cdefFilterSecondaryOnly16xSIMD(dst []byte, dstStride int, input []byte, so int, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	cdefSecWindowArgs(so, height, dstStride)
	base := so - cdefSecReach
	for y := 0; y < height; y++ {
		o := base + y*BStride
		w := (*[cdefSecWin16]byte)(input[o : o+cdefSecWin16])
		x := cdefSecLoad16(w, 0, 0)
		// The four weight-2 taps sum before doubling (2a+2b+2c+2d wraps to the
		// same int8 lanes as 2(a+b+c+d)), and every tap is consumed on the line
		// that loads it, so no inlined helper call leaves an inline-mark NOP.
		d := cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 1, 0), x, strength, shift).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, -1, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, 1), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, -1), x, strength, shift))
		sum := d.Add(d).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, -2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, 2), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, -2), x, strength, shift))
		pixels := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits())
		do := y * dstStride
		pixels.StoreArray((*[16]uint8)(dst[do : do+16]))
	}
}

func cdefFilterSecondaryOnly8xSIMD(dst []byte, dstStride int, input []byte, so int, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	cdefSecWindowArgs(so, height, dstStride)
	base := so - cdefSecReach
	// One iteration per row pair, counted by pair (a step-2 counter keeps a
	// second check per window).
	for i := 0; i < height>>1; i++ {
		y := 2 * i
		o := base + y*BStride
		w := (*[cdefSecWin8]byte)(input[o : o+cdefSecWin8])
		x := cdefSecLoad8x2(w, 0, 0)
		// The four weight-2 taps sum before doubling (2a+2b+2c+2d wraps to the
		// same int8 lanes as 2(a+b+c+d)), and every tap is consumed on the line
		// that loads it, so no inlined helper call leaves an inline-mark NOP.
		d := cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 1, 0), x, strength, shift).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, -1, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, 1), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, -1), x, strength, shift))
		sum := d.Add(d).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, -2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, 2), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, -2), x, strength, shift))
		packed := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits()).ReshapeToUint64s()
		do := y * dstStride
		cdefSecPut64(dst[do:do+8], packed.GetElem(0))
		cdefSecPut64(dst[do+dstStride:do+dstStride+8], packed.GetElem(1))
	}
}

func cdefFilterSecondaryOnly4xSIMD(dst []byte, dstStride int, input []byte, so int, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	cdefSecWindowArgs(so, height, dstStride)
	base := so - cdefSecReach
	for i := 0; i < height>>2; i++ {
		y := 4 * i
		o := base + y*BStride
		w := (*[cdefSecWin4]byte)(input[o : o+cdefSecWin4])
		x := cdefSecLoad4x4(w, 0, 0)
		// The four weight-2 taps sum before doubling (2a+2b+2c+2d wraps to the
		// same int8 lanes as 2(a+b+c+d)), and every tap is consumed on the line
		// that loads it, so no inlined helper call leaves an inline-mark NOP.
		d := cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 1, 0), x, strength, shift).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, -1, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, 1), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, -1), x, strength, shift))
		sum := d.Add(d).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, -2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, 2), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, -2), x, strength, shift))
		packed := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits()).ReshapeToUint32s()
		do := y * dstStride
		cdefSecPut32(dst[do:do+4], packed.GetElem(0))
		cdefSecPut32(dst[do+dstStride:do+dstStride+4], packed.GetElem(1))
		cdefSecPut32(dst[do+2*dstStride:do+2*dstStride+4], packed.GetElem(2))
		cdefSecPut32(dst[do+3*dstStride:do+3*dstStride+4], packed.GetElem(3))
	}
}
