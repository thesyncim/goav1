// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package bitstream

import "errors"

var (
	ErrNotEnoughBits       = errors.New("bitstream: not enough bits")
	ErrInvalidBitCount     = errors.New("bitstream: invalid bit count")
	ErrInvalidTrailingBits = errors.New("bitstream: invalid trailing bits")
	ErrUVLCOverflow        = errors.New("bitstream: uvlc overflow")
)

// Reader is an allocation-free MSB-first bit reader.
type Reader struct {
	src []byte
	bit int
}

func NewReader(src []byte) Reader {
	return Reader{src: src}
}

func (r *Reader) BitsRead() int {
	return r.bit
}

func (r *Reader) BitsRemaining() int {
	return len(r.src)*8 - r.bit
}

func (r *Reader) ByteAligned() bool {
	return r.bit&7 == 0
}

func (r *Reader) ReadBit() (uint8, error) {
	if r.BitsRemaining() < 1 {
		return 0, ErrNotEnoughBits
	}
	byteIndex := r.bit >> 3
	shift := 7 - (r.bit & 7)
	r.bit++
	return (r.src[byteIndex] >> shift) & 1, nil
}

func (r *Reader) ReadBool() (bool, error) {
	bit, err := r.ReadBit()
	return bit != 0, err
}

func (r *Reader) ReadBits(n uint8) (uint64, error) {
	if n > 64 {
		return 0, ErrInvalidBitCount
	}
	if n == 0 {
		return 0, nil
	}
	if r.BitsRemaining() < int(n) {
		return 0, ErrNotEnoughBits
	}

	var v uint64
	bits := int(n)
	byteIndex := r.bit >> 3
	bitInByte := r.bit & 7

	if bitInByte != 0 {
		take := 8 - bitInByte
		if take > bits {
			take = bits
		}
		shift := 8 - bitInByte - take
		mask := byte((1 << take) - 1)
		v = uint64((r.src[byteIndex] >> shift) & mask)
		bits -= take
		byteIndex++
	}

	for bits >= 8 {
		v = (v << 8) | uint64(r.src[byteIndex])
		bits -= 8
		byteIndex++
	}

	if bits > 0 {
		v = (v << uint(bits)) | uint64(r.src[byteIndex]>>(8-bits))
	}

	r.bit += int(n)
	return v, nil
}

func (r *Reader) SkipBits(n int) error {
	if n < 0 {
		return ErrInvalidBitCount
	}
	if r.BitsRemaining() < n {
		return ErrNotEnoughBits
	}
	r.bit += n
	return nil
}

// ReadUVLC reads the AV1 uvlc() syntax element.
func (r *Reader) ReadUVLC() (uint32, error) {
	leadingZeros := 0
	for {
		bit, err := r.ReadBit()
		if err != nil {
			return 0, err
		}
		if bit == 1 {
			break
		}
		leadingZeros++
		if leadingZeros >= 32 {
			return 0, ErrUVLCOverflow
		}
	}

	if leadingZeros == 0 {
		return 0, nil
	}

	value, err := r.ReadBits(uint8(leadingZeros))
	if err != nil {
		return 0, err
	}
	return uint32((uint64(1) << uint(leadingZeros)) - 1 + value), nil
}

// ReadTrailingBits validates AV1 trailing bits: one 1 bit followed by zero bits
// until byte alignment, then optional zero padding bytes.
func (r *Reader) ReadTrailingBits() error {
	one, err := r.ReadBit()
	if err != nil {
		return err
	}
	if one != 1 {
		return ErrInvalidTrailingBits
	}

	for !r.ByteAligned() {
		bit, err := r.ReadBit()
		if err != nil {
			return err
		}
		if bit != 0 {
			return ErrInvalidTrailingBits
		}
	}

	for r.BitsRemaining() > 0 {
		b, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		if b != 0 {
			return ErrInvalidTrailingBits
		}
	}
	return nil
}
