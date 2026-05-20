package bitstream

import "errors"

const (
	// MaxLEB128Bytes is the AV1 limit for leb128 syntax elements.
	MaxLEB128Bytes = 8

	// MaxLEB128Value is the maximum value accepted by AV1 leb128 syntax.
	MaxLEB128Value = uint64(1<<32 - 1)
)

var (
	ErrShortBuffer    = errors.New("bitstream: short buffer")
	ErrLEB128Overflow = errors.New("bitstream: leb128 overflow")
	ErrShortLEB128    = errors.New("bitstream: truncated leb128")
)

// ReadLEB128 reads an AV1 leb128 value.
//
// AV1 caps leb128 encodings at 8 bytes and values at uint32 range. The function
// performs no allocation and returns the number of bytes consumed on success.
func ReadLEB128(src []byte) (value uint32, n int, err error) {
	var v uint64
	for i := 0; i < MaxLEB128Bytes; i++ {
		if i >= len(src) {
			return 0, 0, ErrShortLEB128
		}

		b := src[i]
		v |= uint64(b&0x7f) << uint(i*7)
		if b&0x80 == 0 {
			if v > MaxLEB128Value {
				return 0, 0, ErrLEB128Overflow
			}
			return uint32(v), i + 1, nil
		}
	}

	return 0, 0, ErrLEB128Overflow
}

// LEB128Len returns the number of bytes needed to encode value as AV1 leb128.
func LEB128Len(value uint32) int {
	n := 1
	for value >= 0x80 {
		value >>= 7
		n++
	}
	return n
}

// PutLEB128 writes value as AV1 leb128 into dst.
func PutLEB128(dst []byte, value uint32) (int, error) {
	need := LEB128Len(value)
	if len(dst) < need {
		return 0, ErrShortBuffer
	}

	i := 0
	for value >= 0x80 {
		dst[i] = byte(value) | 0x80
		value >>= 7
		i++
	}
	dst[i] = byte(value)
	return need, nil
}

// AppendLEB128 appends value as AV1 leb128 to dst.
//
// Callers on hot paths should reserve capacity first or use PutLEB128.
func AppendLEB128(dst []byte, value uint32) []byte {
	for value >= 0x80 {
		dst = append(dst, byte(value)|0x80)
		value >>= 7
	}
	return append(dst, byte(value))
}
