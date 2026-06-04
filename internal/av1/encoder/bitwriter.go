package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

type bitWriter struct {
	dst    []byte
	bit    int
	sizing bool
}

func newBitWriter(dst []byte) bitWriter {
	return bitWriter{dst: dst}
}

func newSizingBitWriter() bitWriter {
	return bitWriter{sizing: true}
}

func (w *bitWriter) bitsWritten() int {
	return w.bit
}

func (w *bitWriter) bytesWritten() int {
	return (w.bit + 7) >> 3
}

func (w *bitWriter) byteAligned() bool {
	return w.bit&7 == 0
}

func (w *bitWriter) writeBit(bit uint8) error {
	if bit > 1 {
		return bitstream.ErrInvalidBitCount
	}
	if w.sizing {
		w.bit++
		return nil
	}

	byteIndex := w.bit >> 3
	if byteIndex >= len(w.dst) {
		return bitstream.ErrShortBuffer
	}

	shift := uint(7 - (w.bit & 7))
	if shift == 7 {
		w.dst[byteIndex] = 0
	}
	if bit != 0 {
		w.dst[byteIndex] |= 1 << shift
	} else {
		w.dst[byteIndex] &^= 1 << shift
	}
	w.bit++
	return nil
}

func (w *bitWriter) writeBool(value bool) error {
	if value {
		return w.writeBit(1)
	}
	return w.writeBit(0)
}

func (w *bitWriter) writeBits(value uint64, n uint8) error {
	if n > 64 {
		return bitstream.ErrInvalidBitCount
	}
	for i := int(n) - 1; i >= 0; i-- {
		if err := w.writeBit(uint8((value >> uint(i)) & 1)); err != nil {
			return err
		}
	}
	return nil
}

func (w *bitWriter) writeTrailingBits() error {
	if err := w.writeBit(1); err != nil {
		return err
	}
	for !w.byteAligned() {
		if err := w.writeBit(0); err != nil {
			return err
		}
	}
	return nil
}
