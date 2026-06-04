package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

func readerBitsRead(r *bitstream.Reader) (uint16, error) {
	bits := r.BitsRead()
	if bits < 0 || bits > int(^uint16(0)) {
		return 0, bitstream.ErrInvalidBitCount
	}
	return uint16(bits), nil
}

func skipBitsRead(r *bitstream.Reader, bits uint16) error {
	return r.SkipBits(int(bits))
}
