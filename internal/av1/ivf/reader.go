// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package ivf

const (
	FileHeaderSize  = 32
	FrameHeaderSize = 12
)

type Header struct {
	Version     uint16
	HeaderSize  uint16
	FourCC      [4]byte
	Width       uint16
	Height      uint16
	TimebaseNum uint32
	TimebaseDen uint32
	FrameCount  uint32
	Unused      uint32
}

type Frame struct {
	Payload   []byte
	Index     uint32
	Offset    uint32
	Timestamp uint64
}

type Iterator struct {
	src    []byte
	header Header
	off    int
	index  uint32
}

func ParseHeader(src []byte) (Header, int, error) {
	if len(src) < FileHeaderSize {
		return Header{}, 0, ErrShortHeader
	}
	if src[0] != 'D' || src[1] != 'K' || src[2] != 'I' || src[3] != 'F' {
		return Header{}, 0, ErrInvalidSignature
	}
	header := Header{
		Version:     le16(src[4:]),
		HeaderSize:  le16(src[6:]),
		FourCC:      [4]byte{src[8], src[9], src[10], src[11]},
		Width:       le16(src[12:]),
		Height:      le16(src[14:]),
		TimebaseNum: le32(src[16:]),
		TimebaseDen: le32(src[20:]),
		FrameCount:  le32(src[24:]),
		Unused:      le32(src[28:]),
	}
	if header.HeaderSize != FileHeaderSize || header.Version != 0 {
		return Header{}, 0, ErrUnsupportedHeader
	}
	if header.FourCC != ([4]byte{'A', 'V', '0', '1'}) {
		return Header{}, 0, ErrUnsupportedCodec
	}
	return header, FileHeaderSize, nil
}

func NewIterator(src []byte) (Iterator, error) {
	header, n, err := ParseHeader(src)
	if err != nil {
		return Iterator{}, err
	}
	return Iterator{
		src:    src,
		header: header,
		off:    n,
	}, nil
}

func (it *Iterator) Header() Header {
	return it.header
}

func (it *Iterator) Next() (Frame, bool, error) {
	if it.off == len(it.src) {
		return Frame{}, false, nil
	}
	if len(it.src)-it.off < FrameHeaderSize {
		return Frame{}, false, ErrShortFrameHeader
	}
	offset := it.off
	size := int(le32(it.src[it.off:]))
	timestamp := le64(it.src[it.off+4:])
	start := it.off + FrameHeaderSize
	end := start + size
	if end < start || end > len(it.src) {
		return Frame{}, false, ErrShortFramePayload
	}
	frame := Frame{
		Payload:   it.src[start:end],
		Index:     it.index,
		Offset:    uint32(offset),
		Timestamp: timestamp,
	}
	it.off = end
	it.index++
	return frame, true, nil
}

func le16(src []byte) uint16 {
	return uint16(src[0]) | uint16(src[1])<<8
}

func le32(src []byte) uint32 {
	return uint32(src[0]) | uint32(src[1])<<8 | uint32(src[2])<<16 | uint32(src[3])<<24
}

func le64(src []byte) uint64 {
	return uint64(le32(src)) | uint64(le32(src[4:]))<<32
}
