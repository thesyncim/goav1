package rtp

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// Element is an OBU element in an AV1 RTP payload.
//
// Data aliases the RTP payload input when returned by Iterator. For outbound
// packetization, Data is copied into the caller-provided output buffer.
type Element struct {
	Data []byte

	Index              int
	ContinuesPrevious  bool
	ContinuesNext      bool
	InferredLastLength bool
}

// Iterator iterates OBU elements in one AV1 RTP payload without allocating.
type Iterator struct {
	payload []byte
	header  AggregationHeader
	off     int
	index   int
	done    bool
}

func NewIterator(payload []byte) (Iterator, error) {
	header, n, err := ParseAggregationHeader(payload)
	if err != nil {
		return Iterator{}, err
	}
	if n == len(payload) {
		return Iterator{}, ErrShortPayload
	}
	return Iterator{
		payload: payload,
		header:  header,
		off:     n,
	}, nil
}

func (it *Iterator) Header() AggregationHeader {
	return it.header
}

func (it *Iterator) Next() (Element, bool, error) {
	if it.done {
		return Element{}, false, nil
	}
	if it.off >= len(it.payload) {
		it.done = true
		return Element{}, false, nil
	}

	index := it.index
	count := int(it.header.ElementCount)
	if count != 0 && index >= count {
		return Element{}, false, ErrInvalidElementCount
	}

	var data []byte
	inferred := false
	if count != 0 && index == count-1 {
		data = it.payload[it.off:]
		it.off = len(it.payload)
		it.done = true
		inferred = true
	} else {
		size, sizeLen, err := bitstream.ReadLEB128(it.payload[it.off:])
		if err != nil {
			return Element{}, false, err
		}
		start := it.off + sizeLen
		end := start + int(size)
		if end < start || end > len(it.payload) {
			return Element{}, false, ErrShortPayload
		}
		data = it.payload[start:end]
		it.off = end
		if count == 0 && it.off == len(it.payload) {
			it.done = true
		}
	}

	if len(data) == 0 {
		return Element{}, false, ErrZeroLengthElement
	}

	elem := Element{
		Data:               data,
		Index:              index,
		ContinuesPrevious:  it.header.ContinuesPrevious && index == 0,
		ContinuesNext:      it.header.ContinuesNext && it.off == len(it.payload),
		InferredLastLength: inferred,
	}
	it.index++
	return elem, true, nil
}

// PutPayload writes one AV1 RTP payload into dst.
//
// If header.ElementCount is zero, every element is preceded by a leb128 length.
// If it is 1..3, it must match len(elements), and the last element length is
// inferred from the payload size.
func PutPayload(dst []byte, header AggregationHeader, elements []Element) (int, error) {
	if len(elements) == 0 {
		return 0, ErrInvalidElementCount
	}
	if header.ElementCount != 0 && int(header.ElementCount) != len(elements) {
		return 0, ErrInvalidElementCount
	}
	off, err := PutAggregationHeader(dst, header)
	if err != nil {
		return 0, err
	}

	for i := range elements {
		data := elements[i].Data
		if len(data) == 0 {
			return 0, ErrZeroLengthElement
		}

		writeLength := header.ElementCount == 0 || i != len(elements)-1
		if writeLength {
			n, err := bitstream.PutLEB128(dst[off:], uint32(len(data)))
			if err != nil {
				return 0, err
			}
			off += n
		}

		if len(dst)-off < len(data) {
			return 0, ErrShortBuffer
		}
		off += copy(dst[off:], data)
	}

	return off, nil
}
