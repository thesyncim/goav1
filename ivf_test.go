package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicIVFHeaderAndIterator(t *testing.T) {
	src := appendPublicIVF(nil, 16, 18, 30, 1, []publicIVFFrame{
		{timestamp: 0, payload: []byte{0x12, 0x00}},
		{timestamp: 3, payload: []byte{0x30, 0xaa, 0xbb}},
	})
	header, consumed, err := av1.ParseIVFHeader(src)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != av1.IVFFileHeaderSize || header.Width != 16 || header.Height != 18 {
		t.Fatalf("header=%+v consumed=%d", header, consumed)
	}
	if header.TimebaseNum != 30 || header.TimebaseDen != 1 || header.FrameCount != 2 {
		t.Fatalf("header timing/count=%+v", header)
	}

	it, err := av1.NewIVFIterator(src)
	if err != nil {
		t.Fatal(err)
	}
	if it.Header() != header {
		t.Fatalf("iterator header=%+v want %+v", it.Header(), header)
	}
	first, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first ok=%v err=%v", ok, err)
	}
	if first.Index != 0 || first.Offset != av1.IVFFileHeaderSize || first.Timestamp != 0 || string(first.Payload) != string([]byte{0x12, 0x00}) {
		t.Fatalf("first=%+v", first)
	}
	second, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("second ok=%v err=%v", ok, err)
	}
	if second.Index != 1 || second.Offset != av1.IVFFileHeaderSize+av1.IVFFrameHeaderSize+2 || second.Timestamp != 3 || string(second.Payload) != string([]byte{0x30, 0xaa, 0xbb}) {
		t.Fatalf("second=%+v", second)
	}
	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected third frame")
	}
}

func TestPublicIVFRejectsInvalid(t *testing.T) {
	_, _, err := av1.ParseIVFHeader([]byte("DKIF"))
	if !errors.Is(err, av1.ErrIVFShortHeader) {
		t.Fatalf("ParseIVFHeader err=%v want %v", err, av1.ErrIVFShortHeader)
	}

	src := appendPublicIVF(nil, 16, 16, 30, 1, nil)
	src[8] = 'V'
	src[9] = 'P'
	src[10] = '9'
	src[11] = '0'
	_, err = av1.NewIVFIterator(src)
	if !errors.Is(err, av1.ErrIVFUnsupportedCodec) {
		t.Fatalf("NewIVFIterator err=%v want %v", err, av1.ErrIVFUnsupportedCodec)
	}

	truncatedFrame := appendPublicIVF(nil, 16, 16, 30, 1, nil)
	truncatedFrame = append(truncatedFrame, 0x01)
	it, err := av1.NewIVFIterator(truncatedFrame)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = it.Next()
	if !errors.Is(err, av1.ErrIVFShortFrameHeader) {
		t.Fatalf("Next err=%v want %v", err, av1.ErrIVFShortFrameHeader)
	}
}

func TestPublicIVFIteratorAllocs(t *testing.T) {
	src := appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{{payload: []byte{0x12, 0x00}}})
	allocs := testing.AllocsPerRun(1000, func() {
		it, err := av1.NewIVFIterator(src)
		if err != nil {
			t.Fatal(err)
		}
		frame, ok, err := it.Next()
		if err != nil || !ok || len(frame.Payload) == 0 {
			t.Fatalf("frame=%+v ok=%v err=%v", frame, ok, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("public IVF iterator allocated: %f", allocs)
	}
}

type publicIVFFrame struct {
	timestamp uint64
	payload   []byte
}

func appendPublicIVF(dst []byte, width uint16, height uint16, timebaseNum uint32, timebaseDen uint32, frames []publicIVFFrame) []byte {
	dst = append(dst,
		'D', 'K', 'I', 'F',
		0, 0,
		byte(av1.IVFFileHeaderSize), 0,
		'A', 'V', '0', '1',
		byte(width), byte(width>>8),
		byte(height), byte(height>>8),
		byte(timebaseNum), byte(timebaseNum>>8), byte(timebaseNum>>16), byte(timebaseNum>>24),
		byte(timebaseDen), byte(timebaseDen>>8), byte(timebaseDen>>16), byte(timebaseDen>>24),
		byte(len(frames)), byte(len(frames)>>8), byte(len(frames)>>16), byte(len(frames)>>24),
		0, 0, 0, 0,
	)
	for _, frame := range frames {
		size := uint32(len(frame.payload))
		dst = append(dst,
			byte(size), byte(size>>8), byte(size>>16), byte(size>>24),
			byte(frame.timestamp), byte(frame.timestamp>>8), byte(frame.timestamp>>16), byte(frame.timestamp>>24),
			byte(frame.timestamp>>32), byte(frame.timestamp>>40), byte(frame.timestamp>>48), byte(frame.timestamp>>56),
		)
		dst = append(dst, frame.payload...)
	}
	return dst
}
