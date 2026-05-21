package ivf

import (
	"errors"
	"testing"
)

func TestParseHeader(t *testing.T) {
	src := appendTestIVF(nil, 16, 18, 30, 1, nil)
	header, consumed, err := ParseHeader(src)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != FileHeaderSize {
		t.Fatalf("consumed=%d want %d", consumed, FileHeaderSize)
	}
	if header.Width != 16 || header.Height != 18 {
		t.Fatalf("size=%dx%d", header.Width, header.Height)
	}
	if header.TimebaseNum != 30 || header.TimebaseDen != 1 {
		t.Fatalf("timebase=%d/%d", header.TimebaseNum, header.TimebaseDen)
	}
}

func TestIterator(t *testing.T) {
	src := appendTestIVF(nil, 16, 16, 30, 1, []testFrame{
		{timestamp: 0, payload: []byte{0x12, 0x00}},
		{timestamp: 3, payload: []byte{0x30, 0xaa, 0xbb}},
	})
	it, err := NewIterator(src)
	if err != nil {
		t.Fatal(err)
	}
	first, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first ok=%v err=%v", ok, err)
	}
	if first.Index != 0 || first.Offset != FileHeaderSize || first.Timestamp != 0 || string(first.Payload) != string([]byte{0x12, 0x00}) {
		t.Fatalf("first=%+v", first)
	}
	second, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("second ok=%v err=%v", ok, err)
	}
	if second.Index != 1 || second.Offset != FileHeaderSize+FrameHeaderSize+2 || second.Timestamp != 3 || string(second.Payload) != string([]byte{0x30, 0xaa, 0xbb}) {
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

func TestRejectsInvalidIVF(t *testing.T) {
	tests := []struct {
		name string
		src  []byte
		err  error
	}{
		{name: "short header", src: []byte("DKIF"), err: ErrShortHeader},
		{name: "signature", src: patchIVFByte(appendTestIVF(nil, 16, 16, 30, 1, nil), 0, 'B'), err: ErrInvalidSignature},
		{name: "version", src: patchIVFHeader(appendTestIVF(nil, 16, 16, 30, 1, nil), 4, 1), err: ErrUnsupportedHeader},
		{name: "header size", src: patchIVFHeader(appendTestIVF(nil, 16, 16, 30, 1, nil), 6, 40), err: ErrUnsupportedHeader},
		{name: "codec", src: patchIVFCodec(appendTestIVF(nil, 16, 16, 30, 1, nil), [4]byte{'V', 'P', '9', '0'}), err: ErrUnsupportedCodec},
		{name: "short frame header", src: append(appendTestIVF(nil, 16, 16, 30, 1, nil), 0x01), err: ErrShortFrameHeader},
		{name: "short frame payload", src: append(appendTestIVF(nil, 16, 16, 30, 1, nil), 0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa), err: ErrShortFramePayload},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			it, err := NewIterator(tt.src)
			if err == nil {
				_, _, err = it.Next()
			}
			if !errors.Is(err, tt.err) {
				t.Fatalf("err=%v want %v", err, tt.err)
			}
		})
	}
}

func TestIteratorAllocs(t *testing.T) {
	src := appendTestIVF(nil, 16, 16, 30, 1, []testFrame{{payload: []byte{0x12, 0x00}}})
	allocs := testing.AllocsPerRun(1000, func() {
		it, err := NewIterator(src)
		if err != nil {
			t.Fatal(err)
		}
		frame, ok, err := it.Next()
		if err != nil || !ok || len(frame.Payload) == 0 {
			t.Fatalf("frame=%+v ok=%v err=%v", frame, ok, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Iterator allocated: %f", allocs)
	}
}

func FuzzIterator(f *testing.F) {
	f.Add(appendTestIVF(nil, 16, 16, 30, 1, []testFrame{{payload: []byte{0x12, 0x00}}}))
	f.Fuzz(func(t *testing.T, src []byte) {
		if len(src) > 4096 {
			src = src[:4096]
		}
		it, err := NewIterator(src)
		if err != nil {
			return
		}
		for i := 0; i < 64; i++ {
			frame, ok, err := it.Next()
			if err != nil {
				return
			}
			if !ok {
				return
			}
			if len(frame.Payload) > len(src) {
				t.Fatalf("payload length=%d src=%d", len(frame.Payload), len(src))
			}
		}
	})
}

type testFrame struct {
	timestamp uint64
	payload   []byte
}

func appendTestIVF(dst []byte, width uint16, height uint16, timebaseNum uint32, timebaseDen uint32, frames []testFrame) []byte {
	dst = append(dst,
		'D', 'K', 'I', 'F',
		0, 0,
		byte(FileHeaderSize), 0,
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

func patchIVFHeader(src []byte, off int, value uint16) []byte {
	src[off] = byte(value)
	src[off+1] = byte(value >> 8)
	return src
}

func patchIVFByte(src []byte, off int, value byte) []byte {
	src[off] = value
	return src
}

func patchIVFCodec(src []byte, codec [4]byte) []byte {
	copy(src[8:12], codec[:])
	return src
}
