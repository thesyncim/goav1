package obu

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

func TestAnnexBIterator(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	seq := []byte{byte(TypeSequenceHeader) << 3, 0xaa}
	frameHeader := []byte{byte(TypeFrameHeader) << 3, 0xbb}
	frame := []byte{byte(TypeFrame) << 3, 0xcc}
	stream := appendAnnexBTestStream(nil,
		[][][]byte{{td, seq}, {frameHeader}},
		[][][]byte{{td, frame}},
	)

	want := []struct {
		raw      []byte
		typ      Type
		temporal uint32
		frame    uint32
		obu      uint32
	}{
		{raw: td, typ: TypeTemporalDelimiter, temporal: 0, frame: 0, obu: 0},
		{raw: seq, typ: TypeSequenceHeader, temporal: 0, frame: 0, obu: 1},
		{raw: frameHeader, typ: TypeFrameHeader, temporal: 0, frame: 1, obu: 0},
		{raw: td, typ: TypeTemporalDelimiter, temporal: 1, frame: 0, obu: 0},
		{raw: frame, typ: TypeFrame, temporal: 1, frame: 0, obu: 1},
	}

	it := NewAnnexBIterator(stream)
	for i, wantUnit := range want {
		unit, ok, err := it.Next()
		if err != nil || !ok {
			t.Fatalf("unit %d ok=%v err=%v", i, ok, err)
		}
		if !bytes.Equal(unit.Raw, wantUnit.raw) {
			t.Fatalf("unit %d raw=%x want %x", i, unit.Raw, wantUnit.raw)
		}
		if unit.OBU.Header.Type != wantUnit.typ {
			t.Fatalf("unit %d type=%d want %d", i, unit.OBU.Header.Type, wantUnit.typ)
		}
		if unit.TemporalUnitIndex != wantUnit.temporal || unit.FrameUnitIndex != wantUnit.frame || unit.OBUIndex != wantUnit.obu {
			t.Fatalf("unit %d indices=%d/%d/%d want %d/%d/%d", i,
				unit.TemporalUnitIndex, unit.FrameUnitIndex, unit.OBUIndex,
				wantUnit.temporal, wantUnit.frame, wantUnit.obu)
		}
		start := int(unit.Offset)
		end := start + len(unit.Raw)
		if end > len(stream) || !bytes.Equal(unit.Raw, stream[start:end]) {
			t.Fatalf("unit %d does not alias stream at offset %d", i, unit.Offset)
		}
	}
	_, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected extra Annex B unit")
	}
}

func TestAnnexBIteratorMultiByteSizes(t *testing.T) {
	payload := make([]byte, 129)
	for i := range payload {
		payload[i] = byte(i)
	}
	raw := append([]byte{byte(TypeFrame) << 3}, payload...)
	stream := appendAnnexBTestStream(nil, [][][]byte{{raw}})

	it := NewAnnexBIterator(stream)
	unit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(unit.Raw, raw) {
		t.Fatalf("raw len=%d want %d", len(unit.Raw), len(raw))
	}
	if unit.OBU.Size != uint32(len(payload)) {
		t.Fatalf("payload size=%d want %d", unit.OBU.Size, len(payload))
	}
	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected extra Annex B unit")
	}
}

func TestParseAnnexBElement(t *testing.T) {
	src := []byte{0x03, byte(TypeSequenceHeader) << 3, 0xaa, 0xbb, 0xcc}
	unit, consumed, err := ParseAnnexBElement(src)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != len(src)-1 || unit.Header.Type != TypeSequenceHeader || !bytes.Equal(unit.Payload, []byte{0xaa, 0xbb}) {
		t.Fatalf("unit=%+v consumed=%d", unit, consumed)
	}

	sized := []byte{0x03, byte(TypeFrame)<<3 | 0x02, 0x01, 0xdd}
	unit, consumed, err = ParseAnnexBElement(sized)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != len(sized) || unit.Header.Type != TypeFrame || !bytes.Equal(unit.Payload, []byte{0xdd}) {
		t.Fatalf("sized unit=%+v consumed=%d", unit, consumed)
	}
}

func TestAnnexBIteratorRejectsInvalid(t *testing.T) {
	for _, tc := range []struct {
		name string
		src  []byte
		err  error
	}{
		{name: "zero temporal unit", src: []byte{0x00}, err: ErrInvalidAnnexB},
		{name: "oversized frame unit", src: []byte{0x01, 0x01}, err: ErrInvalidAnnexB},
		{name: "oversized obu unit", src: []byte{0x02, 0x01, 0x01}, err: ErrInvalidAnnexB},
		{name: "zero obu unit", src: []byte{0x02, 0x01, 0x00}, err: ErrInvalidAnnexB},
		{name: "truncated temporal size", src: []byte{0x80}, err: bitstream.ErrShortLEB128},
		{name: "truncated payload", src: []byte{0x03, 0x02, 0x01}, err: ErrShortPayload},
		{name: "internal size mismatch", src: []byte{0x06, 0x05, 0x04, byte(TypeFrame)<<3 | 0x02, 0x01, 0xaa, 0xbb}, err: ErrSizeMismatch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			it := NewAnnexBIterator(tc.src)
			_, _, err := it.Next()
			if !errors.Is(err, tc.err) {
				t.Fatalf("err=%v want %v", err, tc.err)
			}
		})
	}
}

func TestParseAnnexBElementRejectsInvalid(t *testing.T) {
	for _, tc := range []struct {
		name string
		src  []byte
		err  error
	}{
		{name: "zero unit", src: []byte{0x00}, err: ErrInvalidAnnexB},
		{name: "truncated size", src: []byte{0x80}, err: bitstream.ErrShortLEB128},
		{name: "truncated payload", src: []byte{0x02, byte(TypeFrame) << 3}, err: ErrShortPayload},
		{name: "internal size mismatch", src: []byte{0x04, byte(TypeFrame)<<3 | 0x02, 0x01, 0xaa, 0xbb}, err: ErrSizeMismatch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := ParseAnnexBElement(tc.src)
			if !errors.Is(err, tc.err) {
				t.Fatalf("err=%v want %v", err, tc.err)
			}
		})
	}
}

func TestParseAnnexBElementAllocs(t *testing.T) {
	src := []byte{0x03, byte(TypeFrame)<<3 | 0x02, 0x01, 0xdd}
	allocs := testing.AllocsPerRun(1000, func() {
		unit, consumed, err := ParseAnnexBElement(src)
		if err != nil || consumed != len(src) || len(unit.Payload) != 1 {
			t.Fatalf("unit=%+v consumed=%d err=%v", unit, consumed, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseAnnexBElement allocated: %f", allocs)
	}
}

func TestAnnexBIteratorAllocs(t *testing.T) {
	stream := appendAnnexBTestStream(nil, [][][]byte{{{byte(TypeTemporalDelimiter) << 3}, {byte(TypeFrame) << 3, 0xaa}}})
	allocs := testing.AllocsPerRun(1000, func() {
		it := NewAnnexBIterator(stream)
		for {
			unit, ok, err := it.Next()
			if err != nil {
				t.Fatal(err)
			}
			if !ok {
				return
			}
			if len(unit.Raw) == 0 {
				t.Fatal("empty Annex B unit")
			}
		}
	})
	if allocs != 0 {
		t.Fatalf("AnnexBIterator allocated: %f", allocs)
	}
}

func FuzzAnnexBIterator(f *testing.F) {
	seed := appendAnnexBTestStream(nil, [][][]byte{{{byte(TypeTemporalDelimiter) << 3}, {byte(TypeFrame) << 3, 0xaa}}})
	f.Add(seed)
	f.Fuzz(func(t *testing.T, src []byte) {
		if len(src) > 4096 {
			src = src[:4096]
		}
		it := NewAnnexBIterator(src)
		var prevOffset uint32
		for i := range 64 {
			unit, ok, err := it.Next()
			if err != nil {
				return
			}
			if !ok {
				return
			}
			if len(unit.Raw) == 0 || int(unit.Offset)+len(unit.Raw) > len(src) {
				t.Fatalf("bad unit offset=%d len=%d src=%d", unit.Offset, len(unit.Raw), len(src))
			}
			if i > 0 && unit.Offset < prevOffset {
				t.Fatalf("offset moved backward: %d after %d", unit.Offset, prevOffset)
			}
			prevOffset = unit.Offset
		}
	})
}

func appendAnnexBTestStream(dst []byte, temporalUnits ...[][][]byte) []byte {
	for _, temporalUnit := range temporalUnits {
		var temporal []byte
		for _, frameUnit := range temporalUnit {
			var frame []byte
			for _, obu := range frameUnit {
				frame = bitstream.AppendLEB128(frame, uint32(len(obu)))
				frame = append(frame, obu...)
			}
			temporal = bitstream.AppendLEB128(temporal, uint32(len(frame)))
			temporal = append(temporal, frame...)
		}
		dst = bitstream.AppendLEB128(dst, uint32(len(temporal)))
		dst = append(dst, temporal...)
	}
	return dst
}
