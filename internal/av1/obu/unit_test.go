package obu

import (
	"errors"
	"testing"
)

func TestParseHeader(t *testing.T) {
	header := Header{
		Type:         TypeFrameHeader,
		Extension:    true,
		HasSizeField: true,
		TemporalID:   3,
		SpatialID:    2,
	}

	var buf [2]byte
	n, err := PutHeader(buf[:], header)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("PutHeader len=%d want 2", n)
	}

	got, consumed, err := ParseHeader(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if consumed != n || got != header {
		t.Fatalf("ParseHeader got=%+v consumed=%d want=%+v consumed=%d", got, consumed, header, n)
	}
}

func TestParseHeaderRejectsReservedBits(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		err  error
	}{
		{name: "forbidden", in: []byte{0x80}, err: ErrForbiddenBit},
		{name: "obu reserved", in: []byte{0x01}, err: ErrReservedBit},
		{name: "short extension", in: []byte{0x04}, err: ErrShortHeader},
		{name: "extension reserved", in: []byte{0x04, 0x01}, err: ErrReservedBit},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := ParseHeader(tt.in)
			if !errors.Is(err, tt.err) {
				t.Fatalf("ParseHeader err=%v want %v", err, tt.err)
			}
		})
	}
}

func TestParseLowOverhead(t *testing.T) {
	src := []byte{
		byte(TypeSequenceHeader)<<3 | 0x02,
		0x03,
		0xaa, 0xbb, 0xcc,
		byte(TypeFrameHeader)<<3 | 0x02,
		0x01,
		0xdd,
	}

	it := NewLowOverheadIterator(src)
	first, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !ok || first.Header.Type != TypeSequenceHeader || string(first.Payload) != string([]byte{0xaa, 0xbb, 0xcc}) {
		t.Fatalf("first=%+v ok=%v", first, ok)
	}

	second, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !ok || second.Header.Type != TypeFrameHeader || string(second.Payload) != string([]byte{0xdd}) {
		t.Fatalf("second=%+v ok=%v", second, ok)
	}

	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("iterator returned extra unit")
	}
}

func TestParseLowOverheadRequiresSizeField(t *testing.T) {
	_, _, err := ParseLowOverhead([]byte{byte(TypeFrameHeader) << 3})
	if !errors.Is(err, ErrMissingSizeField) {
		t.Fatalf("ParseLowOverhead err=%v want %v", err, ErrMissingSizeField)
	}
}

func TestParseElementWithoutSizeField(t *testing.T) {
	unit, err := ParseElement([]byte{byte(TypeTileGroup) << 3, 0x01, 0x02})
	if err != nil {
		t.Fatal(err)
	}
	if unit.Header.HasSizeField || unit.Size != 2 || len(unit.Payload) != 2 {
		t.Fatalf("unit=%+v", unit)
	}
}

func TestOBUAllocs(t *testing.T) {
	src := []byte{byte(TypeFrame)<<3 | 0x02, 0x01, 0x55}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err := ParseLowOverhead(src)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("OBU parse allocated: %f", allocs)
	}
}

func BenchmarkParseLowOverhead(b *testing.B) {
	src := []byte{byte(TypeFrame)<<3 | 0x02, 0x03, 0x01, 0x02, 0x03}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = ParseLowOverhead(src)
	}
}
