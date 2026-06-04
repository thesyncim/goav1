package encoder

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

func TestAppendLowOverheadOBU(t *testing.T) {
	unit := OBU{
		Type:       obu.TypeFrame,
		TemporalID: 2,
		SpatialID:  1,
		Payload:    []byte{0xaa, 0xbb},
	}
	size, err := LowOverheadOBUSize(unit)
	if err != nil {
		t.Fatalf("LowOverheadOBUSize: %v", err)
	}
	if size != 5 {
		t.Fatalf("LowOverheadOBUSize = %d; want 5", size)
	}
	var buf [8]byte
	out, err := AppendLowOverheadOBU(buf[:0], unit)
	if err != nil {
		t.Fatalf("AppendLowOverheadOBU: %v", err)
	}
	want := []byte{0x36, 0x48, 0x02, 0xaa, 0xbb}
	if !bytes.Equal(out, want) {
		t.Fatalf("AppendLowOverheadOBU = % x; want % x", out, want)
	}
	parsed, consumed, err := obu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverhead: %v", err)
	}
	if consumed != len(out) || parsed.Header.Type != unit.Type || parsed.Header.TemporalID != unit.TemporalID || parsed.Header.SpatialID != unit.SpatialID {
		t.Fatalf("parsed header=%+v consumed=%d", parsed.Header, consumed)
	}
	if !bytes.Equal(parsed.Payload, unit.Payload) {
		t.Fatalf("payload = % x; want % x", parsed.Payload, unit.Payload)
	}
}

func TestAppendLowOverheadOBUEmptyTemporalDelimiter(t *testing.T) {
	var buf [4]byte
	out, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		t.Fatalf("AppendLowOverheadOBU temporal delimiter: %v", err)
	}
	want := []byte{0x12, 0x00}
	if !bytes.Equal(out, want) {
		t.Fatalf("temporal delimiter = % x; want % x", out, want)
	}
}

func TestAppendLowOverheadOBURejectsInvalid(t *testing.T) {
	var buf [8]byte
	if _, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.TypeReserved}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("reserved type err=%v; want ErrInvalidFrame", err)
	}
	if _, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.Type(16)}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("out-of-range type err=%v; want ErrInvalidFrame", err)
	}
	if _, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.TypeFrame, SpatialID: 4}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad spatial id err=%v; want ErrInvalidFrame", err)
	}
}

func TestAppendLowOverheadOBUShortBuffer(t *testing.T) {
	var buf [4]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLowOverheadOBU(dst, OBU{Type: obu.TypeFrame, Payload: []byte{0xaa, 0xbb, 0xcc}})
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v; want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
	}
}

func TestAppendLowOverheadOBUAllocs(t *testing.T) {
	payload := [...]byte{0xaa, 0xbb, 0xcc}
	unit := OBU{Type: obu.TypeFrame, Payload: payload[:]}
	var buf [8]byte
	if _, err := AppendLowOverheadOBU(buf[:0], unit); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadOBU(buf[:0], unit)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadOBU allocated: %f", allocs)
	}
}

func TestAppendLowOverheadTemporalUnit(t *testing.T) {
	units := [...]OBU{
		{Type: obu.TypeSequenceHeader, Payload: []byte{0x01, 0x02}},
		{Type: obu.TypeFrame, TemporalID: 1, SpatialID: 1, Payload: []byte{0xaa}},
	}
	size, err := LowOverheadTemporalUnitSize(units[:])
	if err != nil {
		t.Fatalf("LowOverheadTemporalUnitSize: %v", err)
	}
	var buf [16]byte
	out, err := AppendLowOverheadTemporalUnit(buf[:0], units[:])
	if err != nil {
		t.Fatalf("AppendLowOverheadTemporalUnit: %v", err)
	}
	if len(out) != size {
		t.Fatalf("temporal unit len=%d want %d", len(out), size)
	}
	want := []byte{
		0x12, 0x00,
		0x0a, 0x02, 0x01, 0x02,
		0x36, 0x28, 0x01, 0xaa,
	}
	if !bytes.Equal(out, want) {
		t.Fatalf("temporal unit = % x; want % x", out, want)
	}
	it := obu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil {
		t.Fatalf("TemporalUnitIterator.Next: %v", err)
	}
	if !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit parsed ok=%v raw=% x", ok, tu.Raw)
	}
	_, ok, err = it.Next()
	if err != nil || ok {
		t.Fatalf("TemporalUnitIterator second ok=%v err=%v", ok, err)
	}
}

func TestAppendLowOverheadTemporalUnitRejectsBeforeWrite(t *testing.T) {
	var buf [16]byte
	dst := buf[:1]
	dst[0] = 0xee
	if _, err := AppendLowOverheadTemporalUnit(dst, nil); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("empty temporal unit err=%v; want ErrInvalidFrame", err)
	}
	if _, err := AppendLowOverheadTemporalUnit(dst, []OBU{{Type: obu.TypeTemporalDelimiter}}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("nested temporal delimiter err=%v; want ErrInvalidFrame", err)
	}
	out, err := AppendLowOverheadTemporalUnit(dst, []OBU{{Type: obu.TypeFrame}, {Type: obu.TypeReserved}})
	if !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad temporal unit err=%v; want ErrInvalidFrame", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("invalid temporal unit mutated output: % x", out)
	}
}

func TestAppendLowOverheadTemporalUnitShortBuffer(t *testing.T) {
	var buf [5]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLowOverheadTemporalUnit(dst, []OBU{{Type: obu.TypeFrame, Payload: []byte{0xaa, 0xbb}}})
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short temporal unit err=%v; want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short temporal unit mutated output: % x", out)
	}
}

func TestAppendLowOverheadTemporalUnitAllocs(t *testing.T) {
	payload := [...]byte{0xaa, 0xbb}
	units := [...]OBU{{Type: obu.TypeFrame, Payload: payload[:]}}
	var buf [8]byte
	if _, err := AppendLowOverheadTemporalUnit(buf[:0], units[:]); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadTemporalUnit(buf[:0], units[:])
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadTemporalUnit allocated: %f", allocs)
	}
}
