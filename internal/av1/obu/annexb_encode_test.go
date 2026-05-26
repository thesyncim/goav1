// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package obu

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

// buildLowOverhead emits one or more OBUs back-to-back with obu_size present.
func buildLowOverhead(dst []byte, obus ...[]byte) []byte {
	for _, payload := range obus {
		// obu_type already encoded in payload[0]; ensure has_size bit is set.
		header := payload[0] | 0x02
		dst = append(dst, header)
		body := payload[1:]
		dst = bitstream.AppendLEB128(dst, uint32(len(body)))
		dst = append(dst, body...)
	}
	return dst
}

func TestLowOverheadToAnnexBSingleTU(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	seq := []byte{byte(TypeSequenceHeader) << 3, 0xaa}
	fh := []byte{byte(TypeFrame) << 3, 0xcc}
	src := buildLowOverhead(nil, td, seq, fh)

	annexB, err := LowOverheadToAnnexB(nil, src, nil)
	if err != nil {
		t.Fatalf("LowOverheadToAnnexB err=%v", err)
	}
	if len(annexB) == 0 {
		t.Fatal("LowOverheadToAnnexB produced empty output")
	}

	it := NewAnnexBIterator(annexB)
	var seen []Type
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("iter err=%v", err)
		}
		if !ok {
			break
		}
		seen = append(seen, unit.OBU.Header.Type)
	}
	wantTypes := []Type{TypeTemporalDelimiter, TypeSequenceHeader, TypeFrame}
	if len(seen) != len(wantTypes) {
		t.Fatalf("annex B types=%v want %v", seen, wantTypes)
	}
	for i := range seen {
		if seen[i] != wantTypes[i] {
			t.Fatalf("type[%d]=%d want %d", i, seen[i], wantTypes[i])
		}
	}
}

func TestLowOverheadToAnnexBMultipleTUs(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	seq := []byte{byte(TypeSequenceHeader) << 3, 0xaa}
	fh1 := []byte{byte(TypeFrame) << 3, 0xcc, 0xcc}
	fh2 := []byte{byte(TypeFrame) << 3, 0xdd}

	src := buildLowOverhead(nil, td, seq, fh1, td, fh2)
	annexB, err := LowOverheadToAnnexB(nil, src, nil)
	if err != nil {
		t.Fatalf("LowOverheadToAnnexB err=%v", err)
	}

	it := NewAnnexBIterator(annexB)
	temporalCounts := map[uint32]int{}
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("iter err=%v", err)
		}
		if !ok {
			break
		}
		temporalCounts[unit.TemporalUnitIndex]++
	}
	if temporalCounts[0] == 0 || temporalCounts[1] == 0 {
		t.Fatalf("temporal counts=%v want at least two TUs", temporalCounts)
	}
}

func TestLowOverheadToAnnexBRoundTrip(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	seq := []byte{byte(TypeSequenceHeader) << 3, 0xaa}
	frame := []byte{byte(TypeFrame) << 3, 0xcc, 0xcd, 0xce}
	src := buildLowOverhead(nil, td, seq, frame)

	annexB, err := LowOverheadToAnnexB(nil, src, nil)
	if err != nil {
		t.Fatalf("LowOverheadToAnnexB err=%v", err)
	}
	roundTripped, err := AnnexBToLowOverhead(nil, annexB)
	if err != nil {
		t.Fatalf("AnnexBToLowOverhead err=%v", err)
	}
	if !bytes.Equal(roundTripped, src) {
		t.Fatalf("round-trip mismatch\n got=%x\nwant=%x", roundTripped, src)
	}
}

func TestLowOverheadToAnnexBScratchReuse(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	frame := []byte{byte(TypeFrame) << 3, 0xaa}
	src := buildLowOverhead(nil, td, frame, td, frame)

	scratch := make([]AnnexBFrameSpan, 0, 8)
	out, err := LowOverheadToAnnexB(nil, src, scratch)
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if len(out) == 0 {
		t.Fatal("empty output")
	}
}

func TestAppendAnnexBHelpers(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	frame := []byte{byte(TypeFrame) << 3, 0xaa}
	tu := AppendAnnexBTemporalUnit(nil, [][][]byte{{td, frame}})
	if len(tu) == 0 {
		t.Fatal("empty temporal unit")
	}
	it := NewAnnexBIterator(tu)
	seen := []Type{}
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("err=%v", err)
		}
		if !ok {
			break
		}
		seen = append(seen, unit.OBU.Header.Type)
	}
	if len(seen) != 2 || seen[0] != TypeTemporalDelimiter || seen[1] != TypeFrame {
		t.Fatalf("seen=%v", seen)
	}
}

func TestAnnexBToLowOverheadRoundTrip(t *testing.T) {
	td := []byte{byte(TypeTemporalDelimiter) << 3}
	frame := []byte{byte(TypeFrame) << 3, 0xbb}
	annexB := AppendAnnexBTemporalUnit(nil, [][][]byte{{td, frame}})
	out, err := AnnexBToLowOverhead(nil, annexB)
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	// Re-parse out via low-overhead iterator and ensure both OBUs appear.
	it := NewLowOverheadIterator(out)
	var types []Type
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("err=%v", err)
		}
		if !ok {
			break
		}
		types = append(types, unit.Header.Type)
	}
	if len(types) != 2 || types[0] != TypeTemporalDelimiter || types[1] != TypeFrame {
		t.Fatalf("types=%v", types)
	}
}
