package obu

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/testvector"
)

func TestOBUCoreVectors(t *testing.T) {
	vector, ok := testvector.CoreVector(testvector.TagOBULowOverheadTemporalDelimiter)
	if !ok {
		t.Fatal("missing temporal delimiter vector")
	}

	unit, consumed, err := ParseLowOverhead(vector.Input)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != len(vector.Input) {
		t.Fatalf("consumed=%d want %d", consumed, len(vector.Input))
	}
	if unit.Header.Type != TypeTemporalDelimiter {
		t.Fatalf("type=%d want %d", unit.Header.Type, TypeTemporalDelimiter)
	}
	if !bytes.Equal(unit.Payload, vector.Want) {
		t.Fatalf("payload=%x want %x", unit.Payload, vector.Want)
	}
	if testvector.OracleEnabled {
		oracle := testvector.NewOracle(testvector.CoreSuite().Manifest)
		if err := oracle.CheckBytes(vector.Tag, unit.Payload); err != nil {
			t.Fatalf("oracle err=%v", err)
		}
	}
}

func TestAnnexBCoreVector(t *testing.T) {
	vector, ok := testvector.CoreVector(testvector.TagOBUAnnexBTemporalUnit)
	if !ok {
		t.Fatal("missing Annex B vector")
	}

	it := NewAnnexBIterator(vector.Input)
	wantTypes := [...]Type{TypeTemporalDelimiter, TypeSequenceHeader, TypeFrameHeader}
	wantFrames := [...]uint32{0, 0, 1}
	var got [64]byte
	wantOff := 0
	for i := range len(wantTypes) {
		unit, ok, err := it.Next()
		if err != nil || !ok {
			t.Fatalf("unit %d ok=%v err=%v", i, ok, err)
		}
		end := wantOff + len(unit.Raw)
		if end > len(vector.Want) {
			t.Fatalf("unit %d raw overrun off=%d len=%d want=%d", i, wantOff, len(unit.Raw), len(vector.Want))
		}
		if end > len(got) {
			t.Fatalf("unit %d fixed buffer overrun end=%d cap=%d", i, end, len(got))
		}
		if !bytes.Equal(unit.Raw, vector.Want[wantOff:end]) {
			t.Fatalf("unit %d raw=%x want=%x", i, unit.Raw, vector.Want[wantOff:end])
		}
		copy(got[wantOff:end], unit.Raw)
		if unit.OBU.Header.Type != wantTypes[i] {
			t.Fatalf("unit %d type=%d want %d", i, unit.OBU.Header.Type, wantTypes[i])
		}
		if unit.FrameUnitIndex != wantFrames[i] {
			t.Fatalf("unit %d frame=%d want %d", i, unit.FrameUnitIndex, wantFrames[i])
		}
		wantOff = end
	}
	_, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected extra Annex B unit")
	}
	if wantOff != len(vector.Want) {
		t.Fatalf("consumed=%d want %d", wantOff, len(vector.Want))
	}
	if testvector.OracleEnabled {
		oracle := testvector.NewOracle(testvector.CoreSuite().Manifest)
		if err := oracle.CheckBytes(vector.Tag, got[:wantOff]); err != nil {
			t.Fatalf("oracle err=%v", err)
		}
	}
}
