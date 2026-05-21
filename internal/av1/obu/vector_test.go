package obu

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/testvector"
)

func TestOBUCoreVectors(t *testing.T) {
	suite := testvector.CoreSuite()
	oracle := testvector.NewOracle(suite.Manifest)
	vector, ok := suite.Manifest.Find(testvector.TagOBULowOverheadTemporalDelimiter)
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
		if err := oracle.CheckBytes(vector.Tag, unit.Payload); err != nil {
			t.Fatalf("oracle err=%v", err)
		}
	}
}
