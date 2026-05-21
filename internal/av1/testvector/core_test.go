package testvector

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/rtp"
)

func TestCoreSuiteVectors(t *testing.T) {
	suite := CoreSuite()
	if suite.Name != "goav1-core" {
		t.Fatalf("suite name=%q", suite.Name)
	}
	if err := suite.Manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, vector := range suite.Manifest.Vectors {
		switch vector.Kind {
		case KindOBU:
			checkOBUVector(t, vector)
		case KindRTP:
			checkRTPVector(t, vector)
		default:
			t.Fatalf("unhandled vector kind=%d", vector.Kind)
		}
	}
}

func checkOBUVector(t *testing.T, vector Vector) {
	t.Helper()
	unit, consumed, err := obu.ParseLowOverhead(vector.Input)
	if err != nil {
		t.Fatalf("%s ParseLowOverhead: %v", vector.Name, err)
	}
	if consumed != len(vector.Input) {
		t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
	}
	if unit.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeTemporalDelimiter)
	}
	if !bytes.Equal(unit.Payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, unit.Payload, vector.Want)
	}
}

func checkRTPVector(t *testing.T, vector Vector) {
	t.Helper()
	it, err := rtp.NewIterator(vector.Input)
	if err != nil {
		t.Fatalf("%s NewIterator: %v", vector.Name, err)
	}
	elem, ok, err := it.Next()
	if err != nil {
		t.Fatalf("%s Next: %v", vector.Name, err)
	}
	if !ok {
		t.Fatalf("%s missing element", vector.Name)
	}
	if !bytes.Equal(elem.Data, vector.Want) {
		t.Fatalf("%s data=%x want %x", vector.Name, elem.Data, vector.Want)
	}
}
