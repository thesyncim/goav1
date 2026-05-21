package testvector

import (
	"bytes"
	"crypto/md5"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/ivf"
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
		case KindIVF:
			checkIVFVector(t, vector)
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

func checkIVFVector(t *testing.T, vector Vector) {
	t.Helper()
	it, err := ivf.NewIterator(vector.Input)
	if err != nil {
		t.Fatalf("%s NewIterator: %v", vector.Name, err)
	}
	frame, ok, err := it.Next()
	if err != nil {
		t.Fatalf("%s Next: %v", vector.Name, err)
	}
	if !ok {
		t.Fatalf("%s missing frame", vector.Name)
	}
	if !bytes.Equal(frame.Payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, frame.Payload, vector.Want)
	}
	digest, ok := CoreSuite().Manifest.FindDigest(vector.Tag, frame.Index)
	if !ok {
		t.Fatalf("%s missing digest", vector.Name)
	}
	if got := MD5(md5.Sum(frame.Payload)); got != digest.MD5 {
		t.Fatalf("%s md5=%x want %x", vector.Name, got, digest.MD5)
	}
}
