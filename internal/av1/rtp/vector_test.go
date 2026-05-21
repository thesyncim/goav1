package rtp

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/testvector"
)

func TestRTPCoreVectors(t *testing.T) {
	suite := testvector.CoreSuite()
	oracle := testvector.NewOracle(suite.Manifest)
	for _, tag := range []testvector.Tag{
		testvector.TagRTPPayloadSingleOBU,
		testvector.TagRTPPayloadFragmentedOBU,
	} {
		vector, ok := suite.Manifest.Find(tag)
		if !ok {
			t.Fatalf("missing vector tag=%d", tag)
		}
		it, err := NewIterator(vector.Input)
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
		if tag == testvector.TagRTPPayloadFragmentedOBU && !elem.ContinuesNext {
			t.Fatalf("%s ContinuesNext=false", vector.Name)
		}
		if testvector.OracleEnabled {
			if err := oracle.CheckBytes(vector.Tag, elem.Data); err != nil {
				t.Fatalf("%s oracle err=%v", vector.Name, err)
			}
		}
	}
}
