//go:build goav1_oracle

package testvector

import (
	"errors"
	"testing"
	"unsafe"
)

func TestOracleEnabledBuildChecksManifest(t *testing.T) {
	if !OracleEnabled {
		t.Fatal("OracleEnabled=false in oracle build")
	}
	if size := unsafe.Sizeof(Oracle{}); size == 0 {
		t.Fatal("enabled oracle unexpectedly has zero size")
	}
	oracle := NewOracle(Manifest{Vectors: []Vector{{
		Tag:  TagRTPPayloadSingleOBU,
		Kind: KindRTP,
		Want: []byte{0xaa, 0xbb},
	}}})
	if err := oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xaa, 0xbb}); err != nil {
		t.Fatalf("CheckBytes err=%v", err)
	}
	if err := oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xaa}); !errors.Is(err, ErrMismatchedBytes) {
		t.Fatalf("mismatch err=%v want %v", err, ErrMismatchedBytes)
	}
	if err := oracle.CheckBytes(TagRTPPayloadFragmentedOBU, nil); !errors.Is(err, ErrMissingVector) {
		t.Fatalf("missing err=%v want %v", err, ErrMissingVector)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if OracleEnabled {
			_ = oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xaa, 0xbb})
		}
	})
	if allocs != 0 {
		t.Fatalf("enabled oracle check allocated: %f", allocs)
	}
}

func TestOracleEnabledBuildChecksCoreSuite(t *testing.T) {
	suite := CoreSuite()
	oracle := NewOracle(suite.Manifest)
	for _, vector := range suite.Manifest.Vectors {
		if err := oracle.CheckBytes(vector.Tag, vector.Want); err != nil {
			t.Fatalf("%s oracle err=%v", vector.Name, err)
		}
	}
}
