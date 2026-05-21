//go:build !goav1_oracle

package testvector

import (
	"testing"
	"unsafe"
)

func TestOracleDisabledBuildIsZeroCost(t *testing.T) {
	if OracleEnabled {
		t.Fatal("OracleEnabled=true in default build")
	}
	if size := unsafe.Sizeof(Oracle{}); size != 0 {
		t.Fatalf("Oracle size=%d want 0", size)
	}
	oracle := NewOracle(Manifest{Vectors: []Vector{{
		Tag:  TagRTPPayloadSingleOBU,
		Kind: KindRTP,
		Want: []byte{0xaa},
	}}})
	if err := oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xbb}); err != nil {
		t.Fatalf("disabled oracle err=%v", err)
	}
	if err := oracle.CheckBytes(Tag(0xffff), nil); err != nil {
		t.Fatalf("disabled oracle missing err=%v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if OracleEnabled {
			_ = oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xbb})
		}
	})
	if allocs != 0 {
		t.Fatalf("disabled oracle branch allocated: %f", allocs)
	}
}
