package testvector

import (
	"errors"
	"testing"
)

func TestManifestFindAndValidate(t *testing.T) {
	manifest := Manifest{Vectors: []Vector{
		{Tag: TagOBULowOverheadTemporalDelimiter, Kind: KindOBU, Name: "temporal delimiter", Input: []byte{0x12}, Want: []byte{0x12}},
		{Tag: TagRTPPayloadSingleOBU, Kind: KindRTP, Name: "single obu", Input: []byte{0x10, 0xaa}, Want: []byte{0xaa}},
	}}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	vector, ok := manifest.Find(TagRTPPayloadSingleOBU)
	if !ok {
		t.Fatal("missing vector")
	}
	if vector.Kind != KindRTP || vector.Name != "single obu" {
		t.Fatalf("vector=%+v", vector)
	}
}

func TestManifestRejectsInvalidTags(t *testing.T) {
	err := (Manifest{Vectors: []Vector{{}}}).Validate()
	if !errors.Is(err, ErrInvalidTag) {
		t.Fatalf("err=%v want %v", err, ErrInvalidTag)
	}
	err = (Manifest{Vectors: []Vector{
		{Tag: TagRTPPayloadSingleOBU},
		{Tag: TagRTPPayloadSingleOBU},
	}}).Validate()
	if !errors.Is(err, ErrDuplicateTag) {
		t.Fatalf("err=%v want %v", err, ErrDuplicateTag)
	}
}
