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

func TestManifestFindDigest(t *testing.T) {
	md5, err := ParseMD5Hex([]byte("5a68de997d60afa9083b17fe00f7cdf2"))
	if err != nil {
		t.Fatal(err)
	}
	manifest := Manifest{Digests: []FrameDigest{{
		Tag:        TagIVFSingleFrameAV1,
		FrameIndex: 3,
		MD5:        md5,
	}}}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	digest, ok := manifest.FindDigest(TagIVFSingleFrameAV1, 3)
	if !ok {
		t.Fatal("missing digest")
	}
	if digest.MD5 != md5 {
		t.Fatalf("digest=%x want %x", digest.MD5, md5)
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
	err = (Manifest{Digests: []FrameDigest{{}}}).Validate()
	if !errors.Is(err, ErrInvalidTag) {
		t.Fatalf("digest err=%v want %v", err, ErrInvalidTag)
	}
	err = (Manifest{Digests: []FrameDigest{
		{Tag: TagIVFSingleFrameAV1, FrameIndex: 0},
		{Tag: TagIVFSingleFrameAV1, FrameIndex: 0},
	}}).Validate()
	if !errors.Is(err, ErrDuplicateTag) {
		t.Fatalf("duplicate digest err=%v want %v", err, ErrDuplicateTag)
	}
}

func TestParseMD5Hex(t *testing.T) {
	md5, err := ParseMD5Hex([]byte("5A68DE997D60AFA9083B17FE00F7CDF2 trailing"))
	if err != nil {
		t.Fatal(err)
	}
	want := MD5{0x5a, 0x68, 0xde, 0x99, 0x7d, 0x60, 0xaf, 0xa9, 0x08, 0x3b, 0x17, 0xfe, 0x00, 0xf7, 0xcd, 0xf2}
	if md5 != want {
		t.Fatalf("md5=%x want %x", md5, want)
	}
	if _, err := ParseMD5Hex([]byte("short")); !errors.Is(err, ErrInvalidMD5) {
		t.Fatalf("short err=%v want %v", err, ErrInvalidMD5)
	}
	if _, err := ParseMD5Hex([]byte("zz68de997d60afa9083b17fe00f7cdf2")); !errors.Is(err, ErrInvalidMD5) {
		t.Fatalf("invalid err=%v want %v", err, ErrInvalidMD5)
	}
}
