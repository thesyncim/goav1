package goav1_test

import (
	"errors"
	"strings"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicRTPSDESHeaderExtensionConstants(t *testing.T) {
	if av1.RTPSDESItemTypeRTPStreamID != 12 ||
		av1.RTPSDESItemTypeRepairedRTPStreamID != 13 ||
		av1.RTPSDESItemTypeMID != 15 {
		t.Fatalf("unexpected RTP SDES item type constants")
	}
	if av1.RTPHeaderExtensionSDESMaxLen != 255 {
		t.Fatalf("RTPHeaderExtensionSDESMaxLen = %d, want 255",
			av1.RTPHeaderExtensionSDESMaxLen)
	}
}

func TestPublicRTPMIDHeaderExtension(t *testing.T) {
	mid := "video-0"
	buf := make([]byte, len(mid)+4)
	n, err := av1.PutRTPMIDHeaderExtension(buf, mid)
	if err != nil {
		t.Fatalf("PutRTPMIDHeaderExtension returned error: %v", err)
	}
	if n != len(mid) || string(buf[:n]) != mid {
		t.Fatalf("PutRTPMIDHeaderExtension wrote %d/%q, want %q",
			n, string(buf[:n]), mid)
	}
	got, err := av1.ParseRTPMIDHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPMIDHeaderExtension returned error: %v", err)
	}
	if got != mid {
		t.Fatalf("ParseRTPMIDHeaderExtension = %q, want %q", got, mid)
	}

	tokenChars := "!#$%&'*+-.09AZaz^_`{|}~"
	if err := av1.ValidateRTPMID(tokenChars); err != nil {
		t.Fatalf("ValidateRTPMID rejected SDP token chars: %v", err)
	}
	for _, value := range []string{
		"",
		strings.Repeat("a", av1.RTPHeaderExtensionSDESMaxLen+1),
		"video 0",
		"video/0",
		"video:0",
		"µ",
	} {
		if err := av1.ValidateRTPMID(value); !errors.Is(err, av1.ErrRTPInvalidSDESItem) {
			t.Fatalf("ValidateRTPMID(%q) error = %v, want ErrRTPInvalidSDESItem",
				value, err)
		}
	}
	if _, err := av1.ParseRTPMIDHeaderExtension([]byte("video/0")); !errors.Is(err, av1.ErrRTPInvalidSDESItem) {
		t.Fatalf("ParseRTPMIDHeaderExtension invalid error = %v, want ErrRTPInvalidSDESItem", err)
	}
	if _, err := av1.PutRTPMIDHeaderExtension(make([]byte, len(mid)-1), mid); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("PutRTPMIDHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPStreamIDHeaderExtension(t *testing.T) {
	id := "A1z9"
	buf := make([]byte, len(id)+4)
	n, err := av1.PutRTPStreamIDHeaderExtension(buf, id)
	if err != nil {
		t.Fatalf("PutRTPStreamIDHeaderExtension returned error: %v", err)
	}
	if n != len(id) || string(buf[:n]) != id {
		t.Fatalf("PutRTPStreamIDHeaderExtension wrote %d/%q, want %q",
			n, string(buf[:n]), id)
	}
	got, err := av1.ParseRTPStreamIDHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPStreamIDHeaderExtension returned error: %v", err)
	}
	if got != id {
		t.Fatalf("ParseRTPStreamIDHeaderExtension = %q, want %q", got, id)
	}

	repaired := "R2"
	n, err = av1.PutRTPRepairedStreamIDHeaderExtension(buf, repaired)
	if err != nil {
		t.Fatalf("PutRTPRepairedStreamIDHeaderExtension returned error: %v", err)
	}
	got, err = av1.ParseRTPRepairedStreamIDHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPRepairedStreamIDHeaderExtension returned error: %v", err)
	}
	if got != repaired {
		t.Fatalf("ParseRTPRepairedStreamIDHeaderExtension = %q, want %q",
			got, repaired)
	}

	for _, value := range []string{
		"",
		strings.Repeat("a", av1.RTPHeaderExtensionSDESMaxLen+1),
		"a-b",
		"a_b",
		"a b",
		"stream.0",
		"µ",
	} {
		if err := av1.ValidateRTPStreamID(value); !errors.Is(err, av1.ErrRTPInvalidSDESItem) {
			t.Fatalf("ValidateRTPStreamID(%q) error = %v, want ErrRTPInvalidSDESItem",
				value, err)
		}
	}
	if _, err := av1.ParseRTPStreamIDHeaderExtension([]byte("a-b")); !errors.Is(err, av1.ErrRTPInvalidSDESItem) {
		t.Fatalf("ParseRTPStreamIDHeaderExtension invalid error = %v, want ErrRTPInvalidSDESItem", err)
	}
	if _, err := av1.PutRTPStreamIDHeaderExtension(make([]byte, len(id)-1), id); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("PutRTPStreamIDHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.PutRTPRepairedStreamIDHeaderExtension(make([]byte, len(repaired)-1), repaired); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("PutRTPRepairedStreamIDHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPSDESSourceDescriptionTextHeaderExtension(t *testing.T) {
	value := "camera-α"
	buf := make([]byte, len(value)+4)
	n, err := av1.PutRTPSDESSourceDescriptionTextHeaderExtension(buf, value)
	if err != nil {
		t.Fatalf("PutRTPSDESSourceDescriptionTextHeaderExtension returned error: %v", err)
	}
	if n != len(value) || string(buf[:n]) != value {
		t.Fatalf("PutRTPSDESSourceDescriptionTextHeaderExtension wrote %d/%q, want %q",
			n, string(buf[:n]), value)
	}
	got, err := av1.ParseRTPSDESSourceDescriptionTextHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPSDESSourceDescriptionTextHeaderExtension returned error: %v", err)
	}
	if got != value {
		t.Fatalf("ParseRTPSDESSourceDescriptionTextHeaderExtension = %q, want %q",
			got, value)
	}
	if err := av1.ValidateRTPSDESSourceDescriptionText(""); err != nil {
		t.Fatalf("ValidateRTPSDESSourceDescriptionText rejected empty generic SDES text: %v", err)
	}

	for _, value := range []string{
		strings.Repeat("a", av1.RTPHeaderExtensionSDESMaxLen+1),
		string([]byte{0xff}),
	} {
		if err := av1.ValidateRTPSDESSourceDescriptionText(value); !errors.Is(err, av1.ErrRTPInvalidSDESItem) {
			t.Fatalf("ValidateRTPSDESSourceDescriptionText(%q) error = %v, want ErrRTPInvalidSDESItem",
				value, err)
		}
	}
	if _, err := av1.ParseRTPSDESSourceDescriptionTextHeaderExtension([]byte{0xff}); !errors.Is(err, av1.ErrRTPInvalidSDESItem) {
		t.Fatalf("ParseRTPSDESSourceDescriptionTextHeaderExtension invalid error = %v, want ErrRTPInvalidSDESItem", err)
	}
	if _, err := av1.PutRTPSDESSourceDescriptionTextHeaderExtension(make([]byte, len(value)-1), value); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("PutRTPSDESSourceDescriptionTextHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
}
