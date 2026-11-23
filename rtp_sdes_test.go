package goav1

import (
	"errors"
	"strings"
	"testing"
)

func TestPublicRTPSDESHeaderExtensionConstants(t *testing.T) {
	if RTPSDESItemTypeRTPStreamID != 12 ||
		RTPSDESItemTypeRepairedRTPStreamID != 13 ||
		RTPSDESItemTypeMID != 15 {
		t.Fatalf("unexpected RTP SDES item type constants")
	}
	if RTPHeaderExtensionSDESMaxLen != 255 {
		t.Fatalf("RTPHeaderExtensionSDESMaxLen = %d, want 255",
			RTPHeaderExtensionSDESMaxLen)
	}
}

func TestPublicRTPMIDHeaderExtension(t *testing.T) {
	mid := "video-0"
	buf := make([]byte, len(mid)+4)
	n, err := PutRTPMIDHeaderExtension(buf, mid)
	if err != nil {
		t.Fatalf("PutRTPMIDHeaderExtension returned error: %v", err)
	}
	if n != len(mid) || string(buf[:n]) != mid {
		t.Fatalf("PutRTPMIDHeaderExtension wrote %d/%q, want %q",
			n, string(buf[:n]), mid)
	}
	got, err := ParseRTPMIDHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPMIDHeaderExtension returned error: %v", err)
	}
	if got != mid {
		t.Fatalf("ParseRTPMIDHeaderExtension = %q, want %q", got, mid)
	}

	tokenChars := "!#$%&'*+-.09AZaz^_`{|}~"
	if err := ValidateRTPMID(tokenChars); err != nil {
		t.Fatalf("ValidateRTPMID rejected SDP token chars: %v", err)
	}
	for _, value := range []string{
		"",
		strings.Repeat("a", RTPHeaderExtensionSDESMaxLen+1),
		"video 0",
		"video/0",
		"video:0",
		"µ",
	} {
		if err := ValidateRTPMID(value); !errors.Is(err, ErrRTPInvalidSDESItem) {
			t.Fatalf("ValidateRTPMID(%q) error = %v, want ErrRTPInvalidSDESItem",
				value, err)
		}
	}
	if _, err := ParseRTPMIDHeaderExtension([]byte("video/0")); !errors.Is(err, ErrRTPInvalidSDESItem) {
		t.Fatalf("ParseRTPMIDHeaderExtension invalid error = %v, want ErrRTPInvalidSDESItem", err)
	}
	if _, err := PutRTPMIDHeaderExtension(make([]byte, len(mid)-1), mid); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("PutRTPMIDHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPStreamIDHeaderExtension(t *testing.T) {
	id := "A1z9"
	buf := make([]byte, len(id)+4)
	n, err := PutRTPStreamIDHeaderExtension(buf, id)
	if err != nil {
		t.Fatalf("PutRTPStreamIDHeaderExtension returned error: %v", err)
	}
	if n != len(id) || string(buf[:n]) != id {
		t.Fatalf("PutRTPStreamIDHeaderExtension wrote %d/%q, want %q",
			n, string(buf[:n]), id)
	}
	got, err := ParseRTPStreamIDHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPStreamIDHeaderExtension returned error: %v", err)
	}
	if got != id {
		t.Fatalf("ParseRTPStreamIDHeaderExtension = %q, want %q", got, id)
	}

	repaired := "R2"
	n, err = PutRTPRepairedStreamIDHeaderExtension(buf, repaired)
	if err != nil {
		t.Fatalf("PutRTPRepairedStreamIDHeaderExtension returned error: %v", err)
	}
	got, err = ParseRTPRepairedStreamIDHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPRepairedStreamIDHeaderExtension returned error: %v", err)
	}
	if got != repaired {
		t.Fatalf("ParseRTPRepairedStreamIDHeaderExtension = %q, want %q",
			got, repaired)
	}

	for _, value := range []string{
		"",
		strings.Repeat("a", RTPHeaderExtensionSDESMaxLen+1),
		"a-b",
		"a_b",
		"a b",
		"stream.0",
		"µ",
	} {
		if err := ValidateRTPStreamID(value); !errors.Is(err, ErrRTPInvalidSDESItem) {
			t.Fatalf("ValidateRTPStreamID(%q) error = %v, want ErrRTPInvalidSDESItem",
				value, err)
		}
	}
	if _, err := ParseRTPStreamIDHeaderExtension([]byte("a-b")); !errors.Is(err, ErrRTPInvalidSDESItem) {
		t.Fatalf("ParseRTPStreamIDHeaderExtension invalid error = %v, want ErrRTPInvalidSDESItem", err)
	}
	if _, err := PutRTPStreamIDHeaderExtension(make([]byte, len(id)-1), id); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("PutRTPStreamIDHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := PutRTPRepairedStreamIDHeaderExtension(make([]byte, len(repaired)-1), repaired); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("PutRTPRepairedStreamIDHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPSDESSourceDescriptionTextHeaderExtension(t *testing.T) {
	value := "camera-α"
	buf := make([]byte, len(value)+4)
	n, err := PutRTPSDESSourceDescriptionTextHeaderExtension(buf, value)
	if err != nil {
		t.Fatalf("PutRTPSDESSourceDescriptionTextHeaderExtension returned error: %v", err)
	}
	if n != len(value) || string(buf[:n]) != value {
		t.Fatalf("PutRTPSDESSourceDescriptionTextHeaderExtension wrote %d/%q, want %q",
			n, string(buf[:n]), value)
	}
	got, err := ParseRTPSDESSourceDescriptionTextHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPSDESSourceDescriptionTextHeaderExtension returned error: %v", err)
	}
	if got != value {
		t.Fatalf("ParseRTPSDESSourceDescriptionTextHeaderExtension = %q, want %q",
			got, value)
	}
	if err := ValidateRTPSDESSourceDescriptionText(""); err != nil {
		t.Fatalf("ValidateRTPSDESSourceDescriptionText rejected empty generic SDES text: %v", err)
	}

	for _, value := range []string{
		strings.Repeat("a", RTPHeaderExtensionSDESMaxLen+1),
		string([]byte{0xff}),
	} {
		if err := ValidateRTPSDESSourceDescriptionText(value); !errors.Is(err, ErrRTPInvalidSDESItem) {
			t.Fatalf("ValidateRTPSDESSourceDescriptionText(%q) error = %v, want ErrRTPInvalidSDESItem",
				value, err)
		}
	}
	if _, err := ParseRTPSDESSourceDescriptionTextHeaderExtension([]byte{0xff}); !errors.Is(err, ErrRTPInvalidSDESItem) {
		t.Fatalf("ParseRTPSDESSourceDescriptionTextHeaderExtension invalid error = %v, want ErrRTPInvalidSDESItem", err)
	}
	if _, err := PutRTPSDESSourceDescriptionTextHeaderExtension(make([]byte, len(value)-1), value); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("PutRTPSDESSourceDescriptionTextHeaderExtension short error = %v, want ErrRTPShortBuffer", err)
	}
}
