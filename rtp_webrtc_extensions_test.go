package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicRTPWebRTCHeaderExtensionConstants(t *testing.T) {
	if av1.RTPTransportWideCCHeaderExtensionSize != 2 {
		t.Fatalf("RTPTransportWideCCHeaderExtensionSize = %d, want 2", av1.RTPTransportWideCCHeaderExtensionSize)
	}
	if av1.RTPAbsoluteSendTimeHeaderExtensionSize != 3 {
		t.Fatalf("RTPAbsoluteSendTimeHeaderExtensionSize = %d, want 3", av1.RTPAbsoluteSendTimeHeaderExtensionSize)
	}
	if av1.RTPAbsoluteSendTimeMaxValue != 0x00ffffff {
		t.Fatalf("RTPAbsoluteSendTimeMaxValue = %#x, want 0x00ffffff", av1.RTPAbsoluteSendTimeMaxValue)
	}
}

func TestPublicRTPTransportWideCCHeaderExtension(t *testing.T) {
	var buf [4]byte
	n, err := av1.PutRTPTransportWideCCHeaderExtension(buf[:], 0x1234)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCCHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPTransportWideCCHeaderExtensionSize || buf[0] != 0x12 || buf[1] != 0x34 {
		t.Fatalf("encoded transport-wide cc n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPTransportWideCCHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCCHeaderExtension returned error: %v", err)
	}
	if got != 0x1234 {
		t.Fatalf("ParseRTPTransportWideCCHeaderExtension = %#x, want 0x1234", got)
	}
	if _, err := av1.ParseRTPTransportWideCCHeaderExtension(buf[:1]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPTransportWideCCHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPTransportWideCCHeaderExtension(buf[:3]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPTransportWideCCHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPTransportWideCCHeaderExtension(buf[:1], 1); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCCHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPAbsoluteSendTimeHeaderExtension(t *testing.T) {
	var buf [4]byte
	n, err := av1.PutRTPAbsoluteSendTimeHeaderExtension(buf[:], 0xabcdef)
	if err != nil {
		t.Fatalf("PutRTPAbsoluteSendTimeHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPAbsoluteSendTimeHeaderExtensionSize ||
		buf[0] != 0xab || buf[1] != 0xcd || buf[2] != 0xef {
		t.Fatalf("encoded abs-send-time n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPAbsoluteSendTimeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPAbsoluteSendTimeHeaderExtension returned error: %v", err)
	}
	if got != 0xabcdef {
		t.Fatalf("ParseRTPAbsoluteSendTimeHeaderExtension = %#x, want 0xabcdef", got)
	}
	if _, err := av1.ParseRTPAbsoluteSendTimeHeaderExtension(buf[:2]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPAbsoluteSendTimeHeaderExtension(buf[:4]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPAbsoluteSendTimeHeaderExtension(buf[:2], 1); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.PutRTPAbsoluteSendTimeHeaderExtension(buf[:], 0x01000000); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("large PutRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
}
