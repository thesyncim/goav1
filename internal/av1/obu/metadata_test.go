// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package obu

import (
	"bytes"
	"errors"
	"testing"
)

func TestParseMetadataITUTT35(t *testing.T) {
	// metadata_type=4 (1 byte leb128 = 0x04), country_code=0xB5 (US),
	// payload "ABC", trailing byte 0x80.
	src := []byte{0x04, 0xB5, 'A', 'B', 'C', 0x80}
	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if meta.Type != MetadataTypeITUTT35 {
		t.Fatalf("type=%v want ITUT_T35", meta.Type)
	}
	if meta.ITUTT35.CountryCode != 0xB5 || meta.ITUTT35.HasCountryExtension {
		t.Fatalf("country=%#v", meta.ITUTT35)
	}
	if !bytes.Equal(meta.ITUTT35.Payload, []byte("ABC")) {
		t.Fatalf("payload=%q", meta.ITUTT35.Payload)
	}
	// Payload must alias src.
	if &meta.ITUTT35.Payload[0] != &src[2] {
		t.Fatalf("ITUTT35 payload should alias src")
	}
}

func TestParseMetadataITUTT35Extension(t *testing.T) {
	// country_code=0xFF triggers an extension byte.
	src := []byte{0x04, 0xFF, 0x42, 'X', 0x80}
	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if !meta.ITUTT35.HasCountryExtension || meta.ITUTT35.CountryCodeExtension != 0x42 {
		t.Fatalf("extension=%#v", meta.ITUTT35)
	}
	if !bytes.Equal(meta.ITUTT35.Payload, []byte{'X'}) {
		t.Fatalf("payload=%q", meta.ITUTT35.Payload)
	}
}

func TestParseMetadataITUTT35Errors(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		err  error
	}{
		{"missing country", []byte{0x04}, ErrMetadataMissingCountry},
		{"missing extension", []byte{0x04, 0xFF}, ErrMetadataMissingCountryEx},
		{"missing trailing", []byte{0x04, 0xB5, 0x00, 0x00}, ErrMetadataMissingTrailing},
		{"invalid trailing", []byte{0x04, 0xB5, 'A', 0x40}, ErrMetadataInvalidTrailing},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseMetadata(tt.in)
			if !errors.Is(err, tt.err) {
				t.Fatalf("err=%v want %v", err, tt.err)
			}
		})
	}
}

func TestParseMetadataHDRCLL(t *testing.T) {
	// metadata_type=1, max_cll=1000 (0x03E8), max_fall=400 (0x0190), then 0x80.
	src := []byte{0x01, 0x03, 0xE8, 0x01, 0x90, 0x80}
	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if meta.Type != MetadataTypeHDRCLL {
		t.Fatalf("type=%v", meta.Type)
	}
	if meta.HDRCLL.MaxCLL != 1000 || meta.HDRCLL.MaxFALL != 400 {
		t.Fatalf("cll=%#v", meta.HDRCLL)
	}
}

func TestParseMetadataHDRCLLErrors(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		err  error
	}{
		{"short", []byte{0x01, 0x00, 0x00, 0x00}, ErrMetadataShortPayload},
		{"missing trailing", []byte{0x01, 0x00, 0x00, 0x00, 0x00}, ErrMetadataMissingTrailing},
		{"invalid trailing", []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x40}, ErrMetadataInvalidTrailing},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseMetadata(tt.in)
			if !errors.Is(err, tt.err) {
				t.Fatalf("err=%v want %v", err, tt.err)
			}
		})
	}
}

func TestParseMetadataHDRMDCV(t *testing.T) {
	// metadata_type=2 followed by 24 fixed bytes + 0x80 trailing.
	src := []byte{
		0x02,
		// Primary R x=0x1234 y=0x5678
		0x12, 0x34, 0x56, 0x78,
		// Primary G x=0x9ABC y=0xDEF0
		0x9A, 0xBC, 0xDE, 0xF0,
		// Primary B x=0x1122 y=0x3344
		0x11, 0x22, 0x33, 0x44,
		// White point x=0x5566 y=0x7788
		0x55, 0x66, 0x77, 0x88,
		// max_lum=0x01020304, min_lum=0x05060708
		0x01, 0x02, 0x03, 0x04,
		0x05, 0x06, 0x07, 0x08,
		0x80,
	}
	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if meta.Type != MetadataTypeHDRMDCV {
		t.Fatalf("type=%v", meta.Type)
	}
	want := MetadataHDRMDCV{
		PrimaryChromaticityX:    [3]uint16{0x1234, 0x9ABC, 0x1122},
		PrimaryChromaticityY:    [3]uint16{0x5678, 0xDEF0, 0x3344},
		WhitePointChromaticityX: 0x5566,
		WhitePointChromaticityY: 0x7788,
		LuminanceMax:            0x01020304,
		LuminanceMin:            0x05060708,
	}
	if meta.HDRMDCV != want {
		t.Fatalf("mdcv=%#v want %#v", meta.HDRMDCV, want)
	}
}

func TestParseMetadataScalability(t *testing.T) {
	// metadata_type=3 (0x03), mode_idc=2 (L2T1), trailing 0x80.
	src := []byte{0x03, 0x02, 0x80}
	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if meta.Type != MetadataTypeScalability {
		t.Fatalf("type=%v", meta.Type)
	}
	if meta.Scalability.ModeIDC != 2 || meta.Scalability.HasStructure {
		t.Fatalf("scalability=%#v", meta.Scalability)
	}
}

func TestParseMetadataScalabilitySS(t *testing.T) {
	// mode_idc=14 (SS) plus an empty scalability_structure: 1 spatial layer
	// (count_minus_1=0), no dimensions/description/temporal-group flags,
	// reserved 3 bits = 0. Bits: 00 0 0 0 000 -> 0x00 byte. Then trailing 0x80.
	src := []byte{0x03, 0x0E, 0x00, 0x80}
	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if !meta.Scalability.HasStructure {
		t.Fatalf("expected scalability structure")
	}
	if meta.Scalability.Structure.SpatialLayersCountMinus1 != 0 {
		t.Fatalf("layers=%d", meta.Scalability.Structure.SpatialLayersCountMinus1)
	}
}

func TestParseMetadataTimecode(t *testing.T) {
	// metadata_type=5, payload built with full_timestamp=1.
	// Bit layout:
	//   counting_type=5 bits (=1)
	//   full_timestamp=1
	//   discontinuity=0
	//   cnt_dropped=0
	//   n_frames=9 bits (=10)
	//   seconds_value=6 (=30)
	//   minutes_value=6 (=15)
	//   hours_value=5 (=2)
	//   time_offset_length=5 (=0)
	//   trailing 1 bit + zero-padding to byte.
	// 5+1+1+1+9+6+6+5+5+1 = 40 bits = 5 bytes.
	// Pack via helper to avoid hand-shifting errors.
	w := newBitWriter()
	w.writeBits(uint64(1), 5)
	w.writeBits(1, 1)
	w.writeBits(0, 1)
	w.writeBits(0, 1)
	w.writeBits(10, 9)
	w.writeBits(30, 6)
	w.writeBits(15, 6)
	w.writeBits(2, 5)
	w.writeBits(0, 5)
	// trailing bits.
	w.writeBits(1, 1)
	for w.bitCount&7 != 0 {
		w.writeBits(0, 1)
	}
	src := append([]byte{0x05}, w.bytes()...)

	meta, err := ParseMetadata(src)
	if err != nil {
		t.Fatalf("ParseMetadata err=%v", err)
	}
	if meta.Type != MetadataTypeTimecode {
		t.Fatalf("type=%v", meta.Type)
	}
	if meta.Timecode.CountingType != 1 || !meta.Timecode.FullTimestamp {
		t.Fatalf("timecode=%#v", meta.Timecode)
	}
	if meta.Timecode.NFrames != 10 || meta.Timecode.Seconds != 30 || meta.Timecode.Minutes != 15 || meta.Timecode.Hours != 2 {
		t.Fatalf("timecode fields=%#v", meta.Timecode)
	}
	if meta.Timecode.TimeOffsetLength != 0 {
		t.Fatalf("offset_len=%d", meta.Timecode.TimeOffsetLength)
	}
}

func TestParseMetadataReservedRequiresPayload(t *testing.T) {
	// metadata_type=6 (reserved). Per spec the OBU is ignored, but at least
	// one nonzero trailing byte must be present.
	if _, err := ParseMetadata([]byte{0x06, 0x00}); !errors.Is(err, ErrMetadataInvalidReserved) {
		t.Fatalf("expected ErrMetadataInvalidReserved, got %v", err)
	}
	if _, err := ParseMetadata([]byte{0x06, 0x80}); err != nil {
		t.Fatalf("expected reserved with trailing to succeed, got %v", err)
	}
}

func TestParseMetadataEmpty(t *testing.T) {
	if _, err := ParseMetadata(nil); !errors.Is(err, ErrMetadataShortPayload) {
		t.Fatalf("err=%v want ErrMetadataShortPayload", err)
	}
}

// bitWriter is a minimal MSB-first bit packer used by the timecode test.
type bitWriter struct {
	buf      []byte
	bitCount int
}

func newBitWriter() *bitWriter { return &bitWriter{} }

func (w *bitWriter) writeBits(v uint64, n int) {
	for i := n - 1; i >= 0; i-- {
		bit := byte((v >> uint(i)) & 1)
		byteIdx := w.bitCount >> 3
		if byteIdx == len(w.buf) {
			w.buf = append(w.buf, 0)
		}
		w.buf[byteIdx] |= bit << uint(7-(w.bitCount&7))
		w.bitCount++
	}
}

func (w *bitWriter) bytes() []byte { return w.buf }
