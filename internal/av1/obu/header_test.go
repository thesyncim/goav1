package obu

import (
	"errors"
	"testing"
)

// TestParseHeaderExtensionLayerIDs exercises the obu_extension_header() byte
// layout per AV1 spec section 5.3.3:
//
//	temporal_id : 3 bits (high)
//	spatial_id  : 2 bits
//	reserved    : 3 bits (must be zero)
func TestParseHeaderExtensionLayerIDs(t *testing.T) {
	// Walk every (temporal_id, spatial_id) pair allowed by the 3+2-bit
	// extension byte: 8 x 4 = 32 combinations.
	for tid := range uint8(8) {
		for sid := range uint8(4) {
			ext := (tid << 5) | (sid << 3)
			src := []byte{
				byte(TypeFrameHeader)<<3 | 0x04, // extension_flag = 1, has_size_field = 0
				ext,
			}
			h, n, err := ParseHeader(src)
			if err != nil {
				t.Fatalf("tid=%d sid=%d ext=%#x: ParseHeader err=%v", tid, sid, ext, err)
			}
			if n != 2 {
				t.Fatalf("tid=%d sid=%d: consumed=%d want 2", tid, sid, n)
			}
			if h.TemporalID != tid || h.SpatialID != sid {
				t.Fatalf("ext=%#x: got T%d S%d want T%d S%d", ext, h.TemporalID, h.SpatialID, tid, sid)
			}
			if !h.Extension {
				t.Fatalf("ext=%#x: Extension flag dropped", ext)
			}
		}
	}
}

// TestParseHeaderIgnoresReservedBits verifies that obu_reserved_1bit and
// every pattern of extension_header_reserved_3bits are ignored, as the spec
// requires of a decoder and as libaom (av1/common/obu_util.c) and dav1d
// (src/obu.c) do: the header still parses with its type and layer IDs.
func TestParseHeaderIgnoresReservedBits(t *testing.T) {
	h, n, err := ParseHeader([]byte{byte(TypeTemporalDelimiter)<<3 | 0x02 | 0x01})
	if err != nil || n != 1 || h.Type != TypeTemporalDelimiter || !h.HasSizeField || h.Extension {
		t.Fatalf("obu_reserved_1bit set: header=%+v n=%d err=%v", h, n, err)
	}
	for reserved := uint8(1); reserved <= 0x07; reserved++ {
		src := []byte{
			byte(TypeFrameHeader)<<3 | 0x04 | 0x01,
			5<<5 | 2<<3 | reserved, // temporal_id=5, spatial_id=2
		}
		h, n, err := ParseHeader(src)
		if err != nil || n != 2 || h.Type != TypeFrameHeader || h.TemporalID != 5 || h.SpatialID != 2 {
			t.Fatalf("reserved=%#x: header=%+v n=%d err=%v", reserved, h, n, err)
		}
	}
}

// TestPutHeaderRoundTripLayerIDs verifies PutHeader writes the 3+2+3-bit
// layout and ParseHeader recovers the same IDs.
func TestPutHeaderRoundTripLayerIDs(t *testing.T) {
	for tid := range uint8(8) {
		for sid := range uint8(4) {
			in := Header{
				Type:       TypeFrame,
				Extension:  true,
				TemporalID: tid,
				SpatialID:  sid,
			}
			var buf [2]byte
			n, err := PutHeader(buf[:], in)
			if err != nil {
				t.Fatalf("PutHeader T%d S%d: %v", tid, sid, err)
			}
			if n != 2 {
				t.Fatalf("PutHeader T%d S%d: n=%d want 2", tid, sid, n)
			}
			// Reserved tail bits must be cleared by the writer.
			if buf[1]&0x07 != 0 {
				t.Fatalf("PutHeader leaked reserved bits: byte=%#x", buf[1])
			}
			got, _, err := ParseHeader(buf[:])
			if err != nil {
				t.Fatalf("ParseHeader T%d S%d: %v", tid, sid, err)
			}
			if got != in {
				t.Fatalf("round-trip T%d S%d: got=%+v", tid, sid, got)
			}
		}
	}
}

// TestPutHeaderRejectsOversizedLayerIDs guards the header layout against
// callers that ask for IDs the AV1 extension byte cannot represent.
func TestPutHeaderRejectsOversizedLayerIDs(t *testing.T) {
	cases := []Header{
		{Type: TypeFrame, Extension: true, TemporalID: 8, SpatialID: 0},
		{Type: TypeFrame, Extension: true, TemporalID: 0, SpatialID: 4},
	}
	for _, h := range cases {
		var buf [2]byte
		if _, err := PutHeader(buf[:], h); !errors.Is(err, ErrInvalidType) {
			t.Fatalf("PutHeader(%+v) err=%v want %v", h, err, ErrInvalidType)
		}
	}
}
