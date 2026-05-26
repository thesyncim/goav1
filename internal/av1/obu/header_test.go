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
	for tid := uint8(0); tid < 8; tid++ {
		for sid := uint8(0); sid < 4; sid++ {
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

// TestParseHeaderExtensionRejectsAllReservedBits verifies that every nonzero
// pattern of the reserved 3-bit tail in obu_extension_header() is rejected
// (one bit at a time and all together). The spec requires the reserved bits
// to be exactly zero; libaom raises a corrupt-frame error otherwise and
// goav1 surfaces ErrReservedBit.
func TestParseHeaderExtensionRejectsAllReservedBits(t *testing.T) {
	for reserved := uint8(1); reserved <= 0x07; reserved++ {
		src := []byte{
			byte(TypeFrameHeader)<<3 | 0x04,
			reserved, // temporal_id=0, spatial_id=0, reserved=reserved
		}
		_, _, err := ParseHeader(src)
		if !errors.Is(err, ErrReservedBit) {
			t.Fatalf("reserved=%#x: err=%v want %v", reserved, err, ErrReservedBit)
		}
	}
}

// TestPutHeaderRoundTripLayerIDs verifies PutHeader writes the 3+2+3-bit
// layout and ParseHeader recovers the same IDs.
func TestPutHeaderRoundTripLayerIDs(t *testing.T) {
	for tid := uint8(0); tid < 8; tid++ {
		for sid := uint8(0); sid < 4; sid++ {
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
