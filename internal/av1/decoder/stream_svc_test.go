package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// appendLowOverheadOBUExt is the extension-header counterpart of
// appendLowOverheadOBU: it sets obu_extension_flag and writes the
// (temporal_id, spatial_id) byte. Used by the SVC propagation tests below.
func appendLowOverheadOBUExt(dst []byte, typ obu.Type, temporalID uint8, spatialID uint8, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{
		Type:         typ,
		HasSizeField: true,
		Extension:    true,
		TemporalID:   temporalID,
		SpatialID:    spatialID,
	})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

// TestStreamPropagatesExtensionLayerIDs verifies that an extension-tagged
// OBU group (the AV1 SVC routing path) lands its (temporal_id, spatial_id)
// onto every Event produced from the group: sequence header, frame OBU,
// and tile-group continuations.
func TestStreamPropagatesExtensionLayerIDs(t *testing.T) {
	cases := []struct {
		name           string
		temporal, spat uint8
	}{
		{"base", 0, 0},
		{"T1S0", 1, 0},
		{"T0S1", 0, 1},
		{"T2S3", 2, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			frame := append([]byte{}, reducedStillFrameHeaderPayload()...)
			frame = append(frame, 0xaa)

			var stream []byte
			stream = appendLowOverheadOBUExt(stream, obu.TypeSequenceHeader, tc.temporal, tc.spat, testSequenceHeaderPayload(16))
			stream = appendLowOverheadOBUExt(stream, obu.TypeFrame, tc.temporal, tc.spat, frame)

			var dec Stream
			var events [2]Event
			count, err := dec.PushLowOverhead(stream, events[:])
			if err != nil {
				t.Fatal(err)
			}
			if count != 2 {
				t.Fatalf("count=%d", count)
			}
			for i, ev := range events[:count] {
				if ev.TemporalID != tc.temporal || ev.SpatialID != tc.spat {
					t.Fatalf("event[%d] T%d S%d want T%d S%d", i, ev.TemporalID, ev.SpatialID, tc.temporal, tc.spat)
				}
			}
		})
	}
}

// TestStreamRejectsExtensionReservedBits verifies that a malformed extension
// byte (with one of the reserved bits set) is rejected at the Stream entry
// point rather than propagated as a "successful" layer-tagged event.
func TestStreamRejectsExtensionReservedBits(t *testing.T) {
	// Manually build a malformed low-overhead unit: extension byte sets the
	// lowest reserved bit. The header parser must reject this before any
	// payload decode.
	raw := []byte{
		byte(obu.TypeSequenceHeader)<<3 | 0x04 /*extension*/ | 0x02, /*has_size_field*/
		0x01, // extension byte: reserved bit set
		0x00, // obu_size = 0
	}
	var dec Stream
	if _, err := pushOBU(&dec, raw, false); err == nil {
		t.Fatal("expected reserved-bit error")
	}
}
