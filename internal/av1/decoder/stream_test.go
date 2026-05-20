package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/rtp"
)

type testBitWriter struct {
	buf [128]byte
	bit int
}

func (w *testBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 != 0 {
			w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *testBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *testBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func testSequenceHeaderPayload(width uint64) []byte {
	var w testBitWriter
	w.writeBits(0, 3)       // seq_profile
	w.writeBool(true)       // still_picture
	w.writeBool(true)       // reduced_still_picture_header
	w.writeBits(5, 5)       // seq_level_idx[0]
	w.writeBits(7, 4)       // frame_width_bits_minus_1
	w.writeBits(3, 4)       // frame_height_bits_minus_1
	w.writeBits(width-1, 8) // max_frame_width_minus_1
	w.writeBits(8, 4)       // max_frame_height_minus_1
	w.writeBool(false)      // use_128x128_superblock
	w.writeBool(true)       // enable_filter_intra
	w.writeBool(true)       // enable_intra_edge_filter
	w.writeBool(false)      // enable_superres
	w.writeBool(true)       // enable_cdef
	w.writeBool(false)      // enable_restoration
	w.writeBool(false)      // high_bitdepth
	w.writeBool(false)      // mono_chrome
	w.writeBool(false)      // color_description_present_flag
	w.writeBool(false)      // color_range
	w.writeBits(0, 2)       // chroma_sample_position
	w.writeBool(true)       // separate_uv_delta_q
	w.writeBool(false)      // film_grain_params_present
	return w.trailingBits()
}

func appendLowOverheadOBU(dst []byte, typ obu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{Type: typ, HasSizeField: true})
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

func appendRTPElement(dst []byte, typ obu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{Type: typ})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	dst = append(dst, payload...)
	return dst
}

func TestStreamLowOverheadState(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, []byte{0x80})
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [4]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	if events[0].Kind != EventSequenceHeader || !events[0].NewCodedVideoSequence {
		t.Fatalf("sequence event=%+v", events[0])
	}
	if events[1].Kind != EventFrameHeader {
		t.Fatalf("frame header event=%+v", events[1])
	}
	if events[1].FrameHeader.FrameType != parser.FrameTypeKey || !events[1].FrameHeader.ShowFrame {
		t.Fatalf("frame header parse=%+v", events[1].FrameHeader)
	}
	if events[2].Kind != EventTileGroup {
		t.Fatalf("tile event=%+v", events[2])
	}
}

func TestStreamRejectsFrameBeforeSequenceHeader(t *testing.T) {
	var dec Stream
	_, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrameHeader, []byte{0x80}), false)
	if !errors.Is(err, ErrMissingSequenceHeader) {
		t.Fatalf("PushOBU err=%v want %v", err, ErrMissingSequenceHeader)
	}
}

func TestStreamRejectsTileBeforeFrameHeader(t *testing.T) {
	var dec Stream
	_, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16)), false)
	if err != nil {
		t.Fatal(err)
	}
	_, err = dec.PushOBU(appendRTPElement(nil, obu.TypeTileGroup, []byte{0x80}), false)
	if !errors.Is(err, ErrMissingFrameHeader) {
		t.Fatalf("PushOBU err=%v want %v", err, ErrMissingFrameHeader)
	}
}

func TestStreamSequenceChange(t *testing.T) {
	var dec Stream
	first, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16)), false)
	if err != nil {
		t.Fatal(err)
	}
	second, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16)), false)
	if err != nil {
		t.Fatal(err)
	}
	third, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(32)), false)
	if err != nil {
		t.Fatal(err)
	}

	if !first.NewCodedVideoSequence || second.NewCodedVideoSequence || !third.NewCodedVideoSequence {
		t.Fatalf("new sequence flags: first=%v second=%v third=%v", first.NewCodedVideoSequence, second.NewCodedVideoSequence, third.NewCodedVideoSequence)
	}
}

func TestStreamRTPPayload(t *testing.T) {
	elements := []rtp.Element{
		{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))},
		{Data: appendRTPElement(nil, obu.TypeFrameHeader, []byte{0x80})},
	}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var dec Stream
	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	used, count, err := dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used == 0 || count != 2 {
		t.Fatalf("used=%d count=%d", used, count)
	}
	if events[0].Kind != EventSequenceHeader || !events[0].NewCodedVideoSequence {
		t.Fatalf("events[0]=%+v", events[0])
	}
	if events[1].Kind != EventFrameHeader {
		t.Fatalf("events[1]=%+v", events[1])
	}
	if events[1].FrameHeader.FrameType != parser.FrameTypeKey || !events[1].FrameHeader.DisableCDFUpdate {
		t.Fatalf("events[1] frame header=%+v", events[1].FrameHeader)
	}
}

func TestStreamAllocs(t *testing.T) {
	raw := appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	allocs := testing.AllocsPerRun(1000, func() {
		var dec Stream
		_, err := dec.PushOBU(raw, false)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Stream allocated: %f", allocs)
	}
}

func BenchmarkStreamPushOBU(b *testing.B) {
	raw := appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var dec Stream
		_, _ = dec.PushOBU(raw, false)
	}
}
