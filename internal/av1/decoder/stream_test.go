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

func (w *testBitWriter) bytes() []byte {
	return w.buf[:(w.bit+7)>>3]
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

func testRealtimeNoOrderSequenceHeaderPayload() []byte {
	var w testBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(0, 12) // operating_point_idc[0]
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // frame_id_numbers_present_flag
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(false) // enable_order_hint
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(false) // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBits(0, 2)  // chroma_sample_position
	w.writeBool(true)  // separate_uv_delta_q
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func shownKeyFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(false)                          // show_existing_frame
	w.writeBits(uint64(parser.FrameTypeKey), 2) // frame_type
	w.writeBool(true)                           // show_frame
	w.writeBool(false)                          // disable_cdf_update
	w.writeBool(false)                          // frame_size_override_flag
	w.writeBool(false)                          // render_and_frame_size_different
	w.writeBool(false)                          // disable_frame_end_update_cdf
	w.writeBool(false)                          // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	return w.bytes()
}

func reducedStillFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(true)  // disable_cdf_update
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	return w.bytes()
}

func interFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(false)                            // show_existing_frame
	w.writeBits(uint64(parser.FrameTypeInter), 2) // frame_type
	w.writeBool(true)                             // show_frame
	w.writeBool(false)                            // error_resilient_mode
	w.writeBool(false)                            // disable_cdf_update
	w.writeBool(false)                            // frame_size_override_flag
	w.writeBits(0, 3)                             // primary_ref_frame
	w.writeBits(0x01, 8)                          // refresh_frame_flags
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(0, 2)  // interpolation_filter = EIGHTTAP
	w.writeBool(false) // is_motion_mode_switchable
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	return w.bytes()
}

func writeZeroQuantParams(w *testBitWriter) {
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // diff_uv_delta
	w.writeBool(false) // u_dc_delta_q
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(false) // using_qmatrix
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
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
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
	if events[1].FrameSize.CodedWidth != 16 || events[1].FrameSize.Height != 9 {
		t.Fatalf("frame size=%+v", events[1].FrameSize)
	}
	if events[1].TileInfo.Cols != 1 || events[1].TileInfo.Rows != 1 {
		t.Fatalf("tile info=%+v", events[1].TileInfo)
	}
	if events[1].Quantization.BaseQIdx != 0 {
		t.Fatalf("quantization=%+v", events[1].Quantization)
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
		{Data: appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())},
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
	if events[1].FrameSize.CodedWidth != 16 || events[1].FrameSize.RenderHeight != 9 {
		t.Fatalf("events[1] frame size=%+v", events[1].FrameSize)
	}
	if events[1].TileInfo.Cols != 1 || events[1].TileInfo.Rows != 1 {
		t.Fatalf("events[1] tile info=%+v", events[1].TileInfo)
	}
	if events[1].Quantization.BaseQIdx != 0 {
		t.Fatalf("events[1] quantization=%+v", events[1].Quantization)
	}
}

func TestStreamInterFrameUsesReferenceState(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, shownKeyFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTemporalDelimiter, nil)
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, interFrameHeaderPayload())

	var dec Stream
	var events [4]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 4 {
		t.Fatalf("count=%d", count)
	}
	if events[1].FrameSize.RefreshFrameFlags != 0xff {
		t.Fatalf("key frame size=%+v", events[1].FrameSize)
	}
	if events[3].FrameHeader.FrameType != parser.FrameTypeInter {
		t.Fatalf("inter header=%+v", events[3].FrameHeader)
	}
	if events[3].FrameSize.RefreshFrameFlags != 0x01 {
		t.Fatalf("inter frame size=%+v", events[3].FrameSize)
	}
	if events[3].TileInfo.Cols != 1 || events[3].TileInfo.Rows != 1 {
		t.Fatalf("inter tile info=%+v", events[3].TileInfo)
	}
	if events[3].Quantization.BaseQIdx != 0 {
		t.Fatalf("inter quantization=%+v", events[3].Quantization)
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if events[3].FrameSize.RefFrameIdx[i] != 0 {
			t.Fatalf("inter RefFrameIdx[%d]=%d", i, events[3].FrameSize.RefFrameIdx[i])
		}
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
