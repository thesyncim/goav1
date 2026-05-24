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
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

func reducedStillFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(true)  // disable_cdf_update
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
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
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func interFrameHeaderPayloadWithReferenceState(update bool) []byte {
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
	writeQuantParams(&w, 40)
	if update {
		writeSegmentationUpdateData(&w)
	} else {
		writeSegmentationCopyData(&w)
	}
	w.writeBool(false) // delta_q_present
	if update {
		writeLoopFilterDeltaUpdate(&w)
	} else {
		writeLoopFilterDeltaCopy(&w)
	}
	writeMinimalCDEF(&w)
	w.writeBool(false) // transform_mode_select
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func showExistingFrameHeaderPayload(index uint8) []byte {
	var w testBitWriter
	w.writeBool(true) // show_existing_frame
	w.writeBits(uint64(index&7), 3)
	return w.bytes()
}

func writeZeroQuantParams(w *testBitWriter) {
	writeQuantParams(w, 0)
}

func writeQuantParams(w *testBitWriter, baseQIdx uint8) {
	w.writeBits(uint64(baseQIdx), 8) // base_q_idx
	w.writeBool(false)               // y_dc_delta_q
	w.writeBool(false)               // diff_uv_delta
	w.writeBool(false)               // u_dc_delta_q
	w.writeBool(false)               // u_ac_delta_q
	w.writeBool(false)               // using_qmatrix
}

func writeSegmentationUpdateData(w *testBitWriter) {
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(true)  // update_data
	w.writeBool(true)  // segment 0 delta_q enabled
	writeSignedBits(w, 5, 9)
	w.writeBool(false) // delta_lf_y_v
	w.writeBool(false) // delta_lf_y_h
	w.writeBool(false) // delta_lf_u
	w.writeBool(false) // delta_lf_v
	w.writeBool(false) // ref_frame
	w.writeBool(false) // skip
	w.writeBool(false) // globalmv
	for i := 1; i < parser.MaxSegments; i++ {
		writeEmptySegmentData(w)
	}
}

func writeSegmentationCopyData(w *testBitWriter) {
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(false) // update_data
}

func writeEmptySegmentData(w *testBitWriter) {
	w.writeBool(false) // delta_q
	w.writeBool(false) // delta_lf_y_v
	w.writeBool(false) // delta_lf_y_h
	w.writeBool(false) // delta_lf_u
	w.writeBool(false) // delta_lf_v
	w.writeBool(false) // ref_frame
	w.writeBool(false) // skip
	w.writeBool(false) // globalmv
}

func writeLoopFilterDeltaUpdate(w *testBitWriter) {
	w.writeBits(10, 6) // loop_filter_level[0]
	w.writeBits(0, 6)  // loop_filter_level[1]
	w.writeBits(5, 6)  // loop_filter_level_u
	w.writeBits(6, 6)  // loop_filter_level_v
	w.writeBits(0, 3)  // loop_filter_sharpness
	w.writeBool(true)  // mode_ref_delta_enabled
	w.writeBool(true)  // mode_ref_delta_update
	for i := 0; i < parser.RefFrames; i++ {
		update := i == 2
		w.writeBool(update)
		if update {
			writeSignedBits(w, 4, 7)
		}
	}
	for i := 0; i < parser.LoopFilterModeDeltas; i++ {
		w.writeBool(false)
	}
}

func writeLoopFilterDeltaCopy(w *testBitWriter) {
	w.writeBits(11, 6) // loop_filter_level[0]
	w.writeBits(0, 6)  // loop_filter_level[1]
	w.writeBits(5, 6)  // loop_filter_level_u
	w.writeBits(6, 6)  // loop_filter_level_v
	w.writeBits(0, 3)  // loop_filter_sharpness
	w.writeBool(true)  // mode_ref_delta_enabled
	w.writeBool(false) // mode_ref_delta_update
}

func writeMinimalCDEF(w *testBitWriter) {
	w.writeBits(0, 2) // cdef_damping_minus_3
	w.writeBits(0, 2) // cdef_bits
	w.writeBits(0, 6) // cdef_y_pri_strength + cdef_y_sec_strength
	w.writeBits(0, 6) // cdef_uv_pri_strength + cdef_uv_sec_strength
}

func writeSignedBits(w *testBitWriter, value int16, n uint8) {
	mask := (uint64(1) << n) - 1
	w.writeBits(uint64(uint16(value))&mask, n)
}

func writeZeroSegmentationParams(w *testBitWriter) {
	w.writeBool(false) // segmentation_enabled
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
	if events[1].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("frame sequence=%+v want %+v", events[1].SequenceHeader, events[0].SequenceHeader)
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
	if events[1].Segmentation.Enabled || !events[1].Segmentation.AllLossless {
		t.Fatalf("segmentation=%+v", events[1].Segmentation)
	}
	if events[1].Delta.DeltaQPresent || events[1].Delta.BitsRead != events[1].Segmentation.BitsRead {
		t.Fatalf("delta=%+v", events[1].Delta)
	}
	if !events[1].LoopFilter.ModeRefDeltaEnabled || !events[1].LoopFilter.ModeRefDeltaUpdate {
		t.Fatalf("loopfilter=%+v", events[1].LoopFilter)
	}
	if events[1].CDEF.BitsRead != events[1].LoopFilter.BitsRead {
		t.Fatalf("cdef=%+v", events[1].CDEF)
	}
	if events[1].Restoration.BitsRead != events[1].CDEF.BitsRead {
		t.Fatalf("restoration=%+v", events[1].Restoration)
	}
	if events[1].TransformRef.TransformMode != parser.TransformMode4x4Only ||
		events[1].TransformRef.ReferenceMode != parser.ReferenceModeSingle ||
		events[1].TransformRef.BitsRead != events[1].Restoration.BitsRead {
		t.Fatalf("transform/reference=%+v", events[1].TransformRef)
	}
	if events[1].SkipMode.Allowed || events[1].SkipMode.Enabled ||
		events[1].SkipMode.BitsRead != events[1].TransformRef.BitsRead {
		t.Fatalf("skip mode=%+v", events[1].SkipMode)
	}
	if events[1].FrameMode.AllowWarpedMotion || events[1].FrameMode.ReducedTxSet ||
		events[1].FrameMode.BitsRead != events[1].SkipMode.BitsRead+1 {
		t.Fatalf("frame mode=%+v", events[1].FrameMode)
	}
	if events[1].GlobalMotion.BitsRead != events[1].FrameMode.BitsRead {
		t.Fatalf("global motion=%+v", events[1].GlobalMotion)
	}
	if events[1].FilmGrain.Apply || events[1].FilmGrain.BitsRead != events[1].GlobalMotion.BitsRead {
		t.Fatalf("film grain=%+v", events[1].FilmGrain)
	}
	if events[2].Kind != EventTileGroup {
		t.Fatalf("tile event=%+v", events[2])
	}
	if events[2].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("tile sequence=%+v want %+v", events[2].SequenceHeader, events[0].SequenceHeader)
	}
	if events[2].FrameHeader != events[1].FrameHeader || events[2].FrameSize != events[1].FrameSize ||
		events[2].Quantization != events[1].Quantization || events[2].Segmentation != events[1].Segmentation ||
		events[2].LoopFilter != events[1].LoopFilter || events[2].FilmGrain != events[1].FilmGrain {
		t.Fatalf("tile frame state=%+v header event=%+v", events[2], events[1])
	}
	if events[2].TileGroup.StartTile != 0 || events[2].TileGroup.EndTile != 0 ||
		events[2].TileGroup.DataOffset != 0 || events[2].TileGroup.DataSize != 1 || !events[2].TileGroup.Final {
		t.Fatalf("tile group=%+v", events[2].TileGroup)
	}
}

func TestStreamFrameOBUParsesImplicitTileGroup(t *testing.T) {
	frame := append([]byte{}, reducedStillFrameHeaderPayload()...)
	frame = append(frame, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, frame)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	if events[1].Kind != EventFrame {
		t.Fatalf("frame event=%+v", events[1])
	}
	if events[1].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("frame sequence=%+v want %+v", events[1].SequenceHeader, events[0].SequenceHeader)
	}
	if events[1].TileGroup.StartTile != 0 || events[1].TileGroup.EndTile != 0 ||
		events[1].TileGroup.DataOffset != len(reducedStillFrameHeaderPayload()) ||
		events[1].TileGroup.DataSize != 1 || !events[1].TileGroup.Final {
		t.Fatalf("frame tile group=%+v", events[1].TileGroup)
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

func TestStreamRejectsTileAfterFinalTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [4]Event
	_, err := dec.PushLowOverhead(stream, events[:])
	if !errors.Is(err, ErrMissingFrameHeader) {
		t.Fatalf("PushLowOverhead err=%v want %v", err, ErrMissingFrameHeader)
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
	if events[1].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("events[1] sequence=%+v want %+v", events[1].SequenceHeader, events[0].SequenceHeader)
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
	if events[1].Segmentation.Enabled || !events[1].Segmentation.AllLossless {
		t.Fatalf("events[1] segmentation=%+v", events[1].Segmentation)
	}
	if events[1].Delta.DeltaQPresent || events[1].Delta.BitsRead != events[1].Segmentation.BitsRead {
		t.Fatalf("events[1] delta=%+v", events[1].Delta)
	}
	if !events[1].LoopFilter.ModeRefDeltaEnabled || !events[1].LoopFilter.ModeRefDeltaUpdate {
		t.Fatalf("events[1] loopfilter=%+v", events[1].LoopFilter)
	}
	if events[1].CDEF.BitsRead != events[1].LoopFilter.BitsRead {
		t.Fatalf("events[1] cdef=%+v", events[1].CDEF)
	}
	if events[1].Restoration.BitsRead != events[1].CDEF.BitsRead {
		t.Fatalf("events[1] restoration=%+v", events[1].Restoration)
	}
	if events[1].TransformRef.TransformMode != parser.TransformMode4x4Only ||
		events[1].TransformRef.ReferenceMode != parser.ReferenceModeSingle ||
		events[1].TransformRef.BitsRead != events[1].Restoration.BitsRead {
		t.Fatalf("events[1] transform/reference=%+v", events[1].TransformRef)
	}
	if events[1].SkipMode.Allowed || events[1].SkipMode.Enabled ||
		events[1].SkipMode.BitsRead != events[1].TransformRef.BitsRead {
		t.Fatalf("events[1] skip mode=%+v", events[1].SkipMode)
	}
	if events[1].FrameMode.AllowWarpedMotion || events[1].FrameMode.ReducedTxSet ||
		events[1].FrameMode.BitsRead != events[1].SkipMode.BitsRead+1 {
		t.Fatalf("events[1] frame mode=%+v", events[1].FrameMode)
	}
	if events[1].GlobalMotion.BitsRead != events[1].FrameMode.BitsRead {
		t.Fatalf("events[1] global motion=%+v", events[1].GlobalMotion)
	}
	if events[1].FilmGrain.Apply || events[1].FilmGrain.BitsRead != events[1].GlobalMotion.BitsRead {
		t.Fatalf("events[1] film grain=%+v", events[1].FilmGrain)
	}
}

func TestStreamRTPPayloadFragmentedFrameHeader(t *testing.T) {
	var dec Stream
	var out [256]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event

	sequence := []rtp.Element{{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))}}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                1,
		StartsNewCodedVideoSequence: true,
	}, sequence)
	if err != nil {
		t.Fatal(err)
	}
	used, count, err := dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used == 0 || count != 1 || events[0].Kind != EventSequenceHeader {
		t.Fatalf("sequence used=%d count=%d event=%+v", used, count, events[0])
	}

	frameHeader := appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	var packet [8]byte
	n, next, more, err := rtp.PutFragment(packet[:], frameHeader, 0, 4, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	used = 0
	used, count, err = dec.PushRTPPayload(out[:], used, spans[:], events[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || !dec.InRTPFragment() {
		t.Fatalf("first fragment count=%d inFragment=%v", count, dec.InRTPFragment())
	}

	n, _, more, err = rtp.PutFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("unexpected third fragment")
	}
	used, count, err = dec.PushRTPPayload(out[:], used, spans[:], events[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != len(frameHeader) || count != 1 || dec.InRTPFragment() {
		t.Fatalf("final fragment used=%d count=%d inFragment=%v", used, count, dec.InRTPFragment())
	}
	if events[0].Kind != EventFrameHeader || events[0].FrameHeader.FrameType != parser.FrameTypeKey {
		t.Fatalf("frame header event=%+v", events[0])
	}
}

func TestStreamRTPPayloadAllocs(t *testing.T) {
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

	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	allocs := testing.AllocsPerRun(1000, func() {
		var dec Stream
		used, count, err := dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
		if err != nil {
			t.Fatal(err)
		}
		if used == 0 || count != 2 {
			t.Fatalf("used=%d count=%d", used, count)
		}
	})
	if allocs != 0 {
		t.Fatalf("Stream.PushRTPPayload allocated: %f", allocs)
	}
}

func BenchmarkStreamRTPPayload(b *testing.B) {
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
		b.Fatal(err)
	}

	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var dec Stream
		_, _, _ = dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
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
	if events[3].Segmentation.Enabled || !events[3].Segmentation.AllLossless {
		t.Fatalf("inter segmentation=%+v", events[3].Segmentation)
	}
	if events[3].Delta.DeltaQPresent || events[3].Delta.BitsRead != events[3].Segmentation.BitsRead {
		t.Fatalf("inter delta=%+v", events[3].Delta)
	}
	if !events[3].LoopFilter.ModeRefDeltaEnabled || !events[3].LoopFilter.ModeRefDeltaUpdate {
		t.Fatalf("inter loopfilter=%+v", events[3].LoopFilter)
	}
	if events[3].CDEF.BitsRead != events[3].LoopFilter.BitsRead {
		t.Fatalf("inter cdef=%+v", events[3].CDEF)
	}
	if events[3].Restoration.BitsRead != events[3].CDEF.BitsRead {
		t.Fatalf("inter restoration=%+v", events[3].Restoration)
	}
	if events[3].TransformRef.TransformMode != parser.TransformMode4x4Only ||
		events[3].TransformRef.ReferenceMode != parser.ReferenceModeSingle ||
		events[3].TransformRef.BitsRead != events[3].Restoration.BitsRead+1 {
		t.Fatalf("inter transform/reference=%+v", events[3].TransformRef)
	}
	if events[3].SkipMode.Allowed || events[3].SkipMode.Enabled ||
		events[3].SkipMode.BitsRead != events[3].TransformRef.BitsRead {
		t.Fatalf("inter skip mode=%+v", events[3].SkipMode)
	}
	if events[3].FrameMode.AllowWarpedMotion || events[3].FrameMode.ReducedTxSet ||
		events[3].FrameMode.BitsRead != events[3].SkipMode.BitsRead+1 {
		t.Fatalf("inter frame mode=%+v", events[3].FrameMode)
	}
	if events[3].GlobalMotion.BitsRead != events[3].FrameMode.BitsRead+parser.InterRefsPerFrame {
		t.Fatalf("inter global motion=%+v", events[3].GlobalMotion)
	}
	if events[3].FilmGrain.Apply || events[3].FilmGrain.BitsRead != events[3].GlobalMotion.BitsRead {
		t.Fatalf("inter film grain=%+v", events[3].FilmGrain)
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if events[3].GlobalMotion.Ref[i].Type != parser.GlobalMotionIdentity {
			t.Fatalf("inter global motion ref[%d]=%+v", i, events[3].GlobalMotion.Ref[i])
		}
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if events[3].FrameSize.RefFrameIdx[i] != 0 {
			t.Fatalf("inter RefFrameIdx[%d]=%d", i, events[3].FrameSize.RefFrameIdx[i])
		}
	}
}

func TestReferenceOrderHintsFromReferenceState(t *testing.T) {
	refs := parser.ReferenceState{}
	refs.Frames[2] = parser.ReferenceFrame{Valid: true, OrderHint: 11}
	refs.Frames[5] = parser.ReferenceFrame{Valid: true, OrderHint: 21}
	size := parser.FrameSize{
		RefFrameIdx: [parser.InterRefsPerFrame]uint8{2, 5, 2, 5, 2, 5, 2},
	}
	got := referenceOrderHints(size, &refs)
	want := [parser.InterRefsPerFrame]uint32{11, 21, 11, 21, 11, 21, 11}
	if got != want {
		t.Fatalf("reference order hints=%v want %v", got, want)
	}
}

func TestStreamCarriesReferenceFrameHeaderState(t *testing.T) {
	var dec Stream
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}

	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, keyFrame), false); err != nil {
		t.Fatal(err)
	}

	updateFrame := append([]byte{}, interFrameHeaderPayloadWithReferenceState(true)...)
	updateFrame = append(updateFrame, 0xbb)
	updateEvent, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, updateFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if !updateEvent.Segmentation.Enabled || !updateEvent.Segmentation.UpdateData ||
		updateEvent.Segmentation.Data.Segments[0].DeltaQ != 5 || updateEvent.Segmentation.QIndex[0] != 45 {
		t.Fatalf("updated segmentation=%+v", updateEvent.Segmentation)
	}
	if !updateEvent.LoopFilter.ModeRefDeltaUpdate || updateEvent.LoopFilter.Deltas.Ref[2] != 4 {
		t.Fatalf("updated loopfilter=%+v", updateEvent.LoopFilter)
	}
	if dec.references.Frames[0].Segmentation.Segments[0].DeltaQ != 5 ||
		dec.references.Frames[0].LoopFilterDeltas.Ref[2] != 4 {
		t.Fatalf("stored reference=%+v", dec.references.Frames[0])
	}

	copyFrame := append([]byte{}, interFrameHeaderPayloadWithReferenceState(false)...)
	copyFrame = append(copyFrame, 0xcc)
	copyEvent, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, copyFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if !copyEvent.Segmentation.Enabled || copyEvent.Segmentation.UpdateData ||
		copyEvent.Segmentation.Data.Segments[0].DeltaQ != 5 || copyEvent.Segmentation.QIndex[0] != 45 {
		t.Fatalf("copied segmentation=%+v", copyEvent.Segmentation)
	}
	if copyEvent.LoopFilter.ModeRefDeltaUpdate || copyEvent.LoopFilter.Deltas.Ref[2] != 4 {
		t.Fatalf("copied loopfilter=%+v", copyEvent.LoopFilter)
	}
}

func TestStreamShowExistingFrame(t *testing.T) {
	var dec Stream
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}

	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, keyFrame), false); err != nil {
		t.Fatal(err)
	}

	interFrame := append([]byte{}, interFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	interEvent, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, interFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if interEvent.Kind != EventFrame || interEvent.FrameHeader.FrameType != parser.FrameTypeInter {
		t.Fatalf("inter event=%+v", interEvent)
	}

	event, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrameHeader, showExistingFrameHeaderPayload(0)), false)
	if err != nil {
		t.Fatal(err)
	}
	if event.Kind != EventExistingFrame || !event.FrameHeader.ShowExistingFrame || event.FrameHeader.ExistingFrameIdx != 0 {
		t.Fatalf("show existing event=%+v", event)
	}
	if !event.ExistingFrame.Valid || event.ExistingFrame.FrameType != parser.FrameTypeInter {
		t.Fatalf("existing frame=%+v", event.ExistingFrame)
	}
	if event.FrameSize != interEvent.FrameSize || event.FilmGrain != interEvent.FilmGrain {
		t.Fatalf("show existing state=%+v inter=%+v", event, interEvent)
	}

	_, err = dec.PushOBU(appendRTPElement(nil, obu.TypeTileGroup, []byte{0x80}), false)
	if !errors.Is(err, ErrMissingFrameHeader) {
		t.Fatalf("PushOBU err=%v want %v", err, ErrMissingFrameHeader)
	}
}

func TestStreamRejectsShowExistingBeforeReference(t *testing.T) {
	var dec Stream
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	_, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrameHeader, showExistingFrameHeaderPayload(0)), false)
	if !errors.Is(err, parser.ErrReferenceFrameNeeded) {
		t.Fatalf("PushOBU err=%v want %v", err, parser.ErrReferenceFrameNeeded)
	}
}

func TestStreamRejectsUnshowableExistingFrame(t *testing.T) {
	var dec Stream
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, keyFrame), false); err != nil {
		t.Fatal(err)
	}

	_, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrameHeader, showExistingFrameHeaderPayload(0)), false)
	if !errors.Is(err, parser.ErrInvalidFrameHeader) {
		t.Fatalf("PushOBU err=%v want %v", err, parser.ErrInvalidFrameHeader)
	}
}

func TestStreamRejectsFrameOBUShowExistingFrame(t *testing.T) {
	var dec Stream
	if _, err := dec.PushOBU(appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	_, err := dec.PushOBU(appendRTPElement(nil, obu.TypeFrame, showExistingFrameHeaderPayload(0)), false)
	if !errors.Is(err, parser.ErrInvalidFrameHeader) {
		t.Fatalf("PushOBU err=%v want %v", err, parser.ErrInvalidFrameHeader)
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
