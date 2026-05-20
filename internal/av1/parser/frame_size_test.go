package parser

import (
	"errors"
	"testing"
)

func TestParseIntraFrameSizeShownKeyFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false) // superres_params: use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0xff {
		t.Fatalf("RefreshFrameFlags=%02x want ff", size.RefreshFrameFlags)
	}
	if size.CodedWidth != seq.MaxFrameWidth || size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight {
		t.Fatalf("dimensions=%+v seq=%+v", size, seq)
	}
	if size.RenderWidth != seq.MaxFrameWidth || size.RenderHeight != seq.MaxFrameHeight {
		t.Fatalf("render dimensions=%+v", size)
	}
	if size.SuperResEnabled || size.SuperResDenominator != 8 {
		t.Fatalf("superres=%+v", size)
	}
	if size.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", size.BitsRead, w.bit)
	}
}

func TestParseIntraFrameSizeOverrideSuperresRender(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 6
	seq.FrameHeightBits = 5
	seq.MaxFrameWidth = 64
	seq.MaxFrameHeight = 32
	payload, prefix := buildShownKeyFramePrefix(t, seq, true)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(47, seq.FrameWidthBits)  // frame_width_minus_1
	w.writeBits(23, seq.FrameHeightBits) // frame_height_minus_1
	w.writeBool(true)                    // use_superres
	w.writeBits(3, 3)                    // coded_denom = 12
	w.writeBool(true)                    // render_and_frame_size_different
	w.writeBits(39, 16)                  // render_width_minus_1
	w.writeBits(19, 16)                  // render_height_minus_1

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.SuperResEnabled || size.SuperResDenominator != 12 {
		t.Fatalf("superres=%+v", size)
	}
	if size.UpscaledWidth != 48 || size.CodedWidth != 32 || size.Height != 24 {
		t.Fatalf("dimensions=%+v", size)
	}
	if !size.HaveRenderSize || size.RenderWidth != 40 || size.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", size)
	}
}

func TestParseIntraFrameSizeHiddenKeyReadsRefreshAndOrderHints(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(false)                   // show_frame
	w.writeBool(true)                    // showable_frame
	w.writeBool(true)                    // error_resilient_mode
	w.writeBool(false)                   // disable_cdf_update
	w.writeBool(false)                   // frame_size_override_flag
	w.writeBits(4, seq.OrderHintBits)    // order_hint

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}

	w.writeBits(0x01, 8) // refresh_frame_flags
	for i := 0; i < refFrames; i++ {
		w.writeBits(uint64(i), seq.OrderHintBits)
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0x01 {
		t.Fatalf("RefreshFrameFlags=%02x", size.RefreshFrameFlags)
	}
	for i := 0; i < refFrames; i++ {
		if size.RefOrderHints[i] != uint32(i) {
			t.Fatalf("RefOrderHints[%d]=%d", i, size.RefOrderHints[i])
		}
	}
}

func TestParseIntraFrameSizeBufferRemovalTime(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.DecoderModelInfoPresent = true
	seq.TimingInfo.EqualPictureInterval = true
	seq.DecoderModelInfo.BufferRemovalTimeLength = 5
	seq.OperatingPointsCount = 2
	seq.OperatingPoints[0].DecoderModelPresent = true
	seq.OperatingPoints[0].IDC = 0
	seq.OperatingPoints[1].DecoderModelPresent = true
	seq.OperatingPoints[1].IDC = 1 << 1

	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(true)  // buffer_removal_time_present
	w.writeBits(17, 5) // operating point 0
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.BufferRemovalTimePresent || size.BufferRemovalTimes[0] != 17 || size.BufferRemovalTimes[1] != 0 {
		t.Fatalf("buffer removal=%+v", size)
	}
}

func TestParseIntraFrameSizeRejectsIntraOnlyRefreshAll(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                         // show_existing_frame
	w.writeBits(uint64(FrameTypeIntraOnly), 2) // frame_type
	w.writeBool(true)                          // show_frame
	w.writeBool(false)                         // error_resilient_mode
	w.writeBool(false)                         // disable_cdf_update
	w.writeBool(false)                         // frame_size_override_flag
	w.writeBits(0, seq.OrderHintBits)          // order_hint

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	w.writeBits(0xff, 8)

	_, err = ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseIntraFrameSize err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestParseIntraFrameSizeRejectsInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(0, seq.OrderHintBits)
	w.writeBits(0, 3)

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ParseIntraFrameSize(w.bytes(), seq, prefix, 0, 0)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseIntraFrameSize err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseIntraFrameSizeAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseIntraFrameSize(payload, seq, prefix, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseIntraFrameSize allocated: %f", allocs)
	}
}

func BenchmarkParseIntraFrameSize(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefixBench(b, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseIntraFrameSize(payload, seq, prefix, 0, 0)
	}
}

func buildShownKeyFramePrefix(t *testing.T, seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix) {
	t.Helper()
	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, frameSizeOverride)
	if err != nil {
		t.Fatal(err)
	}
	return payload, prefix
}

func buildShownKeyFramePrefixBench(b *testing.B, seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix) {
	b.Helper()
	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, frameSizeOverride)
	if err != nil {
		b.Fatal(err)
	}
	return payload, prefix
}

func buildShownKeyFramePrefixRaw(seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix, error) {
	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(true)                    // show_frame
	w.writeBool(false)                   // disable_cdf_update
	w.writeBool(frameSizeOverride)       // frame_size_override_flag
	w.writeBits(3, seq.OrderHintBits)    // order_hint
	payload := w.bytes()
	prefix, err := ParseFrameHeaderPrefix(payload, seq)
	return payload, prefix, err
}
