package parser

import (
	"errors"
	"testing"
)

func TestParseFrameHeaderPrefixShownKeyFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(true)                    // show_frame
	w.writeBool(true)                    // disable_cdf_update
	w.writeBool(false)                   // frame_size_override_flag
	w.writeBits(5, seq.OrderHintBits)    // order_hint

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.ShowExistingFrame || hdr.FrameType != FrameTypeKey || !hdr.ShowFrame {
		t.Fatalf("basic keyframe fields=%+v", hdr)
	}
	if !hdr.ErrorResilientMode || !hdr.DisableCDFUpdate {
		t.Fatalf("flags=%+v", hdr)
	}
	if hdr.FrameSizeOverride || hdr.OrderHint != 5 || hdr.PrimaryRefFrame != PrimaryRefNone {
		t.Fatalf("routing fields=%+v", hdr)
	}
	if hdr.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", hdr.BitsRead, w.bit)
	}
}

func TestParseFrameHeaderPrefixHiddenInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                     // show_existing_frame
	w.writeBits(uint64(FrameTypeInter), 2) // frame_type
	w.writeBool(false)                     // show_frame
	w.writeBool(true)                      // showable_frame
	w.writeBool(false)                     // error_resilient_mode
	w.writeBool(false)                     // disable_cdf_update
	w.writeBool(true)                      // frame_size_override_flag
	w.writeBits(9, seq.OrderHintBits)      // order_hint
	w.writeBits(3, 3)                      // primary_ref_frame

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.FrameType != FrameTypeInter || hdr.ShowFrame || !hdr.ShowableFrame {
		t.Fatalf("visibility fields=%+v", hdr)
	}
	if hdr.ErrorResilientMode || hdr.DisableCDFUpdate {
		t.Fatalf("flags=%+v", hdr)
	}
	if !hdr.FrameSizeOverride || hdr.OrderHint != 9 || hdr.PrimaryRefFrame != 3 {
		t.Fatalf("routing fields=%+v", hdr)
	}
}

func TestParseFrameHeaderPrefixShowExistingFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.DecoderModelInfoPresent = true
	seq.TimingInfo.EqualPictureInterval = false
	seq.DecoderModelInfo.FramePresentationTimeLength = 5
	seq.FrameIDNumbersPresent = true
	seq.DeltaFrameIDLength = 4
	seq.AdditionalFrameIDLength = 2

	var w testBitWriter
	w.writeBool(true)  // show_existing_frame
	w.writeBits(3, 3)  // existing_frame_idx
	w.writeBits(17, 5) // frame_presentation_delay
	w.writeBits(42, seq.FrameIDBits())

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if !hdr.ShowExistingFrame || hdr.ExistingFrameIdx != 3 {
		t.Fatalf("show existing fields=%+v", hdr)
	}
	if hdr.FramePresentationDelay != 17 || hdr.FrameID != 42 {
		t.Fatalf("timing/id fields=%+v", hdr)
	}
	if hdr.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", hdr.BitsRead, w.bit)
	}
}

func TestParseFrameHeaderPrefixReducedStillPicture(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, reducedStillPictureSequenceHeader())

	var w testBitWriter
	w.writeBool(false) // disable_cdf_update

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.ShowExistingFrame || hdr.FrameType != FrameTypeKey || !hdr.ShowFrame {
		t.Fatalf("reduced still fields=%+v", hdr)
	}
	if !hdr.ErrorResilientMode || hdr.DisableCDFUpdate {
		t.Fatalf("flags=%+v", hdr)
	}
	if hdr.BitsRead != 1 {
		t.Fatalf("BitsRead=%d want 1", hdr.BitsRead)
	}
}

func TestParseFrameHeaderPrefixRejectsInvalidStillPicture(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.StillPicture = true

	var w testBitWriter
	w.writeBool(false)                     // show_existing_frame
	w.writeBits(uint64(FrameTypeInter), 2) // frame_type
	w.writeBool(true)                      // show_frame

	_, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseFrameHeaderPrefix err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestParseFrameHeaderPrefixAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(9, seq.OrderHintBits)
	w.writeBits(3, 3)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameHeaderPrefix allocated: %f", allocs)
	}
}

func BenchmarkParseFrameHeaderPrefix(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(9, seq.OrderHintBits)
	w.writeBits(3, 3)
	payload := w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseFrameHeaderPrefix(payload, seq)
	}
}

func mustParseTestSequenceHeader(t *testing.T, payload []byte) SequenceHeader {
	t.Helper()
	seq, err := ParseSequenceHeader(payload)
	if err != nil {
		t.Fatal(err)
	}
	return seq
}

func mustParseBenchSequenceHeader(b *testing.B, payload []byte) SequenceHeader {
	b.Helper()
	seq, err := ParseSequenceHeader(payload)
	if err != nil {
		b.Fatal(err)
	}
	return seq
}
