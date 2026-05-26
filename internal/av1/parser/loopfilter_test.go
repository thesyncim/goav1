package parser

import (
	"errors"
	"testing"
)

func TestParseLoopFilterParamsLosslessDefaults(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, _, seg := buildShownKeyFrameSegmentation(t, seq, 0)
	delta, err := ParseDeltaParams(w.bytes(), size, QuantizationParams{}, seg)
	if err != nil {
		t.Fatal(err)
	}

	lf, err := ParseLoopFilterParams(w.bytes(), seq, FrameHeaderPrefix{}, size, seg, delta, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate {
		t.Fatalf("loopfilter=%+v", lf)
	}
	want := defaultLoopFilterDeltas()
	if lf.Deltas != want {
		t.Fatalf("deltas=%+v want %+v", lf.Deltas, want)
	}
	if lf.BitsRead != delta.BitsRead {
		t.Fatalf("BitsRead=%d want %d", lf.BitsRead, delta.BitsRead)
	}
}

func TestParseLoopFilterParamsLevelsAndUpdates(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size, seg, delta := buildShownKeyFrameDelta(t, seq, 40)
	w.writeBits(10, 6) // loop_filter_level[0]
	w.writeBits(0, 6)  // loop_filter_level[1]
	w.writeBits(5, 6)  // loop_filter_level_u
	w.writeBits(6, 6)  // loop_filter_level_v
	w.writeBits(3, 3)  // loop_filter_sharpness
	w.writeBool(true)  // mode_ref_delta_enabled
	w.writeBool(true)  // mode_ref_delta_update
	w.writeBool(true)  // ref_delta[0] update
	writeSignedBitsTest(&w, -2, 7)
	for i := 1; i < RefFrames; i++ {
		w.writeBool(false)
	}
	w.writeBool(false) // mode_delta[0] update
	w.writeBool(true)  // mode_delta[1] update
	writeSignedBitsTest(&w, 3, 7)

	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lf.LevelY[0] != 10 || lf.LevelY[1] != 0 || lf.LevelU != 5 || lf.LevelV != 6 || lf.Sharpness != 3 {
		t.Fatalf("levels=%+v", lf)
	}
	if !lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate {
		t.Fatalf("delta flags=%+v", lf)
	}
	if lf.Deltas.Ref[0] != -2 || lf.Deltas.Ref[4] != -1 || lf.Deltas.Mode[1] != 3 {
		t.Fatalf("deltas=%+v", lf.Deltas)
	}
	if lf.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", lf.BitsRead, w.bit)
	}
}

func TestParseLoopFilterParamsMonochromeSkipsUVLevels(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	var w testBitWriter
	w.writeBits(12, 6)
	w.writeBits(0, 6)
	w.writeBits(1, 3)
	w.writeBool(false)
	prefix := FrameHeaderPrefix{PrimaryRefFrame: PrimaryRefNone}
	seg := SegmentationParams{AllLossless: false}

	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, FrameSize{}, seg, DeltaParams{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lf.LevelY[0] != 12 || lf.LevelU != 0 || lf.LevelV != 0 || lf.Sharpness != 1 {
		t.Fatalf("loopfilter=%+v", lf)
	}
	if lf.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", lf.BitsRead, w.bit)
	}
}

func TestParseLoopFilterParamsCopiesPreviousDeltas(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	var w testBitWriter
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)

	previous := defaultLoopFilterDeltas()
	previous.Ref[2] = 4
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	seg := SegmentationParams{AllLossless: false}

	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, FrameSize{}, seg, DeltaParams{}, &previous)
	if err != nil {
		t.Fatal(err)
	}
	if lf.Deltas.Ref[2] != 4 || lf.Deltas.Ref[4] != -1 {
		t.Fatalf("deltas=%+v", lf.Deltas)
	}
}

func TestParseLoopFilterParamsRejectsMissingPreviousDeltas(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	var w testBitWriter
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	seg := SegmentationParams{AllLossless: false}

	_, err := ParseLoopFilterParams(w.bytes(), seq, prefix, FrameSize{}, seg, DeltaParams{}, nil)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseLoopFilterParams err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseLoopFilterParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size, seg, delta := buildShownKeyFrameDelta(t, seq, 40)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, nil)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseLoopFilterParams allocated: %f", allocs)
	}
}

func BenchmarkParseLoopFilterParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, prefix, size, seg, delta := buildShownKeyFrameDeltaBench(b, seq, 40)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, nil)
	}
}

func buildShownKeyFrameDelta(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, FrameSize, SegmentationParams, DeltaParams) {
	t.Helper()
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	w.writeBool(false)
	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if baseQIdx != 0 {
		w.writeBool(false)
	}
	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, size, seg, delta
}

func buildShownKeyFrameDeltaBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, FrameSize, SegmentationParams, DeltaParams) {
	b.Helper()
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		b.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		b.Fatal(err)
	}
	w.writeBool(false)
	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		b.Fatal(err)
	}
	if baseQIdx != 0 {
		w.writeBool(false)
	}
	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		b.Fatal(err)
	}
	return w, prefix, size, seg, delta
}
