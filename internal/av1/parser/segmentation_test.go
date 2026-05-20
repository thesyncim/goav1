package parser

import (
	"errors"
	"testing"
)

func TestParseSegmentationParamsDisabledLossless(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 0)
	w.writeBool(false) // segmentation_enabled

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if seg.Enabled || !seg.AllLossless {
		t.Fatalf("segmentation=%+v", seg)
	}
	for i := 0; i < MaxSegments; i++ {
		if seg.Data.Segments[i].RefFrame != -1 || seg.QIndex[i] != 0 || !seg.Lossless[i] {
			t.Fatalf("segment[%d]=%+v q=%d lossless=%v", i, seg.Data.Segments[i], seg.QIndex[i], seg.Lossless[i])
		}
	}
	if seg.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", seg.BitsRead, w.bit)
	}
}

func TestParseSegmentationParamsDisabledNotLossless(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 10)
	w.writeBool(false)

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if seg.AllLossless || seg.QIndex[0] != 10 || seg.Lossless[0] {
		t.Fatalf("segmentation=%+v", seg)
	}
}

func TestParseSegmentationParamsPrimaryRefNoneUpdateData(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 100)
	w.writeBool(true) // segmentation_enabled
	w.writeBool(true) // segment 0 delta_q enabled
	writeSignedBitsTest(&w, 10, 9)
	w.writeBool(false) // delta_lf_y_v
	w.writeBool(false) // delta_lf_y_h
	w.writeBool(false) // delta_lf_u
	w.writeBool(false) // delta_lf_v
	w.writeBool(true)  // ref_frame enabled
	w.writeBits(2, 3)
	w.writeBool(true)  // skip
	w.writeBool(false) // globalmv
	for i := 1; i < MaxSegments; i++ {
		writeEmptySegmentData(&w)
	}

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !seg.Enabled || !seg.UpdateMap || !seg.UpdateData || seg.TemporalUpdate {
		t.Fatalf("segmentation flags=%+v", seg)
	}
	if !seg.Data.Preskip || seg.Data.LastActiveID != 0 {
		t.Fatalf("active/preskip=%+v", seg.Data)
	}
	first := seg.Data.Segments[0]
	if first.DeltaQ != 10 || first.RefFrame != 2 || !first.Skip || first.GlobalMV {
		t.Fatalf("segment 0=%+v", first)
	}
	if seg.QIndex[0] != 110 || seg.QIndex[1] != 100 {
		t.Fatalf("qindex=%v", seg.QIndex[:2])
	}
}

func TestParseSegmentationParamsCopiesPreviousData(t *testing.T) {
	var previous SegmentationData
	clearSegmentationRefs(&previous)
	previous.Segments[0].DeltaQ = 5

	var w testBitWriter
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(false) // update_data
	quant := QuantizationParams{BaseQIdx: 20}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, &previous)
	if err != nil {
		t.Fatal(err)
	}
	if !seg.Enabled || seg.UpdateMap || seg.UpdateData {
		t.Fatalf("segmentation flags=%+v", seg)
	}
	if seg.Data.Segments[0].DeltaQ != 5 || seg.QIndex[0] != 25 {
		t.Fatalf("copied data=%+v q=%d", seg.Data.Segments[0], seg.QIndex[0])
	}
}

func TestParseSegmentationParamsRejectsMissingPreviousData(t *testing.T) {
	var w testBitWriter
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(false) // update_data
	quant := QuantizationParams{BaseQIdx: 20}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}

	_, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseSegmentationParams err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseSegmentationParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 0)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseSegmentationParams(payload, prefix, quant, nil)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseSegmentationParams allocated: %f", allocs)
	}
}

func BenchmarkParseSegmentationParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuantBench(b, seq, 0)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseSegmentationParams(payload, prefix, quant, nil)
	}
}

func buildShownKeyFrameQuant(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, QuantizationParams) {
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
	return w, prefix, quant
}

func buildShownKeyFrameQuantBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, QuantizationParams) {
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
	return w, prefix, quant
}

func writeSharedUVZeroQuant(w *testBitWriter, baseQIdx uint8) {
	w.writeBits(uint64(baseQIdx), 8)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
}

func writeEmptySegmentData(w *testBitWriter) {
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
}
