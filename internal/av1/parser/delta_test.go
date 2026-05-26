package parser

import "testing"

func TestParseDeltaParamsBaseQZeroReadsNoBits(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 0)

	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if delta.DeltaQPresent || delta.BitsRead != seg.BitsRead {
		t.Fatalf("delta=%+v seg bits=%d", delta, seg.BitsRead)
	}
}

func TestParseDeltaParamsQOnly(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 50)
	w.writeBool(true) // delta_q_present
	w.writeBits(2, 2) // delta_q_res_log2
	w.writeBool(false)

	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if !delta.DeltaQPresent || delta.DeltaQResLog2 != 2 || delta.DeltaLFPresent {
		t.Fatalf("delta=%+v", delta)
	}
	if delta.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", delta.BitsRead, w.bit)
	}
}

func TestParseDeltaParamsLoopFilterMulti(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 50)
	w.writeBool(true) // delta_q_present
	w.writeBits(1, 2) // delta_q_res_log2
	w.writeBool(true) // delta_lf_present
	w.writeBits(3, 2) // delta_lf_res_log2
	w.writeBool(true) // delta_lf_multi

	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if !delta.DeltaQPresent || delta.DeltaQResLog2 != 1 ||
		!delta.DeltaLFPresent || delta.DeltaLFResLog2 != 3 || !delta.DeltaLFMulti {
		t.Fatalf("delta=%+v", delta)
	}
}

func TestParseDeltaParamsIntrabcSkipsLoopFilter(t *testing.T) {
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(2, 2)

	size := FrameSize{AllowIntrabc: true}
	quant := QuantizationParams{BaseQIdx: 50}
	seg := SegmentationParams{}
	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if !delta.DeltaQPresent || delta.DeltaQResLog2 != 2 || delta.DeltaLFPresent || delta.BitsRead != w.bit {
		t.Fatalf("delta=%+v bits=%d", delta, w.bit)
	}
}

func TestParseDeltaParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 50)
	w.writeBool(true)
	w.writeBits(1, 2)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseDeltaParams allocated: %f", allocs)
	}
}

func BenchmarkParseDeltaParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentationBench(b, seq, 50)
	w.writeBool(true)
	w.writeBits(1, 2)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseDeltaParams(payload, size, quant, seg)
	}
}

func buildShownKeyFrameSegmentation(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, QuantizationParams, SegmentationParams) {
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
	return w, size, quant, seg
}

func buildShownKeyFrameSegmentationBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, QuantizationParams, SegmentationParams) {
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
	return w, size, quant, seg
}
