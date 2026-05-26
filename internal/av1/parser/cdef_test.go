package parser

import "testing"

func TestParseCDEFParamsSkipsLossless(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, _, seg := buildShownKeyFrameSegmentation(t, seq, 0)
	lf := LoopFilterParams{BitsRead: w.bit}

	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		t.Fatal(err)
	}
	if cdef.StrengthCount != 0 || cdef.BitsRead != lf.BitsRead {
		t.Fatalf("cdef=%+v lf bits=%d", cdef, lf.BitsRead)
	}
}

func TestParseCDEFParamsStrengths(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, lf := buildShownKeyFrameLoopFilter(t, seq, 40)
	w.writeBits(2, 2) // cdef_damping_minus_3
	w.writeBits(2, 2) // cdef_bits
	for i := range 4 {
		w.writeBits(uint64(10+i), 6)
		w.writeBits(uint64(20+i), 6)
	}

	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		t.Fatal(err)
	}
	if cdef.Damping != 5 || cdef.Bits != 2 || cdef.StrengthCount != 4 {
		t.Fatalf("cdef=%+v", cdef)
	}
	if cdef.YStrength[3] != 13 || cdef.UVStrength[3] != 23 {
		t.Fatalf("strengths y=%v uv=%v", cdef.YStrength[:4], cdef.UVStrength[:4])
	}
	if cdef.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", cdef.BitsRead, w.bit)
	}
}

func TestParseCDEFParamsMonochromeSkipsUVStrengths(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	var w testBitWriter
	w.writeBits(1, 2)
	w.writeBits(1, 2)
	w.writeBits(7, 6)
	w.writeBits(8, 6)
	seg := SegmentationParams{AllLossless: false}

	cdef, err := ParseCDEFParams(w.bytes(), seq, FrameSize{}, seg, LoopFilterParams{})
	if err != nil {
		t.Fatal(err)
	}
	if cdef.StrengthCount != 2 || cdef.YStrength[0] != 7 || cdef.YStrength[1] != 8 {
		t.Fatalf("cdef=%+v", cdef)
	}
	if cdef.UVStrength[0] != 0 || cdef.UVStrength[1] != 0 || cdef.BitsRead != w.bit {
		t.Fatalf("uv/bits cdef=%+v want bits %d", cdef, w.bit)
	}
}

func TestParseCDEFParamsDisabledBySequence(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.EnableCDEF = false
	var w testBitWriter
	seg := SegmentationParams{AllLossless: false}

	cdef, err := ParseCDEFParams(w.bytes(), seq, FrameSize{}, seg, LoopFilterParams{})
	if err != nil {
		t.Fatal(err)
	}
	if cdef.BitsRead != 0 || cdef.StrengthCount != 0 {
		t.Fatalf("cdef=%+v", cdef)
	}
}

func TestParseCDEFParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, lf := buildShownKeyFrameLoopFilter(t, seq, 40)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	w.writeBits(9, 6)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseCDEFParams allocated: %f", allocs)
	}
}

func BenchmarkParseCDEFParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, size, seg, lf := buildShownKeyFrameLoopFilterBench(b, seq, 40)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	w.writeBits(9, 6)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseCDEFParams(payload, seq, size, seg, lf)
	}
}

func buildShownKeyFrameLoopFilter(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, LoopFilterParams) {
	t.Helper()
	w, prefix, size, seg, delta := buildShownKeyFrameDelta(t, seq, baseQIdx)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatal(err)
	}
	return w, size, seg, lf
}

func buildShownKeyFrameLoopFilterBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, LoopFilterParams) {
	b.Helper()
	w, prefix, size, seg, delta := buildShownKeyFrameDeltaBench(b, seq, baseQIdx)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, size, seg, delta, nil)
	if err != nil {
		b.Fatal(err)
	}
	return w, size, seg, lf
}
