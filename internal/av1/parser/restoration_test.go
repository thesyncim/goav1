package parser

import "testing"

func TestParseRestorationParamsSkipsLosslessWithoutSuperres(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, _, seg := buildShownKeyFrameSegmentation(t, seq, 0)
	cdef := CDEFParams{BitsRead: w.bit}

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.BitsRead != cdef.BitsRead || restoration.UnitSizeY != 0 {
		t.Fatalf("restoration=%+v cdef bits=%d", restoration, cdef.BitsRead)
	}
}

func TestParseRestorationParamsAllNone(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.Type[0] != RestorationNone || restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone {
		t.Fatalf("types=%+v", restoration.Type)
	}
	if restoration.UnitSizeY != RestorationUnitMax || restoration.UnitSizeUV != RestorationUnitMax ||
		restoration.UnitSizeYLog2 != 8 || restoration.UnitSizeUVLog2 != 8 {
		t.Fatalf("unit sizes=%+v", restoration)
	}
	if restoration.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", restoration.BitsRead, w.bit)
	}
}

func TestParseRestorationParamsTypesAndUnitSizes(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.SubsamplingX = true
	seq.ColorConfig.SubsamplingY = true
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationWiener), 2)
	w.writeBits(uint64(RestorationSGRProj), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBool(true) // grow from 64 to 128
	w.writeBool(false)
	w.writeBool(true) // chroma unit is half luma for 4:2:0

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.Type[0] != RestorationWiener || restoration.Type[1] != RestorationSGRProj || restoration.Type[2] != RestorationNone {
		t.Fatalf("types=%+v", restoration.Type)
	}
	if restoration.UnitSizeY != 128 || restoration.UnitSizeUV != 64 ||
		restoration.UnitSizeYLog2 != 7 || restoration.UnitSizeUVLog2 != 6 {
		t.Fatalf("unit sizes=%+v", restoration)
	}
}

func TestParseRestorationParams128SuperblockUnitSize(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.Use128x128Superblock = true
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationSGRProj), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBool(true) // grow from 128 to 256

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.UnitSizeY != 256 || restoration.UnitSizeUV != 256 || restoration.UnitSizeYLog2 != 8 {
		t.Fatalf("unit sizes=%+v", restoration)
	}
}

func TestParseRestorationParamsMonochromeReadsYOnly(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	var w testBitWriter
	w.writeBits(uint64(RestorationSwitchable), 2)
	w.writeBool(false)
	seg := SegmentationParams{AllLossless: false}

	restoration, err := ParseRestorationParams(w.bytes(), seq, FrameSize{}, seg, CDEFParams{})
	if err != nil {
		t.Fatal(err)
	}
	if restoration.Type[0] != RestorationSwitchable || restoration.Type[1] != RestorationNone || restoration.UnitSizeY != 64 {
		t.Fatalf("restoration=%+v", restoration)
	}
	if restoration.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", restoration.BitsRead, w.bit)
	}
}

func TestParseRestorationParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseRestorationParams allocated: %f", allocs)
	}
}

func BenchmarkParseRestorationParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, size, seg, cdef := buildShownKeyFrameCDEFBench(b, seq, 40)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseRestorationParams(payload, seq, size, seg, cdef)
	}
}

func buildShownKeyFrameCDEF(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, CDEFParams) {
	t.Helper()
	w, size, seg, lf := buildShownKeyFrameLoopFilter(t, seq, baseQIdx)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	if !seq.ColorConfig.MonoChrome {
		w.writeBits(9, 6)
	}
	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		t.Fatal(err)
	}
	return w, size, seg, cdef
}

func buildShownKeyFrameCDEFBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, CDEFParams) {
	b.Helper()
	w, size, seg, lf := buildShownKeyFrameLoopFilterBench(b, seq, baseQIdx)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	if !seq.ColorConfig.MonoChrome {
		w.writeBits(9, 6)
	}
	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		b.Fatal(err)
	}
	return w, size, seg, cdef
}
