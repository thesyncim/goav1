package parser

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

func TestParseQuantizationParamsZeroDeltas(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // u_dc_delta_q
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(false) // using_qmatrix

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.BaseQIdx != 0 || quant.YDCDelta != 0 || quant.UDCDelta != 0 || quant.UACDelta != 0 {
		t.Fatalf("quant=%+v", quant)
	}
	if quant.VDCDelta != quant.UDCDelta || quant.VACDelta != quant.UACDelta {
		t.Fatalf("uv copy=%+v", quant)
	}
	if quant.UsingQMatrix || quant.BitsRead != w.bit {
		t.Fatalf("qmatrix/bits=%+v want bits %d", quant, w.bit)
	}
}

func TestParseQuantizationParamsSeparateUVAndQMatrix(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.SeparateUVDeltaQ = true
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(96, 8) // base_q_idx
	w.writeBool(true)
	writeSignedBitsTest(&w, -2, 7)
	w.writeBool(true) // diff_uv_delta
	w.writeBool(true)
	writeSignedBitsTest(&w, 5, 7)
	w.writeBool(true)
	writeSignedBitsTest(&w, -3, 7)
	w.writeBool(true)
	writeSignedBitsTest(&w, 7, 7)
	w.writeBool(true)
	writeSignedBitsTest(&w, -9, 7)
	w.writeBool(true) // using_qmatrix
	w.writeBits(2, 4)
	w.writeBits(3, 4)
	w.writeBits(4, 4)

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.BaseQIdx != 96 || quant.YDCDelta != -2 {
		t.Fatalf("base/y=%+v", quant)
	}
	if !quant.DiffUVDeltas || quant.UDCDelta != 5 || quant.UACDelta != -3 || quant.VDCDelta != 7 || quant.VACDelta != -9 {
		t.Fatalf("uv deltas=%+v", quant)
	}
	if !quant.UsingQMatrix || quant.QMatrixLevelY != 2 || quant.QMatrixLevelU != 3 || quant.QMatrixLevelV != 4 {
		t.Fatalf("qmatrix=%+v", quant)
	}
}

func TestParseQuantizationParamsSharedUV(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.SeparateUVDeltaQ = false
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(10, 8)
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(true)  // u_dc_delta_q
	writeSignedBitsTest(&w, 4, 7)
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(true)  // using_qmatrix
	w.writeBits(1, 4)
	w.writeBits(6, 4)

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.DiffUVDeltas || quant.UDCDelta != 4 || quant.VDCDelta != 4 || quant.VACDelta != 0 {
		t.Fatalf("shared uv=%+v", quant)
	}
	if quant.QMatrixLevelY != 1 || quant.QMatrixLevelU != 6 || quant.QMatrixLevelV != 6 {
		t.Fatalf("qmatrix=%+v", quant)
	}
}

func TestParseQuantizationParamsMonoChrome(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(3, 8)
	w.writeBool(true)
	writeSignedBitsTest(&w, -1, 7)
	w.writeBool(false) // using_qmatrix

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.BaseQIdx != 3 || quant.YDCDelta != -1 {
		t.Fatalf("mono quant=%+v", quant)
	}
	if quant.UDCDelta != 0 || quant.UACDelta != 0 || quant.VDCDelta != 0 || quant.VACDelta != 0 {
		t.Fatalf("mono uv deltas=%+v", quant)
	}
}

func TestReadSignedBits(t *testing.T) {
	cases := []struct {
		bits uint64
		want int16
	}{
		{bits: 0b0000101, want: 5},
		{bits: 0b1111111, want: -1},
		{bits: 0b1111110, want: -2},
		{bits: 0b1000000, want: -64},
	}
	for _, tc := range cases {
		var w testBitWriter
		w.writeBits(tc.bits, 7)
		r := bitstream.NewReader(w.bytes())
		got, err := readSignedBits(&r, 7)
		if err != nil {
			t.Fatal(err)
		}
		if got != tc.want {
			t.Fatalf("readSignedBits(%07b)=%d want %d", tc.bits, got, tc.want)
		}
	}
}

func TestParseQuantizationParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(0, 8)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseQuantizationParams allocated: %f", allocs)
	}
}

func BenchmarkParseQuantizationParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, tiles := buildShownKeyFrameTileBench(b, seq)
	w.writeBits(0, 8)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseQuantizationParams(payload, seq, tiles)
	}
}

func buildShownKeyFrameTile(t *testing.T, seq SequenceHeader) (testBitWriter, TileInfo) {
	t.Helper()
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	return w, tiles
}

func buildShownKeyFrameTileBench(b *testing.B, seq SequenceHeader) (testBitWriter, TileInfo) {
	b.Helper()
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		b.Fatal(err)
	}
	return w, tiles
}

func writeSignedBitsTest(w *testBitWriter, value int16, n uint8) {
	mask := int32((uint32(1) << n) - 1)
	w.writeBits(uint64(int32(value)&mask), n)
}
