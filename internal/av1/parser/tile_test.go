package parser

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

func TestParseTileInfoSingleTileExplicit(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag

	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	if !tiles.RefreshContext {
		t.Fatal("RefreshContext=false want true")
	}
	if tiles.UniformSpacing {
		t.Fatal("UniformSpacing=true want false")
	}
	if tiles.SBCols != 1 || tiles.SBRows != 1 || tiles.Cols != 1 || tiles.Rows != 1 {
		t.Fatalf("tile grid=%+v", tiles)
	}
	if tiles.ColStartSB[0] != 0 || tiles.ColStartSB[1] != 1 || tiles.RowStartSB[0] != 0 || tiles.RowStartSB[1] != 1 {
		t.Fatalf("tile starts col=%v row=%v", tiles.ColStartSB[:2], tiles.RowStartSB[:2])
	}
	if tiles.TileSizeBytes != 0 || tiles.ContextUpdateTileID != 0 {
		t.Fatalf("tile data fields=%+v", tiles)
	}
	if tiles.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", tiles.BitsRead, w.bit)
	}
}

func TestParseTileInfoUniformSplit(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 8
	seq.FrameHeightBits = 7
	seq.MaxFrameWidth = 256
	seq.MaxFrameHeight = 128

	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(true)  // uniform_tile_spacing_flag
	w.writeBool(true)  // increment tile_cols_log2 to 1
	w.writeBool(false) // stop before tile_cols_log2 2
	w.writeBool(true)  // increment tile_rows_log2 to 1
	w.writeBits(1, 2)  // context_update_tile_id
	w.writeBits(2, 2)  // tile_size_bytes_minus_1

	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	if !tiles.UniformSpacing || tiles.Log2Cols != 1 || tiles.Log2Rows != 1 {
		t.Fatalf("uniform log2 grid=%+v", tiles)
	}
	if tiles.SBCols != 4 || tiles.SBRows != 2 || tiles.Cols != 2 || tiles.Rows != 2 {
		t.Fatalf("tile grid=%+v", tiles)
	}
	if tiles.ColStartSB[0] != 0 || tiles.ColStartSB[1] != 2 || tiles.ColStartSB[2] != 4 {
		t.Fatalf("col starts=%v", tiles.ColStartSB[:3])
	}
	if tiles.RowStartSB[0] != 0 || tiles.RowStartSB[1] != 1 || tiles.RowStartSB[2] != 2 {
		t.Fatalf("row starts=%v", tiles.RowStartSB[:3])
	}
	if tiles.ContextUpdateTileID != 1 || tiles.TileSizeBytes != 3 {
		t.Fatalf("tile data fields=%+v", tiles)
	}
}

func TestParseTileInfoInterMotionControls(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	refs := oneReferenceState(seq, 0, 1)
	w, prefix, size := buildInterFrameSize(t, seq, &refs)
	w.writeBool(true)  // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(uint64(InterpolationSharp), 2)
	w.writeBool(true)  // is_motion_mode_switchable
	w.writeBool(true)  // use_ref_frame_mvs
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag

	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	if !tiles.AllowHighPrecisionMV || tiles.InterpolationFilter != InterpolationSharp {
		t.Fatalf("motion controls=%+v", tiles)
	}
	if !tiles.SwitchableMotionMode || !tiles.UseRefFrameMVS || !tiles.RefreshContext {
		t.Fatalf("frame controls=%+v", tiles)
	}
}

func TestParseTileInfoRejectsContextTileOutOfRange(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 8
	seq.FrameHeightBits = 6
	seq.MaxFrameWidth = 192
	seq.MaxFrameHeight = 64

	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	w.writeBool(false) // explicit col 0 width 1 from ns(3)
	w.writeBool(false) // explicit col 1 width 1 from ns(2)
	w.writeBits(3, 2)  // context_update_tile_id, invalid for 3 tiles

	_, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseTileInfo err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestReadUniformMatchesDav1dNSCoding(t *testing.T) {
	cases := []struct {
		name string
		bits uint64
		n    uint8
		want uint32
	}{
		{name: "zero", bits: 0b0, n: 1, want: 0},
		{name: "one", bits: 0b10, n: 2, want: 1},
		{name: "two", bits: 0b11, n: 2, want: 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var w testBitWriter
			w.writeBits(tc.bits, tc.n)
			r := bitstream.NewReader(w.bytes())
			got, err := readUniform(&r, 3)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("readUniform=%d want %d", got, tc.want)
			}
		})
	}
}

func TestParseTileInfoAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseTileInfo allocated: %f", allocs)
	}
}

func BenchmarkParseTileInfo(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseTileInfo(payload, seq, prefix, size)
	}
}

func buildShownKeyFrameSize(t *testing.T, seq SequenceHeader) (testBitWriter, FrameHeaderPrefix, FrameSize) {
	t.Helper()
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different
	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, size
}

func buildShownKeyFrameSizeBench(b *testing.B, seq SequenceHeader) (testBitWriter, FrameHeaderPrefix, FrameSize) {
	b.Helper()
	payload, prefix := buildShownKeyFramePrefixBench(b, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		b.Fatal(err)
	}
	return w, prefix, size
}

func buildInterFrameSize(t *testing.T, seq SequenceHeader, refs *ReferenceState) (testBitWriter, FrameHeaderPrefix, FrameSize) {
	t.Helper()
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for i := 0; i < InterRefsPerFrame; i++ {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different
	size, err := ParseFrameSize(w.bytes(), seq, prefix, refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, size
}
