package parser

import (
	"errors"
	"testing"
)

func TestParseTileGroupHeaderSingleTile(t *testing.T) {
	tiles := TileInfo{Cols: 1, Rows: 1}
	payload := []byte{0xaa}

	group, err := ParseTileGroupHeader(payload, tiles, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if group.StartTile != 0 || group.EndTile != 0 || group.TileCount != 1 || !group.Final {
		t.Fatalf("group=%+v", group)
	}
	if group.HeaderBits != 0 || group.BitsRead != 0 || group.DataOffset != 0 || group.DataSize != len(payload) {
		t.Fatalf("header fields=%+v", group)
	}

	var spans [1]TileSpan
	n, err := SplitTileGroup(payload, tiles, group, spans[:])
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || spans[0].Tile != 0 || spans[0].Offset != 0 || spans[0].Size != 1 {
		t.Fatalf("spans n=%d spans=%+v", n, spans[:])
	}
}

func TestParseTileGroupHeaderExplicitAndSplit(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 2,
	}
	payload := explicitTileGroupPayload(1, 2)
	payload = append(payload,
		0x02, 0x00, // first tile size minus 1, little-endian
		0xaa, 0xbb, 0xcc,
		0xdd, 0xee, 0xff, 0x11, 0x22,
	)

	group, err := ParseTileGroupHeader(payload, tiles, 0, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if !group.TileStartAndEndPresent || group.StartTile != 1 || group.EndTile != 2 || group.TileCount != 2 || group.Final {
		t.Fatalf("group=%+v", group)
	}
	if group.HeaderBits != 8 || group.BitsRead != 8 || group.DataOffset != 1 || group.DataSize != 10 {
		t.Fatalf("header fields=%+v", group)
	}

	var spans [2]TileSpan
	n, err := SplitTileGroup(payload, tiles, group, spans[:])
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("span count=%d", n)
	}
	if spans[0] != (TileSpan{Tile: 1, Row: 0, Col: 1, Offset: 3, Size: 3}) {
		t.Fatalf("span[0]=%+v", spans[0])
	}
	if spans[1] != (TileSpan{Tile: 2, Row: 1, Col: 0, Offset: 6, Size: 5}) {
		t.Fatalf("span[1]=%+v", spans[1])
	}
}

func TestParseTileGroupHeaderRejectsExplicitFrameOBU(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	_, err := ParseTileGroupHeader(explicitTileGroupPayload(0, 3), tiles, 0, 0, true)
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("ParseTileGroupHeader err=%v want %v", err, ErrInvalidTileGroup)
	}
}

func TestParseTileGroupHeaderRejectsOutOfOrderStart(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	_, err := ParseTileGroupHeader(explicitTileGroupPayload(1, 2), tiles, 0, 2, false)
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("ParseTileGroupHeader err=%v want %v", err, ErrInvalidTileGroup)
	}
}

func TestParseTileGroupHeaderRejectsEndBeforeStart(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	_, err := ParseTileGroupHeader(explicitTileGroupPayload(3, 2), tiles, 0, 3, false)
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("ParseTileGroupHeader err=%v want %v", err, ErrInvalidTileGroup)
	}
}

func TestParseTileGroupHeaderRejectsNonZeroAlignment(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBool(true) // byte_alignment zero_bit violation

	_, err := ParseTileGroupHeader(w.bytes(), tiles, 0, 0, false)
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("ParseTileGroupHeader err=%v want %v", err, ErrInvalidTileGroup)
	}
}

func TestSplitTileGroupRejectsShortPayload(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 2,
	}
	payload := explicitTileGroupPayload(0, 1)
	payload = append(payload, 0xff, 0x00)

	group, err := ParseTileGroupHeader(payload, tiles, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	var spans [2]TileSpan
	_, err = SplitTileGroup(payload, tiles, group, spans[:])
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("SplitTileGroup err=%v want %v", err, ErrInvalidTileGroup)
	}
}

// TestParseTileGroupHeaderFrameOBUByteAligns guards the frame_obu()
// byte_alignment() that sits between frame_header_obu() and tile_group_obu()
// (spec 5.10.1). For multi-tile frame OBUs whose frame header does not end on a
// byte boundary (e.g. asymmetric uniform tile layouts where TileColsLog2 !=
// TileRowsLog2), startBits is not a multiple of 8. The parser must round it up
// to the next byte so the tile group header — and therefore DataOffset and the
// first tile_size_minus_1 — land on the same byte libaom reads. Without the
// alignment the decoder reads a frame-header byte as tile 0's size, producing
// an absurd 1-byte tile 0 / "invalid tile group".
func TestParseTileGroupHeaderFrameOBUByteAligns(t *testing.T) {
	// 2 tile rows, 1 tile column (the 2x1 asymmetric uniform layout).
	tiles := TileInfo{Cols: 1, Rows: 2, Log2Cols: 0, Log2Rows: 1, TileSizeBytes: 2}

	// Pretend the frame header consumed 95 bits (matches a real 256x256 2x1
	// clip). Byte 11 (bits 88..95) holds the unaligned tail of the frame
	// header; the tile group must start at byte 12, not byte 11.
	const frameHeaderBits = 95
	payload := make([]byte, 24)
	payload[11] = 0xff // frame-header tail: must NOT be read as tile size
	// Tile group OBU starts at byte 12. With numTiles>1 the first bit is
	// tile_start_and_end_present_flag (0 for a frame OBU), then byte_alignment
	// advances to byte 13 where tile 0's 2-byte size lives.
	payload[12] = 0x00 // present flag 0 + alignment padding
	payload[13] = 0x04 // tile0 size_minus_1 low byte -> size 5
	payload[14] = 0x00 // tile0 size_minus_1 high byte

	group, err := ParseTileGroupHeader(payload, tiles, frameHeaderBits, 0, true)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	if group.DataOffset != 13 {
		t.Fatalf("DataOffset=%d want 13 (byte-aligned tile group data)", group.DataOffset)
	}

	var spans [2]TileSpan
	n, err := SplitTileGroup(payload, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != 2 {
		t.Fatalf("span count=%d want 2", n)
	}
	// tile0: 2 size bytes at 13..14 (=5), payload at 15..19.
	if spans[0].Tile != 0 || spans[0].Row != 0 || spans[0].Col != 0 ||
		spans[0].Offset != 15 || spans[0].Size != 5 {
		t.Fatalf("tile0 span=%+v want offset 15 size 5", spans[0])
	}
	// tile1: remainder 20..23.
	if spans[1].Tile != 1 || spans[1].Row != 1 || spans[1].Col != 0 ||
		spans[1].Offset != 20 || spans[1].Size != 4 {
		t.Fatalf("tile1 span=%+v want offset 20 size 4", spans[1])
	}
}

func TestParseTileGroupHeaderAllocs(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1, TileSizeBytes: 1}
	payload := explicitTileGroupPayload(0, 1)
	payload = append(payload, 0x00, 0xaa, 0xbb)

	allocs := testing.AllocsPerRun(1000, func() {
		group, err := ParseTileGroupHeader(payload, tiles, 0, 0, false)
		if err != nil {
			t.Fatal(err)
		}
		var spans [2]TileSpan
		_, err = SplitTileGroup(payload, tiles, group, spans[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("tile group parse allocated: %f", allocs)
	}
}

func BenchmarkParseTileGroupHeader(b *testing.B) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1, TileSizeBytes: 1}
	payload := explicitTileGroupPayload(0, 1)
	payload = append(payload, 0x00, 0xaa, 0xbb)

	b.ReportAllocs()
	for b.Loop() {
		group, _ := ParseTileGroupHeader(payload, tiles, 0, 0, false)
		var spans [2]TileSpan
		_, _ = SplitTileGroup(payload, tiles, group, spans[:])
	}
}

func explicitTileGroupPayload(start uint64, end uint64) []byte {
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(start, 2)
	w.writeBits(end, 2)
	for w.bit&7 != 0 {
		w.writeBool(false)
	}
	return w.bytes()
}
