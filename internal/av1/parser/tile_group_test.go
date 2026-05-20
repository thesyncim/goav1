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
	for i := 0; i < b.N; i++ {
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
