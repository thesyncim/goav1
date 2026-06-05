package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendTileInfoPayloadSingleTileExplicit(t *testing.T) {
	seq := tileInfoSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
	}
	payload, parsed := appendAndParseTileInfo(t, seq, prefix, 64, 64, tiles)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.RefreshContext || parsed.UniformSpacing || parsed.Cols != 1 || parsed.Rows != 1 {
		t.Fatalf("tile info=%+v", parsed)
	}
}

func TestAppendTileInfoPayloadUniformSplit(t *testing.T) {
	seq := tileInfoSequenceHeader()
	seq.MaxFrameWidth = 256
	seq.MaxFrameHeight = 128
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := TileInfo{
		RefreshContext:      true,
		UniformSpacing:      true,
		SBCols:              4,
		SBRows:              2,
		MinLog2Cols:         0,
		MaxLog2Cols:         2,
		MinLog2Rows:         0,
		MaxLog2Rows:         1,
		MinLog2Tiles:        0,
		Log2Cols:            1,
		Log2Rows:            1,
		Cols:                2,
		Rows:                2,
		ContextUpdateTileID: 1,
		TileSizeBytes:       3,
	}
	tiles.ColStartSB[0], tiles.ColStartSB[1], tiles.ColStartSB[2] = 0, 2, 4
	tiles.RowStartSB[0], tiles.RowStartSB[1], tiles.RowStartSB[2] = 0, 1, 2
	_, parsed := appendAndParseTileInfo(t, seq, prefix, 256, 128, tiles)
	if !parsed.UniformSpacing || parsed.Log2Cols != 1 || parsed.Log2Rows != 1 || parsed.ContextUpdateTileID != 1 || parsed.TileSizeBytes != 3 {
		t.Fatalf("tile info=%+v", parsed)
	}
}

func TestAppendTileInfoPayloadInterMotionControls(t *testing.T) {
	seq := tileInfoSequenceHeader()
	seq.EnableOrderHint = true
	seq.EnableRefFrameMVS = true
	seq.OrderHintBits = 8
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	tiles := TileInfo{
		AllowHighPrecisionMV: true,
		InterpolationFilter:  InterpolationSharp,
		SwitchableMotionMode: true,
		UseRefFrameMVS:       true,
		RefreshContext:       true,
		SBCols:               1,
		SBRows:               1,
		Cols:                 1,
		Rows:                 1,
		ColStartSB:           [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:           [MaxTileRows + 1]uint16{0, 1},
	}
	_, parsed := appendAndParseTileInfo(t, seq, prefix, 64, 64, tiles)
	if !parsed.AllowHighPrecisionMV || parsed.InterpolationFilter != parser.InterpolationSharp ||
		!parsed.SwitchableMotionMode || !parsed.UseRefFrameMVS || !parsed.RefreshContext {
		t.Fatalf("tile info=%+v", parsed)
	}
}

func TestAppendTileInfoPayloadRejectsInvalid(t *testing.T) {
	seq := tileInfoSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	valid := TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
	}
	badContext := valid
	badContext.ContextUpdateTileID = 1
	badMotion := valid
	badMotion.InterpolationFilter = InterpolationSharp
	badStarts := valid
	badStarts.ColStartSB[1] = 0
	cases := [...]TileInfo{badContext, badMotion, badStarts}
	var buf [4]byte
	for _, tiles := range cases {
		if _, err := AppendTileInfoPayload(buf[:0], seq, prefix, 64, 64, tiles); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendTileInfoPayload err=%v want ErrInvalidFrame", err)
		}
	}
}

func TestAppendTileInfoPayloadShortBuffer(t *testing.T) {
	seq := tileInfoSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
	}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendTileInfoPayload(dst, seq, prefix, 64, 64, tiles)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendTileInfoPayloadAllocs(t *testing.T) {
	seq := tileInfoSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
	}
	var buf [2]byte
	if _, err := AppendTileInfoPayload(buf[:0], seq, prefix, 64, 64, tiles); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = TileInfoPayloadSize(seq, prefix, 64, 64, tiles)
		_, _ = AppendTileInfoPayload(buf[:0], seq, prefix, 64, 64, tiles)
	})
	if allocs != 0 {
		t.Fatalf("AppendTileInfoPayload allocated: %f", allocs)
	}
}

func TestAppendTileGroupPayloadSingleTile(t *testing.T) {
	tiles := TileInfo{
		Cols: 1,
		Rows: 1,
	}
	payloads := [...]TilePayload{{Data: []byte{0x80}}}
	var buf [8]byte
	size, err := TileGroupPayloadSize(tiles, 0, 0, payloads[:])
	if err != nil {
		t.Fatalf("TileGroupPayloadSize: %v", err)
	}
	out, err := AppendTileGroupPayload(buf[:0], tiles, 0, 0, payloads[:])
	if err != nil {
		t.Fatalf("AppendTileGroupPayload: %v", err)
	}
	if len(out) != size || len(out) != 1 || out[0] != 0x80 {
		t.Fatalf("payload=% x size=%d", out, size)
	}
	group, err := parser.ParseTileGroupHeader(out, tiles, 0, 0, false)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	var spans [1]parser.TileSpan
	n, err := parser.SplitTileGroup(out, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != 1 || spans[0].Offset != 0 || spans[0].Size != 1 {
		t.Fatalf("spans=%+v n=%d group=%+v", spans, n, group)
	}
}

func TestAppendTileGroupPayloadMultiTileRoundTrip(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 2,
	}
	payloads := [...]TilePayload{
		{Data: []byte{0x11, 0x12}},
		{Data: []byte{0x21, 0x22, 0x23}},
		{Data: []byte{0x31}},
		{Data: []byte{0x41, 0x42, 0x43, 0x44}},
	}
	var buf [32]byte
	size, err := TileGroupPayloadSize(tiles, 0, 3, payloads[:])
	if err != nil {
		t.Fatalf("TileGroupPayloadSize: %v", err)
	}
	out, err := AppendTileGroupPayload(buf[:0], tiles, 0, 3, payloads[:])
	if err != nil {
		t.Fatalf("AppendTileGroupPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("payload len=%d want %d", len(out), size)
	}
	group, err := parser.ParseTileGroupHeader(out, tiles, 0, 0, false)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	if group.TileStartAndEndPresent || group.DataOffset != 1 || group.TileCount != 4 || !group.Final {
		t.Fatalf("group=%+v", group)
	}
	var spans [4]parser.TileSpan
	n, err := parser.SplitTileGroup(out, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != len(payloads) {
		t.Fatalf("span count=%d want %d", n, len(payloads))
	}
	for i := range payloads {
		span := spans[i]
		got := out[span.Offset : span.Offset+span.Size]
		if string(got) != string(payloads[i].Data) || span.Tile != uint16(i) {
			t.Fatalf("span %d=%+v data=% x want % x", i, span, got, payloads[i].Data)
		}
	}
}

func TestAppendTileGroupPayloadSubsetRoundTrip(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]TilePayload{
		{Data: []byte{0x51, 0x52}},
		{Data: []byte{0x61}},
	}
	var buf [16]byte
	out, err := AppendTileGroupPayload(buf[:0], tiles, 1, 2, payloads[:])
	if err != nil {
		t.Fatalf("AppendTileGroupPayload: %v", err)
	}
	group, err := parser.ParseTileGroupHeader(out, tiles, 0, 1, false)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	if !group.TileStartAndEndPresent || group.StartTile != 1 || group.EndTile != 2 || group.DataOffset != 1 {
		t.Fatalf("group=%+v", group)
	}
	var spans [2]parser.TileSpan
	n, err := parser.SplitTileGroup(out, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != 2 || spans[0].Tile != 1 || spans[1].Tile != 2 || spans[0].Size != 2 || spans[1].Size != 1 {
		t.Fatalf("spans=%+v n=%d", spans, n)
	}
}

func TestAppendTileGroupPayloadRejectsInvalid(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	valid := [...]TilePayload{{Data: []byte{0x80}}, {Data: []byte{0x81}}}
	empty := [...]TilePayload{{Data: []byte{}}, {Data: []byte{0x81}}}
	tooLargeForSizeByte := [...]TilePayload{{Data: make([]byte, 257)}, {Data: []byte{0x81}}}
	cases := []struct {
		name     string
		tiles    TileInfo
		start    uint16
		end      uint16
		payloads []TilePayload
	}{
		{name: "range", tiles: tiles, start: 1, end: 0, payloads: valid[:]},
		{name: "count", tiles: tiles, start: 0, end: 1, payloads: valid[:1]},
		{name: "empty", tiles: tiles, start: 0, end: 1, payloads: empty[:]},
		{name: "size", tiles: tiles, start: 0, end: 1, payloads: tooLargeForSizeByte[:]},
	}
	var buf [8]byte
	for _, tc := range cases {
		if _, err := AppendTileGroupPayload(buf[:0], tc.tiles, tc.start, tc.end, tc.payloads); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("%s err=%v want ErrInvalidFrame", tc.name, err)
		}
	}
}

func TestAppendTileGroupPayloadShortBuffer(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]TilePayload{{Data: []byte{0x80}}, {Data: []byte{0x81}}}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendTileGroupPayload(dst, tiles, 0, 1, payloads[:])
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendTileGroupPayloadAllocs(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]TilePayload{{Data: []byte{0x80}}, {Data: []byte{0x81}}}
	var buf [8]byte
	if _, err := AppendTileGroupPayload(buf[:0], tiles, 0, 1, payloads[:]); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = TileGroupPayloadSize(tiles, 0, 1, payloads[:])
		_, _ = AppendTileGroupPayload(buf[:0], tiles, 0, 1, payloads[:])
	})
	if allocs != 0 {
		t.Fatalf("AppendTileGroupPayload allocated: %f", allocs)
	}
}

func appendAndParseTileInfo(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) ([]byte, parser.TileInfo) {
	t.Helper()
	payloadSize, err := TileInfoPayloadSize(seq, prefix, codedWidth, height, tiles)
	if err != nil {
		t.Fatalf("TileInfoPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := AppendTileInfoPayload(buf[:0], seq, prefix, codedWidth, height, tiles)
	if err != nil {
		t.Fatalf("AppendTileInfoPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseTileInfo(
		payload,
		parser.SequenceHeader{
			Use128x128Superblock: seq.Use128x128Superblock,
			EnableOrderHint:      seq.EnableOrderHint,
			EnableRefFrameMVS:    seq.EnableRefFrameMVS,
		},
		parser.FrameHeaderPrefix{
			FrameType:          parser.FrameType(prefix.FrameType),
			ForceIntegerMV:     prefix.ForceIntegerMV,
			ErrorResilientMode: prefix.ErrorResilientMode,
			DisableCDFUpdate:   prefix.DisableCDFUpdate,
		},
		parser.FrameSize{CodedWidth: codedWidth, Height: height},
	)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	return payload, parsed
}

func tileInfoSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        64,
		MaxFrameHeight:       64,
		Use128x128Superblock: false,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}
