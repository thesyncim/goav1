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
