package goav1

import (
	"errors"
	"testing"

	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPublicTemporalUnitIterator(t *testing.T) {
	var stream []byte
	firstStart := len(stream)
	stream = appendOBUForTest(stream, internalobu.TypeTemporalDelimiter, nil)
	stream = appendOBUForTest(stream, internalobu.TypeSequenceHeader, []byte{0xaa})
	stream = appendOBUForTest(stream, internalobu.TypeFrameHeader, []byte{0xbb})
	firstEnd := len(stream)
	stream = appendOBUForTest(stream, internalobu.TypeTemporalDelimiter, nil)
	stream = appendOBUForTest(stream, internalobu.TypeFrame, []byte{0xcc})
	secondEnd := len(stream)

	it := internalobu.NewTemporalUnitIterator(stream)
	first, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first ok=%v err=%v", ok, err)
	}
	if first.Index != 0 || string(first.Raw) != string(stream[firstStart:firstEnd]) {
		t.Fatalf("first=%+v want=%x", first, stream[firstStart:firstEnd])
	}
	second, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("second ok=%v err=%v", ok, err)
	}
	if second.Index != 1 || string(second.Raw) != string(stream[firstEnd:secondEnd]) {
		t.Fatalf("second=%+v want=%x", second, stream[firstEnd:secondEnd])
	}
	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected third temporal unit")
	}
}

func TestPublicTemporalUnitIteratorRejectsMissingDelimiter(t *testing.T) {
	stream := appendOBUForTest(nil, internalobu.TypeSequenceHeader, []byte{0xaa})
	it := internalobu.NewTemporalUnitIterator(stream)
	_, _, err := it.Next()
	if !errors.Is(err, internalobu.ErrMissingTemporalDelimiter) {
		t.Fatalf("err=%v want %v", err, internalobu.ErrMissingTemporalDelimiter)
	}
}

func TestPublicAnnexBIterator(t *testing.T) {
	td := []byte{byte(internalobu.TypeTemporalDelimiter) << 3}
	seq := []byte{byte(internalobu.TypeSequenceHeader) << 3, 0xaa}
	frameHeader := []byte{byte(internalobu.TypeFrameHeader) << 3, 0xbb}
	frame := []byte{byte(internalobu.TypeFrame) << 3, 0xcc}
	stream := appendPublicAnnexBStream(nil,
		[][][]byte{{td, seq}, {frameHeader}},
		[][][]byte{{td, frame}},
	)

	want := []struct {
		raw      []byte
		typ      internalobu.Type
		temporal uint32
		frame    uint32
		obu      uint32
	}{
		{raw: td, typ: internalobu.TypeTemporalDelimiter, temporal: 0, frame: 0, obu: 0},
		{raw: seq, typ: internalobu.TypeSequenceHeader, temporal: 0, frame: 0, obu: 1},
		{raw: frameHeader, typ: internalobu.TypeFrameHeader, temporal: 0, frame: 1, obu: 0},
		{raw: td, typ: internalobu.TypeTemporalDelimiter, temporal: 1, frame: 0, obu: 0},
		{raw: frame, typ: internalobu.TypeFrame, temporal: 1, frame: 0, obu: 1},
	}

	it := internalobu.NewAnnexBIterator(stream)
	for i, wantUnit := range want {
		unit, ok, err := it.Next()
		if err != nil || !ok {
			t.Fatalf("unit %d ok=%v err=%v", i, ok, err)
		}
		if string(unit.Raw) != string(wantUnit.raw) || unit.OBU.Header.Type != wantUnit.typ {
			t.Fatalf("unit %d raw=%x type=%d want raw=%x type=%d", i, unit.Raw, unit.OBU.Header.Type, wantUnit.raw, wantUnit.typ)
		}
		if unit.TemporalUnitIndex != wantUnit.temporal || unit.FrameUnitIndex != wantUnit.frame || unit.OBUIndex != wantUnit.obu {
			t.Fatalf("unit %d indices=%d/%d/%d want %d/%d/%d", i,
				unit.TemporalUnitIndex, unit.FrameUnitIndex, unit.OBUIndex,
				wantUnit.temporal, wantUnit.frame, wantUnit.obu)
		}
		start := int(unit.Offset)
		end := start + len(unit.Raw)
		if end > len(stream) || string(unit.Raw) != string(stream[start:end]) {
			t.Fatalf("unit %d does not alias stream at offset %d", i, unit.Offset)
		}
	}
	_, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected extra Annex B unit")
	}
}

func TestPublicParseAnnexBElement(t *testing.T) {
	src := []byte{0x03, byte(internalobu.TypeFrame)<<3 | 0x02, 0x01, 0xdd}
	unit, consumed, err := internalobu.ParseAnnexBElement(src)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != len(src) || unit.Header.Type != internalobu.TypeFrame || string(unit.Payload) != string([]byte{0xdd}) {
		t.Fatalf("unit=%+v consumed=%d", unit, consumed)
	}
}

func TestPublicAnnexBIteratorRejectsInvalid(t *testing.T) {
	it := internalobu.NewAnnexBIterator([]byte{0x00})
	_, _, err := it.Next()
	if !errors.Is(err, internalobu.ErrInvalidAnnexB) {
		t.Fatalf("err=%v want %v", err, internalobu.ErrInvalidAnnexB)
	}
}

func TestPublicOBUStreamIteratorsAllocs(t *testing.T) {
	lowOverhead := appendOBUForTest(nil, internalobu.TypeTemporalDelimiter, nil)
	lowOverhead = appendOBUForTest(lowOverhead, internalobu.TypeFrame, []byte{0xaa})
	annexB := appendPublicAnnexBStream(nil, [][][]byte{{{byte(internalobu.TypeTemporalDelimiter) << 3}, {byte(internalobu.TypeFrame) << 3, 0xaa}}})
	annexElement := []byte{0x02, byte(internalobu.TypeFrame) << 3, 0xdd}

	allocs := testing.AllocsPerRun(1000, func() {
		tuIt := internalobu.NewTemporalUnitIterator(lowOverhead)
		unit, ok, err := tuIt.Next()
		if err != nil || !ok || len(unit.Raw) == 0 {
			t.Fatalf("temporal unit=%+v ok=%v err=%v", unit, ok, err)
		}

		annexIt := internalobu.NewAnnexBIterator(annexB)
		count := 0
		for {
			annexUnit, ok, err := annexIt.Next()
			if err != nil {
				t.Fatal(err)
			}
			if !ok {
				break
			}
			if len(annexUnit.Raw) == 0 {
				t.Fatal("empty Annex B unit")
			}
			count++
		}
		if count != 2 {
			t.Fatalf("annex count=%d want 2", count)
		}

		element, consumed, err := internalobu.ParseAnnexBElement(annexElement)
		if err != nil || consumed != 3 || element.Header.Type != internalobu.TypeFrame {
			t.Fatalf("element=%+v consumed=%d err=%v", element, consumed, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("public OBU stream iterators allocated: %f", allocs)
	}
}

func FuzzPublicOBUStreamIterators(f *testing.F) {
	lowOverhead := appendOBUForTest(nil, internalobu.TypeTemporalDelimiter, nil)
	lowOverhead = appendOBUForTest(lowOverhead, internalobu.TypeFrame, []byte{0xaa})
	annexB := appendPublicAnnexBStream(nil, [][][]byte{{{byte(internalobu.TypeTemporalDelimiter) << 3}, {byte(internalobu.TypeFrame) << 3, 0xaa}}})

	for _, seed := range [][]byte{
		nil,
		{0x00},
		{0x12, 0x00},
		{0x10},
		{0x80},
		{0x13, 0x00},
		lowOverhead,
		annexB,
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 8<<10 {
			data = data[:8<<10]
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in public OBU stream path (len=%d): %v", len(data), r)
			}
		}()

		_, _, _ = internalobu.ParseHeader(data)
		_, _ = internalobu.ParseElement(data)
		_, _, _ = internalobu.ParseLowOverhead(data)
		_, _, _ = internalobu.ParseAnnexBElement(data)
		_, _ = internalobu.NormalizeLowOverhead(make([]byte, len(data)*2+16), data)
		fuzzPublicDriveLowOverhead(data)
		fuzzPublicDriveTemporalUnit(data)
		fuzzPublicDriveAnnexB(data)
	})
}

func fuzzPublicDriveLowOverhead(data []byte) {
	it := internalobu.NewLowOverheadIterator(data)
	for range 1 << 12 {
		if _, ok, err := it.Next(); err != nil || !ok {
			return
		}
	}
}

func fuzzPublicDriveTemporalUnit(data []byte) {
	it := internalobu.NewTemporalUnitIterator(data)
	for range 1 << 12 {
		if _, ok, err := it.Next(); err != nil || !ok {
			return
		}
	}
}

func fuzzPublicDriveAnnexB(data []byte) {
	it := internalobu.NewAnnexBIterator(data)
	for range 1 << 12 {
		if _, ok, err := it.Next(); err != nil || !ok {
			return
		}
	}
}

func TestPublicParseMetadataOBU(t *testing.T) {
	// HDR-CLL: type=1, max_cll=1000, max_fall=400, trailing 0x80.
	payload := []byte{0x01, 0x03, 0xE8, 0x01, 0x90, 0x80}
	meta, err := internalobu.ParseMetadata(payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU err=%v", err)
	}
	if meta.Type != internalobu.MetadataTypeHDRCLL {
		t.Fatalf("type=%v", meta.Type)
	}
	if meta.HDRCLL.MaxCLL != 1000 || meta.HDRCLL.MaxFALL != 400 {
		t.Fatalf("cll=%#v", meta.HDRCLL)
	}

	if _, err := internalobu.ParseMetadata(nil); !errors.Is(err, internalobu.ErrMetadataShortPayload) {
		t.Fatalf("nil payload err=%v want ErrMetadataShortPayload", err)
	}
}

func TestPublicParseTileListOBU(t *testing.T) {
	// 2 tiles in a 2x1 output layout.
	tile0 := []byte{0x10, 0x20}
	tile1 := []byte{0x30}
	payload := []byte{
		0x01,       // output_frame_width_in_tiles_minus_1 = 1 (width = 2)
		0x00,       // output_frame_height_in_tiles_minus_1 = 0 (height = 1)
		0x00, 0x01, // tile_count_minus_1 = 1 (2 tiles)
		0x00, 0x00, 0x00, 0x00, 0x01, 0x10, 0x20, // entry 0: anchor=0, row=0, col=0, size_m1=1, data
		0x07, 0x00, 0x01, 0x00, 0x00, 0x30, // entry 1
	}

	var scratch [4]internalparser.TileListEntry
	list, err := internalparser.ParseTileListOBU(payload, scratch[:0])
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if list.TileCount() != 2 || list.OutputFrameWidthInTiles() != 2 || list.OutputFrameHeightInTiles() != 1 {
		t.Fatalf("list header=%+v", list)
	}
	if string(list.Entries[0].TileData) != string(tile0) ||
		string(list.Entries[1].TileData) != string(tile1) {
		t.Fatalf("tile data mismatch: %x %x", list.Entries[0].TileData, list.Entries[1].TileData)
	}
	if list.Entries[1].AnchorFrameIdx != 7 {
		t.Fatalf("entry1 anchor=%d want 7", list.Entries[1].AnchorFrameIdx)
	}

	// Round-trip via AppendTileListOBU.
	encoded := internalparser.AppendTileListOBU(nil, list)
	if string(encoded) != string(payload) {
		t.Fatalf("AppendTileListOBU mismatch\n got=%x\nwant=%x", encoded, payload)
	}

	if _, err := internalparser.ParseTileListOBU([]byte{0x00}, nil); !errors.Is(err, internalparser.ErrTileListShortHeader) {
		t.Fatalf("short err=%v want ErrTileListShortHeader", err)
	}
	referenceGridAnchor := []byte{0x00, 0x00, 0x00, 0x00, 0, 1, 0, 0, 0, 0xaa}
	anchored, err := internalparser.ParseTileListOBU(referenceGridAnchor, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU reference-grid anchor err=%v", err)
	}
	if err := internalparser.ValidateTileListAnchors(anchored, internalparser.TileInfo{Cols: 1, Rows: 2}); err != nil {
		t.Fatalf("ValidateTileListAnchors err=%v", err)
	}
	if err := internalparser.ValidateTileListAnchors(anchored, internalparser.TileInfo{Cols: 1, Rows: 1}); !errors.Is(err, internalparser.ErrTileListInvalidAnchorTile) {
		t.Fatalf("anchor tile err=%v want ErrTileListInvalidAnchorTile", err)
	}
}

func TestPublicTileListDecodeLayout(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x00, 0, 1, 1, 0, 0, 0xaa}
	list, err := internalparser.ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU: %v", err)
	}
	tiles := internalparser.TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1, 2},
	}
	width, height, ok := internalparser.TileListUniformTileSize(tiles)
	if !ok || width != 2 || height != 1 {
		t.Fatalf("TileListUniformTileSize=%d,%d,%v want 2,1,true", width, height, ok)
	}
	width, height, err = internalparser.ValidateTileListDecodeLayout(list, tiles)
	if err != nil || width != 2 || height != 1 {
		t.Fatalf("ValidateTileListDecodeLayout=%d,%d,%v want 2,1,nil", width, height, err)
	}

	tiles.ColStartSB[2] = 5
	if _, _, err := internalparser.ValidateTileListDecodeLayout(list, tiles); !errors.Is(err, internalparser.ErrTileListNonUniformTileSize) {
		t.Fatalf("non-uniform err=%v want %v", err, internalparser.ErrTileListNonUniformTileSize)
	}
}

func TestPublicTileListOutputGeometryAndRegions(t *testing.T) {
	payload := []byte{0x01, 0x01, 0x00, 0x02}
	payload = append(payload, 0, 0, 0, 0, 0, 0xa0)
	payload = append(payload, 0, 0, 1, 0, 0, 0xa1)
	payload = append(payload, 0, 1, 1, 0, 0, 0xa2)
	list, err := internalparser.ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU: %v", err)
	}
	tiles := internalparser.TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1, 2},
	}
	geom, err := internalparser.TileListOutputGeometryForGrid(list, tiles, false)
	if err != nil {
		t.Fatalf("TileListOutputGeometryForGrid: %v", err)
	}
	if geom.OutputFrameWidth != 256 || geom.OutputFrameHeight != 128 ||
		geom.TileWidthPixels != 128 || geom.TileHeightPixels != 64 ||
		geom.TileCount != 3 {
		t.Fatalf("geometry=%+v", geom)
	}
	region, err := internalparser.TileListOutputTileRegion(list, geom, 2)
	if err != nil {
		t.Fatalf("TileListOutputTileRegion: %v", err)
	}
	if region != (internalparser.TileListTileRegion{SourceX: 128, SourceY: 64, DestX: 0, DestY: 64, Width: 128, Height: 64}) {
		t.Fatalf("region=%+v", region)
	}
	geom128, err := internalparser.TileListOutputGeometryForGrid(list, tiles, true)
	if err != nil {
		t.Fatalf("TileListOutputGeometryForGrid 128: %v", err)
	}
	if geom128.OutputFrameWidth != 512 || geom128.OutputFrameHeight != 256 ||
		geom128.TileWidthPixels != 256 || geom128.TileHeightPixels != 128 {
		t.Fatalf("128 geometry=%+v", geom128)
	}

	sequence := SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:     10,
			SubsamplingX: true,
		},
	}
	format, err := tileListOutputFrameFormat(sequence, geom128, 32)
	if err != nil {
		t.Fatalf("TileListOutputFrameFormat: %v", err)
	}
	if format != (FrameFormat{
		Width:        geom128.OutputFrameWidth,
		Height:       geom128.OutputFrameHeight,
		BitDepth:     10,
		SubsamplingX: true,
		SBSizeLog2:   7,
		Align:        32,
	}) {
		t.Fatalf("TileListOutputFrameFormat=%+v", format)
	}
	if _, err := internalframe.RequiredSize(format); err != nil {
		t.Fatalf("FrameRequiredSize(tile-list output): %v", err)
	}
	if _, err := tileListOutputFrameFormat(sequence, internalparser.TileListOutputGeometry{}, 32); !errors.Is(err, internalframe.ErrInvalidFormat) {
		t.Fatalf("empty TileListOutputFrameFormat err=%v want %v", err, internalframe.ErrInvalidFormat)
	}
}

func TestPublicCopyTileListEntryToOutputFrame(t *testing.T) {
	list, geom := publicTileListCopyFixture(t)
	for _, tc := range []struct {
		name         string
		bitDepth     uint8
		mono         bool
		subsamplingX bool
		subsamplingY bool
	}{
		{name: "i420-10", bitDepth: 10, subsamplingX: true, subsamplingY: true},
		{name: "i422-10", bitDepth: 10, subsamplingX: true},
		{name: "i444-8", bitDepth: 8},
		{name: "mono-12", bitDepth: 12, mono: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			format := FrameFormat{
				Width:        geom.OutputFrameWidth,
				Height:       geom.OutputFrameHeight,
				BitDepth:     tc.bitDepth,
				MonoChrome:   tc.mono,
				SubsamplingX: tc.subsamplingX,
				SubsamplingY: tc.subsamplingY,
				Align:        1,
			}
			src := bindPublicTileListFrame(t, format)
			dst := bindPublicTileListFrame(t, format)
			fillTileListCopySource(src)

			if err := copyTileListEntryToOutputFrame(&dst, &src, list, geom, 2); err != nil {
				t.Fatalf("CopyTileListEntryToOutputFrame: %v", err)
			}
			if got, want := getPublicFrameSample(dst.Y, dst.Layout.BytesPerSample, 0, 64), getPublicFrameSample(src.Y, src.Layout.BytesPerSample, 128, 64); got != want {
				t.Fatalf("Y top-left got=%d want %d", got, want)
			}
			if got, want := getPublicFrameSample(dst.Y, dst.Layout.BytesPerSample, 127, 127), getPublicFrameSample(src.Y, src.Layout.BytesPerSample, 255, 127); got != want {
				t.Fatalf("Y bottom-right got=%d want %d", got, want)
			}
			if got := getPublicFrameSample(dst.Y, dst.Layout.BytesPerSample, 128, 64); got != 0 {
				t.Fatalf("unexpected copy outside destination Y tile: %d", got)
			}
			if tc.mono {
				return
			}
			shiftX, shiftY := 0, 0
			if tc.subsamplingX {
				shiftX = 1
			}
			if tc.subsamplingY {
				shiftY = 1
			}
			if got, want := getPublicFrameSample(dst.U, dst.Layout.BytesPerSample, 0, 64>>shiftY), getPublicFrameSample(src.U, src.Layout.BytesPerSample, 128>>shiftX, 64>>shiftY); got != want {
				t.Fatalf("U top-left got=%d want %d", got, want)
			}
			if got, want := getPublicFrameSample(dst.V, dst.Layout.BytesPerSample, 127>>shiftX, 127>>shiftY), getPublicFrameSample(src.V, src.Layout.BytesPerSample, 255>>shiftX, 127>>shiftY); got != want {
				t.Fatalf("V bottom-right got=%d want %d", got, want)
			}
			if got := getPublicFrameSample(dst.U, dst.Layout.BytesPerSample, 128>>shiftX, 64>>shiftY); got != 0 {
				t.Fatalf("unexpected copy outside destination U tile: %d", got)
			}
		})
	}
}

func TestPublicCopyTileListEntryToOutputFrameRejectsInvalid(t *testing.T) {
	list, geom := publicSingleTileListCopyFixture(t)
	dst := bindPublicTileListFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 1})
	src := bindPublicTileListFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 1})
	if err := copyTileListEntryToOutputFrame(&dst, &src, list, geom, 0); !errors.Is(err, internalframe.ErrInvalidFormat) {
		t.Fatalf("mismatch err=%v want %v", err, internalframe.ErrInvalidFormat)
	}
	src = bindPublicTileListFrame(t, dst.Format)
	if err := copyTileListEntryToOutputFrame(&dst, &src, list, geom, 1); !errors.Is(err, internalparser.ErrTileListInvalidTileCount) {
		t.Fatalf("index err=%v want %v", err, internalparser.ErrTileListInvalidTileCount)
	}
}

func TestPublicCopyTileListEntryToOutputFrameRejectsShortPlanesBeforeCopy(t *testing.T) {
	list, geom := publicSingleTileListCopyFixture(t)
	format := FrameFormat{
		Width:        64,
		Height:       64,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        1,
	}

	t.Run("source chroma", func(t *testing.T) {
		dst := bindPublicTileListFrame(t, format)
		src := bindPublicTileListFrame(t, format)
		fillTileListCopySource(src)
		src.U.Pix = src.U.Pix[:len(src.U.Pix)-1]

		if err := copyTileListEntryToOutputFrame(&dst, &src, list, geom, 0); !errors.Is(err, internalframe.ErrInvalidFormat) {
			t.Fatalf("source chroma err=%v want %v", err, internalframe.ErrInvalidFormat)
		}
		if got := getPublicFrameSample(dst.Y, dst.Layout.BytesPerSample, 0, 0); got != 0 {
			t.Fatalf("destination Y mutated before source chroma failure: %d", got)
		}
	})

	t.Run("destination chroma", func(t *testing.T) {
		dst := bindPublicTileListFrame(t, format)
		src := bindPublicTileListFrame(t, format)
		fillTileListCopySource(src)
		dst.U.Pix = dst.U.Pix[:len(dst.U.Pix)-1]

		if err := copyTileListEntryToOutputFrame(&dst, &src, list, geom, 0); !errors.Is(err, internalframe.ErrInvalidFormat) {
			t.Fatalf("destination chroma err=%v want %v", err, internalframe.ErrInvalidFormat)
		}
		if got := getPublicFrameSample(dst.Y, dst.Layout.BytesPerSample, 0, 0); got != 0 {
			t.Fatalf("destination Y mutated before destination chroma failure: %d", got)
		}
	})
}

func TestPublicCopyTileListEntryToOutputFrameAllocs(t *testing.T) {
	list, geom := publicTileListCopyFixture(t)
	format := FrameFormat{
		Width:        geom.OutputFrameWidth,
		Height:       geom.OutputFrameHeight,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        1,
	}
	src := bindPublicTileListFrame(t, format)
	dst := bindPublicTileListFrame(t, format)
	fillTileListCopySource(src)

	allocs := testing.AllocsPerRun(1000, func() {
		if err := copyTileListEntryToOutputFrame(&dst, &src, list, geom, 2); err != nil {
			t.Fatalf("CopyTileListEntryToOutputFrame: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("CopyTileListEntryToOutputFrame allocated: %f", allocs)
	}
}

func publicTileListCopyFixture(t *testing.T) (internalparser.TileList, internalparser.TileListOutputGeometry) {
	t.Helper()
	payload := []byte{0x01, 0x01, 0x00, 0x02}
	payload = append(payload, 0, 0, 0, 0, 0, 0xa0)
	payload = append(payload, 0, 0, 1, 0, 0, 0xa1)
	payload = append(payload, 0, 1, 1, 0, 0, 0xa2)
	list, err := internalparser.ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU: %v", err)
	}
	tiles := internalparser.TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1, 2},
	}
	geom, err := internalparser.TileListOutputGeometryForGrid(list, tiles, false)
	if err != nil {
		t.Fatalf("TileListOutputGeometryForGrid: %v", err)
	}
	return list, geom
}

func publicSingleTileListCopyFixture(t *testing.T) (internalparser.TileList, internalparser.TileListOutputGeometry) {
	t.Helper()
	list := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorTileRow: 0,
			AnchorTileCol: 0,
			TileData:      []byte{0xaa},
		}},
	}
	tiles := internalparser.TileInfo{
		Cols:       1,
		Rows:       1,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 1},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1},
	}
	geom, err := internalparser.TileListOutputGeometryForGrid(list, tiles, false)
	if err != nil {
		t.Fatalf("TileListOutputGeometryForGrid: %v", err)
	}
	return list, geom
}

func bindPublicTileListFrame(t *testing.T, format FrameFormat) Frame {
	t.Helper()
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatalf("FrameRequiredSize: %v", err)
	}
	frame, err := internalframe.Bind(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatalf("BindFrame: %v", err)
	}
	return frame
}

func fillTileListCopySource(frame Frame) {
	for y := 0; y < frame.Y.Height; y++ {
		for x := 0; x < frame.Y.Width; x++ {
			setPublicFrameSample(frame.Y, frame.Layout.BytesPerSample, x, y, uint16(0x100+((x*3+y*5)&0x2ff)))
		}
	}
	for y := 0; y < frame.U.Height; y++ {
		for x := 0; x < frame.U.Width; x++ {
			setPublicFrameSample(frame.U, frame.Layout.BytesPerSample, x, y, uint16(0x200+((x*7+y*11)&0x1ff)))
			setPublicFrameSample(frame.V, frame.Layout.BytesPerSample, x, y, uint16(0x300+((x*13+y*17)&0xff)))
		}
	}
}

func TestPublicParseTileListOBUAllocs(t *testing.T) {
	payload := []byte{
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x01, 0xaa, 0xbb,
	}
	scratch := make([]internalparser.TileListEntry, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		list, err := internalparser.ParseTileListOBU(payload, scratch[:0])
		if err != nil || len(list.Entries) != 1 {
			t.Fatalf("err=%v entries=%d", err, len(list.Entries))
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseTileListOBU allocated: %f", allocs)
	}
}

func FuzzPublicParseTileListOBU(f *testing.F) {
	for _, seed := range [][]byte{
		nil,
		{0x00},
		{0x00, 0x00, 0x00},
		{0x00, 0x00, 0x00, 0x00},
		{
			0x01, 0x00, 0x00, 0x01,
			0x00, 0x00, 0x00, 0x00, 0x01, 0x10, 0x20,
			0x07, 0x00, 0x01, 0x00, 0x00, 0x30,
		},
		{0x00, 0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0, 0xaa},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		var scratch [internalparser.TileListMaxTiles]internalparser.TileListEntry
		list, err := internalparser.ParseTileListOBU(payload, scratch[:0])
		if err != nil {
			return
		}
		if list.TileCount() != len(list.Entries) {
			t.Fatalf("tile count=%d entries=%d", list.TileCount(), len(list.Entries))
		}
		if list.TileCount() <= 0 || list.TileCount() > internalparser.TileListMaxTiles {
			t.Fatalf("invalid tile count=%d", list.TileCount())
		}
		if list.TileCount() > list.OutputFrameWidthInTiles()*list.OutputFrameHeightInTiles() {
			t.Fatalf("tile count=%d exceeds output %dx%d", list.TileCount(), list.OutputFrameWidthInTiles(), list.OutputFrameHeightInTiles())
		}
		for i, entry := range list.Entries {
			if entry.AnchorFrameIdx >= internalparser.TileListMaxExternalReferences {
				t.Fatalf("entry %d anchor_frame_idx=%d", i, entry.AnchorFrameIdx)
			}
			if len(entry.TileData) != entry.TileDataSize() {
				t.Fatalf("entry %d data len=%d want %d", i, len(entry.TileData), entry.TileDataSize())
			}
		}
		if got := internalparser.AppendTileListOBU(nil, list); string(got) != string(payload) {
			t.Fatalf("round trip mismatch\n got=%x\nwant=%x", got, payload)
		}
	})
}

func TestPublicAnnexBRoundTrip(t *testing.T) {
	td := []byte{byte(internalobu.TypeTemporalDelimiter) << 3}
	seq := []byte{byte(internalobu.TypeSequenceHeader) << 3, 0xaa}
	frame := []byte{byte(internalobu.TypeFrame) << 3, 0xcc, 0xdd}

	// Build a low-overhead stream first (with obu_size fields).
	low := appendOBUForTest(nil, internalobu.TypeTemporalDelimiter, nil)
	low = appendOBUForTest(low, internalobu.TypeSequenceHeader, seq[1:])
	low = appendOBUForTest(low, internalobu.TypeFrame, frame[1:])

	annexB, err := internalobu.LowOverheadToAnnexB(nil, low, nil)
	if err != nil {
		t.Fatalf("LowOverheadToAnnexB err=%v", err)
	}
	if len(annexB) == 0 {
		t.Fatal("empty annex b output")
	}

	roundTripped, err := internalobu.AnnexBToLowOverhead(nil, annexB)
	if err != nil {
		t.Fatalf("AnnexBToLowOverhead err=%v", err)
	}
	if string(roundTripped) != string(low) {
		t.Fatalf("round-trip mismatch\n got=%x\nwant=%x", roundTripped, low)
	}

	// AppendAnnexBTemporalUnit + AnnexBToLowOverhead also lets callers
	// transcode externally-supplied frame_units into the low-overhead form.
	tu := internalobu.AppendAnnexBTemporalUnit(nil, [][][]byte{{td, seq, frame}})
	if len(tu) == 0 {
		t.Fatal("empty temporal unit")
	}
	if _, err := internalobu.AnnexBToLowOverhead(nil, tu); err != nil {
		t.Fatalf("AnnexBToLowOverhead err=%v", err)
	}
}

func TestPublicParseMetadataOBUAllocs(t *testing.T) {
	payload := []byte{0x02,
		0x12, 0x34, 0x56, 0x78,
		0x9A, 0xBC, 0xDE, 0xF0,
		0x11, 0x22, 0x33, 0x44,
		0x55, 0x66, 0x77, 0x88,
		0x01, 0x02, 0x03, 0x04,
		0x05, 0x06, 0x07, 0x08,
		0x80,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		meta, err := internalobu.ParseMetadata(payload)
		if err != nil || meta.Type != internalobu.MetadataTypeHDRMDCV {
			t.Fatalf("err=%v type=%v", err, meta.Type)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseMetadataOBU allocated: %f", allocs)
	}
}

func appendOBUForTest(dst []byte, typ internalobu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := internalobu.PutHeader(header[:], internalobu.Header{Type: typ, HasSizeField: true})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	dst = appendPublicLEB128(dst, uint32(len(payload)))
	dst = append(dst, payload...)
	return dst
}

func appendPublicAnnexBStream(dst []byte, temporalUnits ...[][][]byte) []byte {
	for _, temporalUnit := range temporalUnits {
		var temporal []byte
		for _, frameUnit := range temporalUnit {
			var frame []byte
			for _, obu := range frameUnit {
				frame = appendPublicLEB128(frame, uint32(len(obu)))
				frame = append(frame, obu...)
			}
			temporal = appendPublicLEB128(temporal, uint32(len(frame)))
			temporal = append(temporal, frame...)
		}
		dst = appendPublicLEB128(dst, uint32(len(temporal)))
		dst = append(dst, temporal...)
	}
	return dst
}

func appendPublicLEB128(dst []byte, value uint32) []byte {
	for {
		b := byte(value & 0x7f)
		value >>= 7
		if value != 0 {
			b |= 0x80
		}
		dst = append(dst, b)
		if value == 0 {
			return dst
		}
	}
}
