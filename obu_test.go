package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicTemporalUnitIterator(t *testing.T) {
	var stream []byte
	firstStart := len(stream)
	stream = appendPublicLowOverheadOBU(stream, av1.OBUTemporalDelimiter, nil)
	stream = appendPublicLowOverheadOBU(stream, av1.OBUSequenceHeader, []byte{0xaa})
	stream = appendPublicLowOverheadOBU(stream, av1.OBUFrameHeader, []byte{0xbb})
	firstEnd := len(stream)
	stream = appendPublicLowOverheadOBU(stream, av1.OBUTemporalDelimiter, nil)
	stream = appendPublicLowOverheadOBU(stream, av1.OBUFrame, []byte{0xcc})
	secondEnd := len(stream)

	it := av1.NewTemporalUnitIterator(stream)
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
	stream := appendPublicLowOverheadOBU(nil, av1.OBUSequenceHeader, []byte{0xaa})
	it := av1.NewTemporalUnitIterator(stream)
	_, _, err := it.Next()
	if !errors.Is(err, av1.ErrOBUMissingTemporalDelimiter) {
		t.Fatalf("err=%v want %v", err, av1.ErrOBUMissingTemporalDelimiter)
	}
}

func TestPublicAnnexBIterator(t *testing.T) {
	td := []byte{byte(av1.OBUTemporalDelimiter) << 3}
	seq := []byte{byte(av1.OBUSequenceHeader) << 3, 0xaa}
	frameHeader := []byte{byte(av1.OBUFrameHeader) << 3, 0xbb}
	frame := []byte{byte(av1.OBUFrame) << 3, 0xcc}
	stream := appendPublicAnnexBStream(nil,
		[][][]byte{{td, seq}, {frameHeader}},
		[][][]byte{{td, frame}},
	)

	want := []struct {
		raw      []byte
		typ      av1.OBUType
		temporal uint32
		frame    uint32
		obu      uint32
	}{
		{raw: td, typ: av1.OBUTemporalDelimiter, temporal: 0, frame: 0, obu: 0},
		{raw: seq, typ: av1.OBUSequenceHeader, temporal: 0, frame: 0, obu: 1},
		{raw: frameHeader, typ: av1.OBUFrameHeader, temporal: 0, frame: 1, obu: 0},
		{raw: td, typ: av1.OBUTemporalDelimiter, temporal: 1, frame: 0, obu: 0},
		{raw: frame, typ: av1.OBUFrame, temporal: 1, frame: 0, obu: 1},
	}

	it := av1.NewAnnexBIterator(stream)
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
	src := []byte{0x03, byte(av1.OBUFrame)<<3 | 0x02, 0x01, 0xdd}
	unit, consumed, err := av1.ParseAnnexBElement(src)
	if err != nil {
		t.Fatal(err)
	}
	if consumed != len(src) || unit.Header.Type != av1.OBUFrame || string(unit.Payload) != string([]byte{0xdd}) {
		t.Fatalf("unit=%+v consumed=%d", unit, consumed)
	}
}

func TestPublicAnnexBIteratorRejectsInvalid(t *testing.T) {
	it := av1.NewAnnexBIterator([]byte{0x00})
	_, _, err := it.Next()
	if !errors.Is(err, av1.ErrOBUInvalidAnnexB) {
		t.Fatalf("err=%v want %v", err, av1.ErrOBUInvalidAnnexB)
	}
}

func TestPublicOBUStreamIteratorsAllocs(t *testing.T) {
	lowOverhead := appendPublicLowOverheadOBU(nil, av1.OBUTemporalDelimiter, nil)
	lowOverhead = appendPublicLowOverheadOBU(lowOverhead, av1.OBUFrame, []byte{0xaa})
	annexB := appendPublicAnnexBStream(nil, [][][]byte{{{byte(av1.OBUTemporalDelimiter) << 3}, {byte(av1.OBUFrame) << 3, 0xaa}}})
	annexElement := []byte{0x02, byte(av1.OBUFrame) << 3, 0xdd}

	allocs := testing.AllocsPerRun(1000, func() {
		tuIt := av1.NewTemporalUnitIterator(lowOverhead)
		unit, ok, err := tuIt.Next()
		if err != nil || !ok || len(unit.Raw) == 0 {
			t.Fatalf("temporal unit=%+v ok=%v err=%v", unit, ok, err)
		}

		annexIt := av1.NewAnnexBIterator(annexB)
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

		element, consumed, err := av1.ParseAnnexBElement(annexElement)
		if err != nil || consumed != 3 || element.Header.Type != av1.OBUFrame {
			t.Fatalf("element=%+v consumed=%d err=%v", element, consumed, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("public OBU stream iterators allocated: %f", allocs)
	}
}

func FuzzPublicOBUStreamIterators(f *testing.F) {
	lowOverhead := appendPublicLowOverheadOBU(nil, av1.OBUTemporalDelimiter, nil)
	lowOverhead = appendPublicLowOverheadOBU(lowOverhead, av1.OBUFrame, []byte{0xaa})
	annexB := appendPublicAnnexBStream(nil, [][][]byte{{{byte(av1.OBUTemporalDelimiter) << 3}, {byte(av1.OBUFrame) << 3, 0xaa}}})

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

		_, _, _ = av1.ParseOBUHeader(data)
		_, _ = av1.ParseOBUElement(data)
		_, _, _ = av1.ParseLowOverheadOBU(data)
		_, _, _ = av1.ParseAnnexBElement(data)
		_, _ = av1.NormalizeLowOverheadOBU(make([]byte, len(data)*2+16), data)
		fuzzPublicDriveLowOverhead(data)
		fuzzPublicDriveTemporalUnit(data)
		fuzzPublicDriveAnnexB(data)
	})
}

func fuzzPublicDriveLowOverhead(data []byte) {
	it := av1.NewLowOverheadIterator(data)
	for range 1 << 12 {
		if _, ok, err := it.Next(); err != nil || !ok {
			return
		}
	}
}

func fuzzPublicDriveTemporalUnit(data []byte) {
	it := av1.NewTemporalUnitIterator(data)
	for range 1 << 12 {
		if _, ok, err := it.Next(); err != nil || !ok {
			return
		}
	}
}

func fuzzPublicDriveAnnexB(data []byte) {
	it := av1.NewAnnexBIterator(data)
	for range 1 << 12 {
		if _, ok, err := it.Next(); err != nil || !ok {
			return
		}
	}
}

func TestPublicParseMetadataOBU(t *testing.T) {
	// HDR-CLL: type=2, max_cll=1000, max_fall=400, trailing 0x80.
	payload := []byte{0x02, 0x03, 0xE8, 0x01, 0x90, 0x80}
	meta, err := av1.ParseMetadataOBU(payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU err=%v", err)
	}
	if meta.Type != av1.MetadataTypeHDRCLL {
		t.Fatalf("type=%v", meta.Type)
	}
	if meta.HDRCLL.MaxCLL != 1000 || meta.HDRCLL.MaxFALL != 400 {
		t.Fatalf("cll=%#v", meta.HDRCLL)
	}

	if _, err := av1.ParseMetadataOBU(nil); !errors.Is(err, av1.ErrMetadataShortPayload) {
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

	var scratch [4]av1.TileListEntry
	list, err := av1.ParseTileListOBU(payload, scratch[:0])
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
	encoded := av1.AppendTileListOBU(nil, list)
	if string(encoded) != string(payload) {
		t.Fatalf("AppendTileListOBU mismatch\n got=%x\nwant=%x", encoded, payload)
	}

	if _, err := av1.ParseTileListOBU([]byte{0x00}, nil); !errors.Is(err, av1.ErrTileListShortHeader) {
		t.Fatalf("short err=%v want ErrTileListShortHeader", err)
	}
	outOfGridAnchor := []byte{0x00, 0x00, 0x00, 0x00, 0, 1, 0, 0, 0, 0xaa}
	if _, err := av1.ParseTileListOBU(outOfGridAnchor, nil); !errors.Is(err, av1.ErrTileListInvalidAnchorTile) {
		t.Fatalf("anchor tile err=%v want ErrTileListInvalidAnchorTile", err)
	}
}

func TestPublicParseTileListOBUAllocs(t *testing.T) {
	payload := []byte{
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x01, 0xaa, 0xbb,
	}
	scratch := make([]av1.TileListEntry, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		list, err := av1.ParseTileListOBU(payload, scratch[:0])
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
		var scratch [av1.TileListMaxTiles]av1.TileListEntry
		list, err := av1.ParseTileListOBU(payload, scratch[:0])
		if err != nil {
			return
		}
		if list.TileCount() != len(list.Entries) {
			t.Fatalf("tile count=%d entries=%d", list.TileCount(), len(list.Entries))
		}
		if list.TileCount() <= 0 || list.TileCount() > av1.TileListMaxTiles {
			t.Fatalf("invalid tile count=%d", list.TileCount())
		}
		if list.TileCount() > list.OutputFrameWidthInTiles()*list.OutputFrameHeightInTiles() {
			t.Fatalf("tile count=%d exceeds output %dx%d", list.TileCount(), list.OutputFrameWidthInTiles(), list.OutputFrameHeightInTiles())
		}
		for i, entry := range list.Entries {
			if entry.AnchorFrameIdx >= av1.TileListMaxExternalReferences {
				t.Fatalf("entry %d anchor_frame_idx=%d", i, entry.AnchorFrameIdx)
			}
			if len(entry.TileData) != entry.TileDataSize() {
				t.Fatalf("entry %d data len=%d want %d", i, len(entry.TileData), entry.TileDataSize())
			}
		}
		if got := av1.AppendTileListOBU(nil, list); string(got) != string(payload) {
			t.Fatalf("round trip mismatch\n got=%x\nwant=%x", got, payload)
		}
	})
}

func TestPublicAnnexBRoundTrip(t *testing.T) {
	td := []byte{byte(av1.OBUTemporalDelimiter) << 3}
	seq := []byte{byte(av1.OBUSequenceHeader) << 3, 0xaa}
	frame := []byte{byte(av1.OBUFrame) << 3, 0xcc, 0xdd}

	// Build a low-overhead stream first (with obu_size fields).
	low := appendPublicLowOverheadOBU(nil, av1.OBUTemporalDelimiter, nil)
	low = appendPublicLowOverheadOBU(low, av1.OBUSequenceHeader, seq[1:])
	low = appendPublicLowOverheadOBU(low, av1.OBUFrame, frame[1:])

	annexB, err := av1.LowOverheadToAnnexB(nil, low, nil)
	if err != nil {
		t.Fatalf("LowOverheadToAnnexB err=%v", err)
	}
	if len(annexB) == 0 {
		t.Fatal("empty annex b output")
	}

	roundTripped, err := av1.AnnexBToLowOverhead(nil, annexB)
	if err != nil {
		t.Fatalf("AnnexBToLowOverhead err=%v", err)
	}
	if string(roundTripped) != string(low) {
		t.Fatalf("round-trip mismatch\n got=%x\nwant=%x", roundTripped, low)
	}

	// AppendAnnexBTemporalUnit + AnnexBToLowOverhead also lets callers
	// transcode externally-supplied frame_units into the low-overhead form.
	tu := av1.AppendAnnexBTemporalUnit(nil, [][][]byte{{td, seq, frame}})
	if len(tu) == 0 {
		t.Fatal("empty temporal unit")
	}
	if _, err := av1.AnnexBToLowOverhead(nil, tu); err != nil {
		t.Fatalf("AnnexBToLowOverhead err=%v", err)
	}
}

func TestPublicParseMetadataOBUAllocs(t *testing.T) {
	payload := []byte{0x03,
		0x12, 0x34, 0x56, 0x78,
		0x9A, 0xBC, 0xDE, 0xF0,
		0x11, 0x22, 0x33, 0x44,
		0x55, 0x66, 0x77, 0x88,
		0x01, 0x02, 0x03, 0x04,
		0x05, 0x06, 0x07, 0x08,
		0x80,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		meta, err := av1.ParseMetadataOBU(payload)
		if err != nil || meta.Type != av1.MetadataTypeHDRMDCV {
			t.Fatalf("err=%v type=%v", err, meta.Type)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseMetadataOBU allocated: %f", allocs)
	}
}

func appendPublicLowOverheadOBU(dst []byte, typ av1.OBUType, payload []byte) []byte {
	var header [2]byte
	n, err := av1.PutOBUHeader(header[:], av1.OBUHeader{Type: typ, HasSizeField: true})
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
