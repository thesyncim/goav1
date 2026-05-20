package parser

import "testing"

func FuzzParseSequenceHeader(f *testing.F) {
	f.Add(reducedStillPictureSequenceHeader())
	f.Add(realtimeSequenceHeader())
	f.Add([]byte{0xe0})
	f.Add([]byte{0x00})

	f.Fuzz(func(t *testing.T, payload []byte) {
		sh, err := ParseSequenceHeader(payload)
		if err != nil {
			return
		}
		if sh.SeqProfile > 2 {
			t.Fatalf("accepted invalid profile %d", sh.SeqProfile)
		}
		if sh.ReducedStillPictureHeader && !sh.StillPicture {
			t.Fatal("accepted reduced_still_picture_header without still_picture")
		}
		if sh.OperatingPointsCount == 0 || sh.OperatingPointsCount > 32 {
			t.Fatalf("bad operating point count %d", sh.OperatingPointsCount)
		}
		if sh.MaxFrameWidth == 0 || sh.MaxFrameHeight == 0 {
			t.Fatalf("bad dimensions %dx%d", sh.MaxFrameWidth, sh.MaxFrameHeight)
		}
	})
}

func FuzzParseFrameHeaderPrefix(f *testing.F) {
	f.Add([]byte{0x12, 0x80})
	f.Add([]byte{0x48, 0xc0})
	f.Add([]byte{0x80})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		hdr, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		if hdr.BitsRead < 0 || hdr.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d len=%d", hdr.BitsRead, len(payload))
		}
		if hdr.PrimaryRefFrame > PrimaryRefNone {
			t.Fatalf("PrimaryRefFrame=%d", hdr.PrimaryRefFrame)
		}
	})
}

func FuzzParseIntraFrameSize(f *testing.F) {
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x08, 0x00})
	f.Add([]byte{0x04, 0x00})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseIntraFrameSize(payload, seq, prefix, 0, 0)
		if err != nil {
			return
		}
		if size.BitsRead < prefix.BitsRead || size.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d prefix=%d len=%d", size.BitsRead, prefix.BitsRead, len(payload))
		}
		if size.CodedWidth == 0 || size.UpscaledWidth == 0 || size.Height == 0 {
			t.Fatalf("bad dimensions=%+v", size)
		}
		if size.CodedWidth > size.UpscaledWidth {
			t.Fatalf("coded width=%d upscaled=%d", size.CodedWidth, size.UpscaledWidth)
		}
	})
}

func FuzzParseFrameSize(f *testing.F) {
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x12, 0x00, 0x00, 0x00, 0x00})
	f.Add([]byte{0x09, 0x00, 0x00, 0x00, 0x00})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}
	var refs ReferenceState
	for i := 0; i < RefFrames; i++ {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint32(i),
			Size: FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		if size.BitsRead < prefix.BitsRead || size.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d prefix=%d len=%d", size.BitsRead, prefix.BitsRead, len(payload))
		}
		if size.CodedWidth == 0 || size.UpscaledWidth == 0 || size.Height == 0 {
			t.Fatalf("bad dimensions=%+v", size)
		}
	})
}

func FuzzParseTileInfo(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x12, 0x00, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := 0; i < RefFrames; i++ {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint32(i),
			Size: FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		if tiles.BitsRead < size.BitsRead || tiles.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d size=%d len=%d", tiles.BitsRead, size.BitsRead, len(payload))
		}
		if tiles.SBCols == 0 || tiles.SBRows == 0 || tiles.Cols == 0 || tiles.Rows == 0 {
			t.Fatalf("empty tile grid=%+v", tiles)
		}
		if tiles.ColStartSB[0] != 0 || tiles.ColStartSB[tiles.Cols] != tiles.SBCols {
			t.Fatalf("bad col bounds=%+v", tiles)
		}
		if tiles.RowStartSB[0] != 0 || tiles.RowStartSB[tiles.Rows] != tiles.SBRows {
			t.Fatalf("bad row bounds=%+v", tiles)
		}
		for i := uint8(1); i <= tiles.Cols; i++ {
			if tiles.ColStartSB[i] <= tiles.ColStartSB[i-1] {
				t.Fatalf("non-increasing col starts=%+v", tiles)
			}
		}
		for i := uint8(1); i <= tiles.Rows; i++ {
			if tiles.RowStartSB[i] <= tiles.RowStartSB[i-1] {
				t.Fatalf("non-increasing row starts=%+v", tiles)
			}
		}
		if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
			tileCount := uint16(tiles.Cols) * uint16(tiles.Rows)
			if tiles.ContextUpdateTileID >= tileCount || tiles.TileSizeBytes == 0 {
				t.Fatalf("tile data fields=%+v", tiles)
			}
		}
	})
}
