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

func FuzzParseQuantizationParams(f *testing.F) {
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
	seed.writeBits(0, 8)  // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		if quant.BitsRead < tiles.BitsRead || quant.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d tile=%d len=%d", quant.BitsRead, tiles.BitsRead, len(payload))
		}
		if !seq.ColorConfig.MonoChrome && !quant.DiffUVDeltas {
			if quant.VDCDelta != quant.UDCDelta || quant.VACDelta != quant.UACDelta {
				t.Fatalf("uv deltas not copied=%+v", quant)
			}
		}
		if quant.UsingQMatrix && !seq.ColorConfig.SeparateUVDeltaQ && quant.QMatrixLevelV != quant.QMatrixLevelU {
			t.Fatalf("qmatrix v not copied=%+v", quant)
		}
	})
}

func FuzzParseSegmentationParams(f *testing.F) {
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
	seed.writeBits(0, 8)  // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previous SegmentationData
	clearSegmentationRefs(&previous)

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previous)
		if err != nil {
			return
		}
		if seg.BitsRead < quant.BitsRead || seg.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d quant=%d len=%d", seg.BitsRead, quant.BitsRead, len(payload))
		}
		if !seg.Enabled && (seg.UpdateMap || seg.UpdateData || seg.TemporalUpdate) {
			t.Fatalf("disabled segmentation has update flags=%+v", seg)
		}
		for i := 0; i < MaxSegments; i++ {
			if seg.Lossless[i] && seg.QIndex[i] != 0 {
				t.Fatalf("lossless segment has qindex=%+v", seg)
			}
			if !seg.Enabled && seg.Data.Segments[i].RefFrame != -1 {
				t.Fatalf("disabled segmentation ref=%+v", seg.Data.Segments[i])
			}
		}
	})
}

func FuzzParseDeltaParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(true)  // delta_q_present
	seed.writeBits(1, 2)  // delta_q_res_log2
	seed.writeBool(false) // delta_lf_present
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previous SegmentationData
	clearSegmentationRefs(&previous)

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previous)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		if delta.BitsRead < seg.BitsRead || delta.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d seg=%d len=%d", delta.BitsRead, seg.BitsRead, len(payload))
		}
		if !delta.DeltaQPresent && (delta.DeltaQResLog2 != 0 || delta.DeltaLFPresent || delta.DeltaLFResLog2 != 0 || delta.DeltaLFMulti) {
			t.Fatalf("delta fields set while absent=%+v", delta)
		}
		if size.AllowIntrabc && delta.DeltaLFPresent {
			t.Fatalf("intrabc frame has delta lf=%+v", delta)
		}
	})
}

func FuzzParseLoopFilterParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		if lf.BitsRead < delta.BitsRead || lf.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d delta=%d len=%d", lf.BitsRead, delta.BitsRead, len(payload))
		}
		if seg.AllLossless || size.AllowIntrabc {
			if lf.BitsRead != delta.BitsRead || !lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate {
				t.Fatalf("lossless/intrabc loopfilter=%+v delta bits=%d", lf, delta.BitsRead)
			}
		}
		if seq.ColorConfig.MonoChrome && (lf.LevelU != 0 || lf.LevelV != 0) {
			t.Fatalf("monochrome uv levels=%+v", lf)
		}
	})
}

func FuzzParseCDEFParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		if cdef.BitsRead < lf.BitsRead || cdef.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d lf=%d len=%d", cdef.BitsRead, lf.BitsRead, len(payload))
		}
		if seg.AllLossless || !seq.EnableCDEF || size.AllowIntrabc {
			if cdef.BitsRead != lf.BitsRead || cdef.StrengthCount != 0 {
				t.Fatalf("skipped cdef=%+v lf bits=%d", cdef, lf.BitsRead)
			}
		}
		if cdef.StrengthCount > MaxCDEFStrengths {
			t.Fatalf("too many strengths=%+v", cdef)
		}
		if seq.ColorConfig.MonoChrome {
			for i := uint8(0); i < cdef.StrengthCount; i++ {
				if cdef.UVStrength[i] != 0 {
					t.Fatalf("monochrome uv strength=%+v", cdef)
				}
			}
		}
	})
}

func FuzzParseRestorationParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		if restoration.BitsRead < cdef.BitsRead || restoration.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d cdef=%d len=%d", restoration.BitsRead, cdef.BitsRead, len(payload))
		}
		if (seg.AllLossless && !size.SuperResEnabled) || !seq.EnableRestoration || size.AllowIntrabc {
			if restoration.BitsRead != cdef.BitsRead || restoration.UnitSizeY != 0 {
				t.Fatalf("skipped restoration=%+v cdef bits=%d", restoration, cdef.BitsRead)
			}
		}
		if seq.ColorConfig.MonoChrome && (restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) {
			t.Fatalf("monochrome restoration types=%+v", restoration)
		}
		if restoration.UnitSizeY != 0 && restoration.UnitSizeY != 64 && restoration.UnitSizeY != 128 && restoration.UnitSizeY != 256 {
			t.Fatalf("bad y unit size=%+v", restoration)
		}
	})
}

func FuzzParseTransformReferenceParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		if transformRef.BitsRead < restoration.BitsRead || transformRef.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d restoration=%d len=%d", transformRef.BitsRead, restoration.BitsRead, len(payload))
		}
		if seg.AllLossless && transformRef.TransformMode != TransformMode4x4Only {
			t.Fatalf("lossless transform/reference=%+v", transformRef)
		}
		if !frameTypeIsInterOrSwitch(prefix.FrameType) && transformRef.ReferenceMode != ReferenceModeSingle {
			t.Fatalf("intra transform/reference=%+v", transformRef)
		}
	})
}

func FuzzParseSkipModeParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		if skipMode.BitsRead < transformRef.BitsRead || skipMode.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d transform=%d len=%d", skipMode.BitsRead, transformRef.BitsRead, len(payload))
		}
		if !skipMode.Allowed {
			if skipMode.Enabled || skipMode.BitsRead != transformRef.BitsRead {
				t.Fatalf("disabled skip mode=%+v transform bits=%d", skipMode, transformRef.BitsRead)
			}
		}
		if skipMode.Allowed && (skipMode.RefFrameIdx[0] >= InterRefsPerFrame || skipMode.RefFrameIdx[1] >= InterRefsPerFrame) {
			t.Fatalf("skip refs=%+v", skipMode)
		}
		if !seq.EnableOrderHint || !frameTypeIsInterOrSwitch(prefix.FrameType) || transformRef.ReferenceMode == ReferenceModeSingle {
			if skipMode.Allowed {
				t.Fatalf("unexpected skip mode=%+v", skipMode)
			}
		}
	})
}

func FuzzParseFrameModeParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	seed.writeBool(false) // reduced_tx_set
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		frameMode, err := ParseFrameModeParams(payload, seq, prefix, skipMode)
		if err != nil {
			return
		}
		if frameMode.BitsRead < skipMode.BitsRead || frameMode.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d skip=%d len=%d", frameMode.BitsRead, skipMode.BitsRead, len(payload))
		}
		if prefix.ErrorResilientMode || !frameTypeIsInterOrSwitch(prefix.FrameType) || !seq.EnableWarpedMotion {
			if frameMode.AllowWarpedMotion {
				t.Fatalf("unexpected warped motion=%+v prefix=%+v", frameMode, prefix)
			}
			if frameMode.BitsRead != skipMode.BitsRead+1 {
				t.Fatalf("frame mode bits=%+v skip bits=%d", frameMode, skipMode.BitsRead)
			}
		}
	})
}

func FuzzParseGlobalMotionParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	seed.writeBool(false) // reduced_tx_set
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	defaultGlobal := DefaultGlobalMotionParams()
	var refs ReferenceState
	for i := 0; i < RefFrames; i++ {
		refs.Frames[i] = ReferenceFrame{
			Valid:        true,
			OrderHint:    uint32(i),
			GlobalMotion: defaultGlobal,
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		frameMode, err := ParseFrameModeParams(payload, seq, prefix, skipMode)
		if err != nil {
			return
		}
		globalMotion, err := ParseGlobalMotionParams(payload, prefix, size, tiles, &refs, frameMode)
		if err != nil {
			return
		}
		if globalMotion.BitsRead < frameMode.BitsRead || globalMotion.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d frame mode=%d len=%d", globalMotion.BitsRead, frameMode.BitsRead, len(payload))
		}
		if !frameTypeIsInterOrSwitch(prefix.FrameType) && globalMotion.BitsRead != frameMode.BitsRead {
			t.Fatalf("intra global motion=%+v frame bits=%d", globalMotion, frameMode.BitsRead)
		}
		for i := 0; i < InterRefsPerFrame; i++ {
			if globalMotion.Ref[i].Type > GlobalMotionAffine {
				t.Fatalf("global motion ref[%d]=%+v", i, globalMotion.Ref[i])
			}
		}
	})
}

func FuzzParseFilmGrainParams(f *testing.F) {
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
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	seed.writeBool(false) // reduced_tx_set
	seed.writeBool(false) // apply_grain
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	defaultGlobal := DefaultGlobalMotionParams()
	defaultFilmGrain := FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		BitDepth:      seq.ColorConfig.BitDepth,
	}
	var refs ReferenceState
	for i := 0; i < RefFrames; i++ {
		refs.Frames[i] = ReferenceFrame{
			Valid:        true,
			OrderHint:    uint32(i),
			GlobalMotion: defaultGlobal,
			FilmGrain:    defaultFilmGrain,
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

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
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		frameMode, err := ParseFrameModeParams(payload, seq, prefix, skipMode)
		if err != nil {
			return
		}
		globalMotion, err := ParseGlobalMotionParams(payload, prefix, size, tiles, &refs, frameMode)
		if err != nil {
			return
		}
		filmGrain, err := ParseFilmGrainParams(payload, seq, prefix, size, &refs, globalMotion)
		if err != nil {
			return
		}
		if filmGrain.BitsRead < globalMotion.BitsRead || filmGrain.BitsRead > len(payload)*8 {
			t.Fatalf("BitsRead=%d global=%d len=%d", filmGrain.BitsRead, globalMotion.BitsRead, len(payload))
		}
		if filmGrain.BitDepth != seq.ColorConfig.BitDepth {
			t.Fatalf("bit depth=%+v seq=%+v", filmGrain, seq.ColorConfig)
		}
		if !seq.FilmGrainParamsPresent || (!prefix.ShowFrame && !prefix.ShowableFrame) {
			if filmGrain.Apply || filmGrain.BitsRead != globalMotion.BitsRead {
				t.Fatalf("skipped film grain=%+v global bits=%d", filmGrain, globalMotion.BitsRead)
			}
		} else if !filmGrain.Apply && filmGrain.BitsRead != globalMotion.BitsRead+1 {
			t.Fatalf("disabled film grain=%+v global bits=%d", filmGrain, globalMotion.BitsRead)
		}
		if filmGrain.NumYPoints > MaxFilmGrainYPoints ||
			filmGrain.NumCbPoints > MaxFilmGrainUVPoints ||
			filmGrain.NumCrPoints > MaxFilmGrainUVPoints {
			t.Fatalf("too many film grain points=%+v", filmGrain)
		}
		if seq.ColorConfig.MonoChrome && (filmGrain.ChromaScalingFromLuma || filmGrain.NumCbPoints != 0 || filmGrain.NumCrPoints != 0) {
			t.Fatalf("monochrome film grain=%+v", filmGrain)
		}
		if seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY &&
			((filmGrain.NumCbPoints == 0) != (filmGrain.NumCrPoints == 0)) {
			t.Fatalf("4:2:0 chroma film grain=%+v", filmGrain)
		}
	})
}
