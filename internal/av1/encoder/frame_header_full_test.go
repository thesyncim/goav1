package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendIntraFrameHeaderPayloadRoundTripKeyFrame(t *testing.T) {
	seq := fullHeaderSequence()
	header := fullHeaderKeyFrame()
	payload, parsed := appendAndParseIntraFrameHeader(t, seq, header)

	if len(payload) == 0 {
		t.Fatal("payload is empty")
	}
	if parsed.Prefix.FrameType != parser.FrameTypeKey || !parsed.Prefix.ShowFrame {
		t.Fatalf("prefix=%+v", parsed.Prefix)
	}
	if parsed.Size.UpscaledWidth != 64 || parsed.Size.Height != 64 || parsed.Size.CodedWidth != 64 {
		t.Fatalf("size=%+v", parsed.Size)
	}
	if parsed.Tile.Cols != 1 || parsed.Tile.Rows != 1 || !parsed.Tile.RefreshContext {
		t.Fatalf("tile=%+v", parsed.Tile)
	}
	if parsed.Quant.BaseQIdx != 50 || parsed.Seg.Enabled || parsed.Delta.DeltaQPresent {
		t.Fatalf("quant/seg/delta=%+v %+v %+v", parsed.Quant, parsed.Seg, parsed.Delta)
	}
	if parsed.LoopFilter.LevelY != [2]uint8{4, 4} || parsed.CDEF.Bits != 0 || parsed.Restoration.Type[0] != parser.RestorationNone {
		t.Fatalf("filters=%+v %+v %+v", parsed.LoopFilter, parsed.CDEF, parsed.Restoration)
	}
	if parsed.TransformRef.TransformMode != parser.TransformModeLargest || parsed.FrameMode.ReducedTxSet || parsed.GlobalMotion.Ref[0] != parser.DefaultWarpedMotionParams() {
		t.Fatalf("motion=%+v %+v %+v", parsed.TransformRef, parsed.FrameMode, parsed.GlobalMotion)
	}
	if parsed.FilmGrain.Apply {
		t.Fatalf("film grain=%+v", parsed.FilmGrain)
	}
}

func TestAppendInterFrameHeaderPayloadRoundTrip(t *testing.T) {
	seq := fullHeaderSequence()
	header, refs := fullHeaderInterFrame(seq)
	payload, parsed := appendAndParseInterFrameHeader(t, seq, header, &refs)

	if len(payload) == 0 {
		t.Fatal("payload is empty")
	}
	if parsed.Prefix.FrameType != parser.FrameTypeInter || !parsed.Prefix.ShowFrame {
		t.Fatalf("prefix=%+v", parsed.Prefix)
	}
	if parsed.Size.RefreshFrameFlags != 0x02 || parsed.Size.RefFrameIdx[0] != 0 || parsed.Size.UpscaledWidth != 64 {
		t.Fatalf("size=%+v", parsed.Size)
	}
	if parsed.Tile.InterpolationFilter != parser.InterpolationEightTap || !parsed.Tile.RefreshContext {
		t.Fatalf("tile=%+v", parsed.Tile)
	}
	if parsed.TransformRef.ReferenceMode != parser.ReferenceModeSingle || parsed.SkipMode.Allowed {
		t.Fatalf("transform/skip=%+v %+v", parsed.TransformRef, parsed.SkipMode)
	}
	if parsed.GlobalMotion.Ref[0] != parser.DefaultWarpedMotionParams() || parsed.FilmGrain.Apply {
		t.Fatalf("tail=%+v %+v", parsed.GlobalMotion, parsed.FilmGrain)
	}
}

func TestAppendIntraFrameHeaderPayloadRejectsInvalid(t *testing.T) {
	seq := fullHeaderSequence()
	header := fullHeaderKeyFrame()
	header.Tile.SBCols = 2
	var buf [256]byte
	if _, err := AppendIntraFrameHeaderPayload(buf[:0], seq, header); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("AppendIntraFrameHeaderPayload err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendInterFrameHeaderPayloadRejectsInvalid(t *testing.T) {
	seq := fullHeaderSequence()
	header, refs := fullHeaderInterFrame(seq)
	header.References = &refs
	header.Size.RefFrameIdx[0] = parser.RefFrames
	var buf [256]byte
	if _, err := AppendInterFrameHeaderPayload(buf[:0], seq, header); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("AppendInterFrameHeaderPayload err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendIntraFrameHeaderPayloadShortBuffer(t *testing.T) {
	seq := fullHeaderSequence()
	header := fullHeaderKeyFrame()
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendIntraFrameHeaderPayload(dst, seq, header)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendInterFrameHeaderPayloadShortBuffer(t *testing.T) {
	seq := fullHeaderSequence()
	header, refs := fullHeaderInterFrame(seq)
	header.References = &refs
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendInterFrameHeaderPayload(dst, seq, header)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendIntraFrameHeaderPayloadAllocs(t *testing.T) {
	seq := fullHeaderSequence()
	header := fullHeaderKeyFrame()
	var buf [256]byte
	if _, err := AppendIntraFrameHeaderPayload(buf[:0], seq, header); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = IntraFrameHeaderPayloadSize(seq, header)
		_, _ = AppendIntraFrameHeaderPayload(buf[:0], seq, header)
	})
	if allocs != 0 {
		t.Fatalf("AppendIntraFrameHeaderPayload allocated: %f", allocs)
	}
}

func TestAppendInterFrameHeaderPayloadAllocs(t *testing.T) {
	seq := fullHeaderSequence()
	header, refs := fullHeaderInterFrame(seq)
	header.References = &refs
	var buf [256]byte
	if _, err := AppendInterFrameHeaderPayload(buf[:0], seq, header); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = InterFrameHeaderPayloadSize(seq, header)
		_, _ = AppendInterFrameHeaderPayload(buf[:0], seq, header)
	})
	if allocs != 0 {
		t.Fatalf("AppendInterFrameHeaderPayload allocated: %f", allocs)
	}
}

type parsedIntraFrameHeader struct {
	Prefix       parser.FrameHeaderPrefix
	Size         parser.FrameSize
	Tile         parser.TileInfo
	Quant        parser.QuantizationParams
	Seg          parser.SegmentationParams
	Delta        parser.DeltaParams
	LoopFilter   parser.LoopFilterParams
	CDEF         parser.CDEFParams
	Restoration  parser.RestorationParams
	TransformRef parser.TransformReferenceParams
	SkipMode     parser.SkipModeParams
	FrameMode    parser.FrameModeParams
	GlobalMotion parser.GlobalMotionParams
	FilmGrain    parser.FilmGrainParams
}

func appendAndParseIntraFrameHeader(t *testing.T, seq SequenceHeader, header IntraFrameHeaderParams) ([]byte, parsedIntraFrameHeader) {
	t.Helper()
	payloadSize, err := IntraFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("IntraFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := AppendIntraFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("AppendIntraFrameHeaderPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}

	var seqBuf [128]byte
	seqPayload, err := AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}

	prefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := parser.ParseIntraFrameSize(payload, parsedSeq, prefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	tiles, err := parser.ParseTileInfo(payload, parsedSeq, prefix, size)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	quant, err := parser.ParseQuantizationParams(payload, parsedSeq, tiles)
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	seg, err := parser.ParseSegmentationParams(payload, prefix, quant, nil)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	delta, err := parser.ParseDeltaParams(payload, size, quant, seg)
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	lf, err := parser.ParseLoopFilterParams(payload, parsedSeq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	cdef, err := parser.ParseCDEFParams(payload, parsedSeq, size, seg, lf)
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	restoration, err := parser.ParseRestorationParams(payload, parsedSeq, size, seg, cdef)
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	transformRef, err := parser.ParseTransformReferenceParams(payload, prefix, seg, restoration)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	skipMode, err := parser.ParseSkipModeParams(payload, parsedSeq, prefix, size, nil, transformRef)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	frameMode, err := parser.ParseFrameModeParams(payload, parsedSeq, prefix, skipMode)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	globalMotion, err := parser.ParseGlobalMotionParams(payload, prefix, size, tiles, nil, frameMode)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	filmGrain, err := parser.ParseFilmGrainParams(payload, parsedSeq, prefix, size, nil, globalMotion)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	return payload, parsedIntraFrameHeader{
		Prefix:       prefix,
		Size:         size,
		Tile:         tiles,
		Quant:        quant,
		Seg:          seg,
		Delta:        delta,
		LoopFilter:   lf,
		CDEF:         cdef,
		Restoration:  restoration,
		TransformRef: transformRef,
		SkipMode:     skipMode,
		FrameMode:    frameMode,
		GlobalMotion: globalMotion,
		FilmGrain:    filmGrain,
	}
}

func appendAndParseInterFrameHeader(t *testing.T, seq SequenceHeader, header InterFrameHeaderParams, refs *parser.ReferenceState) ([]byte, parsedIntraFrameHeader) {
	t.Helper()
	header.References = refs
	payloadSize, err := InterFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("InterFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := AppendInterFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("AppendInterFrameHeaderPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}

	var seqBuf [128]byte
	seqPayload, err := AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}

	prefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := parser.ParseFrameSize(payload, parsedSeq, prefix, refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	tiles, err := parser.ParseTileInfo(payload, parsedSeq, prefix, size)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	quant, err := parser.ParseQuantizationParams(payload, parsedSeq, tiles)
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	seg, err := parser.ParseSegmentationParams(payload, prefix, quant, nil)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	delta, err := parser.ParseDeltaParams(payload, size, quant, seg)
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	lf, err := parser.ParseLoopFilterParams(payload, parsedSeq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	cdef, err := parser.ParseCDEFParams(payload, parsedSeq, size, seg, lf)
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	restoration, err := parser.ParseRestorationParams(payload, parsedSeq, size, seg, cdef)
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	transformRef, err := parser.ParseTransformReferenceParams(payload, prefix, seg, restoration)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	skipMode, err := parser.ParseSkipModeParams(payload, parsedSeq, prefix, size, refs, transformRef)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	frameMode, err := parser.ParseFrameModeParams(payload, parsedSeq, prefix, skipMode)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	globalMotion, err := parser.ParseGlobalMotionParams(payload, prefix, size, tiles, refs, frameMode)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	filmGrain, err := parser.ParseFilmGrainParams(payload, parsedSeq, prefix, size, refs, globalMotion)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	return payload, parsedIntraFrameHeader{
		Prefix:       prefix,
		Size:         size,
		Tile:         tiles,
		Quant:        quant,
		Seg:          seg,
		Delta:        delta,
		LoopFilter:   lf,
		CDEF:         cdef,
		Restoration:  restoration,
		TransformRef: transformRef,
		SkipMode:     skipMode,
		FrameMode:    frameMode,
		GlobalMotion: globalMotion,
		FilmGrain:    filmGrain,
	}
}

func fullHeaderSequence() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:              64,
		MaxFrameHeight:             64,
		Use128x128Superblock:       false,
		SeqForceScreenContentTools: 1,
		SeqForceIntegerMV:          1,
		EnableCDEF:                 true,
		EnableRestoration:          true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func fullHeaderKeyFrame() IntraFrameHeaderParams {
	header := IntraFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:               FrameHeaderTypeKey,
			ShowFrame:               true,
			ErrorResilientMode:      true,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         EncoderPrimaryRefNone,
		},
		Size: IntraFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		Tile: TileInfo{
			RefreshContext: true,
			SBCols:         1,
			SBRows:         1,
			Cols:           1,
			Rows:           1,
			ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
			RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: QuantizationParams{
			BaseQIdx: 50,
		},
		LoopFilter: LoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			Sharpness:           0,
			ModeRefDeltaEnabled: false,
			Deltas:              defaultLoopFilterDeltas(),
		},
		CDEF: CDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		Restoration: RestorationParams{
			Type:       [3]RestorationType{RestorationNone, RestorationNone, RestorationNone},
			UnitSizeY:  0,
			UnitSizeUV: 0,
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformModeLargest,
			ReferenceMode: ReferenceModeSingle,
		},
		FrameMode: FrameModeParams{},
		FilmGrain: FilmGrainParams{},
	}
	return header
}

func fullHeaderInterFrame(seq SequenceHeader) (InterFrameHeaderParams, parser.ReferenceState) {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := uint8(0); i < parser.RefFrames; i++ {
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			OrderHint:    i,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}
	header := InterFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:               FrameHeaderTypeInter,
			ShowFrame:               true,
			ShowableFrame:           true,
			ErrorResilientMode:      false,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         EncoderPrimaryRefNone,
		},
		Size: InterFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0x02,
			RefFrameIdx:         [7]uint8{0, 0, 0, 0, 0, 0, 0},
		},
		Tile: TileInfo{
			InterpolationFilter: InterpolationEightTap,
			RefreshContext:      true,
			SBCols:              1,
			SBRows:              1,
			Cols:                1,
			Rows:                1,
			ColStartSB:          [MaxTileCols + 1]uint16{0, 1},
			RowStartSB:          [MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: QuantizationParams{
			BaseQIdx: 50,
		},
		LoopFilter: LoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			Sharpness:           0,
			ModeRefDeltaEnabled: false,
			Deltas:              defaultLoopFilterDeltas(),
		},
		CDEF: CDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		Restoration: RestorationParams{
			Type:       [3]RestorationType{RestorationNone, RestorationNone, RestorationNone},
			UnitSizeY:  0,
			UnitSizeUV: 0,
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformModeLargest,
			ReferenceMode: ReferenceModeSingle,
		},
		GlobalMotion: DefaultGlobalMotionParams(),
	}
	return header, refs
}
