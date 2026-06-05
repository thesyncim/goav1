package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendFilmGrainParamsPayloadSkippedBySequence(t *testing.T) {
	seq := filmGrainSequenceHeader()
	seq.FilmGrainParamsPresent = false
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	payload, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, FilmGrainParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.ParamsPresent || parsed.Apply || parsed.BitsRead != 0 || parsed.BitDepth != 8 {
		t.Fatalf("film grain=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadApplyFalse(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	payload, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, FilmGrainParams{})
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.ParamsPresent || parsed.Apply || parsed.BitsRead != 1 || parsed.BitDepth != 8 {
		t.Fatalf("film grain=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadUpdateMinimal(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	params := minimalFilmGrainUpdate()
	_, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, params)
	if !parsed.Apply || !parsed.Update || parsed.Seed != 0x1234 {
		t.Fatalf("film grain=%+v", parsed)
	}
	if parsed.NumYPoints != 1 || parsed.YPoints[0] != [2]uint8{10, 20} {
		t.Fatalf("y points=%+v", parsed.YPoints[:parsed.NumYPoints])
	}
	if parsed.ScalingShift != 9 || parsed.ARCoeffLag != 0 || parsed.ARCoeffShift != 8 || parsed.GrainScaleShift != 3 {
		t.Fatalf("shift fields=%+v", parsed)
	}
	if !parsed.Overlap || parsed.ClipToRestrictedRange {
		t.Fatalf("range flags=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadChromaAndCoeffs(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true}
	params := minimalFilmGrainUpdate()
	params.NumCbPoints = 1
	params.CbPoints[0] = [2]uint8{4, 5}
	params.NumCrPoints = 1
	params.CrPoints[0] = [2]uint8{6, 7}
	params.ARCoeffLag = 1
	params.ARCoeffsY[0] = -2
	params.ARCoeffsY[1] = 3
	params.ARCoeffsY[2] = 4
	params.ARCoeffsY[3] = -5
	params.ARCoeffsCb[0] = 6
	params.ARCoeffsCb[1] = -7
	params.ARCoeffsCb[2] = 8
	params.ARCoeffsCb[3] = -9
	params.ARCoeffsCb[4] = 10
	params.ARCoeffsCr[0] = -11
	params.ARCoeffsCr[1] = 12
	params.ARCoeffsCr[2] = -13
	params.ARCoeffsCr[3] = 14
	params.ARCoeffsCr[4] = -15
	params.CbMult = 2
	params.CbLumaMult = 3
	params.CbOffset = 257
	params.CrMult = 4
	params.CrLumaMult = 5
	params.CrOffset = 258

	_, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, params)
	if parsed.NumCbPoints != 1 || parsed.CbPoints[0] != [2]uint8{4, 5} || parsed.NumCrPoints != 1 || parsed.CrPoints[0] != [2]uint8{6, 7} {
		t.Fatalf("chroma points=%+v", parsed)
	}
	if parsed.ARCoeffLag != 1 || parsed.ARCoeffsY[0] != -2 || parsed.ARCoeffsCb[4] != 10 || parsed.ARCoeffsCr[4] != -15 {
		t.Fatalf("coeffs=%+v", parsed)
	}
	if parsed.CbOffset != 257 || parsed.CrOffset != 258 {
		t.Fatalf("offsets=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadCopiesReference(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true}
	var size InterFrameSize
	size.RefFrameIdx[0] = 2
	var refs parser.ReferenceState
	refs.Frames[2] = parser.ReferenceFrame{
		Valid: true,
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Update:        true,
			Seed:          0x1111,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{40, 50}},
			ScalingShift:  9,
		},
	}
	params := FilmGrainParams{Apply: true, Seed: 0x2244, RefIdx: 2}
	_, parsed := appendAndParseFilmGrainParams(t, seq, prefix, size, &refs, params)
	if !parsed.Apply || parsed.Update || parsed.RefIdx != 2 || parsed.Seed != 0x2244 {
		t.Fatalf("reference copy flags=%+v", parsed)
	}
	if parsed.NumYPoints != 1 || parsed.YPoints[0] != [2]uint8{40, 50} {
		t.Fatalf("copied y points=%+v", parsed.YPoints)
	}
}

func TestAppendFilmGrainParamsPayloadRejectsInvalid(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	nonIncreasing := minimalFilmGrainUpdate()
	nonIncreasing.NumYPoints = 2
	nonIncreasing.YPoints[0] = [2]uint8{10, 20}
	nonIncreasing.YPoints[1] = [2]uint8{10, 30}
	badShift := minimalFilmGrainUpdate()
	badShift.ScalingShift = 12
	badChroma := minimalFilmGrainUpdate()
	badChroma.NumCbPoints = 1
	badChroma.CbPoints[0] = [2]uint8{1, 2}
	badReference := FilmGrainParams{Apply: true, RefIdx: 2}
	cases := [...]struct {
		prefix FrameHeaderPrefix
		size   InterFrameSize
		refs   *parser.ReferenceState
		params FilmGrainParams
	}{
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderType(9), ShowFrame: true}, params: FilmGrainParams{}},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}, params: minimalFilmGrainUpdate()},
		{prefix: prefix, params: nonIncreasing},
		{prefix: prefix, params: badShift},
		{prefix: prefix, params: badChroma},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true}, params: badReference},
	}
	var buf [64]byte
	for _, tc := range cases {
		if _, err := AppendFilmGrainParamsPayload(buf[:0], seq, tc.prefix, tc.size, tc.refs, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendFilmGrainParamsPayload(%+v) err=%v want ErrInvalidFrame", tc.params, err)
		}
	}
}

func TestAppendFilmGrainParamsPayloadShortBuffer(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendFilmGrainParamsPayload(dst, seq, prefix, InterFrameSize{}, nil, minimalFilmGrainUpdate())
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendFilmGrainParamsPayloadAllocs(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	params := minimalFilmGrainUpdate()
	var buf [16]byte
	if _, err := AppendFilmGrainParamsPayload(buf[:0], seq, prefix, InterFrameSize{}, nil, params); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = FilmGrainParamsPayloadSize(seq, prefix, InterFrameSize{}, nil, params)
		_, _ = AppendFilmGrainParamsPayload(buf[:0], seq, prefix, InterFrameSize{}, nil, params)
	})
	if allocs != 0 {
		t.Fatalf("AppendFilmGrainParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseFilmGrainParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) ([]byte, parser.FilmGrainParams) {
	t.Helper()
	payloadSize, err := FilmGrainParamsPayloadSize(seq, prefix, size, refs, params)
	if err != nil {
		t.Fatalf("FilmGrainParamsPayloadSize: %v", err)
	}
	var buf [128]byte
	payload, err := AppendFilmGrainParamsPayload(buf[:0], seq, prefix, size, refs, params)
	if err != nil {
		t.Fatalf("AppendFilmGrainParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseFilmGrainParams(
		payload,
		parser.SequenceHeader{
			FilmGrainParamsPresent: seq.FilmGrainParamsPresent,
			ColorConfig: parser.ColorConfig{
				BitDepth:     seq.ColorConfig.BitDepth,
				MonoChrome:   seq.ColorConfig.MonoChrome,
				SubsamplingX: seq.ColorConfig.SubsamplingX,
				SubsamplingY: seq.ColorConfig.SubsamplingY,
			},
		},
		parser.FrameHeaderPrefix{
			ShowExistingFrame: prefix.ShowExistingFrame,
			FrameType:         parser.FrameType(prefix.FrameType),
			ShowFrame:         prefix.ShowFrame,
			ShowableFrame:     prefix.ShowableFrame,
		},
		parser.FrameSize{RefFrameIdx: size.RefFrameIdx},
		refs,
		parser.GlobalMotionParams{},
	)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	return payload, parsed
}

func filmGrainSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:                Profile0,
		FilmGrainParamsPresent: true,
		OperatingPointsCount:   1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        64,
		MaxFrameHeight:       64,
		Use128x128Superblock: true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func minimalFilmGrainUpdate() FilmGrainParams {
	params := FilmGrainParams{
		Apply:           true,
		Update:          true,
		Seed:            0x1234,
		NumYPoints:      1,
		ScalingShift:    9,
		ARCoeffShift:    8,
		GrainScaleShift: 3,
		Overlap:         true,
	}
	params.YPoints[0] = [2]uint8{10, 20}
	return params
}
