package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendLoopFilterParamsPayloadLosslessDefaults(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	lf := LoopFilterParams{
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  true,
		Deltas:              defaultLoopFilterDeltas(),
	}
	payload, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, true, lf, nil)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if !parsed.ModeRefDeltaEnabled || !parsed.ModeRefDeltaUpdate || parsed.Deltas != parserDefaultLoopFilterDeltas() {
		t.Fatalf("parsed lossless loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsPayloadLevelsAndUpdates(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	deltas := defaultLoopFilterDeltas()
	deltas.Ref[0] = -2
	deltas.Mode[1] = 3
	lf := LoopFilterParams{
		LevelY:              [2]uint8{10, 0},
		LevelU:              5,
		LevelV:              6,
		Sharpness:           3,
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  true,
		Deltas:              deltas,
	}
	_, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, false, lf, nil)
	if parsed.LevelY[0] != 10 || parsed.LevelY[1] != 0 || parsed.LevelU != 5 || parsed.LevelV != 6 || parsed.Sharpness != 3 {
		t.Fatalf("parsed levels=%+v", parsed)
	}
	if !parsed.ModeRefDeltaEnabled || !parsed.ModeRefDeltaUpdate ||
		parsed.Deltas.Ref[0] != -2 || parsed.Deltas.Ref[4] != -1 || parsed.Deltas.Mode[1] != 3 {
		t.Fatalf("parsed deltas=%+v", parsed.Deltas)
	}
}

func TestAppendLoopFilterParamsPayloadMonochromeSkipsUVLevels(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.MonoChrome = true
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	lf := LoopFilterParams{
		LevelY:    [2]uint8{12, 0},
		Sharpness: 1,
		Deltas:    defaultLoopFilterDeltas(),
	}
	_, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, false, lf, nil)
	if parsed.LevelY[0] != 12 || parsed.LevelU != 0 || parsed.LevelV != 0 || parsed.Sharpness != 1 {
		t.Fatalf("parsed monochrome loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsPayloadCopiesPreviousDeltas(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	previous := defaultLoopFilterDeltas()
	previous.Ref[2] = 4
	lf := LoopFilterParams{
		Sharpness:           1,
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  false,
		Deltas:              previous,
	}
	_, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, false, lf, &previous)
	if parsed.Deltas.Ref[2] != 4 || parsed.Deltas.Ref[4] != -1 || parsed.Sharpness != 1 {
		t.Fatalf("parsed previous loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsInterPayload(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	lf := LoopFilterParams{
		LevelY:              [2]uint8{8, 7},
		LevelU:              6,
		LevelV:              5,
		Sharpness:           2,
		ModeRefDeltaEnabled: false,
		Deltas:              defaultLoopFilterDeltas(),
	}
	payloadSize, err := LoopFilterParamsInterPayloadSize(seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("LoopFilterParamsInterPayloadSize: %v", err)
	}
	var buf [8]byte
	payload, err := AppendLoopFilterParamsInterPayload(buf[:0], seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("AppendLoopFilterParamsInterPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseLoopFilterParams(payload, parseEncoderSequenceHeader(t, seq), parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, parser.FrameSize{}, parser.SegmentationParams{}, parser.DeltaParams{}, nil)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	if parsed.LevelY != [2]uint8{8, 7} || parsed.LevelU != 6 || parsed.LevelV != 5 || parsed.Sharpness != 2 || parsed.ModeRefDeltaEnabled {
		t.Fatalf("parsed inter loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	base := defaultLoopFilterDeltas()
	cases := [...]LoopFilterParams{
		{LevelY: [2]uint8{64, 0}, Deltas: base},
		{LevelU: 1, Deltas: base},
		{ModeRefDeltaEnabled: false, ModeRefDeltaUpdate: true, Deltas: base},
		{ModeRefDeltaEnabled: false, Deltas: LoopFilterDeltas{}},
	}
	var buf [8]byte
	for _, lf := range cases {
		if _, err := AppendLoopFilterParamsPayload(buf[:0], seq, prefix, size, false, lf, nil); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendLoopFilterParamsPayload(%+v) err=%v want ErrInvalidFrame", lf, err)
		}
	}
	lossless := LoopFilterParams{Deltas: base}
	if _, err := AppendLoopFilterParamsPayload(buf[:0], seq, prefix, size, true, lossless, nil); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("lossless missing default flags err=%v want ErrInvalidFrame", err)
	}
	nonNone := FrameHeaderPrefix{PrimaryRefFrame: 0}
	if _, err := AppendLoopFilterParamsPayload(buf[:0], seq, nonNone, size, false, LoopFilterParams{}, nil); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing previous err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendLoopFilterParamsPayloadShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	lf := LoopFilterParams{LevelY: [2]uint8{1, 0}, Deltas: defaultLoopFilterDeltas()}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLoopFilterParamsPayload(dst, seq, prefix, size, false, lf, nil)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendLoopFilterParamsPayloadAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	lf := LoopFilterParams{LevelY: [2]uint8{1, 0}, Deltas: defaultLoopFilterDeltas()}
	var buf [8]byte
	if _, err := AppendLoopFilterParamsPayload(buf[:0], seq, prefix, size, false, lf, nil); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = LoopFilterParamsPayloadSize(seq, prefix, size, false, lf, nil)
		_, _ = AppendLoopFilterParamsPayload(buf[:0], seq, prefix, size, false, lf, nil)
	})
	if allocs != 0 {
		t.Fatalf("AppendLoopFilterParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseLoopFilterParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, parser.LoopFilterParams) {
	t.Helper()
	payloadSize, err := LoopFilterParamsPayloadSize(seq, prefix, size, allLossless, lf, previous)
	if err != nil {
		t.Fatalf("LoopFilterParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := AppendLoopFilterParamsPayload(buf[:0], seq, prefix, size, allLossless, lf, previous)
	if err != nil {
		t.Fatalf("AppendLoopFilterParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsedPrevious := parserLoopFilterDeltas(previous)
	parsed, err := parser.ParseLoopFilterParams(
		payload,
		parseEncoderSequenceHeader(t, seq),
		parser.FrameHeaderPrefix{PrimaryRefFrame: prefix.PrimaryRefFrame},
		parser.FrameSize{AllowIntrabc: size.AllowIntrabc},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.DeltaParams{},
		parsedPrevious,
	)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	return payload, parsed
}

func parserLoopFilterDeltas(previous *LoopFilterDeltas) *parser.LoopFilterDeltas {
	if previous == nil {
		return nil
	}
	var out parser.LoopFilterDeltas
	for i := uint8(0); i < encoderRefFrames; i++ {
		out.Ref[i] = previous.Ref[i]
	}
	for i := uint8(0); i < LoopFilterModeDeltas; i++ {
		out.Mode[i] = previous.Mode[i]
	}
	return &out
}

func parserDefaultLoopFilterDeltas() parser.LoopFilterDeltas {
	return *parserLoopFilterDeltasPtr(defaultLoopFilterDeltas())
}

func parserLoopFilterDeltasPtr(deltas LoopFilterDeltas) *parser.LoopFilterDeltas {
	return parserLoopFilterDeltas(&deltas)
}
