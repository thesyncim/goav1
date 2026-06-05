package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendSkipModeParamsPayloadDisabledForSingleReference(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 8}
	params := SkipModeParams{}
	payload, parsed := appendAndParseSkipModeParams(t, seq, prefix, InterFrameSize{}, nil, TransformReferenceParams{ReferenceMode: ReferenceModeSingle}, params)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.Allowed || parsed.Enabled || parsed.BitsRead != 0 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadBeforeAfterRefs(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{15, 17, 14, 18, 13, 19, 12})
	params := SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{0, 1}}
	payload, parsed := appendAndParseSkipModeParams(t, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.Allowed || !parsed.Enabled || parsed.RefFrameIdx != [2]uint8{0, 1} || parsed.BitsRead != 1 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadSecondBeforeRef(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{12, 14, 11, 10, 9, 8, 7})
	params := SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{0, 1}}
	_, parsed := appendAndParseSkipModeParams(t, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if !parsed.Allowed || parsed.Enabled || parsed.RefFrameIdx != [2]uint8{0, 1} || parsed.BitsRead != 1 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadNoLegalPair(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{16, 16, 16, 16, 16, 16, 16})
	params := SkipModeParams{}
	payload, parsed := appendAndParseSkipModeParams(t, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.Allowed || parsed.Enabled || parsed.BitsRead != 0 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadRejectsInvalid(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{15, 17, 14, 18, 13, 19, 12})
	cases := [...]struct {
		size   InterFrameSize
		refs   *parser.ReferenceState
		params SkipModeParams
	}{
		{size: size, refs: nil, params: SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{0, 1}}},
		{size: size, refs: &refs, params: SkipModeParams{Enabled: true}},
		{size: size, refs: &refs, params: SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{1, 2}}},
		{size: invalidSkipModeRefSize(size), refs: &refs, params: SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{0, 1}}},
	}
	var buf [1]byte
	for _, tc := range cases {
		if _, err := AppendSkipModeParamsPayload(buf[:0], seq, prefix, tc.size, tc.refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendSkipModeParamsPayload(%+v) err=%v want ErrInvalidFrame", tc.params, err)
		}
	}

	intraPrefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ErrorResilientMode: true}
	if _, err := AppendSkipModeParamsPayload(buf[:0], seq, intraPrefix, InterFrameSize{}, nil, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, SkipModeParams{Allowed: true}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("intra skip err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendSkipModeParamsPayloadShortBuffer(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{15, 17, 14, 18, 13, 19, 12})
	params := SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{0, 1}}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendSkipModeParamsPayload(dst, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendSkipModeParamsPayloadAllocs(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{15, 17, 14, 18, 13, 19, 12})
	transformRef := TransformReferenceParams{ReferenceMode: ReferenceModeSelect}
	params := SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{0, 1}}
	var buf [1]byte
	if _, err := AppendSkipModeParamsPayload(buf[:0], seq, prefix, size, &refs, transformRef, params); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = SkipModeParamsPayloadSize(seq, prefix, size, &refs, transformRef, params)
		_, _ = AppendSkipModeParamsPayload(buf[:0], seq, prefix, size, &refs, transformRef, params)
	})
	if allocs != 0 {
		t.Fatalf("AppendSkipModeParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseSkipModeParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) ([]byte, parser.SkipModeParams) {
	t.Helper()
	payloadSize, err := SkipModeParamsPayloadSize(seq, prefix, size, refs, transformRef, params)
	if err != nil {
		t.Fatalf("SkipModeParamsPayloadSize: %v", err)
	}
	var buf [1]byte
	payload, err := AppendSkipModeParamsPayload(buf[:0], seq, prefix, size, refs, transformRef, params)
	if err != nil {
		t.Fatalf("AppendSkipModeParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseSkipModeParams(
		payload,
		parser.SequenceHeader{EnableOrderHint: seq.EnableOrderHint, OrderHintBits: seq.OrderHintBits},
		parser.FrameHeaderPrefix{FrameType: parser.FrameType(prefix.FrameType), OrderHint: prefix.OrderHint},
		parser.FrameSize{RefFrameIdx: size.RefFrameIdx},
		refs,
		parser.TransformReferenceParams{ReferenceMode: parser.ReferenceMode(transformRef.ReferenceMode)},
	)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	return payload, parsed
}

func skipModeSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        64,
		MaxFrameHeight:       64,
		Use128x128Superblock: true,
		EnableOrderHint:      true,
		OrderHintBits:        5,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func encoderSkipModeRefs(seq SequenceHeader, orderHints [7]uint8) (InterFrameSize, parser.ReferenceState) {
	var size InterFrameSize
	var refs parser.ReferenceState
	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:     true,
			OrderHint: orderHints[i],
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
	return size, refs
}

func invalidSkipModeRefSize(size InterFrameSize) InterFrameSize {
	size.RefFrameIdx[0] = parser.RefFrames
	return size
}
