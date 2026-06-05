package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendGlobalMotionParamsPayloadIntraReadsNoBits(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
	payload, parsed := appendAndParseGlobalMotionParams(t, prefix, InterFrameSize{}, parser.TileInfo{}, nil, DefaultGlobalMotionParams())
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.BitsRead != 0 {
		t.Fatalf("global motion bits=%d", parsed.BitsRead)
	}
	for i := range parser.InterRefsPerFrame {
		if parsed.Ref[i] != parser.DefaultWarpedMotionParams() {
			t.Fatalf("ref[%d]=%+v", i, parsed.Ref[i])
		}
	}
}

func TestAppendGlobalMotionParamsPayloadInterIdentity(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	payload, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{}, &refs, DefaultGlobalMotionParams())
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.BitsRead != parser.InterRefsPerFrame {
		t.Fatalf("BitsRead=%d want %d", parsed.BitsRead, parser.InterRefsPerFrame)
	}
	for i := range parser.InterRefsPerFrame {
		if parsed.Ref[i] != parser.DefaultWarpedMotionParams() {
			t.Fatalf("ref[%d]=%+v", i, parsed.Ref[i])
		}
	}
}

func TestAppendGlobalMotionParamsPayloadTranslation(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	params := DefaultGlobalMotionParams()
	params.Ref[0] = DefaultWarpedMotionParams()
	params.Ref[0].Type = GlobalMotionTranslation
	params.Ref[0].Matrix[0] = 2 << gmTransOnlyPrecDiff
	params.Ref[0].Matrix[1] = -1 << gmTransOnlyPrecDiff

	_, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if parsed.Ref[0].Type != parser.GlobalMotionTranslation ||
		parsed.Ref[0].Matrix[0] != params.Ref[0].Matrix[0] ||
		parsed.Ref[0].Matrix[1] != params.Ref[0].Matrix[1] {
		t.Fatalf("ref[0]=%+v want %+v", parsed.Ref[0], params.Ref[0])
	}
}

func TestAppendGlobalMotionParamsPayloadRotZoomAndAffine(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	params := DefaultGlobalMotionParams()
	params.Ref[0] = DefaultWarpedMotionParams()
	params.Ref[0].Type = GlobalMotionRotZoom
	params.Ref[0].Matrix[2] = (1 << warpedModelPrecBits) + 4
	params.Ref[0].Matrix[3] = 2
	params.Ref[0].Matrix[4] = -2
	params.Ref[0].Matrix[5] = (1 << warpedModelPrecBits) + 4
	params.Ref[1] = DefaultWarpedMotionParams()
	params.Ref[1].Type = GlobalMotionAffine
	params.Ref[1].Matrix[2] = (1 << warpedModelPrecBits) - 6
	params.Ref[1].Matrix[3] = 4
	params.Ref[1].Matrix[4] = 2
	params.Ref[1].Matrix[5] = (1 << warpedModelPrecBits) + 8

	_, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	for i := range 2 {
		if parsed.Ref[i].Type != parser.GlobalMotionType(params.Ref[i].Type) || parsed.Ref[i].Matrix != params.Ref[i].Matrix {
			t.Fatalf("ref[%d]=+%v want %+v", i, parsed.Ref[i], params.Ref[i])
		}
	}
}

func TestAppendGlobalMotionParamsPayloadRejectsInvalid(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	invalidIntra := DefaultGlobalMotionParams()
	invalidIntra.Ref[0].Type = GlobalMotionTranslation
	invalidTranslation := DefaultGlobalMotionParams()
	invalidTranslation.Ref[0].Type = GlobalMotionTranslation
	invalidTranslation.Ref[0].Matrix[0] = 1
	invalidRotZoom := DefaultGlobalMotionParams()
	invalidRotZoom.Ref[0].Type = GlobalMotionRotZoom
	invalidRotZoom.Ref[0].Matrix[2] = 1 << warpedModelPrecBits
	invalidRotZoom.Ref[0].Matrix[3] = 2
	invalidRotZoom.Ref[0].Matrix[4] = 0
	invalidRotZoom.Ref[0].Matrix[5] = 1 << warpedModelPrecBits
	badRefSize := size
	badRefSize.RefFrameIdx[0] = parser.RefFrames
	cases := [...]struct {
		prefix FrameHeaderPrefix
		size   InterFrameSize
		refs   *parser.ReferenceState
		params GlobalMotionParams
	}{
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}, size: size, refs: nil, params: invalidIntra},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderType(9)}, size: size, refs: &refs, params: DefaultGlobalMotionParams()},
		{prefix: prefix, size: size, refs: &refs, params: invalidTranslation},
		{prefix: prefix, size: size, refs: &refs, params: invalidRotZoom},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: 0}, size: badRefSize, refs: &refs, params: DefaultGlobalMotionParams()},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: 0}, size: size, refs: nil, params: DefaultGlobalMotionParams()},
	}
	var buf [8]byte
	for _, tc := range cases {
		if _, err := AppendGlobalMotionParamsPayload(buf[:0], tc.prefix, tc.size, parser.TileInfo{}, tc.refs, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendGlobalMotionParamsPayload(%+v) err=%v want ErrInvalidFrame", tc.prefix, err)
		}
	}
}

func TestAppendGlobalMotionParamsPayloadShortBuffer(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendGlobalMotionParamsPayload(dst, prefix, size, parser.TileInfo{}, &refs, DefaultGlobalMotionParams())
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendGlobalMotionParamsPayloadAllocs(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	params := DefaultGlobalMotionParams()
	var buf [1]byte
	if _, err := AppendGlobalMotionParamsPayload(buf[:0], prefix, size, parser.TileInfo{}, &refs, params); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = GlobalMotionParamsPayloadSize(prefix, size, parser.TileInfo{}, &refs, params)
		_, _ = AppendGlobalMotionParamsPayload(buf[:0], prefix, size, parser.TileInfo{}, &refs, params)
	})
	if allocs != 0 {
		t.Fatalf("AppendGlobalMotionParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseGlobalMotionParams(t *testing.T, prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) ([]byte, parser.GlobalMotionParams) {
	t.Helper()
	payloadSize, err := GlobalMotionParamsPayloadSize(prefix, size, tiles, refs, params)
	if err != nil {
		t.Fatalf("GlobalMotionParamsPayloadSize: %v", err)
	}
	var buf [64]byte
	payload, err := AppendGlobalMotionParamsPayload(buf[:0], prefix, size, tiles, refs, params)
	if err != nil {
		t.Fatalf("AppendGlobalMotionParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseGlobalMotionParams(
		payload,
		parser.FrameHeaderPrefix{FrameType: parser.FrameType(prefix.FrameType), PrimaryRefFrame: prefix.PrimaryRefFrame},
		parser.FrameSize{RefFrameIdx: size.RefFrameIdx},
		tiles,
		refs,
		parser.FrameModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	return payload, parsed
}

func encoderGlobalMotionRefs() (InterFrameSize, parser.ReferenceState) {
	var size InterFrameSize
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			GlobalMotion: defaultGlobal,
		}
	}
	return size, refs
}
