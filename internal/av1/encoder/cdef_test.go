package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendCDEFParamsPayloadSkipsLossless(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	payload, parsed := appendAndParseCDEFParams(t, seq, size, true, CDEFParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.StrengthCount != 0 || parsed.BitsRead != 0 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
}

func TestAppendCDEFParamsPayloadStrengths(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	cdef := CDEFParams{Damping: 5, Bits: 2}
	for i := uint8(0); i < 4; i++ {
		cdef.YStrength[i] = 10 + i
		cdef.UVStrength[i] = 20 + i
	}
	_, parsed := appendAndParseCDEFParams(t, seq, size, false, cdef)
	if parsed.Damping != 5 || parsed.Bits != 2 || parsed.StrengthCount != 4 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
	if parsed.YStrength[3] != 13 || parsed.UVStrength[3] != 23 {
		t.Fatalf("parsed strengths y=%v uv=%v", parsed.YStrength[:4], parsed.UVStrength[:4])
	}
}

func TestAppendCDEFParamsPayloadMonochromeSkipsUVStrengths(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.MonoChrome = true
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	cdef := CDEFParams{Damping: 4, Bits: 1}
	cdef.YStrength[0] = 7
	cdef.YStrength[1] = 8
	_, parsed := appendAndParseCDEFParams(t, seq, size, false, cdef)
	if parsed.StrengthCount != 2 || parsed.YStrength[0] != 7 || parsed.YStrength[1] != 8 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
	if parsed.UVStrength[0] != 0 || parsed.UVStrength[1] != 0 {
		t.Fatalf("parsed uv strengths=%v", parsed.UVStrength[:2])
	}
}

func TestAppendCDEFParamsPayloadDisabledBySequence(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.EnableCDEF = false
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	payload, parsed := appendAndParseCDEFParams(t, seq, size, false, CDEFParams{})
	if len(payload) != 0 || parsed.StrengthCount != 0 {
		t.Fatalf("payload=% x parsed=%+v", payload, parsed)
	}
}

func TestAppendCDEFParamsInterPayload(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	cdef := CDEFParams{Damping: 3, Bits: 0}
	cdef.YStrength[0] = 11
	cdef.UVStrength[0] = 12
	payloadSize, err := CDEFParamsInterPayloadSize(seq, size, false, cdef)
	if err != nil {
		t.Fatalf("CDEFParamsInterPayloadSize: %v", err)
	}
	var buf [4]byte
	payload, err := AppendCDEFParamsInterPayload(buf[:0], seq, size, false, cdef)
	if err != nil {
		t.Fatalf("AppendCDEFParamsInterPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseCDEFParams(payload, parseEncoderSequenceHeader(t, seq), parser.FrameSize{}, parser.SegmentationParams{}, parser.LoopFilterParams{})
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	if parsed.Damping != 3 || parsed.Bits != 0 || parsed.YStrength[0] != 11 || parsed.UVStrength[0] != 12 {
		t.Fatalf("parsed inter cdef=%+v", parsed)
	}
}

func TestAppendCDEFParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	nonzeroDisabled := CDEFParams{Damping: 3}
	unused := CDEFParams{Damping: 3, Bits: 0}
	unused.YStrength[1] = 1
	tooHigh := CDEFParams{Damping: 3, Bits: 0}
	tooHigh.YStrength[0] = 64
	cases := [...]CDEFParams{
		{Damping: 2},
		{Damping: 7},
		{Damping: 3, Bits: 4},
		unused,
		tooHigh,
	}
	var buf [8]byte
	for _, cdef := range cases {
		if _, err := AppendCDEFParamsPayload(buf[:0], seq, size, false, cdef); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendCDEFParamsPayload(%+v) err=%v want ErrInvalidFrame", cdef, err)
		}
	}
	if _, err := AppendCDEFParamsPayload(buf[:0], seq, size, true, nonzeroDisabled); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("lossless nonzero cdef err=%v want ErrInvalidFrame", err)
	}
	mono := seq
	mono.ColorConfig.MonoChrome = true
	uv := CDEFParams{Damping: 3, Bits: 0}
	uv.UVStrength[0] = 1
	if _, err := AppendCDEFParamsPayload(buf[:0], mono, size, false, uv); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono uv cdef err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendCDEFParamsPayloadShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	cdef := CDEFParams{Damping: 3, Bits: 0}
	cdef.YStrength[0] = 7
	cdef.UVStrength[0] = 9
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendCDEFParamsPayload(dst, seq, size, false, cdef)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendCDEFParamsPayloadAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	cdef := CDEFParams{Damping: 3, Bits: 0}
	cdef.YStrength[0] = 7
	cdef.UVStrength[0] = 9
	var buf [4]byte
	if _, err := AppendCDEFParamsPayload(buf[:0], seq, size, false, cdef); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = CDEFParamsPayloadSize(seq, size, false, cdef)
		_, _ = AppendCDEFParamsPayload(buf[:0], seq, size, false, cdef)
	})
	if allocs != 0 {
		t.Fatalf("AppendCDEFParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseCDEFParams(t *testing.T, seq SequenceHeader, size IntraFrameSize, allLossless bool, cdef CDEFParams) ([]byte, parser.CDEFParams) {
	t.Helper()
	payloadSize, err := CDEFParamsPayloadSize(seq, size, allLossless, cdef)
	if err != nil {
		t.Fatalf("CDEFParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := AppendCDEFParamsPayload(buf[:0], seq, size, allLossless, cdef)
	if err != nil {
		t.Fatalf("AppendCDEFParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseCDEFParams(
		payload,
		parseEncoderSequenceHeader(t, seq),
		parser.FrameSize{AllowIntrabc: size.AllowIntrabc},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.LoopFilterParams{},
	)
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	return payload, parsed
}
