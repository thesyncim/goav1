package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendFrameHeaderIntraPayloadShownKeyFrame(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	_, parsedPrefix, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if parsedPrefix.BitsRead >= parsedSize.BitsRead {
		t.Fatalf("bits did not advance: prefix=%d size=%d", parsedPrefix.BitsRead, parsedSize.BitsRead)
	}
	if parsedSize.RefreshFrameFlags != 0xff ||
		parsedSize.UpscaledWidth != seq.MaxFrameWidth ||
		parsedSize.CodedWidth != seq.MaxFrameWidth ||
		parsedSize.Height != seq.MaxFrameHeight ||
		parsedSize.RenderWidth != seq.MaxFrameWidth ||
		parsedSize.RenderHeight != seq.MaxFrameHeight ||
		parsedSize.SuperResEnabled ||
		parsedSize.SuperResDenominator != 8 {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
}

func TestAppendFrameHeaderIntraPayloadOverrideSuperresRender(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.MaxFrameWidth = 64
	seq.MaxFrameHeight = 32
	prefix := shownKeyFrameHeaderPrefix(true)
	size := IntraFrameSize{
		UpscaledWidth:       48,
		Height:              24,
		RenderWidth:         40,
		RenderHeight:        20,
		SuperResDenominator: 12,
		HaveRenderSize:      true,
		RefreshFrameFlags:   0xff,
	}
	_, _, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if !parsedSize.SuperResEnabled || parsedSize.SuperResDenominator != 12 {
		t.Fatalf("superres=%+v", parsedSize)
	}
	if parsedSize.UpscaledWidth != 48 || parsedSize.CodedWidth != 32 || parsedSize.Height != 24 {
		t.Fatalf("dimensions=%+v", parsedSize)
	}
	if !parsedSize.HaveRenderSize || parsedSize.RenderWidth != 40 || parsedSize.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", parsedSize)
	}
}

func TestAppendFrameHeaderIntraPayloadHiddenKeyRefreshAndOrderHints(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		OrderHint:          4,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	for i := uint8(0); i < encoderRefFrames; i++ {
		size.RefOrderHints[i] = i
	}
	_, _, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if parsedSize.RefreshFrameFlags != 0x01 {
		t.Fatalf("RefreshFrameFlags=%02x", parsedSize.RefreshFrameFlags)
	}
	for i := uint8(0); i < encoderRefFrames; i++ {
		if parsedSize.RefOrderHints[i] != i {
			t.Fatalf("RefOrderHints[%d]=%d", i, parsedSize.RefOrderHints[i])
		}
	}
}

func TestAppendFrameHeaderIntraPayloadAllowIntrabc(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
	seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	prefix := shownKeyFrameHeaderPrefix(false)
	prefix.AllowScreenContentTools = true
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		AllowIntrabc:        true,
		RefreshFrameFlags:   0xff,
	}
	_, _, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if !parsedSize.AllowIntrabc {
		t.Fatalf("AllowIntrabc=false parsed size=%+v", parsedSize)
	}
}

func TestAppendFrameHeaderIntraPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	basePrefix := shownKeyFrameHeaderPrefix(false)
	baseSize := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	tests := []struct {
		name string
		mut  func(*FrameHeaderPrefix, *IntraFrameSize)
	}{
		{name: "inter prefix", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			*prefix = FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true, PrimaryRefFrame: 0}
		}},
		{name: "bad dimensions without override", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.UpscaledWidth--
		}},
		{name: "shown key refresh flags", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.RefreshFrameFlags = 0x7f
		}},
		{name: "bad superres denom", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.SuperResDenominator = 17
		}},
		{name: "bad render size", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.HaveRenderSize = true
			size.RenderWidth = 0
			size.RenderHeight = 20
		}},
		{name: "intrabc with superres", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			prefix.AllowScreenContentTools = true
			size.SuperResDenominator = 12
			size.AllowIntrabc = true
		}},
	}
	var buf [32]byte
	for _, tt := range tests {
		prefix := basePrefix
		size := baseSize
		tt.mut(&prefix, &size)
		if _, err := appendFrameHeaderIntraPayload(buf[:0], seq, prefix, size); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFrame)
		}
	}
}

func TestAppendFrameHeaderInterPayloadDirectSize(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := interFrameHeaderPrefix(false)
	size := InterFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	_, parsedPrefix, parsedSize := appendAndParseFrameHeaderInter(t, seq, prefix, size)
	if parsedPrefix.FrameType != parser.FrameTypeInter || !parsedPrefix.ShowableFrame {
		t.Fatalf("prefix=%+v", parsedPrefix)
	}
	if parsedSize.RefreshFrameFlags != 0x01 || parsedSize.FrameRefsShortSignaling ||
		parsedSize.UpscaledWidth != seq.MaxFrameWidth || parsedSize.Height != seq.MaxFrameHeight ||
		parsedSize.SuperResEnabled || parsedSize.SuperResDenominator != 8 {
		t.Fatalf("parsed inter size=%+v", parsedSize)
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if parsedSize.RefFrameIdx[i] != 0 {
			t.Fatalf("RefFrameIdx[%d]=%d want 0", i, parsedSize.RefFrameIdx[i])
		}
	}
}

func TestAppendFrameHeaderInterPayloadOverrideSuperresRender(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := interFrameHeaderPrefix(true)
	size := InterFrameSize{
		UpscaledWidth:       12,
		Height:              6,
		RenderWidth:         10,
		RenderHeight:        5,
		SuperResDenominator: 12,
		HaveRenderSize:      true,
		RefreshFrameFlags:   0x02,
		RefFrameIdx:         [7]uint8{0, 1, 1, 0, 1, 0, 1},
	}
	_, _, parsedSize := appendAndParseFrameHeaderInter(t, seq, prefix, size)
	if parsedSize.RefreshFrameFlags != 0x02 ||
		parsedSize.UpscaledWidth != 12 || parsedSize.Height != 6 ||
		!parsedSize.SuperResEnabled || parsedSize.SuperResDenominator != 12 ||
		!parsedSize.HaveRenderSize || parsedSize.RenderWidth != 10 || parsedSize.RenderHeight != 5 {
		t.Fatalf("parsed inter size=%+v", parsedSize)
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if parsedSize.RefFrameIdx[i] != size.RefFrameIdx[i] {
			t.Fatalf("RefFrameIdx[%d]=%d want %d", i, parsedSize.RefFrameIdx[i], size.RefFrameIdx[i])
		}
	}
}

func TestAppendFrameHeaderInterPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := interFrameHeaderPrefix(false)
	size := InterFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	var buf [16]byte
	if _, err := appendFrameHeaderInterPayload(buf[:0], seq, shownKeyFrameHeaderPrefix(false), size); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("intra prefix err=%v want %v", err, ErrInvalidFrame)
	}
	bad := size
	bad.RefFrameIdx[0] = 8
	if _, err := appendFrameHeaderInterPayload(buf[:0], seq, prefix, bad); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad ref idx err=%v want %v", err, ErrInvalidFrame)
	}
}

func appendAndParseFrameHeaderIntra(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, parser.FrameHeaderPrefix, parser.FrameSize) {
	t.Helper()
	parsedSeq := parseEncoderSequenceHeader(t, seq)

	var buf [64]byte
	payload, err := appendFrameHeaderIntraPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("appendFrameHeaderIntraPayload: %v", err)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := parser.ParseFrameSize(payload, parsedSeq, parsedPrefix, nil, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	return payload, parsedPrefix, parsedSize
}

func appendAndParseFrameHeaderInter(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) ([]byte, parser.FrameHeaderPrefix, parser.FrameSize) {
	t.Helper()
	parsedSeq := parseEncoderSequenceHeader(t, seq)

	var buf [64]byte
	payload, err := appendFrameHeaderInterPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("appendFrameHeaderInterPayload: %v", err)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	refs := parserReferenceState(seq, size.RefFrameIdx[:])
	parsedSize, err := parser.ParseFrameSize(payload, parsedSeq, parsedPrefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	return payload, parsedPrefix, parsedSize
}

func parseEncoderSequenceHeader(t *testing.T, seq SequenceHeader) parser.SequenceHeader {
	t.Helper()
	var seqBuf [128]byte
	seqPayload, err := appendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	return parsedSeq
}

func parserReferenceState(seq SequenceHeader, indices []uint8) parser.ReferenceState {
	var refs parser.ReferenceState
	for _, idx := range indices {
		refs.Frames[idx].Valid = true
		refs.Frames[idx].OrderHint = 1
		refs.Frames[idx].Size.UpscaledWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.CodedWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.Height = seq.MaxFrameHeight
		refs.Frames[idx].Size.RenderWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.RenderHeight = seq.MaxFrameHeight
		refs.Frames[idx].Size.SuperResDenominator = 8
	}
	return refs
}

func shownKeyFrameHeaderPrefix(frameSizeOverride bool) FrameHeaderPrefix {
	return FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		FrameSizeOverride:  frameSizeOverride,
		OrderHint:          3,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
}

func interFrameHeaderPrefix(frameSizeOverride bool) FrameHeaderPrefix {
	return FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeInter,
		ShowFrame:          true,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		FrameSizeOverride:  frameSizeOverride,
		OrderHint:          4,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
}
