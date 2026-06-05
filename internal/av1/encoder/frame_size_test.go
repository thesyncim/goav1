package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
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
	payload, parsedPrefix, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	payloadSize, err := FrameHeaderIntraPayloadSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("FrameHeaderIntraPayloadSize: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d bytes=% x want %d", len(payload), payload, payloadSize)
	}
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

func TestAppendLowOverheadFrameHeaderIntraOBU(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	obuSize, err := LowOverheadFrameHeaderIntraOBUSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("LowOverheadFrameHeaderIntraOBUSize: %v", err)
	}
	var buf [64]byte
	out, err := AppendLowOverheadFrameHeaderIntraOBU(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendLowOverheadFrameHeaderIntraOBU: %v", err)
	}
	if len(out) != obuSize {
		t.Fatalf("obu len=%d want %d", len(out), obuSize)
	}
	unit, consumed, err := obu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverhead: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != obu.TypeFrameHeader || !unit.Header.HasSizeField {
		t.Fatalf("parsed obu header=%+v consumed=%d", unit.Header, consumed)
	}
	parsedSeq := parseEncoderSequenceHeader(t, seq)
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := parser.ParseIntraFrameSize(unit.Payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != seq.MaxFrameWidth || parsedSize.Height != seq.MaxFrameHeight {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
}

func TestAppendLowOverheadIntraHeaderTemporalUnit(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	unitSize, err := LowOverheadIntraHeaderTemporalUnitSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("LowOverheadIntraHeaderTemporalUnitSize: %v", err)
	}
	var buf [192]byte
	out, err := AppendLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendLowOverheadIntraHeaderTemporalUnit: %v", err)
	}
	if len(out) != unitSize {
		t.Fatalf("temporal unit len=%d want %d", len(out), unitSize)
	}

	it := obu.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("temporal delimiter ok=%v err=%v", ok, err)
	}
	if td.Header.Type != obu.TypeTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter=%+v payload=% x", td.Header, td.Payload)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("sequence header ok=%v err=%v", ok, err)
	}
	if seqUnit.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("sequence unit type=%v", seqUnit.Header.Type)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	frameUnit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("frame header ok=%v err=%v", ok, err)
	}
	if frameUnit.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("frame unit type=%v", frameUnit.Header.Type)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := parser.ParseIntraFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != seq.MaxFrameWidth || parsedSize.Height != seq.MaxFrameHeight {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra unit ok=%v err=%v unit=%+v", ok, err, extra.Header)
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

func TestAppendFrameHeaderIntraPayloadShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	var buf [2]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendFrameHeaderIntraPayload(dst, seq, prefix, size)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
	}
}

func TestAppendLowOverheadFrameHeaderIntraOBUShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	var buf [2]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLowOverheadFrameHeaderIntraOBU(dst, seq, prefix, size)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
	}
}

func TestAppendLowOverheadIntraHeaderTemporalUnitShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	var buf [8]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLowOverheadIntraHeaderTemporalUnit(dst, seq, prefix, size)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
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
		if _, err := AppendFrameHeaderIntraPayload(buf[:0], seq, prefix, size); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFrame)
		}
	}
}

func TestAppendFrameHeaderIntraPayloadAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	var buf [32]byte
	if _, err := AppendFrameHeaderIntraPayload(buf[:0], seq, prefix, size); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendFrameHeaderIntraPayload(buf[:0], seq, prefix, size)
	})
	if allocs != 0 {
		t.Fatalf("AppendFrameHeaderIntraPayload allocated: %f", allocs)
	}
}

func TestAppendLowOverheadFrameHeaderIntraOBUAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	var buf [64]byte
	if _, err := AppendLowOverheadFrameHeaderIntraOBU(buf[:0], seq, prefix, size); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadFrameHeaderIntraOBU(buf[:0], seq, prefix, size)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadFrameHeaderIntraOBU allocated: %f", allocs)
	}
}

func TestAppendLowOverheadIntraHeaderTemporalUnitAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	var buf [192]byte
	if _, err := AppendLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadIntraHeaderTemporalUnit allocated: %f", allocs)
	}
}

func appendAndParseFrameHeaderIntra(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, parser.FrameHeaderPrefix, parser.FrameSize) {
	t.Helper()
	parsedSeq := parseEncoderSequenceHeader(t, seq)

	var buf [64]byte
	payload, err := AppendFrameHeaderIntraPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendFrameHeaderIntraPayload: %v", err)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := parser.ParseIntraFrameSize(payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	return payload, parsedPrefix, parsedSize
}

func parseEncoderSequenceHeader(t *testing.T, seq SequenceHeader) parser.SequenceHeader {
	t.Helper()
	var seqBuf [128]byte
	seqPayload, err := AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	return parsedSeq
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
