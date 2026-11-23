package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendRestorationParamsPayloadSkipsLosslessWithoutSuperres(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	payload, parsed := appendAndParseRestorationParams(t, seq, size, true, RestorationParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.BitsRead != 0 || parsed.UnitSizeY != 0 {
		t.Fatalf("parsed restoration=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadAllNone(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		UnitSizeYLog2:  8,
		UnitSizeUVLog2: 8,
		UnitSizeY:      RestorationUnitMax,
		UnitSizeUV:     RestorationUnitMax,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.Type[0] != parser.RestorationNone || parsed.Type[1] != parser.RestorationNone || parsed.Type[2] != parser.RestorationNone {
		t.Fatalf("parsed types=%+v", parsed.Type)
	}
	if parsed.UnitSizeY != parser.RestorationUnitMax || parsed.UnitSizeUV != parser.RestorationUnitMax ||
		parsed.UnitSizeYLog2 != 8 || parsed.UnitSizeUVLog2 != 8 {
		t.Fatalf("parsed unit sizes=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadTypesAndUnitSizes(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{Resolution: Resolution{Width: 64, Height: 64}})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	seq.EnableRestoration = true
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationWiener, RestorationSGRProj, RestorationNone},
		UnitSizeYLog2:  7,
		UnitSizeUVLog2: 6,
		UnitSizeY:      128,
		UnitSizeUV:     64,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.Type[0] != parser.RestorationWiener || parsed.Type[1] != parser.RestorationSGRProj || parsed.Type[2] != parser.RestorationNone {
		t.Fatalf("parsed types=%+v", parsed.Type)
	}
	if parsed.UnitSizeY != 128 || parsed.UnitSizeUV != 64 ||
		parsed.UnitSizeYLog2 != 7 || parsed.UnitSizeUVLog2 != 6 {
		t.Fatalf("parsed unit sizes=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayload128SuperblockUnitSize(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.Use128x128Superblock = true
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationSGRProj, RestorationNone, RestorationNone},
		UnitSizeYLog2:  8,
		UnitSizeUVLog2: 8,
		UnitSizeY:      256,
		UnitSizeUV:     256,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.UnitSizeY != 256 || parsed.UnitSizeUV != 256 || parsed.UnitSizeYLog2 != 8 {
		t.Fatalf("parsed unit sizes=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadMonochromeReadsYOnly(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.MonoChrome = true
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationSwitchable, RestorationNone, RestorationNone},
		UnitSizeYLog2:  6,
		UnitSizeUVLog2: 6,
		UnitSizeY:      64,
		UnitSizeUV:     64,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.Type[0] != parser.RestorationSwitchable || parsed.Type[1] != parser.RestorationNone || parsed.UnitSizeY != 64 {
		t.Fatalf("parsed restoration=%+v", parsed)
	}
}

func TestAppendRestorationParamsInterPayload(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationNone, RestorationWiener, RestorationNone},
		UnitSizeYLog2:  6,
		UnitSizeUVLog2: 6,
		UnitSizeY:      64,
		UnitSizeUV:     64,
	}
	var buf [4]byte
	payload, err := appendRestorationParamsInterPayload(buf[:0], seq, size, false, restoration)
	if err != nil {
		t.Fatalf("appendRestorationParamsInterPayload: %v", err)
	}
	parsed, err := parser.ParseRestorationParams(payload, parseEncoderSequenceHeader(t, seq), parser.FrameSize{}, parser.SegmentationParams{}, parser.CDEFParams{})
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	if parsed.Type[1] != parser.RestorationWiener || parsed.UnitSizeY != 64 || parsed.UnitSizeUV != 64 {
		t.Fatalf("parsed inter restoration=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	active := RestorationParams{
		Type:           [3]RestorationType{RestorationWiener, RestorationNone, RestorationNone},
		UnitSizeYLog2:  6,
		UnitSizeUVLog2: 6,
		UnitSizeY:      64,
		UnitSizeUV:     64,
	}
	cases := [...]RestorationParams{
		{Type: [3]RestorationType{4, 0, 0}},
		{Type: [3]RestorationType{RestorationWiener, 0, 0}, UnitSizeYLog2: 5},
		{Type: [3]RestorationType{RestorationWiener, 0, 0}, UnitSizeYLog2: 9},
		{Type: [3]RestorationType{RestorationWiener, 0, 0}, UnitSizeYLog2: 6, UnitSizeY: 128},
		{UnitSizeYLog2: 7},
	}
	var buf [8]byte
	for _, restoration := range cases {
		if _, err := appendRestorationParamsPayload(buf[:0], seq, size, false, restoration); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendRestorationParamsPayload(%+v) err=%v want ErrInvalidFrame", restoration, err)
		}
	}
	mono := seq
	mono.ColorConfig.MonoChrome = true
	if _, err := appendRestorationParamsPayload(buf[:0], mono, size, false, RestorationParams{Type: [3]RestorationType{0, RestorationWiener, 0}}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono chroma restoration err=%v want ErrInvalidFrame", err)
	}
	if _, err := appendRestorationParamsPayload(buf[:0], seq, size, true, active); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("lossless disabled restoration err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseRestorationParams(t *testing.T, seq SequenceHeader, size IntraFrameSize, allLossless bool, restoration RestorationParams) ([]byte, parser.RestorationParams) {
	t.Helper()
	var buf [8]byte
	payload, err := appendRestorationParamsPayload(buf[:0], seq, size, allLossless, restoration)
	if err != nil {
		t.Fatalf("appendRestorationParamsPayload: %v", err)
	}
	parsed, err := parser.ParseRestorationParams(
		payload,
		parseEncoderSequenceHeader(t, seq),
		parser.FrameSize{AllowIntrabc: size.AllowIntrabc, SuperResEnabled: size.SuperResDenominator != 8},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.CDEFParams{},
	)
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	return payload, parsed
}
