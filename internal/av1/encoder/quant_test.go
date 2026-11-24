package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendQuantizationParamsPayloadZeroDeltas(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	payload, parsed := appendAndParseQuantizationParams(t, seq, QuantizationParams{})
	if len(payload) != 2 {
		t.Fatalf("payload len=%d want 2", len(payload))
	}
	if parsed.BaseQIdx != 0 || parsed.YDCDelta != 0 || parsed.UDCDelta != 0 || parsed.UACDelta != 0 {
		t.Fatalf("parsed quant=%+v", parsed)
	}
	if parsed.UsingQMatrix {
		t.Fatalf("qmatrix set in parsed quant=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadSeparateUVAndQMatrix(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.SeparateUVDeltaQ = true
	quant := QuantizationParams{
		BaseQIdx:      96,
		YDCDelta:      -2,
		UDCDelta:      5,
		UACDelta:      -3,
		VDCDelta:      7,
		VACDelta:      -9,
		DiffUVDeltas:  true,
		UsingQMatrix:  true,
		QMatrixLevelY: 2,
		QMatrixLevelU: 3,
		QMatrixLevelV: 4,
	}
	_, parsed := appendAndParseQuantizationParams(t, seq, quant)
	if parsed.BaseQIdx != quant.BaseQIdx || parsed.YDCDelta != quant.YDCDelta ||
		parsed.UDCDelta != quant.UDCDelta || parsed.UACDelta != quant.UACDelta ||
		parsed.VDCDelta != quant.VDCDelta || parsed.VACDelta != quant.VACDelta {
		t.Fatalf("parsed deltas=%+v want %+v", parsed, quant)
	}
	if !parsed.DiffUVDeltas || !parsed.UsingQMatrix ||
		parsed.QMatrixLevelY != 2 || parsed.QMatrixLevelU != 3 || parsed.QMatrixLevelV != 4 {
		t.Fatalf("parsed qmatrix=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadSharedUV(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.SeparateUVDeltaQ = false
	quant := QuantizationParams{
		BaseQIdx:      10,
		UDCDelta:      4,
		VDCDelta:      4,
		UsingQMatrix:  true,
		QMatrixLevelY: 1,
		QMatrixLevelU: 6,
		QMatrixLevelV: 6,
	}
	_, parsed := appendAndParseQuantizationParams(t, seq, quant)
	if parsed.DiffUVDeltas || parsed.UDCDelta != 4 || parsed.VDCDelta != 4 || parsed.VACDelta != 0 {
		t.Fatalf("parsed shared uv=%+v", parsed)
	}
	if parsed.QMatrixLevelY != 1 || parsed.QMatrixLevelU != 6 || parsed.QMatrixLevelV != 6 {
		t.Fatalf("parsed qmatrix=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadMonoChrome(t *testing.T) {
	seq := realtimeEncoderMonochromeSequenceHeader()
	quant := QuantizationParams{
		BaseQIdx: 3,
		YDCDelta: -1,
	}
	_, parsed := appendAndParseQuantizationParams(t, seq, quant)
	if parsed.BaseQIdx != 3 || parsed.YDCDelta != -1 {
		t.Fatalf("parsed mono quant=%+v", parsed)
	}
	if parsed.UDCDelta != 0 || parsed.UACDelta != 0 || parsed.VDCDelta != 0 || parsed.VACDelta != 0 {
		t.Fatalf("parsed mono uv=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	cases := [...]QuantizationParams{
		{YDCDelta: -65},
		{YDCDelta: 64},
		{DiffUVDeltas: true},
		{UDCDelta: 1},
		{UsingQMatrix: true, QMatrixLevelY: 16},
		{UsingQMatrix: false, QMatrixLevelY: 1},
		{UsingQMatrix: true, QMatrixLevelU: 3, QMatrixLevelV: 4},
	}
	var buf [8]byte
	for _, quant := range cases {
		if _, err := appendQuantizationParamsPayload(buf[:0], seq, quant); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendQuantizationParamsPayload(%+v) err=%v want ErrInvalidFrame", quant, err)
		}
	}

	mono := seq
	mono.ColorConfig.MonoChrome = true
	if _, err := appendQuantizationParamsPayload(buf[:0], mono, QuantizationParams{UDCDelta: 1}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono uv delta err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseQuantizationParams(t *testing.T, seq SequenceHeader, quant QuantizationParams) ([]byte, parser.QuantizationParams) {
	t.Helper()
	var buf [16]byte
	out, err := appendQuantizationParamsPayload(buf[:0], seq, quant)
	if err != nil {
		t.Fatalf("appendQuantizationParamsPayload: %v", err)
	}
	parsedSeq := parseEncoderSequenceHeader(t, seq)
	parsed, err := parser.ParseQuantizationParams(out, parsedSeq, parser.TileInfo{})
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	return out, parsed
}
