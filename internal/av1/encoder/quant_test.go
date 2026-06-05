package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
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
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.MonoChrome = true
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
		if _, err := AppendQuantizationParamsPayload(buf[:0], seq, quant); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendQuantizationParamsPayload(%+v) err=%v want ErrInvalidFrame", quant, err)
		}
	}

	mono := seq
	mono.ColorConfig.MonoChrome = true
	if _, err := AppendQuantizationParamsPayload(buf[:0], mono, QuantizationParams{UDCDelta: 1}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono uv delta err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendQuantizationParamsPayloadShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendQuantizationParamsPayload(dst, seq, QuantizationParams{})
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendQuantizationParamsPayloadAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	quant := QuantizationParams{BaseQIdx: 37}
	var buf [4]byte
	if _, err := AppendQuantizationParamsPayload(buf[:0], seq, quant); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = QuantizationParamsPayloadSize(seq, quant)
		_, _ = AppendQuantizationParamsPayload(buf[:0], seq, quant)
	})
	if allocs != 0 {
		t.Fatalf("AppendQuantizationParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseQuantizationParams(t *testing.T, seq SequenceHeader, quant QuantizationParams) ([]byte, parser.QuantizationParams) {
	t.Helper()
	size, err := QuantizationParamsPayloadSize(seq, quant)
	if err != nil {
		t.Fatalf("QuantizationParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	out, err := AppendQuantizationParamsPayload(buf[:0], seq, quant)
	if err != nil {
		t.Fatalf("AppendQuantizationParamsPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("payload len=%d want %d", len(out), size)
	}
	parsedSeq := parseEncoderSequenceHeader(t, seq)
	parsed, err := parser.ParseQuantizationParams(out, parsedSeq, parser.TileInfo{})
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	return out, parsed
}
