package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendDeltaParamsPayloadBaseQZero(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{}
	payload, parsed := appendAndParseDeltaParams(t, size, quant, DeltaParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.DeltaQPresent || parsed.BitsRead != 0 {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadQOnly(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 2}
	_, parsed := appendAndParseDeltaParams(t, size, quant, delta)
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 2 || parsed.DeltaLFPresent {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadLoopFilterMulti(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1, DeltaLFPresent: true, DeltaLFResLog2: 3, DeltaLFMulti: true}
	_, parsed := appendAndParseDeltaParams(t, size, quant, delta)
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 1 ||
		!parsed.DeltaLFPresent || parsed.DeltaLFResLog2 != 3 || !parsed.DeltaLFMulti {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadIntrabcSkipsLoopFilter(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff, AllowIntrabc: true}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 2}
	payload, parsed := appendAndParseDeltaParams(t, size, quant, delta)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 2 || parsed.DeltaLFPresent {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsInterPayload(t *testing.T) {
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	quant := QuantizationParams{BaseQIdx: 31}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1, DeltaLFPresent: true, DeltaLFResLog2: 2}
	payloadSize, err := DeltaParamsInterPayloadSize(size, quant, delta)
	if err != nil {
		t.Fatalf("DeltaParamsInterPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := AppendDeltaParamsInterPayload(buf[:0], size, quant, delta)
	if err != nil {
		t.Fatalf("AppendDeltaParamsInterPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := parser.ParseDeltaParams(payload, parser.FrameSize{}, parser.QuantizationParams{BaseQIdx: 31}, parser.SegmentationParams{})
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	if !parsed.DeltaLFPresent || parsed.DeltaLFResLog2 != 2 || parsed.DeltaLFMulti {
		t.Fatalf("parsed inter delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadRejectsInvalid(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quantZero := QuantizationParams{}
	quant := QuantizationParams{BaseQIdx: 50}
	cases := [...]struct {
		quant QuantizationParams
		delta DeltaParams
	}{
		{quant: quantZero, delta: DeltaParams{DeltaQPresent: true}},
		{quant: quant, delta: DeltaParams{DeltaQResLog2: 1}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaQResLog2: 4}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaLFPresent: true, DeltaLFResLog2: 4}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaLFResLog2: 1}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaLFMulti: true}},
	}
	var buf [2]byte
	for _, tc := range cases {
		if _, err := AppendDeltaParamsPayload(buf[:0], size, tc.quant, tc.delta); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendDeltaParamsPayload(%+v,%+v) err=%v want ErrInvalidFrame", tc.quant, tc.delta, err)
		}
	}
	intrabc := size
	intrabc.AllowIntrabc = true
	if _, err := AppendDeltaParamsPayload(buf[:0], intrabc, quant, DeltaParams{DeltaQPresent: true, DeltaLFPresent: true}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("intrabc delta lf err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendDeltaParamsPayloadShortBuffer(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 2}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendDeltaParamsPayload(dst, size, quant, delta)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendDeltaParamsPayloadAllocs(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1}
	var buf [2]byte
	if _, err := AppendDeltaParamsPayload(buf[:0], size, quant, delta); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = DeltaParamsPayloadSize(size, quant, delta)
		_, _ = AppendDeltaParamsPayload(buf[:0], size, quant, delta)
	})
	if allocs != 0 {
		t.Fatalf("AppendDeltaParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseDeltaParams(t *testing.T, size IntraFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, parser.DeltaParams) {
	t.Helper()
	payloadSize, err := DeltaParamsPayloadSize(size, quant, delta)
	if err != nil {
		t.Fatalf("DeltaParamsPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := AppendDeltaParamsPayload(buf[:0], size, quant, delta)
	if err != nil {
		t.Fatalf("AppendDeltaParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsedSize := parser.FrameSize{AllowIntrabc: size.AllowIntrabc}
	parsedQuant := parser.QuantizationParams{BaseQIdx: quant.BaseQIdx}
	parsed, err := parser.ParseDeltaParams(payload, parsedSize, parsedQuant, parser.SegmentationParams{})
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	return payload, parsed
}
