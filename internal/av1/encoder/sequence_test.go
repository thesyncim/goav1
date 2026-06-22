package encoder

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

type sequenceTestBitWriter struct {
	buf [128]byte
	bit uint16
}

func (w *sequenceTestBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 != 0 {
			w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *sequenceTestBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *sequenceTestBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func TestBitWriterMatchesMSBFirstShape(t *testing.T) {
	var buf [4]byte
	w := newBitWriter(buf[:])
	for _, op := range []struct {
		value uint64
		n     uint8
	}{
		{0b101, 3},
		{0, 1},
		{0b1110_0011, 8},
	} {
		if err := w.writeBits(op.value, op.n); err != nil {
			t.Fatalf("writeBits: %v", err)
		}
	}
	if err := w.writeTrailingBits(); err != nil {
		t.Fatalf("writeTrailingBits: %v", err)
	}
	want := []byte{0xae, 0x38}
	if got := buf[:w.bytesWritten()]; !bytes.Equal(got, want) {
		t.Fatalf("bit writer = %08b want %08b", got, want)
	}
	if w.bitsWritten() != 16 || !w.byteAligned() {
		t.Fatalf("writer position bits=%d aligned=%v", w.bitsWritten(), w.byteAligned())
	}
}

func TestAppendSequenceHeaderPayloadRealtime(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	want := realtimeEncoderSequenceHeaderPayload()

	size, err := SequenceHeaderPayloadSize(seq)
	if err != nil {
		t.Fatalf("SequenceHeaderPayloadSize: %v", err)
	}
	if size != len(want) {
		t.Fatalf("payload size=%d want %d", size, len(want))
	}
	var buf [64]byte
	out, err := AppendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatalf("payload = % x; want % x", out, want)
	}

	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsed.SeqProfile != 0 || parsed.OperatingPointsCount != 1 || parsed.OperatingPoints[0].SeqLevelIdx != 5 {
		t.Fatalf("parsed operating point/profile: %+v", parsed)
	}
	if parsed.MaxFrameWidth != 16 || parsed.MaxFrameHeight != 9 {
		t.Fatalf("parsed dimensions=%dx%d", parsed.MaxFrameWidth, parsed.MaxFrameHeight)
	}
	if !parsed.EnableOrderHint || parsed.OrderHintBits != 7 || !parsed.EnableSuperRes || !parsed.EnableCDEF || !parsed.EnableRestoration {
		t.Fatalf("parsed tools: %+v", parsed)
	}
	if !parsed.ColorConfig.ColorRange || parsed.ColorConfig.SubsamplingX || parsed.ColorConfig.SubsamplingY {
		t.Fatalf("parsed color config: %+v", parsed.ColorConfig)
	}
	if !parsed.FilmGrainParamsPresent {
		t.Fatal("film_grain_params_present=false")
	}
}

func TestAppendLowOverheadSequenceHeaderOBU(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size, err := LowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		t.Fatalf("LowOverheadSequenceHeaderOBUSize: %v", err)
	}
	var buf [80]byte
	out, err := AppendLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendLowOverheadSequenceHeaderOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("obu size=%d want %d", len(out), size)
	}

	unit, consumed, err := obu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverhead: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != obu.TypeSequenceHeader || !unit.Header.HasSizeField {
		t.Fatalf("parsed obu header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := parser.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}

func TestAppendSequenceHeaderRejectsInvalid(t *testing.T) {
	base := realtimeEncoderSequenceHeader()
	tests := []struct {
		name string
		mut  func(*SequenceHeader)
		want error
	}{
		{name: "dimension", mut: func(seq *SequenceHeader) { seq.MaxFrameWidth = 0 }, want: ErrInvalidConfig},
		{name: "decoder model without timing", mut: func(seq *SequenceHeader) { seq.DecoderModelInfoPresent = true }, want: ErrInvalidConfig},
		{name: "decoder model bad length", mut: func(seq *SequenceHeader) {
			seq.TimingInfoPresent = true
			seq.DecoderModelInfoPresent = true
			seq.DecoderModelInfo = SequenceDecoderModelInfo{
				BufferDelayLength:           0,
				BufferRemovalTimeLength:     4,
				FramePresentationTimeLength: 4,
			}
		}, want: ErrInvalidConfig},
		{name: "operating point display delay without header flag", mut: func(seq *SequenceHeader) {
			seq.OperatingPoints[0].InitialDisplayDelayPresent = true
		}, want: ErrInvalidConfig},
		{name: "level", mut: func(seq *SequenceHeader) { seq.OperatingPoints[0].SeqLevelIdx = 10 }, want: ErrInvalidConfig},
		{name: "profile-bitdepth", mut: func(seq *SequenceHeader) {
			seq.Profile = Profile0
			seq.ColorConfig.BitDepth = 12
		}, want: ErrInvalidConfig},
		{name: "order-hint", mut: func(seq *SequenceHeader) {
			seq.EnableOrderHint = false
			seq.EnableJNTComp = true
		}, want: ErrInvalidConfig},
		{name: "color", mut: func(seq *SequenceHeader) { seq.ColorConfig.SubsamplingX = true }, want: ErrInvalidConfig},
	}
	var buf [80]byte
	for _, tt := range tests {
		seq := base
		tt.mut(&seq)
		if _, err := AppendLowOverheadSequenceHeaderOBU(buf[:0], seq); !errors.Is(err, tt.want) {
			t.Fatalf("%s err=%v want %v", tt.name, err, tt.want)
		}
	}
}

func TestAppendSequenceHeaderTimingAndDisplayDelay(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.TimingInfoPresent = true
	seq.TimingInfo = SequenceTimingInfo{
		NumUnitsInDisplayTick:    1001,
		TimeScale:                30000,
		EqualPictureInterval:     false,
		NumTicksPerPictureMinus1: 0,
	}
	seq.DecoderModelInfoPresent = true
	seq.DecoderModelInfo = SequenceDecoderModelInfo{
		BufferDelayLength:           8,
		NumUnitsInDecodingTick:      1001,
		BufferRemovalTimeLength:     5,
		FramePresentationTimeLength: 6,
	}
	seq.InitialDisplayDelayPresent = true
	seq.OperatingPoints[0].DecoderModelPresent = true
	seq.OperatingPoints[0].DecoderBufferDelay = 17
	seq.OperatingPoints[0].EncoderBufferDelay = 19
	seq.OperatingPoints[0].LowDelayMode = true
	seq.OperatingPoints[0].InitialDisplayDelayPresent = true
	seq.OperatingPoints[0].InitialDisplayDelayMinus1 = 3

	var buf [128]byte
	out, err := AppendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if !parsed.TimingInfoPresent || !parsed.DecoderModelInfoPresent || !parsed.InitialDisplayDelayPresent {
		t.Fatalf("parsed timing flags: %+v", parsed)
	}
	if parsed.TimingInfo.NumUnitsInDisplayTick != 1001 || parsed.TimingInfo.TimeScale != 30000 ||
		parsed.TimingInfo.EqualPictureInterval {
		t.Fatalf("parsed timing info: %+v", parsed.TimingInfo)
	}
	if parsed.DecoderModelInfo.BufferDelayLength != 8 || parsed.DecoderModelInfo.NumUnitsInDecodingTick != 1001 ||
		parsed.DecoderModelInfo.BufferRemovalTimeLength != 5 || parsed.DecoderModelInfo.FramePresentationTimeLength != 6 {
		t.Fatalf("parsed decoder model info: %+v", parsed.DecoderModelInfo)
	}
	op := parsed.OperatingPoints[0]
	if !op.DecoderModelPresent || op.DecoderBufferDelay != 17 || op.EncoderBufferDelay != 19 || !op.LowDelayMode ||
		!op.InitialDisplayDelayPresent || op.InitialDisplayDelayMinus1 != 3 {
		t.Fatalf("parsed operating point: %+v", op)
	}
}

func TestAppendSequenceHeaderShortBuffer(t *testing.T) {
	var buf [4]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendSequenceHeaderPayload(dst, realtimeEncoderSequenceHeader())
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
	}
}

func TestAppendSequenceHeaderAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	var buf [80]byte
	if _, err := AppendLowOverheadSequenceHeaderOBU(buf[:0], seq); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadSequenceHeaderOBU(buf[:0], seq)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadSequenceHeaderOBU allocated: %f", allocs)
	}
}

func TestSequenceHeaderForConfigSingleLayer(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{
		Resolution: Resolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	if seq.Profile != Profile0 || seq.MaxFrameWidth != 640 || seq.MaxFrameHeight != 360 {
		t.Fatalf("sequence basics: %+v", seq)
	}
	if seq.OperatingPointsCount != 1 || seq.OperatingPoints[0].IDC != 0 || seq.OperatingPoints[0].SeqLevelIdx != SequenceLevelMax {
		t.Fatalf("single-layer operating point: count=%d op=%+v", seq.OperatingPointsCount, seq.OperatingPoints[0])
	}
	if !seq.EnableOrderHint || seq.OrderHintBits != webRTCDefaultOrderHintBits || !seq.EnableCDEF {
		t.Fatalf("realtime tools: %+v", seq)
	}

	var buf [128]byte
	out, err := AppendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsed.MaxFrameWidth != 640 || parsed.MaxFrameHeight != 360 || parsed.OperatingPoints[0].SeqLevelIdx != parser.SeqLevelMax {
		t.Fatalf("parsed sequence: %+v", parsed)
	}
	if parsed.ColorConfig.BitDepth != 8 || !parsed.ColorConfig.SubsamplingX || !parsed.ColorConfig.SubsamplingY {
		t.Fatalf("parsed color config: %+v", parsed.ColorConfig)
	}
}

func TestSequenceHeaderForConfigSVCOperatingPoints(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	if seq.OperatingPointsCount != 4 {
		t.Fatalf("operating point count=%d want 4", seq.OperatingPointsCount)
	}
	wantIDC := [...]uint16{0x303, 0x301, 0x103, 0x101}
	for i := range wantIDC {
		if seq.OperatingPoints[i].IDC != wantIDC[i] || seq.OperatingPoints[i].SeqLevelIdx != SequenceLevelMax {
			t.Fatalf("op[%d]=%+v want idc=%#x level=%d", i, seq.OperatingPoints[i], wantIDC[i], SequenceLevelMax)
		}
	}

	var buf [160]byte
	out, err := AppendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	for i := range wantIDC {
		if parsed.OperatingPoints[i].IDC != wantIDC[i] {
			t.Fatalf("parsed op[%d]=%+v want idc=%#x", i, parsed.OperatingPoints[i], wantIDC[i])
		}
	}
	if index, ok := parser.SelectOperatingPoint(parsed, 1, 1); !ok || index != 0 {
		t.Fatalf("SelectOperatingPoint top layer = %d,%v; want 0,true", index, ok)
	}
	if index, ok := parser.SelectOperatingPoint(parsed, 0, 0); !ok || index != 0 {
		t.Fatalf("SelectOperatingPoint base layer first match = %d,%v; want 0,true", index, ok)
	}
}

func TestSequenceHeaderForConfigRequestedLayers(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{
		Resolution:         Resolution{Width: 640, Height: 360},
		SpatialLayerCount:  2,
		TemporalLayerCount: 3,
	})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	if seq.OperatingPointsCount != 6 || seq.OperatingPoints[0].IDC != 0x307 || seq.OperatingPoints[5].IDC != 0x101 {
		t.Fatalf("requested-layer operating points: count=%d first=%#x last=%#x",
			seq.OperatingPointsCount, seq.OperatingPoints[0].IDC, seq.OperatingPoints[5].IDC)
	}
}

func TestSequenceHeaderForConfigExplicitColorConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want SequenceColorConfig
	}{
		{
			name: "profile2-420-12bit",
			cfg: Config{
				Resolution:     Resolution{Width: 640, Height: 360},
				Profile:        Profile2,
				BitDepth:       12,
				ColorConfigSet: true,
				ColorConfig: SequenceColorConfig{
					BitDepth:             12,
					SubsamplingX:         true,
					SubsamplingY:         true,
					ChromaSamplePosition: 1,
				},
			},
			want: SequenceColorConfig{
				BitDepth:             12,
				SubsamplingX:         true,
				SubsamplingY:         true,
				ChromaSamplePosition: 1,
			},
		},
		{
			name: "profile2-444-12bit-color-bitdepth-only",
			cfg: Config{
				Resolution:     Resolution{Width: 640, Height: 360},
				Profile:        Profile2,
				ColorConfigSet: true,
				ColorConfig:    SequenceColorConfig{BitDepth: 12},
			},
			want: SequenceColorConfig{BitDepth: 12},
		},
		{
			name: "profile0-mono-10bit",
			cfg: Config{
				Resolution:     Resolution{Width: 64, Height: 64},
				Profile:        Profile0,
				BitDepth:       10,
				ColorConfigSet: true,
				ColorConfig: SequenceColorConfig{
					BitDepth:   10,
					MonoChrome: true,
				},
			},
			want: SequenceColorConfig{
				BitDepth:     10,
				MonoChrome:   true,
				SubsamplingX: true,
				SubsamplingY: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			seq, err := SequenceHeaderForConfig(tc.cfg)
			if err != nil {
				t.Fatalf("SequenceHeaderForConfig: %v", err)
			}
			if seq.ColorConfig != tc.want {
				t.Fatalf("sequence color=%+v want %+v", seq.ColorConfig, tc.want)
			}
			var buf [160]byte
			out, err := AppendSequenceHeaderPayload(buf[:0], seq)
			if err != nil {
				t.Fatalf("AppendSequenceHeaderPayload: %v", err)
			}
			parsed, err := parser.ParseSequenceHeader(out)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			got := parsed.ColorConfig
			if got.BitDepth != tc.want.BitDepth ||
				got.MonoChrome != tc.want.MonoChrome ||
				got.SubsamplingX != tc.want.SubsamplingX ||
				got.SubsamplingY != tc.want.SubsamplingY ||
				got.ChromaSamplePosition != tc.want.ChromaSamplePosition {
				t.Fatalf("parsed color=%+v want %+v", got, tc.want)
			}
		})
	}
}

func TestSequenceHeaderForConfigRejectsInvalidSequenceCombination(t *testing.T) {
	_, err := SequenceHeaderForConfig(Config{
		Resolution: Resolution{Width: 640, Height: 360},
		Profile:    Profile0,
		BitDepth:   12,
	})
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("SequenceHeaderForConfig err=%v want %v", err, ErrInvalidConfig)
	}
}

func TestSequenceHeaderForConfigAllocs(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	if _, err := SequenceHeaderForConfig(cfg); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = SequenceHeaderForConfig(cfg)
	})
	if allocs != 0 {
		t.Fatalf("SequenceHeaderForConfig allocated: %f", allocs)
	}
}

func realtimeEncoderSequenceHeader() SequenceHeader {
	var seq SequenceHeader
	seq.Profile = Profile0
	seq.OperatingPointsCount = 1
	seq.OperatingPoints[0].SeqLevelIdx = 5
	seq.MaxFrameWidth = 16
	seq.MaxFrameHeight = 9
	seq.EnableFilterIntra = true
	seq.EnableIntraEdgeFilter = true
	seq.EnableInterIntraCompound = true
	seq.EnableMaskedCompound = true
	seq.EnableDualFilter = true
	seq.EnableOrderHint = true
	seq.EnableJNTComp = true
	seq.EnableRefFrameMVS = true
	seq.OrderHintBits = 7
	seq.EnableSuperRes = true
	seq.EnableCDEF = true
	seq.EnableRestoration = true
	seq.ColorConfig = SequenceColorConfig{
		BitDepth:                8,
		ColorDescriptionPresent: true,
		ColorPrimaries:          SequenceColorPrimariesBT709,
		TransferCharacteristics: SequenceTransferCharacteristicsSRGB,
		MatrixCoefficients:      SequenceMatrixCoefficientsIdentity,
		ColorRange:              true,
	}
	seq.FilmGrainParamsPresent = true
	return seq
}

func realtimeEncoderSequenceHeaderPayload() []byte {
	var w sequenceTestBitWriter
	w.writeBits(0, 3)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(0, 5)
	w.writeBits(0, 12)
	w.writeBits(5, 5)
	w.writeBits(3, 4)
	w.writeBits(3, 4)
	w.writeBits(15, 4)
	w.writeBits(8, 4)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBits(0, 1)
	w.writeBits(6, 3)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBits(SequenceColorPrimariesBT709, 8)
	w.writeBits(SequenceTransferCharacteristicsSRGB, 8)
	w.writeBits(SequenceMatrixCoefficientsIdentity, 8)
	w.writeBool(false)
	w.writeBool(true)
	return w.trailingBits()
}
