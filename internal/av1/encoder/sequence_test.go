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
		{name: "timing", mut: func(seq *SequenceHeader) { seq.TimingInfoPresent = true }, want: ErrUnsupported},
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
