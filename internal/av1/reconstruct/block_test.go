package reconstruct

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestScratchLen(t *testing.T) {
	tests := []struct {
		name      string
		cfg       Block
		wantInt32 int
		wantInt16 int
	}{
		{
			name: "dct 8x8",
			cfg: Block{
				Size:      transform.Size{Width: 8, Height: 8},
				Transform: transform.TypeDCTDCT,
			},
			wantInt32: 128,
			wantInt16: 64,
		},
		{
			name: "dct 16x16",
			cfg: Block{
				Size:      transform.Size{Width: 16, Height: 16},
				Transform: transform.TypeDCTDCT,
			},
			wantInt32: 512,
			wantInt16: 256,
		},
		{
			name: "idtx 16x16",
			cfg: Block{
				Size:      transform.Size{Width: 16, Height: 16},
				Transform: transform.TypeIDTX,
			},
			wantInt32: 256,
			wantInt16: 256,
		},
		{
			name: "adst 8x8",
			cfg: Block{
				Size:      transform.Size{Width: 8, Height: 8},
				Transform: transform.TypeADSTDCT,
			},
			wantInt32: 128,
			wantInt16: 64,
		},
		{
			name: "dct 64x64",
			cfg: Block{
				Size:      transform.Size{Width: 64, Height: 64},
				Transform: transform.TypeDCTDCT,
			},
			wantInt32: 8192,
			wantInt16: 4096,
		},
	}
	for _, tt := range tests {
		got32, got16, err := ScratchLen(tt.cfg)
		if err != nil {
			t.Fatalf("%s ScratchLen err=%v", tt.name, err)
		}
		if got32 != tt.wantInt32 || got16 != tt.wantInt16 {
			t.Fatalf("%s ScratchLen=%d/%d want %d/%d", tt.name, got32, got16, tt.wantInt32, tt.wantInt16)
		}
	}
	if _, _, err := ScratchLen(Block{Size: transform.Size{Width: 64, Height: 8}, Transform: transform.TypeDCTDCT}); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("unsupported ScratchLen err=%v want %v", err, ErrInvalidBlock)
	}
}

func TestReconstructPlaneBlockDCT4x4(t *testing.T) {
	plane, _ := testPlane(6, 5, 1, 8)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 4*4)
	quantized[0] = 4
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 4},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	if err := ReconstructPlaneBlock(plane, 1, 8, 1, 1, quantized, 4, int32Scratch, residualScratch, cfg); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(100)
			if x >= 1 && x < 5 && y >= 1 && y < 5 {
				want = 101
			}
			if got := getSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestReconstructPlaneBlockVisibleClipsFrameEdge(t *testing.T) {
	plane, _ := testPlane(6, 6, 1, 8)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 4*4)
	quantized[0] = 4
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 4},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	if err := ReconstructPlaneBlockVisible(plane, 1, 8, 4, 3, 2, 3, quantized, 4, int32Scratch, residualScratch, cfg); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(100)
			if x >= 4 && y >= 3 {
				want = 101
			}
			if got := getSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestReconstructPlaneBlockIDTX4x8(t *testing.T) {
	plane, _ := testPlane(4, 8, 1, 4)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 4*8)
	quantized[0] = 16
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 8},
		Transform: transform.TypeIDTX,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch := make([]int32, int32Len+3)
	residualScratch := make([]int16, int16Len+2)
	if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 8, int32Scratch, residualScratch, cfg); err != nil {
		t.Fatal(err)
	}
	if got := getSample(plane, 1, 0, 0); got != 108 {
		t.Fatalf("dc sample=%d want 108", got)
	}
	for i := int32Len; i < len(int32Scratch); i++ {
		if int32Scratch[i] != 0 {
			t.Fatalf("int32 scratch padding[%d]=%d want 0", i, int32Scratch[i])
		}
	}
	for i := int16Len; i < len(residualScratch); i++ {
		if residualScratch[i] != 0 {
			t.Fatalf("residual scratch padding[%d]=%d want 0", i, residualScratch[i])
		}
	}
}

func TestReconstructPlaneBlockADST8x8(t *testing.T) {
	plane, _ := testPlane(8, 8, 1, 8)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 8*8)
	quantized[0] = 4
	quantized[1*8+0] = -2
	quantized[0*8+1] = 3
	cfg := Block{
		Size:      transform.Size{Width: 8, Height: 8},
		Transform: transform.TypeADSTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 8, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	changed := false
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			if got := getSample(plane, 1, x, y); got != 100 {
				changed = true
			}
		}
	}
	if !changed {
		t.Fatal("ADST reconstruction left the predicted block unchanged")
	}
}

func TestReconstructPlaneBlockUsesAV1CoeffOrder(t *testing.T) {
	plane, _ := testPlane(4, 4, 1, 4)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 4*4)
	quantized[2*4+1] = 1
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 4},
		Transform: transform.TypeIDTX,
		Quantizer: quantize.Quantizer{DC: 16, AC: 16},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 4, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(100)
			if x == 2 && y == 1 {
				want = 102
			}
			if got := getSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestReconstructPlaneBlockLosslessWHT4x4(t *testing.T) {
	plane, _ := testPlane(4, 4, 1, 4)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 4*4)
	quantized[0] = 4
	quantized[1] = 2
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 4},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 4},
		Lossless:  true,
		EOB:       2,
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 4, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(100)
			if y == 0 {
				want = 102
			} else if y == 1 {
				want = 101
			}
			if got := getSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestReconstructPlaneBlockDCT16x16(t *testing.T) {
	plane, _ := testPlane(18, 17, 1, 20)
	fillPlane(plane, 1, 90)
	quantized := make([]int16, 16*16)
	quantized[0] = 16
	cfg := Block{
		Size:      transform.Size{Width: 16, Height: 16},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(plane, 1, 8, 1, 1, quantized, 16, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(90)
			if x >= 1 && x < 17 && y >= 1 && y < 17 {
				want = 91
			}
			if got := getSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestReconstructPlaneBlockDCT64x64ReducedCoefficients(t *testing.T) {
	plane, _ := testPlane(64, 64, 1, 64)
	fillPlane(plane, 1, 100)
	quantized := make([]int16, 32*32)
	quantized[0] = 64
	cfg := Block{
		Size:      transform.Size{Width: 64, Height: 64},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 32, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	if got := getSample(plane, 1, 0, 0); got == 100 {
		t.Fatal("DCT64 reconstruction left the first sample unchanged")
	}
}

func TestReconstructPlaneBlockHighBitDepthClips(t *testing.T) {
	plane, _ := testPlane(4, 4, 2, 8)
	fillPlane(plane, 2, 1020)
	quantized := make([]int16, 4*4)
	quantized[0] = 64
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 4},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 16, AC: 16},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(plane, 2, 10, 0, 0, quantized, 4, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			if got := getSample(plane, 2, x, y); got != 1023 {
				t.Fatalf("sample(%d,%d)=%d want 1023", x, y, got)
			}
		}
	}
}

func TestReconstructPlaneBlockRejectsInvalidInputs(t *testing.T) {
	plane, _ := testPlane(4, 4, 1, 4)
	quantized := make([]int16, 4*4)
	cfg := Block{
		Size:      transform.Size{Width: 4, Height: 4},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Scratch := make([]int32, 32)
	residualScratch := make([]int16, 16)
	tests := []struct {
		name            string
		plane           frame.Plane
		bytesPerSample  int
		bitDepth        uint8
		x               int
		y               int
		quantized       []int16
		quantizedStride int
		int32Scratch    []int32
		residualScratch []int16
		cfg             Block
	}{
		{name: "unsupported transform", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: Block{Size: transform.Size{Width: 64, Height: 8}, Transform: transform.TypeDCTDCT, Quantizer: cfg.Quantizer}},
		{name: "lossless non 4x4", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: Block{Size: transform.Size{Width: 8, Height: 8}, Transform: transform.TypeDCTDCT, Quantizer: cfg.Quantizer, Lossless: true}},
		{name: "lossless negative eob", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: Block{Size: cfg.Size, Transform: cfg.Transform, Quantizer: cfg.Quantizer, Lossless: true, EOB: -1}},
		{name: "short int32 scratch", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch[:31], residualScratch: residualScratch, cfg: cfg},
		{name: "short residual scratch", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch[:15], cfg: cfg},
		{name: "short quantized stride", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 3, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: cfg},
		{name: "outside plane", plane: plane, bytesPerSample: 1, bitDepth: 8, x: 1, y: 1, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: cfg},
		{name: "invalid quantizer", plane: plane, bytesPerSample: 1, bitDepth: 8, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: Block{Size: cfg.Size, Transform: cfg.Transform, Quantizer: quantize.Quantizer{AC: 8}}},
		{name: "bitdepth mismatch", plane: plane, bytesPerSample: 1, bitDepth: 10, quantized: quantized, quantizedStride: 4, int32Scratch: int32Scratch, residualScratch: residualScratch, cfg: cfg},
	}
	for _, tt := range tests {
		err := ReconstructPlaneBlock(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.x, tt.y, tt.quantized, tt.quantizedStride, tt.int32Scratch, tt.residualScratch, tt.cfg)
		if !errors.Is(err, ErrInvalidBlock) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidBlock)
		}
	}
}

func TestReconstructPlaneBlockAllocs(t *testing.T) {
	plane, _ := testPlane(8, 8, 1, 8)
	quantized := make([]int16, 8*8)
	cfg := Block{
		Size:      transform.Size{Width: 8, Height: 8},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 8, int32Scratch, residualScratch, cfg); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ReconstructPlaneBlock allocated: %f", allocs)
	}
}

func FuzzReconstructPlaneBlock(f *testing.F) {
	f.Add(uint8(0), uint8(0), int16(0), int16(0))
	f.Add(uint8(1), uint8(0), int16(4), int16(0))
	f.Add(uint8(2), uint8(1), int16(16), int16(-2))
	f.Add(uint8(3), uint8(1), int16(-16), int16(3))

	f.Fuzz(func(t *testing.T, rawMode uint8, rawBPS uint8, dc int16, acDelta int16) {
		bytesPerSample := int(rawBPS%2) + 1
		bitDepth := uint8(8)
		fill := uint16(100)
		if bytesPerSample == 2 {
			bitDepth = 10
			fill = 512
		}
		cfg := Block{
			Size:      transform.Size{Width: 4, Height: 4},
			Transform: transform.TypeDCTDCT,
			Quantizer: quantize.Quantizer{DC: 4, AC: 8},
		}
		if rawMode&1 == 1 {
			cfg.Size = transform.Size{Width: 8, Height: 8}
		}
		if rawMode&2 != 0 {
			cfg.Size = transform.Size{Width: 16, Height: 16}
		}
		if rawMode&4 != 0 {
			cfg.Size = transform.Size{Width: 16, Height: 16}
			cfg.Transform = transform.TypeIDTX
		}
		plane, _ := testPlane(cfg.Size.Width+2, cfg.Size.Height+1, bytesPerSample, (cfg.Size.Width+5)*bytesPerSample)
		fillPlane(plane, bytesPerSample, fill)
		coeffSize, err := transform.ScanSize(cfg.Size)
		if err != nil {
			t.Fatalf("ScanSize err=%v", err)
		}
		quantizedStride := coeffSize.Height + 2
		quantized := make([]int16, quantizedStride*coeffSize.Width)
		for row := 0; row < cfg.Size.Height; row++ {
			for col := 0; col < cfg.Size.Width; col++ {
				if row < coeffSize.Height && col < coeffSize.Width {
					quantized[col*quantizedStride+row] = dc + acDelta*int16((row+col)&3)
				}
			}
		}
		int32Len, int16Len, err := ScratchLen(cfg)
		if err != nil {
			t.Fatalf("ScratchLen err=%v", err)
		}
		int32Scratch := make([]int32, int32Len+3)
		residualScratch := make([]int16, int16Len+2)
		if err := ReconstructPlaneBlock(plane, bytesPerSample, bitDepth, 1, 0, quantized, quantizedStride, int32Scratch, residualScratch, cfg); err != nil {
			t.Fatalf("ReconstructPlaneBlock err=%v", err)
		}
		max := uint16(0xff)
		if bytesPerSample == 2 {
			max = 0x3ff
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				got := getSample(plane, bytesPerSample, x, y)
				if got > max {
					t.Fatalf("sample(%d,%d)=%d exceeds max %d", x, y, got, max)
				}
				if (x == 0 || x > cfg.Size.Width || y == cfg.Size.Height) && got != fill {
					t.Fatalf("padding sample(%d,%d)=%d want %d", x, y, got, fill)
				}
			}
		}
		for i := int32Len; i < len(int32Scratch); i++ {
			if int32Scratch[i] != 0 {
				t.Fatalf("int32 scratch padding[%d]=%d want 0", i, int32Scratch[i])
			}
		}
		for i := int16Len; i < len(residualScratch); i++ {
			if residualScratch[i] != 0 {
				t.Fatalf("residual scratch padding[%d]=%d want 0", i, residualScratch[i])
			}
		}
	})
}

func BenchmarkReconstructPlaneBlockDCT8x8(b *testing.B) {
	plane, _ := testPlane(8, 8, 1, 8)
	quantized := make([]int16, 8*8)
	cfg := Block{
		Size:      transform.Size{Width: 8, Height: 8},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, _ := ScratchLen(cfg)
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 8, int32Scratch, residualScratch, cfg)
	}
}

func BenchmarkReconstructPlaneBlockIDTX16x16(b *testing.B) {
	plane, _ := testPlane(16, 16, 1, 16)
	quantized := make([]int16, 16*16)
	cfg := Block{
		Size:      transform.Size{Width: 16, Height: 16},
		Transform: transform.TypeIDTX,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, _ := ScratchLen(cfg)
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 16, int32Scratch, residualScratch, cfg)
	}
}

func testPlane(width int, height int, bytesPerSample int, stride int) (frame.Plane, []byte) {
	buf := make([]byte, stride*height)
	return frame.Plane{
		Pix:    buf,
		Stride: stride,
		Width:  width,
		Height: height,
	}, buf
}

func fillPlane(plane frame.Plane, bytesPerSample int, value uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, bytesPerSample, x, y, value)
		}
	}
}

func getSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func setSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}
