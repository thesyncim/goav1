package reconstruct

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp"
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

func TestReconstructPlaneBlockVisibleWithGeometryMatchesDefault(t *testing.T) {
	cfg := Block{
		Size:      transform.Size{Width: 8, Height: 8},
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
		EOB:       4,
	}
	scanSize, err := transform.ScanSize(cfg.Size)
	if err != nil {
		t.Fatal(err)
	}
	txScale, err := quantize.TransformScale(int(cfg.Size.Width), int(cfg.Size.Height))
	if err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	quantized := make([]int16, int(cfg.Size.Width)*int(cfg.Size.Height))
	quantized[0] = 8
	quantized[1] = -3
	quantized[8] = 5
	quantized[9] = -2

	want, _ := testPlane(10, 10, 1, 10)
	got, _ := testPlane(10, 10, 1, 10)
	gotSparse, _ := testPlane(10, 10, 1, 10)
	fillPlane(want, 1, 100)
	fillPlane(got, 1, 100)
	fillPlane(gotSparse, 1, 100)
	if err := ReconstructPlaneBlockVisible(want, 1, 8, 2, 1, 6, 7, quantized, int(scanSize.Height), make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlockVisibleWithGeometry(got, 1, 8, 2, 1, 6, 7, quantized, int(scanSize.Height), scanSize, txScale, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Pix, want.Pix) {
		t.Fatalf("geometry path output differs from default")
	}
	scan := []int16{0, 1, 8, 9}
	if err := ReconstructPlaneBlockVisibleWithGeometryAndScan(gotSparse, 1, 8, 2, 1, 6, 7, quantized, int(scanSize.Height), scan, scanSize, txScale, make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotSparse.Pix, want.Pix) {
		t.Fatalf("scan geometry path output differs from default")
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

// TestReconstructPlaneBlockDCT64x64SparseInterShape pins the reconstruct path
// for a TX_64X64 DCT_DCT block with the exact sparse-coefficient shape
// observed in libaom monochrome frame 3 mi=(32,16) (compound NEAREST_NEAREST
// inter block, EOB=3, three non-zero quantized coefficients at column-major
// buffer indices 0:5, 1:1, 32:-3). After the IDCT64 column-pass fix
// (e514b7f), the inverse-transform produces a smooth row+column-monotonic
// residual; this test pins the full dequant+IDCT+add integration so the
// glue (scratch slicing, txScale, residual stride) can't regress for
// sparse 64x64 inter residuals without being caught.
//
// The expected values are computed by running the same dequant+IDCT+add
// pipeline directly through the standalone primitives, then asserted against
// the ReconstructPlaneBlock output. Any subtle regression of the
// integration (e.g. wrong residualStride into AddResidualPlaneBlock, wrong
// dequant region for the 32x32 coded scan inside the 64x64 transform, or a
// scratch-buffer overlap that corrupts the dequant before IDCT reads it)
// breaks this pin.
func TestReconstructPlaneBlockDCT64x64SparseInterShape(t *testing.T) {
	const W, H = 64, 64

	// Frame-3 mi=(32,16) shape: three non-zero quantized coefficients at
	// column-major indices 0 (DC), 1 (col=0,row=1), and 32 (col=1,row=0).
	quantized := make([]int16, 32*32)
	quantized[0*32+0] = 5
	quantized[0*32+1] = 1
	quantized[1*32+0] = -3

	q := quantize.Quantizer{DC: 88, AC: 106}
	cfg := Block{
		Size:      transform.Size{Width: W, Height: H},
		Transform: transform.TypeDCTDCT,
		Quantizer: q,
		EOB:       3,
	}

	// Build the expected output by running the primitives directly: dequant
	// the 32x32 coded region, run InverseBlockBitDepth into a 64x64 residual,
	// then add to a flat-116 predictor.
	const predictor = 116
	wantPix := make([]byte, W*H)
	for i := range wantPix {
		wantPix[i] = predictor
	}
	wantPlane := frame.Plane{Pix: wantPix, Stride: W, Width: W, Height: H}

	txScale, err := quantize.TransformScale(W, H)
	if err != nil {
		t.Fatal(err)
	}
	dq := make([]int32, 32*32)
	if err := quantize.DequantizeBlockScaledBitDepth(dq, 32, quantized, 32, 32, 32, q, txScale, 8); err != nil {
		t.Fatal(err)
	}
	wantResidual := make([]int16, W*H)
	idctScratch := make([]int32, W*H)
	if err := transform.InverseBlockBitDepth(wantResidual, W, dq, 32, idctScratch, transform.Size{Width: W, Height: H}, transform.TypeDCTDCT, 8); err != nil {
		t.Fatal(err)
	}
	if err := dsp.AddResidualPlaneBlock(wantPlane, 1, 8, 0, 0, W, H, wantResidual, W); err != nil {
		t.Fatal(err)
	}

	// Pin a few characteristic residual samples so any change to the IDCT64
	// shape (which the prior agent identified as the source of frame-3
	// monochrome divergence) is caught here too.
	if wantResidual[0] != 0 {
		t.Fatalf("residual[0,0] expected 0 (low-DC sparse case) got %d", wantResidual[0])
	}
	if wantResidual[63] != 2 {
		t.Fatalf("residual[0,63] expected 2 (col=1 cosine peak) got %d", wantResidual[63])
	}
	if wantResidual[15*W+0] != 0 {
		t.Fatalf("residual[15,0] expected 0 got %d", wantResidual[15*W+0])
	}

	// Now run ReconstructPlaneBlock on a flat-116 predictor and assert
	// byte-exact equality with the directly-composed reference.
	gotPix := make([]byte, W*H)
	for i := range gotPix {
		gotPix[i] = predictor
	}
	gotPlane := frame.Plane{Pix: gotPix, Stride: W, Width: W, Height: H}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(gotPlane, 1, 8, 0, 0, quantized, 32,
		make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	for r := range H {
		for c := range W {
			if gotPix[r*W+c] != wantPix[r*W+c] {
				t.Fatalf("sparse 64x64 reconstruct mismatch (r=%d c=%d): got=%d want=%d residual=%d",
					r, c, gotPix[r*W+c], wantPix[r*W+c], wantResidual[r*W+c])
			}
		}
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
		width := int(cfg.Size.Width)
		height := int(cfg.Size.Height)
		plane, _ := testPlane(width+2, height+1, bytesPerSample, (width+5)*bytesPerSample)
		fillPlane(plane, bytesPerSample, fill)
		coeffSize, err := transform.ScanSize(cfg.Size)
		if err != nil {
			t.Fatalf("ScanSize err=%v", err)
		}
		coeffWidth := int(coeffSize.Width)
		coeffHeight := int(coeffSize.Height)
		quantizedStride := coeffHeight + 2
		quantized := make([]int16, quantizedStride*coeffWidth)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				if row < coeffHeight && col < coeffWidth {
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
				if (x == 0 || x > width || y == height) && got != fill {
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
