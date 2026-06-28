package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPlaneBlockSizeMatchesLibaomSSSizeLookup(t *testing.T) {
	tests := []struct {
		name  string
		block BlockSize
		color parser.ColorConfig
		plane int
		want  BlockSize
	}{
		{name: "luma unchanged", block: BlockSize16x8, plane: 0, want: BlockSize16x8},
		{name: "128x128 420", block: BlockSize128x128, color: parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, plane: 1, want: BlockSize64x64},
		{name: "64x64 422", block: BlockSize64x64, color: parser.ColorConfig{SubsamplingX: true}, plane: 1, want: BlockSize32x64},
		{name: "64x64 440", block: BlockSize64x64, color: parser.ColorConfig{SubsamplingY: true}, plane: 2, want: BlockSize64x32},
		{name: "16x16 420", block: BlockSize16x16, color: parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, plane: 1, want: BlockSize8x8},
		{name: "8x8 420", block: BlockSize8x8, color: parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, plane: 2, want: BlockSize4x4},
		{name: "16x4 420 keeps valid chroma shape", block: BlockSize16x4, color: parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, plane: 1, want: BlockSize8x4},
		{name: "4x8 420", block: BlockSize4x8, color: parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, plane: 2, want: BlockSize4x4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PlaneBlockSize(tt.block, tt.color, tt.plane)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("PlaneBlockSize=%d want %d", got, tt.want)
			}
		})
	}

	if _, err := PlaneBlockSize(BlockSize64x128, parser.ColorConfig{SubsamplingX: true}, 1); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("invalid 422 plane block err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := PlaneBlockSize(BlockSize4x8, parser.ColorConfig{SubsamplingX: true}, 1); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("invalid narrow 422 plane block err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := PlaneBlockSize(BlockSize8x8, parser.ColorConfig{MonoChrome: true}, 1); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("monochrome chroma block err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestHasChromaBlockMatchesDav1dCondition(t *testing.T) {
	color420 := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	if HasChromaBlock(TransformTreeRequest{Size: BlockSize4x4, X4: 0, Y4: 0}, color420) {
		t.Fatal("even 4x4 420 block unexpectedly has chroma")
	}
	if !HasChromaBlock(TransformTreeRequest{Size: BlockSize4x4, X4: 1, Y4: 1}, color420) {
		t.Fatal("odd 4x4 420 block should have chroma")
	}
	if !HasChromaBlock(TransformTreeRequest{Size: BlockSize8x8}, color420) {
		t.Fatal("8x8 420 block should have chroma")
	}
	if HasChromaBlock(TransformTreeRequest{Size: BlockSize8x8}, parser.ColorConfig{MonoChrome: true}) {
		t.Fatal("monochrome block unexpectedly has chroma")
	}
	for _, size := range []BlockSize{BlockSize4x4, BlockSize4x8, BlockSize8x4, BlockSize8x8, BlockSize16x16} {
		for y4 := 0; y4 < 4; y4++ {
			for x4 := 0; x4 < 4; x4++ {
				req := TransformTreeRequest{Size: size, X4: uint8(x4), Y4: uint8(y4)}
				if got, want := HasChromaBlockAt(size, x4, y4, color420), HasChromaBlock(req, color420); got != want {
					t.Fatalf("HasChromaBlockAt(%d,%d,%d)=%v want %v", size, x4, y4, got, want)
				}
			}
		}
	}
}

func BenchmarkHasChromaForBlock420(b *testing.B) {
	color := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	sizes := [...]BlockSize{BlockSize4x4, BlockSize4x8, BlockSize8x4, BlockSize8x8, BlockSize16x16}
	sum := 0
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		size := sizes[i%len(sizes)]
		if hasChromaForBlock(size, i&3, (i>>2)&3, color) {
			sum++
		}
	}
	planeBlockSink = sum
}

var planeBlockSink int
