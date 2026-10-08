package encoder

import (
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func overflowingStride(width int) (height, stride int, ok bool) {
	maxInt := int(^uint(0) >> 1)
	for height = 8; height <= 1<<20; height += 8 {
		rows := height - 1
		stride = (maxInt - (width - 1)) / rows
		if stride > (maxInt-width)/rows {
			return height, stride, true
		}
	}
	return 0, 0, false
}

func TestCheckedPlaneLengthBoundaries(t *testing.T) {
	if got, ok := checkedPlaneLength(8, 8, 8); !ok || got != 64 {
		t.Fatalf("minimal plane: got (%d, %v), want (64, true)", got, ok)
	}
	if got, ok := checkedPlaneLength(8, 8, 10); !ok || got != 78 {
		t.Fatalf("padded plane: got (%d, %v), want (78, true)", got, ok)
	}
	if got, ok := checkedPlaneLength(8, 1, 10); !ok || got != 8 {
		t.Fatalf("single-row plane: got (%d, %v), want (8, true)", got, ok)
	}
	maxInt := int(^uint(0) >> 1)
	if got, ok := checkedPlaneLength(8, 2, maxInt-8); !ok || got != maxInt {
		t.Fatalf("exact MaxInt plane: got (%d, %v), want (%d, true)", got, ok, maxInt)
	}

	height, stride, found := overflowingStride(8)
	if !found {
		t.Fatal("could not find a portable overflowing stride case")
	}
	if got, ok := checkedPlaneLength(8, height, stride); ok {
		t.Fatalf("overflowing plane accepted with length %d (height %d, stride %d)", got, height, stride)
	}
}

func TestMonochromeRenderValidatorsRejectOverflowingStride(t *testing.T) {
	const width = 8
	height, stride, found := overflowingStride(width)
	if !found {
		t.Fatal("could not find a portable overflowing stride case")
	}

	mono := &MonochromeVideoEncoder{renderWidth: width, renderHeight: height}
	if err := mono.validateRenderSource(SourceFrameMono{
		YStride: stride,
		Width:   width,
		Height:  height,
	}); err == nil || !strings.Contains(err.Error(), "dimensions overflow int") {
		t.Fatalf("8-bit render validator error = %v, want plane-length overflow", err)
	}

	mono16 := &HighBitDepthMonochromeVideoEncoder{
		renderWidth:  width,
		renderHeight: height,
		bitDepth:     10,
	}
	if err := mono16.validateRenderSource(SourceFrameMono16{
		YStride:  stride,
		Width:    width,
		Height:   height,
		BitDepth: 10,
	}); err == nil || !strings.Contains(err.Error(), "dimensions overflow int") {
		t.Fatalf("high-bit-depth render validator error = %v, want plane-length overflow", err)
	}
}

func TestMonochromeRenderValidatorsAcceptSingleRow(t *testing.T) {
	const width, height, stride = 8, 1, 10
	mono := &MonochromeVideoEncoder{renderWidth: width, renderHeight: height}
	if err := mono.validateRenderSource(SourceFrameMono{
		Y:       make([]byte, width),
		YStride: stride,
		Width:   width,
		Height:  height,
	}); err != nil {
		t.Fatalf("8-bit single-row render: %v", err)
	}

	mono16 := &HighBitDepthMonochromeVideoEncoder{
		renderWidth:  width,
		renderHeight: height,
		bitDepth:     10,
	}
	if err := mono16.validateRenderSource(SourceFrameMono16{
		Y:        make([]uint16, width),
		YStride:  stride,
		Width:    width,
		Height:   height,
		BitDepth: 10,
	}); err != nil {
		t.Fatalf("high-bit-depth single-row render: %v", err)
	}
}

func TestColor16ValidatorChecksChromaPlaneLength(t *testing.T) {
	const width = 8
	height, stride, found := overflowingStride(width)
	if !found {
		t.Fatal("could not find a portable overflowing stride case")
	}
	color := parser.ColorConfig{BitDepth: 10}
	src := SourceFrame42016{
		Y:            make([]uint16, (height-1)*width+width),
		YStride:      width,
		ChromaStride: stride,
		Width:        width,
		Height:       height,
		BitDepth:     10,
	}
	if err := validateSourceFrameColor16(src, color, "4:4:4 test"); err == nil || !strings.Contains(err.Error(), "chroma plane dimensions overflow int") {
		t.Fatalf("color validator error = %v, want chroma-plane length overflow", err)
	}
}

func TestColor16RenderValidatorAcceptsSingleRowSubsampledGeometry(t *testing.T) {
	src := SourceFrame42016{
		Y:            []uint16{0, 0},
		U:            []uint16{0},
		V:            []uint16{0},
		YStride:      2,
		ChromaStride: 1,
		Width:        2,
		Height:       1,
		BitDepth:     10,
	}
	color := parser.ColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
	if err := validateSourceRenderFrameColor16(src, color, "4:2:0 test"); err != nil {
		t.Fatalf("single-row subsampled render: %v", err)
	}
}
