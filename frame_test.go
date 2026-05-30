package goav1

import (
	"errors"
	"testing"
)

func TestFrameFormatFromHeaders(t *testing.T) {
	format, err := FrameFormatFromHeaders(
		SequenceHeader{ColorConfig: ColorConfig{
			BitDepth:     10,
			SubsamplingX: true,
			SubsamplingY: true,
		}},
		FrameSize{CodedWidth: 1920, Height: 1080},
		64,
	)
	if err != nil {
		t.Fatal(err)
	}
	if format.Width != 1920 || format.Height != 1080 || format.BitDepth != 10 ||
		!format.SubsamplingX || !format.SubsamplingY || format.Align != 64 {
		t.Fatalf("format=%+v", format)
	}
}

func TestFrameCodedAndOutputFormatFromHeaders(t *testing.T) {
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	size := FrameSize{
		CodedWidth:    960,
		UpscaledWidth: 1920,
		Height:        1080,
	}

	coded, err := FrameCodedFormatFromHeaders(sequence, size, 64)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := FrameFormatFromHeaders(sequence, size, 64)
	if err != nil {
		t.Fatal(err)
	}
	if coded != legacy || coded.Width != 960 {
		t.Fatalf("coded=%+v legacy=%+v", coded, legacy)
	}

	output, err := FrameOutputFormatFromHeaders(sequence, size, 64)
	if err != nil {
		t.Fatal(err)
	}
	if output.Width != 1920 || output.Height != 1080 || output.BitDepth != 10 ||
		!output.SubsamplingX || !output.SubsamplingY || output.Align != 64 {
		t.Fatalf("output format=%+v", output)
	}
}

func TestFrameFormatFromHeadersMonochrome(t *testing.T) {
	format, err := FrameFormatFromHeaders(
		SequenceHeader{ColorConfig: ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		}},
		FrameSize{CodedWidth: 640, Height: 360},
		32,
	)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	// Monochrome allocates no chroma planes: the U and V planes are zero-length,
	// so both their offsets sit at the end of the (superblock-aligned) luma
	// plane and the total size equals the luma plane size.
	if !format.MonoChrome || layout.ChromaWidth != 0 ||
		layout.UOffset != layout.Size || layout.VOffset != layout.Size {
		t.Fatalf("format=%+v layout=%+v", format, layout)
	}
}

func TestFrameFormatFromHeadersRejectsMissingState(t *testing.T) {
	_, err := FrameFormatFromHeaders(SequenceHeader{}, FrameSize{CodedWidth: 16, Height: 16}, 1)
	if !errors.Is(err, ErrFrameInvalidFormat) {
		t.Fatalf("FrameFormatFromHeaders err=%v want %v", err, ErrFrameInvalidFormat)
	}
	_, err = FrameOutputFormatFromHeaders(
		SequenceHeader{ColorConfig: ColorConfig{BitDepth: 8}},
		FrameSize{CodedWidth: 16, Height: 16},
		1,
	)
	if !errors.Is(err, ErrFrameInvalidFormat) {
		t.Fatalf("FrameOutputFormatFromHeaders err=%v want %v", err, ErrFrameInvalidFormat)
	}
}
