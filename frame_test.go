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
	if !format.MonoChrome || layout.ChromaWidth != 0 || layout.Size != layout.YStride*format.Height {
		t.Fatalf("format=%+v layout=%+v", format, layout)
	}
}

func TestFrameFormatFromHeadersRejectsMissingState(t *testing.T) {
	_, err := FrameFormatFromHeaders(SequenceHeader{}, FrameSize{CodedWidth: 16, Height: 16}, 1)
	if !errors.Is(err, ErrFrameInvalidFormat) {
		t.Fatalf("FrameFormatFromHeaders err=%v want %v", err, ErrFrameInvalidFormat)
	}
}
