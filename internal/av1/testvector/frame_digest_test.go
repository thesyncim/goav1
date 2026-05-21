package testvector

import (
	"crypto/md5"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestFrameMD5MatchesVisibleRows(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:        5,
		Height:       3,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        8,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5MatchesMonochrome(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:      4,
		Height:     2,
		BitDepth:   8,
		MonoChrome: true,
		Align:      8,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5MatchesHighBitDepthRows(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:        3,
		Height:       2,
		BitDepth:     10,
		SubsamplingX: true,
		Align:        16,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5RejectsInvalidPlane(t *testing.T) {
	_, err := FrameMD5(frame.Frame{
		Format: frame.Format{Width: 2, Height: 1, BitDepth: 8},
		Layout: frame.Layout{
			BytesPerSample: 1,
		},
		Y: frame.Plane{Pix: []byte{0x01}, Stride: 1, Width: 2, Height: 1},
	})
	if !errors.Is(err, frame.ErrInvalidPlane) {
		t.Fatalf("FrameMD5 err=%v want %v", err, frame.ErrInvalidPlane)
	}
}

func testDigestFrame(t *testing.T, format frame.Format) (frame.Frame, []byte) {
	t.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	for i := range backing {
		backing[i] = 0xee
	}
	f, err := frame.Bind(backing, format)
	if err != nil {
		t.Fatal(err)
	}

	expected := make([]byte, 0, visibleFrameBytes(f))
	fillDigestPlane(&f.Y, f.Layout.BytesPerSample, 0x10, &expected)
	if !f.Format.MonoChrome {
		fillDigestPlane(&f.U, f.Layout.BytesPerSample, 0x70, &expected)
		fillDigestPlane(&f.V, f.Layout.BytesPerSample, 0xa0, &expected)
	}
	return f, expected
}

func fillDigestPlane(plane *frame.Plane, bytesPerSample int, seed byte, expected *[]byte) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			off := y*plane.Stride + x*bytesPerSample
			v := uint16(seed) + uint16(y*7+x)
			plane.Pix[off] = byte(v)
			if bytesPerSample == 2 {
				plane.Pix[off+1] = byte(v >> 8)
			}
		}
		start := y * plane.Stride
		end := start + plane.Width*bytesPerSample
		*expected = append(*expected, plane.Pix[start:end]...)
	}
}

func visibleFrameBytes(f frame.Frame) int {
	bytes := f.Y.Width * f.Y.Height * f.Layout.BytesPerSample
	if !f.Format.MonoChrome {
		bytes += (f.U.Width*f.U.Height + f.V.Width*f.V.Height) * f.Layout.BytesPerSample
	}
	return bytes
}
