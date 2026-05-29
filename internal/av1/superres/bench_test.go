package superres

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func benchUpscale(b *testing.B, bitDepth uint8, srcWidth, dstWidth, height int) {
	b.Helper()
	src := make([]uint16, srcWidth*height)
	maxVal := uint16((1 << bitDepth) - 1)
	for i := range src {
		src[i] = uint16(i*97) & maxVal
	}
	dst := make([]uint16, dstWidth*height)
	srcPlane := frame.SamplePlane{Pix: src, Stride: srcWidth, Width: srcWidth, Height: height}
	dstPlane := frame.SamplePlane{Pix: dst, Stride: dstWidth, Width: dstWidth, Height: height}
	b.SetBytes(int64(dstWidth * height * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := UpscalePlane(srcPlane, dstPlane, bitDepth); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUpscalePlane8bit(b *testing.B)  { benchUpscale(b, 8, 480, 640, 360) }
func BenchmarkUpscalePlane10bit(b *testing.B) { benchUpscale(b, 10, 480, 640, 360) }
