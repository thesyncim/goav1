package encoder_test

import (
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

func TestEncodeScaledReferencePFrameLayerPoolDecodeMatchesRecon(t *testing.T) {
	const (
		baseW, baseH = 64, 48
		topW, topH   = 128, 96
		keyQ         = 64
	)

	baseSrc := scaledReferenceDecodeBaseFrame(baseW, baseH)
	keyTU, baseRecon, err := encoder.EncodeKeyframeWithSequenceMax(baseSrc, keyQ, topW, topH)
	if err != nil {
		t.Fatalf("EncodeKeyframeWithSequenceMax: %v", err)
	}

	topSrc := scaleSource420(baseRecon, topW, topH)
	for _, tc := range []struct {
		name   string
		qIndex uint8
	}{
		{name: "lossless", qIndex: 0},
		{name: "lossy", qIndex: 72},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scaledTU, topRecon, err := encoder.EncodeScaledReferencePFrame(topSrc, baseRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeScaledReferencePFrame: %v", err)
			}
			if tc.qIndex == 0 {
				compareBytePlane(t, "lossless source Y", topRecon.Y, topRecon.YStride, topSrc.Y, topSrc.YStride, topW, topH)
				compareBytePlane(t, "lossless source U", topRecon.U, topRecon.ChromaStride, topSrc.U, topSrc.ChromaStride, topW/2, topH/2)
				compareBytePlane(t, "lossless source V", topRecon.V, topRecon.ChromaStride, topSrc.V, topSrc.ChromaStride, topW/2, topH/2)
			}

			frames := decodeLayerPoolLowOverheads(t, keyTU, scaledTU)
			if len(frames) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(frames))
			}
			comparePlane(t, "base Y", frames[0].Y, baseRecon.Y, baseW, baseH, baseW)
			comparePlane(t, "base U", frames[0].U, baseRecon.U, baseW/2, baseH/2, baseW/2)
			comparePlane(t, "base V", frames[0].V, baseRecon.V, baseW/2, baseH/2, baseW/2)
			comparePlane(t, "top Y", frames[1].Y, topRecon.Y, topW, topH, topW)
			comparePlane(t, "top U", frames[1].U, topRecon.U, topW/2, topH/2, topW/2)
			comparePlane(t, "top V", frames[1].V, topRecon.V, topW/2, topH/2, topW/2)
		})
	}
}

func compareBytePlane(t *testing.T, name string, got []byte, gotStride int, want []byte, wantStride int, w, h int) {
	t.Helper()
	for y := range h {
		gotRow := got[y*gotStride : y*gotStride+w]
		wantRow := want[y*wantStride : y*wantStride+w]
		for x := range w {
			if gotRow[x] != wantRow[x] {
				t.Fatalf("%s differs at (%d,%d): got %d want %d", name, x, y, gotRow[x], wantRow[x])
			}
		}
	}
}

// decodeLayerPoolLowOverheads decodes the temporal units with LayeredDecoder,
// which keeps one surface pool per coded size so the scaled frame can
// reference the smaller one.
func decodeLayerPoolLowOverheads(t *testing.T, payloads ...[]byte) []*goav1.Frame {
	t.Helper()
	dec, err := goav1.NewLayeredDecoder(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoder: %v", err)
	}
	t.Cleanup(dec.Close)
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("DecodeAll: %v", err)
	}
	return frames
}

func scaledReferenceDecodeBaseFrame(width, height int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	for y := range height {
		for x := range width {
			f.Y[y*width+x] = byte(47 + 3*x + 5*y + x*y)
		}
	}
	for y := range ch {
		for x := range cw {
			f.U[y*cw+x] = byte(96 + 3*x + 7*y)
			f.V[y*cw+x] = byte(151 + 5*x + 2*y)
		}
	}
	return f
}

func scaleSource420(src encoder.SourceFrame420, width, height int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	dst := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	scalePlane(dst.Y, dst.YStride, width, height, src.Y, src.YStride, src.Width, src.Height)
	scalePlane(dst.U, dst.ChromaStride, cw, ch, src.U, src.ChromaStride, src.Width/2, src.Height/2)
	scalePlane(dst.V, dst.ChromaStride, cw, ch, src.V, src.ChromaStride, src.Width/2, src.Height/2)
	return dst
}

func scalePlane(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	for y := range dstHeight {
		sy := y * srcHeight / dstHeight
		for x := range dstWidth {
			sx := x * srcWidth / dstWidth
			dst[y*dstStride+x] = src[sy*srcStride+sx]
		}
	}
}
