package encoder_test

import (
	"fmt"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestWebRTCStreamInterLayerPredictionDecodeMatchesRecon codes two spatial
// SVC pictures, so the upper layer predicts from the half-size base layer
// through the scaled inter predictor, and checks that every decoded frame
// equals its layer encoder's reconstruction bit for bit.
func TestWebRTCStreamInterLayerPredictionDecodeMatchesRecon(t *testing.T) {
	const topW, topH = 128, 96
	for _, mode := range []encoder.ScalabilityMode{encoder.ScalabilityModeL2T1, encoder.ScalabilityModeL2T2} {
		t.Run(mode.String(), func(t *testing.T) {
			stream, err := encoder.NewWebRTCStreamConfig(encoder.Config{
				Resolution:        encoder.Resolution{Width: topW, Height: topH},
				MaxFramerate:      encoder.Rational{Num: 30, Den: 1},
				MinBitrateKbps:    100,
				MaxBitrateKbps:    900,
				TargetBitrateKbps: 500,
				Scalability:       mode,
			})
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			src := scaledReferenceDecodeBaseFrame(topW, topH)
			var tus [][]byte
			var recons []encoder.SourceFrame420
			for pic := range 2 {
				if pic == 1 {
					src = shiftSource420(src, 2, 1)
				}
				picture, err := stream.EncodePicture(src, pic == 0)
				if err != nil {
					t.Fatalf("EncodePicture %d: %v", pic, err)
				}
				for i := range picture.FrameNum {
					frame := picture.Frames[i]
					tus = append(tus, append([]byte(nil), frame.TU...))
					recons = append(recons, cloneFrame(stream.LayerRecon(frame.Info.SpatialID)))
				}
			}
			frames := decodeLayerPoolLowOverheads(t, tus...)
			if len(frames) != len(recons) {
				t.Fatalf("decoded %d frames, want %d", len(frames), len(recons))
			}
			for i, f := range frames {
				r := recons[i]
				cw, ch := r.Width/2, r.Height/2
				comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, r.Y, r.Width, r.Height, r.YStride)
				comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, r.U, cw, ch, r.ChromaStride)
				comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, r.V, cw, ch, r.ChromaStride)
			}
		})
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

// shiftSource420 moves src by (dx, dy) samples, replicating the top and left
// edges.
func shiftSource420(src encoder.SourceFrame420, dx, dy int) encoder.SourceFrame420 {
	dst := cloneFrame(src)
	shiftPlane(dst.Y, src.Y, src.YStride, src.Width, src.Height, dx, dy)
	shiftPlane(dst.U, src.U, src.ChromaStride, src.Width/2, src.Height/2, dx/2, dy/2)
	shiftPlane(dst.V, src.V, src.ChromaStride, src.Width/2, src.Height/2, dx/2, dy/2)
	return dst
}

func shiftPlane(dst, src []byte, stride, w, h, dx, dy int) {
	for y := range h {
		for x := range w {
			dst[y*stride+x] = src[max(0, y-dy)*stride+max(0, x-dx)]
		}
	}
}
