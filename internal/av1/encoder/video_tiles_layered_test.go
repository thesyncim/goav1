package encoder_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderLayeredTilesParity decodes a full-HD two-layer stream and
// compares every frame against the encoder reconstruction. Layered frames
// carry per-layer quantizers, and multi-tile frames split the quantizer
// plumbing per tile column - a tile coded at the wrong layer's quantizer
// reconstructs differently in the decoder, which only a wide, layered,
// multi-tile stream exercises.
func TestVideoEncoderLayeredTilesParity(t *testing.T) {
	const w, h, n = 1920, 1080, 6
	rng := rand.New(rand.NewSource(3))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = byte(40 + rng.Intn(120))
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 6_000_000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	var recons []encoder.SourceFrame420
	for i := 0; i < n; i++ {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		dx := i * 3
		for y := 0; y < h; y++ {
			copy(f.Y[y*w:y*w+w-dx], bg[y*w+dx:])
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 132
		}
		tu, _, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		tus = append(tus, append([]byte(nil), tu...))
		rc := enc.Recon()
		recons = append(recons, encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
		})
	}
	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if f.Y.Pix[y*f.Y.Stride+x] != recons[i].Y[y*w+x] {
						t.Fatalf("frame %d differs at (%d,%d)", i, x, y)
					}
				}
			}
			i++
		}
	}
	if i != n {
		t.Fatalf("decoded %d frames, want %d", i, n)
	}
}
