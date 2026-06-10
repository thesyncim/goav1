package encoder_test

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderSceneCutAutoKey proves the hierarchical search doubles as a
// scene-cut detector: a hard content change mid-stream restarts the chain
// with an automatic keyframe, slow pans never trigger one, and the whole
// stream still decodes bit-exact against the encoder reconstructions.
func TestVideoEncoderSceneCutAutoKey(t *testing.T) {
	const w, h = 320, 192
	sceneA := make([]byte, (w+64)*h)
	sceneB := make([]byte, w*h)
	rngA := rand.New(rand.NewSource(21))
	rngB := rand.New(rand.NewSource(22))
	for y := range h {
		for x := 0; x < w+64; x++ {
			sceneA[y*(w+64)+x] = uint8(60 + (x/5+y/7)%60 + rngA.Intn(40))
		}
	}
	for i := range sceneB {
		sceneB[i] = uint8(140 + (i/9)%50 + rngB.Intn(50))
	}
	makeFrame := func(idx int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		if idx < 6 {
			// Scene A panning 6px/frame.
			off := idx * 6
			for y := range h {
				copy(f.Y[y*w:(y+1)*w], sceneA[y*(w+64)+off:])
			}
		} else {
			copy(f.Y, sceneB)
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 128
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 110)
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	var recons []encoder.SourceFrame420
	var keys []bool
	for i := range 9 {
		tu, key, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		tus = append(tus, tu)
		keys = append(keys, key)
		rc := enc.Recon()
		recons = append(recons, encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
			YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: rc.Width, Height: rc.Height,
		})
	}
	t.Logf("keys: %v", keys)
	if !keys[0] {
		t.Fatal("frame 0 must be a keyframe")
	}
	for i := 1; i < 6; i++ {
		if keys[i] {
			t.Fatalf("spurious keyframe on pan frame %d", i)
		}
	}
	if !keys[6] {
		t.Fatal("scene cut at frame 6 did not trigger a keyframe")
	}
	for i := 7; i < 9; i++ {
		if keys[i] {
			t.Fatalf("spurious keyframe after the cut at frame %d", i)
		}
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
			for y := range h {
				for x := range w {
					if f.Y.Pix[y*f.Y.Stride+x] != recons[i].Y[y*w+x] {
						t.Fatalf("frame %d differs at (%d,%d)", i, x, y)
					}
				}
			}
			i++
		}
	}
	if i != len(tus) {
		t.Fatalf("decoded %d frames, want %d", i, len(tus))
	}
}
