package encoder

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestVideoEncoderOverlapDeterminism proves the backgrounded in-loop filter
// pass cannot change the emitted stream: an encoder whose reconstruction is
// polled after every frame (forcing the filter join before the next encode
// begins) and one that is never polled (maximal overlap) must produce
// byte-identical temporal units, across P-frames, a mid-stream keyframe,
// and golden refreshes.
func TestVideoEncoderOverlapDeterminism(t *testing.T) {
	const w, h, n = 320, 192, 24
	rng := rand.New(rand.NewSource(7))
	frames := make([]SourceFrame420, n)
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = byte(rng.Intn(256))
	}
	for i := range frames {
		f := SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		dx := (i * 2) % 32
		for y := 0; y < h; y++ {
			copy(f.Y[y*w:y*w+w-dx], bg[y*w+dx:])
		}
		for j := range f.U {
			f.U[j] = 110
			f.V[j] = 140
		}
		frames[i] = f
	}
	newEnc := func() *VideoEncoder {
		e, err := NewVideoEncoder(w, h, 160)
		if err != nil {
			t.Fatal(err)
		}
		e.SetGoldenInterval(8)
		return e
	}
	polled, free := newEnc(), newEnc()
	for i, f := range frames {
		forceKey := i == 13
		tuA, keyA, errA := polled.Encode(f, forceKey)
		// Polling the reconstruction joins the filter pass immediately.
		ra := polled.Recon()
		if ra.Y == nil {
			t.Fatalf("frame %d: polled recon unavailable", i)
		}
		tuB, keyB, errB := free.Encode(f, forceKey)
		if errA != nil || errB != nil {
			t.Fatalf("frame %d: encode errors %v / %v", i, errA, errB)
		}
		if keyA != keyB || !bytes.Equal(tuA, tuB) {
			t.Fatalf("frame %d: temporal units diverge (key %v/%v, %d/%d bytes)", i, keyA, keyB, len(tuA), len(tuB))
		}
	}
	// Both end states expose identical reconstructions.
	ra, rb := polled.Recon(), free.Recon()
	if !bytes.Equal(ra.Y, rb.Y) || !bytes.Equal(ra.U, rb.U) || !bytes.Equal(ra.V, rb.V) {
		t.Fatal("final reconstructions diverge")
	}
}
