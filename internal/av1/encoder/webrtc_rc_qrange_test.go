package encoder

import (
	"math/rand"
	"testing"
)

// TestWebRTCStreamQRangeReachesSVTRC: a stream's MinQIndex/MaxQIndex must
// bound the qindex SVT's rate control picks (best/worst quality), not only
// the encoder's legacy rcMinQ/rcMaxQ fields.
func TestWebRTCStreamQRangeReachesSVTRC(t *testing.T) {
	const w, h = 192, 128
	const minQ, maxQ = 120, 140
	stream, err := NewWebRTCStream(w, h, RateControlConfig{
		TargetBitsPerSecond: 5_000_000, // far more than needed: q would fall to the floor
		FramesPerSecond:     30,
		MinQIndex:           minQ,
		MaxQIndex:           maxQ,
	})
	if err != nil {
		t.Fatal(err)
	}
	enc := stream.encoders[0]
	if enc == nil {
		t.Fatal("no 8-bit encoder")
	}
	if got := [2]int{enc.svtRC.bestQuality, enc.svtRC.worstQuality}; got != [2]int{minQ, maxQ} {
		t.Fatalf("svtRC q range = %v, want [%d %d]", got, minQ, maxQ)
	}
	rng := rand.New(rand.NewSource(1))
	cw, ch := w/2, h/2
	for i := range 8 {
		f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch), YStride: w, ChromaStride: cw, Width: w, Height: h}
		for j := range f.Y {
			f.Y[j] = byte(rng.Intn(256))
		}
		if _, err := stream.Encode(f, false); err != nil {
			t.Fatal(err)
		}
		if q := enc.svtCurQ; q < minQ || q > maxQ {
			t.Fatalf("frame %d qindex %d outside [%d, %d]", i, q, minQ, maxQ)
		}
	}
}
