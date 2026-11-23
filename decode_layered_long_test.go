package goav1

import (
	"testing"
)

// Droppable layer frames refresh no reference slot, so their surfaces are
// owned by nothing once shown; sixty frames overflow a pool that leaks them
// (the public decoder previously died at frame fourteen).
func TestLongL1T2DecodeNoPoolLeak(t *testing.T) {
	const w, h, n = 320, 192, 60
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 500_000, Framerate: 30, TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	f := I420Frame{
		Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
	for i := 0; i < n; i++ {
		for j := range f.Y {
			f.Y[j] = byte(40 + (j+i*7)%160)
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatal(err)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}
	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", frames, err)
		}
		if !ok {
			break
		}
		frames += len(batch)
	}
	if frames != n {
		t.Fatalf("decoded %d frames, want %d", frames, n)
	}
}
