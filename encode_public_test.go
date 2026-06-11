package goav1_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
)

// TestPublicVideoEncoderRoundTrip drives the public encoding surface end to
// end: CBR with two temporal layers, a mid-stream forced keyframe, and decode
// through the public Decoder with every frame bit-exact against the encoder
// reconstruction.
func TestPublicVideoEncoderRoundTrip(t *testing.T) {
	const w, h = 320, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(60 + rng.Intn(80))
	}
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		for y := 20 + n*3; y < 52+n*3 && y < h; y++ {
			for x := 16 + n*5; x < 48+n*5 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 400_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	const frames = 12
	var tus [][]byte
	type snap struct {
		y, u, v []byte
	}
	var recons []snap
	keyAt := map[int]bool{}
	for i := range frames {
		out, err := enc.Encode(makeFrame(i), i == 6) // force a mid-stream keyframe
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if out.Keyframe {
			keyAt[i] = true
		}
		if i >= 1 && i < 6 && i%2 == 1 && out.TemporalID != 1 {
			t.Fatalf("frame %d temporal id %d, want 1", i, out.TemporalID)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
		r := enc.Reconstruction()
		recons = append(recons, snap{
			y: append([]byte(nil), r.Y...),
			u: append([]byte(nil), r.U...),
			v: append([]byte(nil), r.V...),
		})
	}
	if !keyAt[0] || !keyAt[6] {
		t.Fatalf("keyframes at %v, want frames 0 and 6", keyAt)
	}
	if enc.QIndex() == 0 {
		t.Fatal("rate controller reported qindex 0")
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
			t.Fatalf("decode %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			for y := range h {
				for x := range w {
					if f.Y.Pix[y*f.Y.Stride+x] != recons[i].y[y*w+x] {
						t.Fatalf("frame %d Y mismatch at (%d,%d)", i, x, y)
					}
				}
			}
			for y := range ch {
				for x := range cw {
					if f.U.Pix[y*f.U.Stride+x] != recons[i].u[y*cw+x] || f.V.Pix[y*f.V.Stride+x] != recons[i].v[y*cw+x] {
						t.Fatalf("frame %d chroma mismatch at (%d,%d)", i, x, y)
					}
				}
			}
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}

// TestPublicRTCEncoder checks the WebRTC surface: every frame carries a
// dependency descriptor and the temporal units decode.
func TestPublicRTCEncoder(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	for i := range 6 {
		f := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for j := range f.Y {
			f.Y[j] = uint8(40 + (j+i*7)%150)
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if len(out.DependencyDescriptor) == 0 {
			t.Fatalf("frame %d missing dependency descriptor", i)
		}
		if i == 0 && !out.Keyframe {
			t.Fatal("first frame not a keyframe")
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}
	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	n := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		n += len(batch)
	}
	if n != 6 {
		t.Fatalf("decoded %d frames, want 6", n)
	}
}

func TestPublicRTCEncoderGoldenInterval(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	base := make([]byte, w*h)
	for i := range base {
		base[i] = uint8(50 + rng.Intn(170))
	}
	mk := func(boxX int) goav1.I420Frame {
		f := goav1.I420Frame{
			Y: append([]byte(nil), base...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		if boxX >= 0 {
			for y := 32; y < 96; y++ {
				for x := boxX; x < boxX+64 && x < w; x++ {
					f.Y[y*w+x] = 235
				}
			}
		}
		return f
	}
	frames := []goav1.I420Frame{
		mk(-1),
		mk(64),
		mk(-1),
	}
	encode := func(goldenInterval int) [][]byte {
		enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 2_000_000, Framerate: 30,
			MinQIndex:      40,
			MaxQIndex:      160,
			TemporalLayers: 1,
			GoldenInterval: goldenInterval,
		})
		if err != nil {
			t.Fatal(err)
		}
		var tus [][]byte
		for i, f := range frames {
			out, err := enc.Encode(f, false)
			if err != nil {
				t.Fatalf("encode %d: %v", i, err)
			}
			tus = append(tus, append([]byte(nil), out.Data...))
		}
		return tus
	}
	lastOnly := encode(-1)
	withGolden := encode(16)
	t.Logf("rtc reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
	if len(withGolden[2])*5 >= len(lastOnly[2])*4 {
		t.Fatalf("golden interval did not materially reduce reveal frame: last-only %dB, golden %dB", len(lastOnly[2]), len(withGolden[2]))
	}
}
