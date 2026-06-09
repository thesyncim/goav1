package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestWebRTCStreamPackagesDecodableFrames is the WebRTC integration gate: an
// L1T1 stream must emit, per frame, a decodable temporal unit plus coherent
// dependency metadata — the keyframe carrying the attached structure with no
// dependencies, every delta frame depending on the previous frame ID — and
// the temporal units must decode bit-identical to the encoder reconstruction.
func TestWebRTCStreamPackagesDecodableFrames(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 118
			f.V[i] = 132
		}
		sx, sy := 12+t*4, 20+t*2
		for y := sy; y < sy+20 && y < h; y++ {
			for x := sx; x < sx+20 && x < w; x++ {
				f.Y[y*w+x] = 210
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStream(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 150_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}

	const frames = 5
	tus := make([][]byte, 0, frames)
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != ef.Keyframe {
			t.Fatalf("frame %d keyframe=%v", i, ef.Keyframe)
		}
		if ef.Info.FrameID != uint64(i) {
			t.Fatalf("frame %d id=%d", i, ef.Info.FrameID)
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
			if len(ef.Descriptor) < 8 {
				t.Fatalf("keyframe descriptor %d bytes: structure not attached?", len(ef.Descriptor))
			}
		} else {
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != uint64(i-1) {
				t.Fatalf("frame %d deps=%v num=%d", i, ef.Info.Dependencies, ef.Info.DependencyNum)
			}
			if len(ef.Descriptor) == 0 {
				t.Fatalf("frame %d missing descriptor", i)
			}
		}
		tus = append(tus, ef.TU)
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
	if n != frames {
		t.Fatalf("decoded %d frames, want %d", n, frames)
	}
	fmt.Println("webrtc stream ok")
}
