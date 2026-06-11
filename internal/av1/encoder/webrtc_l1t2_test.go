package encoder_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestWebRTCStreamL1T2Metadata is the scalable-packaging gate: an L1T2 WebRTC
// stream must mark odd frames as temporal layer 1 with the correct dependency
// chains (T1 and T0 frames both depending on the latest layer-0 frame, never
// on a T1 frame), produce descriptors for every frame, and emit temporal
// units that decode end to end.
func TestWebRTCStreamL1T2Metadata(t *testing.T) {
	const w, h = 128, 64
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(23))
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(80 + rng.Intn(40))
		}
		for i := range f.U {
			f.U[i] = 119
			f.V[i] = 131
		}
		sx := 8 + t*6
		for y := 12; y < 36; y++ {
			for x := sx; x < sx+16 && x < w; x++ {
				f.Y[y*w+x] = 212
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStreamLayers(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 150_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}

	const frames = 7
	tus := make([][]byte, 0, frames)
	lastT0 := uint64(0)
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		wantTID := uint8(0)
		if i%2 == 1 {
			wantTID = 1
		}
		if ef.Info.TemporalID != wantTID {
			t.Fatalf("frame %d temporal id %d want %d", i, ef.Info.TemporalID, wantTID)
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
		} else {
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != lastT0 {
				t.Fatalf("frame %d depends on %v (num %d), want %d", i, ef.Info.Dependencies, ef.Info.DependencyNum, lastT0)
			}
		}
		if len(ef.Descriptor) == 0 {
			t.Fatalf("frame %d missing descriptor", i)
		}
		if wantTID == 0 {
			lastT0 = ef.Info.FrameID
		}
		tus = append(tus, append([]byte(nil), ef.TU...))
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
}
