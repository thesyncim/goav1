package encoder_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestWebRTCStreamL1T3Metadata is the scalable-packaging gate for the
// three-layer pattern: T0/T2/T1/T2, where the trailing T2 must depend on the
// middle layer instead of the older T0 frame.
func TestWebRTCStreamL1T3Metadata(t *testing.T) {
	const w, h = 128, 64
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(31))
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
			f.Y[i] = uint8(70 + (i/11+t*3)%70 + rng.Intn(20))
		}
		for i := range f.U {
			f.U[i] = 119
			f.V[i] = 131
		}
		sx := 8 + t*5
		for y := 12; y < 36; y++ {
			for x := sx; x < sx+16 && x < w; x++ {
				f.Y[y*w+x] = 212
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStreamLayers(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 180_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 3)
	if err != nil {
		t.Fatal(err)
	}

	const frames = 9
	wantTIDs := []uint8{0, 2, 1, 2, 0, 2, 1, 2, 0}
	var tus [][]byte
	lastT0 := uint64(0)
	lastT1 := uint64(0)
	haveT1 := false
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if ef.Info.TemporalID != wantTIDs[i] {
			t.Fatalf("frame %d temporal id %d want %d", i, ef.Info.TemporalID, wantTIDs[i])
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
		} else {
			wantDep := lastT0
			if wantTIDs[i] == 2 && i%4 == 3 {
				if !haveT1 {
					t.Fatalf("frame %d is trailing T2 before a middle-layer frame", i)
				}
				wantDep = lastT1
			}
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != wantDep {
				t.Fatalf("frame %d depends on %v (num %d), want %d", i, ef.Info.Dependencies, ef.Info.DependencyNum, wantDep)
			}
		}
		if len(ef.Descriptor) == 0 {
			t.Fatalf("frame %d missing descriptor", i)
		}
		switch wantTIDs[i] {
		case 0:
			lastT0 = ef.Info.FrameID
		case 1:
			lastT1 = ef.Info.FrameID
			haveT1 = true
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
