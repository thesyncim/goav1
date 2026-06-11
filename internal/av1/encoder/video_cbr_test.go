package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderCBRConvergesAndDecodes is the rate-control gate: a 30-frame
// moving sequence under CBR must (a) decode with every frame bit-identical to
// the encoder reconstruction — now with a different qindex per frame, the
// per-frame header/CDF re-initialization is exercised for real — (b) move the
// working qindex (the controller is alive), and (c) land the steady-state
// bitrate within 30% of target.
func TestVideoEncoderCBRConvergesAndDecodes(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(7))

	bg := make([]byte, w*h)
	for y := range h {
		for x := range w {
			bg[y*w+x] = uint8(60 + (x/4+y/4)%64 + rng.Intn(50))
		}
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
			f.U[i] = 120
			f.V[i] = 130
		}
		sx, sy := 8+t*4, 16+t*2
		for y := sy; y < sy+24 && y < h; y++ {
			for x := sx; x < sx+24 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}

	const fps = 30
	const targetBPS = 120_000 // 500 bytes/frame at 30 fps
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: targetBPS,
		FramesPerSecond:     fps,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}

	// The boosted keyframe repays its debt over the controller's 24-frame
	// memory; steady state is measured after that horizon.
	const frames = 80
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	qSeen := map[uint8]bool{}
	steadyBits := 0
	const warmup = 40
	for i := range frames {
		qSeen[enc.QIndex()] = true
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if i >= warmup {
			steadyBits += len(tu) * 8
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
	}
	if len(qSeen) < 3 {
		t.Fatalf("rate controller never moved qindex: %v", qSeen)
	}
	steadyAvg := steadyBits / (frames - warmup)
	target := targetBPS / fps
	t.Logf("steady-state avg %d bits/frame (target %d), distinct qindexes %d", steadyAvg, target, len(qSeen))
	if steadyAvg < target*70/100 || steadyAvg > target*130/100 {
		t.Fatalf("steady-state %d bits/frame outside 30%% of target %d", steadyAvg, target)
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
			comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
			comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}
