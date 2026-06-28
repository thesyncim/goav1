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

func TestVideoEncoderSingleThreadFiltersStayInline(t *testing.T) {
	const w, h = 320, 192
	makeFrame := func(tick int) SourceFrame420 {
		f := SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f.Y[y*w+x] = uint8(48 + (x/5+y/7+tick)%96)
			}
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 800_000,
		FramesPerSecond:     30,
		MinQIndex:           160,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	enc.SetMaxThreads(1)
	if err := enc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := enc.Encode(makeFrame(0), true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	if _, key, err := enc.Encode(makeFrame(1), false); err != nil {
		t.Fatal(err)
	} else if key {
		t.Fatal("P-frame was coded as keyframe")
	}
	if !enc.lf.bound || !enc.cdefApp.bound {
		t.Fatal("filters did not bind during single-thread encode")
	}
	if enc.filterStarted || enc.tileWorkers != 0 || enc.lf.started || enc.cdefApp.started || enc.hme.started {
		t.Fatalf("single-thread mode started workers: filter=%v tile=%d lf=%v cdef=%v hme=%v",
			enc.filterStarted, enc.tileWorkers, enc.lf.started, enc.cdefApp.started, enc.hme.started)
	}
}

func TestVideoEncoderCloseReleasesPersistentWorkers(t *testing.T) {
	const w, h = 320, 192
	f0 := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	f1 := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			f0.Y[y*w+x] = uint8(48 + (x/5+y/7)%96)
			f1.Y[y*w+x] = uint8(48 + ((x+3)/5+y/7)%96)
		}
	}
	for i := range f0.U {
		f0.U[i], f0.V[i] = 120, 130
		f1.U[i], f1.V[i] = 120, 130
	}
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 800_000,
		FramesPerSecond:     30,
		MinQIndex:           160,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, key, err := enc.Encode(f0, true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	if _, key, err := enc.Encode(f1, false); err != nil {
		t.Fatal(err)
	} else if key {
		t.Fatal("P-frame was coded as keyframe")
	}
	if err := enc.Flush(); err != nil {
		t.Fatal(err)
	}
	if !enc.hme.started || !enc.lf.started || !enc.cdefApp.started {
		t.Fatalf("expected persistent workers to start: hme=%v lf=%v cdef=%v", enc.hme.started, enc.lf.started, enc.cdefApp.started)
	}
	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
	if enc.hme.started || enc.lf.started || enc.cdefApp.started || enc.hme.work != nil || enc.lf.work != nil || enc.cdefApp.work != nil {
		t.Fatalf("close left workers active: hme=%v/%v lf=%v/%v cdef=%v/%v",
			enc.hme.started, enc.hme.work != nil,
			enc.lf.started, enc.lf.work != nil,
			enc.cdefApp.started, enc.cdefApp.work != nil)
	}
}
