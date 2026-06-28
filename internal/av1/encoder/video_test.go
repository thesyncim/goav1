package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderChainDecodesBitExact is the streaming gate: a KEY + 5-P
// chain of a moving scene must decode with EVERY frame bit-identical to the
// encoder's per-frame reconstruction, fidelity above a sanity floor on each
// frame, and the inter frames a fraction of the keyframe — the full encoder
// loop (reference chaining, motion search, skip decisions) working end to end.
func TestVideoEncoderChainDecodesBitExact(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(99))

	// A textured background with a moving bright square: most blocks skip,
	// the square's blocks track motion, and occlusion edges code residuals.
	bg := make([]byte, w*h)
	for y := range h {
		for x := range w {
			bg[y*w+x] = uint8(60 + (x/4+y/4)%64 + rng.Intn(60))
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
		// 24x24 bright square moving diagonally 4px right, 2px down per frame.
		sx, sy := 16+t*4, 24+t*2
		for y := sy; y < sy+24 && y < h; y++ {
			for x := sx; x < sx+24 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 60)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 6
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	keySize := 0
	interTotal := 0
	for i := range frames {
		tu, isKey, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != isKey {
			t.Fatalf("frame %d keyframe=%v", i, isKey)
		}
		if isKey {
			keySize = len(tu)
		} else {
			interTotal += len(tu)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
	}
	avgInter := interTotal / (frames - 1)
	t.Logf("key %d bytes, avg P %d bytes", keySize, avgInter)
	if avgInter*3 >= keySize {
		t.Fatalf("avg P %d bytes not well below key %d bytes", avgInter, keySize)
	}

	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	// Decoded frames alias pooled surfaces that later decodes recycle, so
	// compare each frame as it is produced rather than after DecodeAll.
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
			if i >= frames {
				t.Fatalf("decoded more than %d frames", frames)
			}
			comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
			comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			psnr := planePSNR(makeFrame(i).Y, recons[i].Y)
			t.Logf("frame %d luma PSNR %.2f dB", i, psnr)
			if psnr < 30 {
				t.Fatalf("frame %d luma PSNR %.2f below floor", i, psnr)
			}
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}

func TestVideoEncoder1080pHotPathAllocs(t *testing.T) {
	const w, h = 1920, 1080
	f0 := makeEncoder1080pFrame(0)
	f1 := makeEncoder1080pFrame(1)

	pEnc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000,
		FramesPerSecond:     60,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pEnc.Close() })
	if err := pEnc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := pEnc.Encode(f0, true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	pAllocs := testing.AllocsPerRun(5, func() {
		tu, key, err := pEnc.Encode(f1, false)
		if err != nil {
			t.Fatal(err)
		}
		if key {
			t.Fatal("steady P-frame was coded as keyframe")
		}
		if len(tu) == 0 {
			t.Fatal("empty P-frame temporal unit")
		}
	})
	if pAllocs != 0 {
		t.Fatalf("1080p steady P-frame allocations=%f want 0", pAllocs)
	}

	panFrames := makeEncoder1080pPanFrames(8)
	panEnc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000,
		FramesPerSecond:     60,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = panEnc.Close() })
	if err := panEnc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := panEnc.Encode(panFrames[0], true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial pan frame was not a keyframe")
	}
	panFrameIndex := 1
	panAllocs := testing.AllocsPerRun(5, func() {
		tu, key, err := panEnc.Encode(panFrames[panFrameIndex], false)
		panFrameIndex++
		if panFrameIndex == len(panFrames) {
			panFrameIndex = 1
		}
		if err != nil {
			t.Fatal(err)
		}
		if key {
			t.Fatal("steady pan P-frame was coded as keyframe")
		}
		if len(tu) == 0 {
			t.Fatal("empty pan P-frame temporal unit")
		}
	})
	if panAllocs != 0 {
		t.Fatalf("1080p pan P-frame allocations=%f want 0", panAllocs)
	}

	keyEnc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000,
		FramesPerSecond:     60,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = keyEnc.Close() })
	if err := keyEnc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := keyEnc.Encode(f0, true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	keyAllocs := testing.AllocsPerRun(5, func() {
		tu, key, err := keyEnc.Encode(f1, true)
		if err != nil {
			t.Fatal(err)
		}
		if !key {
			t.Fatal("forced frame was not a keyframe")
		}
		if len(tu) == 0 {
			t.Fatal("empty keyframe temporal unit")
		}
	})
	if keyAllocs != 0 {
		t.Fatalf("1080p forced keyframe allocations=%f want 0", keyAllocs)
	}
}

func makeEncoder1080pFrame(tick int) encoder.SourceFrame420 {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := 0; y < h; y++ {
		row := f.Y[y*w : (y+1)*w]
		for x := range row {
			row[x] = uint8(64 + (x/9+y/11+tick)%72)
		}
	}
	sx, sy := 240+tick*4, 360+tick*2
	for y := sy; y < sy+96; y++ {
		for x := sx; x < sx+96; x++ {
			f.Y[y*w+x] = 220
		}
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	return f
}

func makeEncoder1080pPanFrames(count int) []encoder.SourceFrame420 {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(15))
	wide := make([]byte, (w+512)*h)
	for y := range h {
		for x := 0; x < w+512; x++ {
			wide[y*(w+512)+x] = uint8(60 + (x/7+y/9)%70 + rng.Intn(25))
		}
	}
	for y := range h {
		row := wide[y*(w+512) : (y+1)*(w+512)]
		for x := 1; x < len(row)-1; x++ {
			row[x] = uint8((int(row[x-1]) + 2*int(row[x]) + int(row[x+1])) >> 2)
		}
	}
	frames := make([]encoder.SourceFrame420, count)
	for i := range frames {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		off := (i * 4) % 512
		for y := range h {
			copy(f.Y[y*w:(y+1)*w], wide[y*(w+512)+off:])
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		frames[i] = f
	}
	return frames
}

// cloneFrame deep-copies a reconstruction snapshot; the encoder ping-pongs its
// internal recon buffers, so Recon() contents are only stable until the
// next-but-one Encode call.
func cloneFrame(f encoder.SourceFrame420) encoder.SourceFrame420 {
	f.Y = append([]byte(nil), f.Y...)
	f.U = append([]byte(nil), f.U...)
	f.V = append([]byte(nil), f.V...)
	return f
}
