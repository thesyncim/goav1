package encoder_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderGoldenReference is the multi-ref gate: an object occludes a
// textured region and then reveals it. The revealed pixels exist in the
// golden anchor (the keyframe) but not in the previous frame, so the reveal
// frame must get cheaper with golden references enabled, and every frame must
// decode bit-exact against the encoder reconstruction.
func TestVideoEncoderGoldenReference(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	base := make([]byte, w*h)
	for i := range base {
		base[i] = uint8(50 + rng.Intn(170))
	}
	mk := func(boxX int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
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
	frames := []encoder.SourceFrame420{
		mk(-1), // scene only (keyframe / golden anchor)
		mk(64), // box occludes the center
		mk(-1), // box gone: center revealed, identical to the anchor
	}

	encode := func(goldenInterval int) ([][]byte, []encoder.SourceFrame420) {
		enc, err := encoder.NewVideoEncoder(w, h, 60)
		if err != nil {
			t.Fatal(err)
		}
		enc.SetGoldenInterval(goldenInterval)
		var tus [][]byte
		var recons []encoder.SourceFrame420
		for i, f := range frames {
			tu, _, err := enc.Encode(f, false)
			if err != nil {
				t.Fatalf("encode frame %d: %v", i, err)
			}
			tus = append(tus, append([]byte(nil), tu...))
			recons = append(recons, cloneFrame(enc.Recon()))
		}
		return tus, recons
	}

	lastOnly, _ := encode(0)
	withGolden, recons := encode(32)
	t.Logf("reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
	// The revealed region must get materially cheaper (well under 60% of the
	// last-only cost; the exact ratio drifts with unrelated recon changes).
	if len(withGolden[2])*5 >= len(lastOnly[2])*3 {
		t.Fatalf("golden reveal %dB not well below last-only %dB: golden reference not engaged", len(withGolden[2]), len(lastOnly[2]))
	}

	dec, err := goav1.NewDecoder(withGolden)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
			comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != len(frames) {
		t.Fatalf("decoded %d frames, want %d", i, len(frames))
	}
}

func TestVideoEncoderCompoundGoldenAverageReference(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	mk := func(seed int64) encoder.SourceFrame420 {
		rng := rand.New(rand.NewSource(seed))
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
			f.Y[i] = uint8(16 + rng.Intn(224))
		}
		for i := range f.U {
			f.U[i] = uint8(32 + rng.Intn(192))
			f.V[i] = uint8(32 + rng.Intn(192))
		}
		return f
	}
	blockShift := func(x, y int) (int, int) {
		dx, dy := 2, 2
		if (x/8)&1 != 0 {
			dx = -2
		}
		if (y/8)&1 != 0 {
			dy = -2
		}
		if x+dx < 0 || x+dx+8 > w {
			dx = -dx
		}
		if y+dy < 0 || y+dy+8 > h {
			dy = -dy
		}
		return dx, dy
	}
	avgShifted := func(a, b encoder.SourceFrame420) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, len(a.Y)),
			U:            make([]byte, len(a.U)),
			V:            make([]byte, len(a.V)),
			YStride:      a.YStride,
			ChromaStride: a.ChromaStride,
			Width:        a.Width,
			Height:       a.Height,
		}
		for by := 0; by < h; by += 8 {
			for bx := 0; bx < w; bx += 8 {
				dx, dy := blockShift(bx, by)
				for y := 0; y < 8; y++ {
					dst := (by+y)*w + bx
					aOff := (by+y)*a.YStride + bx
					bOff := (by+dy+y)*b.YStride + bx + dx
					for x := 0; x < 8; x++ {
						f.Y[dst+x] = uint8((int(a.Y[aOff+x]) + int(b.Y[bOff+x]) + 1) >> 1)
					}
				}
				cx, cy := bx/2, by/2
				cdx, cdy := dx/2, dy/2
				for y := 0; y < 4; y++ {
					dst := (cy+y)*cw + cx
					aOff := (cy+y)*a.ChromaStride + cx
					bOff := (cy+cdy+y)*b.ChromaStride + cx + cdx
					for x := 0; x < 4; x++ {
						f.U[dst+x] = uint8((int(a.U[aOff+x]) + int(b.U[bOff+x]) + 1) >> 1)
						f.V[dst+x] = uint8((int(a.V[aOff+x]) + int(b.V[bOff+x]) + 1) >> 1)
					}
				}
			}
		}
		return f
	}

	first, second := mk(101), mk(202)
	withGoldenEnc, err := encoder.NewVideoEncoder(w, h, 72)
	if err != nil {
		t.Fatal(err)
	}
	withGoldenEnc.SetGoldenInterval(32)
	tu0, _, err := withGoldenEnc.Encode(first, false)
	if err != nil {
		t.Fatalf("encode key: %v", err)
	}
	tu0 = append([]byte(nil), tu0...)
	recon0 := cloneFrame(withGoldenEnc.Recon())
	tu1, _, err := withGoldenEnc.Encode(second, false)
	if err != nil {
		t.Fatalf("encode last: %v", err)
	}
	tu1 = append([]byte(nil), tu1...)
	recon1 := cloneFrame(withGoldenEnc.Recon())
	average := avgShifted(recon0, recon1)
	tu2, _, err := withGoldenEnc.Encode(average, false)
	if err != nil {
		t.Fatalf("encode compound average: %v", err)
	}
	tu2 = append([]byte(nil), tu2...)
	recon2 := cloneFrame(withGoldenEnc.Recon())
	withGolden := [][]byte{tu0, tu1, tu2}

	lastOnlyEnc, err := encoder.NewVideoEncoder(w, h, 72)
	if err != nil {
		t.Fatal(err)
	}
	lastOnlyEnc.SetGoldenInterval(0)
	var lastOnly [][]byte
	for i, f := range []encoder.SourceFrame420{first, second, average} {
		tu, _, err := lastOnlyEnc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode last-only frame %d: %v", i, err)
		}
		lastOnly = append(lastOnly, append([]byte(nil), tu...))
	}
	t.Logf("average frame: last-only %dB, with compound %dB", len(lastOnly[2]), len(withGolden[2]))
	if len(withGolden[2])+128 >= len(lastOnly[2]) {
		t.Fatalf("compound average %dB not well below last-only %dB", len(withGolden[2]), len(lastOnly[2]))
	}

	dec, err := goav1.NewDecoder(withGolden)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	recons := []encoder.SourceFrame420{recon0, recon1, recon2}
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
			comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != len(recons) {
		t.Fatalf("decoded %d frames, want %d", i, len(recons))
	}
}
