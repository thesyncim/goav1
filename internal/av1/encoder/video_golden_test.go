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
