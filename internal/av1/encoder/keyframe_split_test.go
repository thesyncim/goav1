package encoder_test

import (
	"bytes"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// keyframeWavefrontSource is a detailed 4:2:0 picture: smooth gradients,
// noise and hard-edged text-like strokes, so the keyframe exercises the merge
// pricing, the directional search and (as screen content) the palette.
func keyframeWavefrontSource(w, h int, seed int64) encoder.SourceFrame420 {
	rng := rand.New(rand.NewSource(seed))
	f := encoder.SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
	for y := range h {
		for x := range w {
			v := (x*3+y*2)/5 + rng.Intn(24)
			switch {
			case (y/9)%7 == 0 && x%13 < 9:
				v = 20
			case (x/11)%6 == 0 && (y/17)%2 == 0:
				v = 235
			}
			f.Y[y*w+x] = byte(min(max(v, 0), 255))
		}
	}
	for y := range h / 2 {
		for x := range w / 2 {
			f.U[y*w/2+x] = byte(96 + (x/7)%5*12 + rng.Intn(6))
			f.V[y*w/2+x] = byte(150 - (y/5)%4*14 + rng.Intn(6))
		}
	}
	return f
}

// TestKeyframeWavefrontMatchesSerial checks the single-tile keyframe's SB-row
// wavefront (keyframe_split.go) reproduces the serial decision pass byte for
// byte at every thread count and run, for camera and screen content, and
// that the stream decodes to the encoder's reconstruction.
func TestKeyframeWavefrontMatchesSerial(t *testing.T) {
	const w, h = 448, 320
	for _, content := range []encoder.ContentHint{encoder.ContentCamera, encoder.ContentScreen} {
		var want [][]byte
		for _, threads := range []int{1, 2, 4, 8, 3} {
			for run := 0; run < 2; run++ {
				enc, err := encoder.NewVideoEncoder(w, h, 110)
				if err != nil {
					t.Fatal(err)
				}
				if err := enc.SetContentHint(content); err != nil {
					t.Fatal(err)
				}
				enc.SetMaxThreads(threads)
				var tus [][]byte
				var recons [][]byte
				for i := range 3 {
					tu, _, err := enc.Encode(keyframeWavefrontSource(w, h, int64(i)), true)
					if err != nil {
						t.Fatalf("content %d threads %d frame %d: %v", content, threads, i, err)
					}
					tus = append(tus, append([]byte(nil), tu...))
					rc := enc.Recon()
					recons = append(recons, append(append(append([]byte(nil), rc.Y...), rc.U...), rc.V...))
				}
				_ = enc.Close()
				if want == nil {
					want = tus
					dec, err := goav1.NewDecoder(tus)
					if err != nil {
						t.Fatal(err)
					}
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
							comparePlane(t, "Y", f.Y, recons[i][:w*h], w, h, w)
							comparePlane(t, "U", f.U, recons[i][w*h:w*h+w*h/4], w/2, h/2, w/2)
							comparePlane(t, "V", f.V, recons[i][w*h+w*h/4:], w/2, h/2, w/2)
							i++
						}
					}
					dec.Close()
					if i != len(tus) {
						t.Fatalf("decoded %d frames, want %d", i, len(tus))
					}
					continue
				}
				for i := range tus {
					if !bytes.Equal(tus[i], want[i]) {
						t.Fatalf("content %d threads %d run %d keyframe %d: %d bytes differ from serial %d bytes", content, threads, run, i, len(tus[i]), len(want[i]))
					}
				}
			}
		}
	}
}
