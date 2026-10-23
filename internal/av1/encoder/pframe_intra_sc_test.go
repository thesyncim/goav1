package encoder_test

import (
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestScreenContentIntraCandidatesDecodeExact encodes a scrolling text-like
// picture as screen content, where newly revealed rows of hard-edged
// stripes favour SVT's V/H/SMOOTH intra candidates in the inter frames, and
// checks every frame decodes bit-exact against the encoder reconstruction,
// serial and on the SB-row wavefront.
func TestScreenContentIntraCandidatesDecodeExact(t *testing.T) {
	const w, h = 320, 192
	page := make([]byte, w*(h+160))
	for y := range h + 160 {
		for x := range w {
			v := byte(245)
			switch {
			case (y/6)%5 == 0:
				v = 30 // horizontal rules
			case (x/7)%9 == 0 && (y/24)%2 == 0:
				v = 60 // vertical rules
			case (x+y/3)%11 < 2 && (y/12)%3 != 0:
				v = 90
			}
			page[y*w+x] = v
		}
	}
	frame := func(i int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		off := (i * 13) % 160
		copy(f.Y, page[off*w:(off+h)*w])
		for y := range h / 2 {
			for x := range w / 2 {
				f.U[y*w/2+x] = 128 + byte((x/9)%3)*20
				f.V[y*w/2+x] = 128 - byte((y/7)%2)*30
			}
		}
		return f
	}
	for _, threads := range []int{1, 4} {
		enc, err := encoder.NewVideoEncoder(w, h, 120)
		if err != nil {
			t.Fatal(err)
		}
		if err := enc.SetContentHint(encoder.ContentScreen); err != nil {
			t.Fatal(err)
		}
		enc.SetMaxThreads(threads)
		var tus [][]byte
		var recons [][]byte
		for i := range 10 {
			tu, _, err := enc.Encode(frame(i), false)
			if err != nil {
				t.Fatalf("threads %d frame %d: %v", threads, i, err)
			}
			tus = append(tus, append([]byte(nil), tu...))
			rc := enc.Recon()
			recons = append(recons, append(append(append([]byte(nil), rc.Y...), rc.U...), rc.V...))
		}
		_ = enc.Close()
		dec, err := goav1.NewDecoder(tus)
		if err != nil {
			t.Fatal(err)
		}
		i := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("threads %d decode: %v", threads, err)
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
			t.Fatalf("threads %d decoded %d frames, want %d", threads, i, len(tus))
		}
	}
}
