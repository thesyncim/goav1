package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderOddDimensions proves frames whose dimensions are not
// multiples of eight encode through the padded coded size with render_size
// carrying the true dimensions: the chain decodes bit-exact over the visible
// region and the streams stay legal for the reference decoders (the TUs ride
// the same cross-check corpus shape as every other oracle test).
func TestVideoEncoderOddDimensions(t *testing.T) {
	for _, dims := range [][2]int{{202, 118}, {158, 94}, {100, 62}} {
		w, h := dims[0], dims[1]
		t.Run(fmtDims(w, h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(w*1000 + h)))
			cw, ch := (w+1)/2, (h+1)/2
			makeFrame := func(idx int) encoder.SourceFrame420 {
				f := encoder.SourceFrame420{
					Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
					YStride: w, ChromaStride: cw, Width: w, Height: h,
				}
				for y := range h {
					for x := range w {
						f.Y[y*w+x] = uint8(60 + (x/5+y/5+idx*3)%70 + rng.Intn(20))
					}
				}
				for i := range f.U {
					f.U[i], f.V[i] = 115, 135
				}
				return f
			}
			enc, err := encoder.NewVideoEncoder(w, h, 120)
			if err != nil {
				t.Fatal(err)
			}
			var tus [][]byte
			var recons []encoder.SourceFrame420
			for i := range 4 {
				tu, _, err := enc.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("frame %d: %v", i, err)
				}
				tus = append(tus, tu)
				rc := enc.Recon()
				if rc.Width != w || rc.Height != h {
					t.Fatalf("recon view %dx%d, want %dx%d", rc.Width, rc.Height, w, h)
				}
				recons = append(recons, encoder.SourceFrame420{
					Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
					YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: w, Height: h,
				})
			}
			dec, err := goav1.NewDecoder(tus)
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			codedW, codedH := (w+7)&^7, (h+7)&^7
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
					if f.Y.Width != codedW || f.Y.Height != codedH {
						t.Fatalf("decoded %dx%d, want coded %dx%d", f.Y.Width, f.Y.Height, codedW, codedH)
					}
					for y := range h {
						for x := range w {
							if f.Y.Pix[y*f.Y.Stride+x] != recons[i].Y[y*recons[i].YStride+x] {
								t.Fatalf("frame %d differs at (%d,%d)", i, x, y)
							}
						}
					}
					i++
				}
			}
			if i != len(tus) {
				t.Fatalf("decoded %d frames, want %d", i, len(tus))
			}
		})
	}
}

func fmtDims(w, h int) string {
	return fmt.Sprintf("%dx%d", w, h)
}
