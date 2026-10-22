package encoder_test

import (
	"bytes"
	"fmt"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestLazyNonReferenceCDEFReconMatchesDecode: non-reference pictures (the
// top temporal layer) defer their CDEF pass until Recon(). Recon() of every
// frame must still equal the decoder's output, and skipping Recon() on some
// frames must leave the bitstream and every other frame's reconstruction
// unchanged (the deferred pass never feeds prediction).
func TestLazyNonReferenceCDEFReconMatchesDecode(t *testing.T) {
	const w, h, frames = 256, 192, 13
	cw, ch := w/2, h/2
	makeFrame := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f.Y[y*w+x] = uint8((x*3 + y*5 + (x*y)%37 + n*4) & 0xff)
			}
		}
		for i := range f.U {
			f.U[i] = uint8(90 + i%50)
			f.V[i] = uint8(160 - i%40)
		}
		return f
	}
	// Pin the searched pictures to a strong single strength so the
	// non-reference pictures inherit a nonzero reference-derived one.
	encoder.SetCDEFDecisionTestHook(func(bits *uint8, y, uv *[encoder.MaxCDEFStrengths]uint8, fbIdx []uint8) {
		*bits = 0
		*y = [encoder.MaxCDEFStrengths]uint8{62}
		*uv = [encoder.MaxCDEFStrengths]uint8{60}
		clear(fbIdx)
	})
	t.Cleanup(func() { encoder.SetCDEFDecisionTestHook(nil) })
	for _, threads := range []int{1, 4} {
		t.Run(fmt.Sprintf("threads%d", threads), func(t *testing.T) {
			run := func(every int) ([][]byte, map[int]encoder.SourceFrame420) {
				enc, err := encoder.NewVideoEncoder(w, h, 120)
				if err != nil {
					t.Fatal(err)
				}
				defer enc.Close()
				enc.SetMaxThreads(threads)
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatal(err)
				}
				var tus [][]byte
				recons := map[int]encoder.SourceFrame420{}
				for i := range frames {
					tu, _, err := enc.Encode(makeFrame(i), false)
					if err != nil {
						t.Fatal(err)
					}
					tus = append(tus, append([]byte(nil), tu...))
					if i%every == 0 {
						recons[i] = cloneFrame(enc.Recon())
					}
				}
				return tus, recons
			}
			tus, recons := run(1)
			tusSparse, reconsSparse := run(3)
			for i := range tus {
				if !bytes.Equal(tus[i], tusSparse[i]) {
					t.Fatalf("frame %d bitstream depends on Recon() calls", i)
				}
			}
			for i, r := range reconsSparse {
				if !bytes.Equal(r.Y, recons[i].Y) || !bytes.Equal(r.U, recons[i].U) || !bytes.Equal(r.V, recons[i].V) {
					t.Fatalf("frame %d reconstruction depends on earlier Recon() calls", i)
				}
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
		})
	}
}
