package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestCDEFSearchMultiStrengthSignalsBitExact forces every searched picture
// (the keyframe and the base-layer inter frames) to signal four CDEF
// strengths with per-64x64 indices, so the tile symbol-log replay splices
// cdef_idx literals into every coded filter block. Every decoded frame must
// equal the encoder's reconstruction, which applied the same indices, for
// both the single-thread fused walk and the multi-thread wavefront.
func TestCDEFSearchMultiStrengthSignalsBitExact(t *testing.T) {
	const w, h = 256, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(7))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(40 + (i%w)/3 + rng.Intn(90))
	}
	makeFrame := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = uint8(100 + (i % cw))
			f.V[i] = uint8(150 - (i % cw))
		}
		sx, sy := 20+n*5, 30+n*3
		for y := sy; y < sy+40 && y < h; y++ {
			for x := sx; x < sx+40 && x < w; x++ {
				f.Y[y*w+x] = uint8(200 + (x+y)%40)
			}
		}
		return f
	}
	forced := 0
	encoder.SetCDEFDecisionTestHook(func(bits *uint8, y, uv *[encoder.MaxCDEFStrengths]uint8, fbIdx []uint8) {
		*bits = 2
		*y = [encoder.MaxCDEFStrengths]uint8{62, 0, 28, 7}
		*uv = [encoder.MaxCDEFStrengths]uint8{60, 2, 0, 33}
		for i := range fbIdx {
			fbIdx[i] = uint8((i*3 + 1) % 4)
		}
		forced++
	})
	t.Cleanup(func() { encoder.SetCDEFDecisionTestHook(nil) })
	// Screen content runs cdef level 7, where the middle layer searches
	// with search_best_ref_fs: the only non-base picture that records.
	for _, cfg := range []struct {
		threads int
		screen  bool
	}{{1, false}, {4, false}, {1, true}, {4, true}} {
		threads := cfg.threads
		t.Run(fmt.Sprintf("threads%d_screen%v", threads, cfg.screen), func(t *testing.T) {
			enc, err := encoder.NewVideoEncoder(w, h, 60)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.screen {
				enc.SetScreenContentSelection(true)
				if err := enc.SetContentHint(encoder.ContentScreen); err != nil {
					t.Fatal(err)
				}
			}
			t.Cleanup(func() { _ = enc.Close() })
			enc.SetMaxThreads(threads)
			if err := enc.SetTemporalLayers(3); err != nil {
				t.Fatal(err)
			}
			const frames = 9
			tus := make([][]byte, 0, frames)
			recons := make([]encoder.SourceFrame420, 0, frames)
			before := forced
			for i := range frames {
				tu, _, err := enc.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("encode frame %d: %v", i, err)
				}
				tus = append(tus, append([]byte(nil), tu...))
				recons = append(recons, cloneFrame(enc.Recon()))
			}
			if forced == before {
				t.Fatal("no picture ran the CDEF search")
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
