package encoder_test

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderFastPan proves the hierarchical coarse search: a 16px/frame
// pan is beyond the +-8px refinement window, so without the quarter-res seeds
// motion compensation collapses to intra coding. With them the pan tracks,
// inter frames stay small, and the whole chain still decodes bit-exact.
func TestVideoEncoderFastPan(t *testing.T) {
	const w, h = 320, 192
	rng := rand.New(rand.NewSource(11))
	wide := make([]byte, (w+256)*h)
	for y := range h {
		for x := 0; x < w+256; x++ {
			wide[y*(w+256)+x] = uint8(70 + (x/6+y/6)%70 + rng.Intn(40))
		}
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		off := t * 16 // 16px/frame pan, double the refinement window
		for y := range h {
			copy(f.Y[y*w:(y+1)*w], wide[y*(w+256)+off:])
		}
		for i := range f.U {
			f.U[i], f.V[i] = 118, 134
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 120)
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	var recons []encoder.SourceFrame420
	var keyBytes, interBytes int
	for i := range 6 {
		tu, key, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if key {
			keyBytes = len(tu)
		} else {
			interBytes += len(tu)
		}
		tus = append(tus, tu)
		rc := enc.Recon()
		cp := encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
			YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: rc.Width, Height: rc.Height,
		}
		recons = append(recons, cp)
	}
	avgInter := interBytes / 5
	t.Logf("fast pan: key %d bytes, avg inter %d bytes", keyBytes, avgInter)
	// Tracked motion codes each pan frame at a small fraction of the key;
	// untracked motion re-codes the texture as intra at near-key cost.
	if avgInter*3 >= keyBytes {
		t.Fatalf("fast pan not tracked: avg inter %d vs key %d", avgInter, keyBytes)
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
			var sse float64
			for y := range h {
				for x := range w {
					d := float64(f.Y.Pix[y*f.Y.Stride+x]) - float64(recons[i].Y[y*w+x])
					sse += d * d
				}
			}
			if sse != 0 {
				t.Fatalf("frame %d decode differs from encoder recon (sse %.0f)", i, sse)
			}
			i++
		}
	}
	if i != len(tus) {
		t.Fatalf("decoded %d frames, want %d", i, len(tus))
	}
}
