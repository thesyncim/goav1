package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// TestVideoEncoderMDS0IntraDC pins the light-PD1 DC_PRED arm: after a cut
// from dense noise to a smooth scene the inter residual variance dwarfs the
// source variance, so whole 16x16/32x32 blocks code as DC intra inside the
// inter frame, and the stream must decode to the encoder reconstruction.
// TestPFrameIntraDCSplitMatchesFused covers the split and wavefront paths.
func TestVideoEncoderMDS0IntraDC(t *testing.T) {
	const w, h = 256, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	mk := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 130
		}
		for y := range h {
			for x := range w {
				if n == 0 {
					f.Y[y*w+x] = uint8(40 + rng.Intn(170))
				} else {
					f.Y[y*w+x] = uint8(60 + (x+y+2*n)/4)
				}
			}
		}
		return f
	}
	frames := []encoder.SourceFrame420{mk(0), mk(1), mk(2)}

	run := func(threads int) ([][]byte, []encoder.SourceFrame420, encoder.EncoderDecisionStats) {
		e, err := encoder.NewVideoEncoder(w, h, 120)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = e.Close() })
		e.SetMaxThreads(threads)
		e.SetGoldenInterval(0)
		e.SetDecisionStatsEnabled(true)
		var tus [][]byte
		var recons []encoder.SourceFrame420
		for i, f := range frames {
			tu, _, err := e.Encode(f, false)
			if err != nil {
				t.Fatalf("threads %d frame %d: %v", threads, i, err)
			}
			tus = append(tus, append([]byte(nil), tu...))
			recons = append(recons, cloneFrame(e.Recon()))
		}
		return tus, recons, e.DecisionStats()
	}
	serialTUs, recons, stats := run(1)
	big := stats.IntraBlockSizes[tile.BlockSize16x16] + stats.IntraBlockSizes[tile.BlockSize32x32]
	t.Logf("intra blocks by size: 8x8=%d 16x16=%d 32x32=%d",
		stats.IntraBlockSizes[tile.BlockSize8x8], stats.IntraBlockSizes[tile.BlockSize16x16], stats.IntraBlockSizes[tile.BlockSize32x32])
	if big == 0 {
		t.Fatal("scene cut coded no 16x16/32x32 DC intra blocks")
	}
	dec, err := goav1.NewDecoder(serialTUs)
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
	if i != len(frames) {
		t.Fatalf("decoded %d frames, want %d", i, len(frames))
	}
}
