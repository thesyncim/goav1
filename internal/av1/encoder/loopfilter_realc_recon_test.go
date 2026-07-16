package encoder_test

import (
	"fmt"
	"os"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// loopfilter_realc_recon_test.go is the multi-tile mask-path safety net: it
// encodes complex real content with the deblocking mask apply forced on for the
// multi-tile (32 tile-column) 1080p layout, then decodes the stream and asserts
// every reconstructed frame is byte-identical to the encoder's own Recon(). The
// decoder re-derives the deblock edges from the shared frame map across tile
// boundaries (the edge-list sweep), so any tile-boundary divergence in the
// forced mask apply shows up as a mismatched pixel.

func loadRealCFrames(t *testing.T, maxFrames int) ([]encoder.SourceFrame420, int, int, bool) {
	t.Helper()
	raw, err := os.ReadFile("/tmp/corpus/realC.yuv")
	if err != nil {
		t.Skipf("realC corpus unavailable: %v", err)
		return nil, 0, 0, false
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	nf := len(raw) / fl
	if nf > maxFrames {
		nf = maxFrames
	}
	if nf < 3 {
		t.Skip("realC too short")
		return nil, 0, 0, false
	}
	frames := make([]encoder.SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = encoder.SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	return frames, w, h, true
}

func lfCloneFrame(f encoder.SourceFrame420) encoder.SourceFrame420 {
	c := f
	c.Y = append([]byte(nil), f.Y...)
	c.U = append([]byte(nil), f.U...)
	c.V = append([]byte(nil), f.V...)
	return c
}

// runRealCReconMatch encodes maxFrames of realC with the given encoder
// configuration and asserts decode == recon for every frame.
func runRealCReconMatch(t *testing.T, maxThreads int) {
	frames, w, h, ok := loadRealCFrames(t, 8)
	if !ok {
		return
	}
	cw, ch := w/2, h/2

	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 5_000_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if maxThreads > 0 {
		enc.SetMaxThreads(maxThreads)
	}

	tus := make([][]byte, 0, len(frames))
	recons := make([]encoder.SourceFrame420, 0, len(frames))
	for i := range frames {
		tu, _, err := enc.Encode(frames[i], i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, lfCloneFrame(enc.Recon()))
	}
	if err := enc.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
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
			if i >= len(frames) {
				t.Fatalf("decoded more than %d frames", len(frames))
			}
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

// TestRealCMaskDefaultMultiTile exercises the DEFAULT (multi-tile, 30 tile-column)
// 1080p encoder, which now routes deblocking through the mask apply, and requires
// decode == recon on complex content across superblock and tile-column boundaries.
func TestRealCMaskDefaultMultiTile(t *testing.T) {
	runRealCReconMatch(t, 0)
}

// TestRealCMaskSingleThread is the single-thread (single-tile) latent-bug gate:
// the serial mask apply must match the decode on complex content.
func TestRealCMaskSingleThread(t *testing.T) {
	runRealCReconMatch(t, 1)
}
