package goav1

// End-to-end verification that the multi-worker SB-row reconstruction wavefront
// produces byte-identical output to the single-worker fused path, with a
// benchmark of the same decode at one and four workers.
//
// The conformance harness (internal/av1/testvector) runs with a one-worker
// pool, so the wavefront never engages there. These tests drive the public
// Decoder with workers > 1 and frame threading off on a
// single-tile clip, which is exactly the configuration that hands the idle
// lanes to the wavefront. The clip's 720p enhancement layer has six 128x128 SB
// rows, enough for the production 2-rows-per-worker threshold to engage
// several lanes.

import (
	"crypto/sha256"
	"os"
	"testing"
)

// wavefrontTestVector is a single-tile two-layer SVC clip whose 1280x720
// enhancement layer has enough superblock rows that a 2-4 worker wavefront
// overlaps at the production threshold.
const wavefrontTestVector = "internal/av1/testdata/libaom/av1-1-b8-22-svc-L2T1.ivf"

func decodeWavefrontFrames(t testing.TB, workers int) []byte {
	t.Helper()
	raw, err := os.ReadFile(wavefrontTestVector)
	if err != nil {
		t.Skipf("read wavefront vector: %v", err)
	}
	it, err := NewIVFIterator(raw)
	if err != nil {
		t.Fatalf("NewIVFIterator: %v", err)
	}
	var payloads [][]byte
	for {
		f, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf next: %v", err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, append([]byte(nil), f.Payload...))
	}
	if len(payloads) == 0 {
		t.Fatal("no frames in wavefront vector")
	}

	// WithMaxFrameDelay(1) keeps one frame in flight, so every worker is a tile
	// or wavefront lane rather than a frame-threading lane.
	dec, err := NewDecoder(payloads, WithWorkers(workers), WithMaxFrameDelay(1))
	if err != nil {
		t.Fatalf("NewDecoder (workers=%d): %v", workers, err)
	}
	defer dec.Close()

	sum := sha256.New()
	// Hash every frame as DecodeNext returns it, before the decoder can recycle
	// its surface.
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode (workers=%d): %v", workers, err)
		}
		if !ok {
			break
		}
		for _, f := range frames {
			hashFramePlanes(sum, f)
		}
	}
	return sum.Sum(nil)
}

func hashFramePlanes(sum interface{ Write([]byte) (int, error) }, f *Frame) {
	if f == nil {
		return
	}
	sum.Write(f.Y.Pix)
	if !f.Format.MonoChrome {
		sum.Write(f.U.Pix)
		sum.Write(f.V.Pix)
	}
}

// TestWavefrontByteIdenticalToSingleWorker is the load-bearing gate: the
// multi-worker SB-row wavefront must reconstruct exactly the same bytes as the
// single-worker fused path. Run it under -race to also prove the wavefront is
// data-race free.
func TestWavefrontByteIdenticalToSingleWorker(t *testing.T) {
	want := decodeWavefrontFrames(t, 1)
	for _, workers := range []int{2, 3, 4} {
		got := decodeWavefrontFrames(t, workers)
		if string(got) != string(want) {
			t.Fatalf("workers=%d output differs from workers=1 (wavefront not byte-identical)", workers)
		}
	}
}

// BenchmarkWavefrontDecode decodes the single-tile clip end to end with one
// worker and with four. The wavefront parallelizes only reconstruction; entropy
// decoding stays serial, so the end-to-end gain on this small clip is modest.
// The reconstruction phase alone is BenchmarkReconWavefrontReplay in the
// threading package.
func BenchmarkWavefrontDecode(b *testing.B) {
	for _, bc := range []struct {
		name    string
		workers int
	}{{"workers1", 1}, {"workers4", 4}} {
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				decodeWavefrontFrames(b, bc.workers)
			}
		})
	}
}
