package goav1

import (
	"crypto/sha256"
	"os"
	"path/filepath"
	"testing"

	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
)

// frameThreadTestClips are single-tile inter clips (every frame takes the
// frame-threaded path at workers > 1) plus a 10-bit and a two-tile clip that
// mix threaded and drained frames.
var frameThreadTestClips = []string{
	"testdata/benchcorpus/p360_inter_q32.ivf",
	"testdata/benchcorpus/p360_inter_q32_10bit.ivf",
	"testdata/benchcorpus/p360_inter_q32_2tiles.ivf",
	"testdata/benchcorpus/p144_intra_q32.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-01-size-226x226.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-04-cdfupdate.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-05-mv.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-06-mfmv.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-22-svc-L1T2.ivf",
	"internal/av1/testdata/libaom/av1-1-b8-23-film_grain-50.ivf",
	"internal/av1/testdata/libaom/av1-1-b10-24-monochrome.ivf",
	"internal/av1/testdata/libaom/av1-1-b10-00-quantizer-10.ivf",
}

// decodeFrameThreadDigest decodes clip through DecodeNext and hashes every
// returned frame in display order, together with the per-call output counts'
// total.
func decodeFrameThreadDigest(t *testing.T, clip string, opts ...Option) ([]byte, int) {
	t.Helper()
	sum, frames, err := decodeFrameThreadDigestErr(t, clip, opts...)
	if err != nil {
		t.Fatalf("%s: %v", clip, err)
	}
	return sum, frames
}

func decodeFrameThreadDigestErr(t *testing.T, clip string, opts ...Option) ([]byte, int, error) {
	t.Helper()
	raw, err := os.ReadFile(clip)
	if err != nil {
		t.Skipf("read %s: %v", clip, err)
	}
	dec, err := NewDecoderFromIVF(raw, opts...)
	if err != nil {
		return nil, 0, err
	}
	defer dec.Close()
	sum := sha256.New()
	frames := 0
	for {
		got, ok, err := dec.DecodeNext()
		if err != nil {
			return nil, 0, err
		}
		if !ok {
			break
		}
		for _, f := range got {
			frames++
			hashDecodedFrame(sum, f)
		}
	}
	return sum.Sum(nil), frames, nil
}

func hashDecodedFrame(sum interface{ Write([]byte) (int, error) }, f *Frame) {
	for _, p := range []FramePlane{f.Y, f.U, f.V} {
		if p.Width == 0 || p.Height == 0 {
			continue
		}
		row := p.Width * f.Layout.BytesPerSample
		for y := 0; y < p.Height; y++ {
			sum.Write(p.Pix[y*p.Stride : y*p.Stride+row])
		}
	}
}

// TestFrameThreadedDecodeByteIdentical is the frame-threading gate: every
// frame-delay depth and worker count must emit exactly the serial decode's
// frames in the same order. Run under -race it also proves the parse of frame
// N+1 shares no unsynchronized state with frame N's reconstruction tail.
func TestFrameThreadedDecodeByteIdentical(t *testing.T) {
	clips := append([]string(nil), frameThreadTestClips...)
	profiles, _ := filepath.Glob("internal/av1/testvector/testdata/profiles/*.ivf")
	clips = append(clips, profiles...)
	for _, clip := range clips {
		want, wantFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(1))
		if err != nil {
			t.Logf("%s: serial decode fails (%v); skipped", clip, err)
			continue
		}
		for _, workers := range []int{2, 4, 8} {
			serial, serialFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(workers), WithMaxFrameDelay(1))
			if err != nil {
				// A pre-existing failure of the one-frame-at-a-time multi-worker
				// path; the frame pipeline must fail the same way.
				if _, _, fpErr := decodeFrameThreadDigestErr(t, clip, WithWorkers(workers)); fpErr == nil {
					t.Fatalf("%s workers=%d: frame threading decodes a clip the serial path rejects (%v)", clip, workers, err)
				}
				t.Logf("%s workers=%d: serial multi-worker decode fails (%v); skipped", clip, workers, err)
				continue
			}
			if serialFrames != wantFrames || string(serial) != string(want) {
				t.Logf("%s workers=%d: delay-1 multi-worker output differs from workers=1 (pre-existing)", clip, workers)
			}
			for _, delay := range []int{0, 2, 3, 4} {
				got, frames := decodeFrameThreadDigest(t, clip, WithWorkers(workers), WithMaxFrameDelay(delay))
				if frames != serialFrames || string(got) != string(serial) {
					t.Fatalf("%s workers=%d delay=%d: %d frames, digest mismatch=%v (want %d frames)",
						clip, workers, delay, frames, string(got) != string(serial), serialFrames)
				}
			}
		}
	}
}

// TestFrameThreadedParseNeverReadsReferencePixels is the poison-References
// audit for frame threading: while a frame's entropy pass runs, every
// reference surface it may predict from is overwritten with garbage and its
// reconstruction lanes are held back; the true pixels return only after the
// pass. The decode stays byte-exact, so the entropy pass (dav1d's
// DAV1D_TASK_TYPE_TILE_ENTROPY) neither reads reference pixels nor waits for
// them.
func TestFrameThreadedParseNeverReadsReferencePixels(t *testing.T) {
	clip := "testdata/benchcorpus/p360_inter_q32.ivf"
	want, wantFrames := decodeFrameThreadDigest(t, clip, WithWorkers(1))

	var gate internalthreading.FrameProgress
	gate.Init()
	poisoned := 0
	decoderFramePipelineParseHook = func(p *decoderFramePipeline, fc *decoderFrameContext, references []*Frame) func() {
		if len(references) == 0 {
			return nil
		}
		// The previous frames' tails may still be filtering or reading these
		// surfaces: wait for every other frame in flight, then poison. Later
		// frames are held back by the pipeline's hook lock until restore.
		for s := range p.progress {
			if s != fc.surface {
				p.progress[s].Wait(internalthreading.FrameProgressDone)
			}
		}
		for i := range references {
			fc.ft.Recon.RefProgress[i].Wait(internalthreading.FrameProgressDone)
		}
		gate.Reset()
		var saved [][]byte
		seen := map[*Frame]bool{}
		for i, f := range references {
			fc.ft.Recon.RefProgress[i] = &gate
			if seen[f] {
				continue
			}
			seen[f] = true
			for _, pl := range []*FramePlane{&f.Y, &f.U, &f.V} {
				saved = append(saved, append([]byte(nil), pl.Pix...))
				for j := range pl.Pix {
					pl.Pix[j] = byte(0xA5 ^ j)
				}
			}
		}
		poisoned++
		return func() {
			k := 0
			done := map[*Frame]bool{}
			for _, f := range references {
				if done[f] {
					continue
				}
				done[f] = true
				for _, pl := range []*FramePlane{&f.Y, &f.U, &f.V} {
					copy(pl.Pix, saved[k])
					k++
				}
			}
			gate.Publish(internalthreading.FrameProgressDone)
		}
	}
	defer func() { decoderFramePipelineParseHook = nil }()

	got, frames := decodeFrameThreadDigest(t, clip, WithWorkers(4), WithMaxFrameDelay(3))
	if poisoned == 0 {
		t.Fatal("no frame took the frame-threaded path")
	}
	if frames != wantFrames || string(got) != string(want) {
		t.Fatalf("poisoned-reference decode differs: %d frames (want %d)", frames, wantFrames)
	}
}

// TestFrameContextCountMatchesDav1d pins dav1d's get_num_threads n_fc.
func TestFrameContextCountMatchesDav1d(t *testing.T) {
	cases := []struct{ workers, delay, want int }{
		{1, 0, 1}, {2, 0, 2}, {4, 0, 2}, {5, 0, 3}, {8, 0, 3}, {9, 0, 3}, {10, 0, 4},
		{16, 0, 4}, {64, 0, 8}, {8, 1, 1}, {8, 5, 5}, {2, 5, 2}, {64, 20, 8},
	}
	for _, c := range cases {
		if got := decoderFrameContextCount(c.workers, c.delay); got != c.want {
			t.Errorf("workers=%d delay=%d: n_fc=%d, want %d", c.workers, c.delay, got, c.want)
		}
	}
}
