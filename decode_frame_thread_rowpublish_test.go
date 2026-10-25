package goav1

import (
	"hash/maphash"
	"os"
	"sync"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
)

// publishedRowsChecker records a hash of every row a frame-threaded
// post-filter publishes as final (with the chroma rows it covers, right
// padding included) and, once the frame's tail finished, checks that no
// published row changed afterwards: a row published before a later stage
// rewrote it would be read by other frames before it was final.
type publishedRowsChecker struct {
	t     *testing.T
	seed  maphash.Seed
	mu    sync.Mutex
	rows  map[*internalframe.Frame][]uint64
	count int
	bad   int
}

func (c *publishedRowsChecker) rowHash(f *internalframe.Frame, y int) uint64 {
	var h maphash.Hash
	h.SetSeed(c.seed)
	h.Write(f.Y.Pix[y*f.Y.Stride : (y+1)*f.Y.Stride])
	if !f.Format.MonoChrome && f.U.Stride != 0 {
		cy := y
		if f.Format.SubsamplingY {
			if y&1 != 0 {
				return h.Sum64()
			}
			cy = y >> 1
		}
		h.Write(f.U.Pix[cy*f.U.Stride : (cy+1)*f.U.Stride])
		h.Write(f.V.Pix[cy*f.V.Stride : (cy+1)*f.V.Stride])
	}
	return h.Sum64()
}

func (c *publishedRowsChecker) publish(f *internalframe.Frame, rows int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	have := c.rows[f]
	for y := len(have); y < rows; y++ {
		have = append(have, c.rowHash(f, y))
	}
	c.rows[f] = have
	c.count++
}

func (c *publishedRowsChecker) done(f *Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for y, want := range c.rows[f] {
		if c.rowHash(f, y) != want {
			c.bad++
			c.t.Errorf("luma row %d of a %dx%d surface changed after it was published", y, f.Y.Width, f.Y.Height)
			break
		}
	}
	delete(c.rows, f)
}

// TestFrameThreadedPublishedRowsFinal checks the publish side of the
// frame-threaded row progress: every row a post-filter publishes while it
// still runs keeps its value until the frame is done, and the output matches
// the serial decode.
func TestFrameThreadedPublishedRowsFinal(t *testing.T) {
	clips := []string{
		// Tracked clips first; the local corpora are skipped when absent.
		"internal/av1/testvector/testdata/profiles/multitile-2x2-rows-framethread-256x256.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-edgemv-130x130.ivf",
		"internal/av1/testvector/testdata/profiles/palette-inter-edge-256x144.ivf",
		"internal/av1/testvector/testdata/profiles/svt-compound-edge-filter-256x136.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-filmgrain-showexisting-128x128.ivf",
		"testdata/benchcorpus/p360_inter_q32.ivf",
		"testdata/benchcorpus/p360_inter_q32_2tiles.ivf",
		"testdata/realcorpus/streams/aomrt_mtbike_360p_600k_s8.ivf",
		"testdata/realcorpus/streams/aomrt_fourpeople_720p_1000k_s10.ivf",
		"testdata/realcorpus/streams/aomrt_wikipedia_1080p_1500k_s11.ivf",
		"testdata/realcorpus/streams/svtrtc_wikipedia_1080p_1000k_p12.ivf",
		"testdata/realcorpus/streams/goav1_johnny_360p_500k.ivf",
		"testdata/realcorpus/streams/aomgood_mtbike_720p_1500k_c6.ivf",
		"testdata/realcorpus/streams/svtra_johnny_720p_crf35_p8.ivf",
	}
	c := &publishedRowsChecker{t: t, seed: maphash.MakeSeed(), rows: map[*internalframe.Frame][]uint64{}}
	internaldecoder.FrameThreadRowsPublishHook = c.publish
	decoderFramePipelineDoneHook = c.done
	defer func() {
		internaldecoder.FrameThreadRowsPublishHook = nil
		decoderFramePipelineDoneHook = nil
	}()
	for _, clip := range clips {
		if _, err := os.Stat(clip); err != nil {
			t.Logf("%s: %v; skipped", clip, err)
			continue
		}
		want, wantFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(1))
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		for _, workers := range []int{4, 8} {
			before := c.count
			got, frames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(workers))
			if err != nil {
				t.Fatalf("%s workers=%d: %v", clip, workers, err)
			}
			if frames != wantFrames || string(got) != string(want) {
				t.Fatalf("%s workers=%d: output differs from the serial decode", clip, workers)
			}
			t.Logf("%s workers=%d: %d row publishes", clip, workers, c.count-before)
			if c.bad != 0 {
				t.FailNow()
			}
		}
	}
}
