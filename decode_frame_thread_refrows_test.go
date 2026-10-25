package goav1

import (
	"testing"
	"time"

	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
)

// poisonRowsTailHook makes every finished frame-threaded surface final one
// band of luma rows at a time: it overwrites all of the surface's samples
// (padding included) with garbage, then restores and publishes the rows top
// down, pausing between bands, and leaves the bottom padding for the final
// publish. A frame that reads a reference row before its band is published
// reads garbage, so any under-estimated reference wait changes the output.
func poisonRowsTailHook(band int) func(*Frame, *internalthreading.FrameProgress) {
	return func(f *Frame, progress *internalthreading.FrameProgress) {
		planes := []*FramePlane{&f.Y, &f.U, &f.V}
		saved := make([][]byte, len(planes))
		for i, pl := range planes {
			if pl.Stride == 0 {
				continue
			}
			saved[i] = append([]byte(nil), pl.Pix...)
			for j := range pl.Pix {
				pl.Pix[j] = byte(0x5A ^ j ^ j>>8)
			}
		}
		lumaH := f.Y.Height
		restore := func(lo, hi int) {
			for i, pl := range planes {
				if saved[i] == nil {
					continue
				}
				shift := 0
				if i > 0 && f.U.Height < f.Y.Height {
					shift = 1
				}
				plo, phi := lo>>shift, (hi+shift)>>shift
				if hi >= lumaH {
					phi = len(pl.Pix) / pl.Stride
				}
				copy(pl.Pix[plo*pl.Stride:phi*pl.Stride], saved[i][plo*pl.Stride:phi*pl.Stride])
			}
		}
		for lo := 0; lo < lumaH; lo += band {
			hi := min(lo+band, lumaH)
			if hi < lumaH {
				restore(lo, hi)
				progress.Publish(int32(hi))
			} else {
				restore(lo, lumaH)
			}
			time.Sleep(20 * time.Microsecond)
		}
	}
}

// TestFrameThreadedReferenceRowWaits checks the per-superblock reference row
// waits (dav1d lowest_px): with every reference surface published one band at
// a time and garbage below the published rows, the frame-threaded decode
// still matches the serial one, so no reconstruction reads a reference row it
// did not wait for.
func TestFrameThreadedReferenceRowWaits(t *testing.T) {
	clips := []string{
		"testdata/benchcorpus/p360_inter_q32.ivf",
		"testdata/benchcorpus/p360_inter_q32_2tiles.ivf",
		"internal/av1/testdata/libaom/av1-1-b8-05-mv.ivf",
		"internal/av1/testdata/libaom/av1-1-b8-06-mfmv.ivf",
		"internal/av1/testvector/testdata/profiles/multitile-2x2-rows-framethread-256x256.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-edgemv-130x130.ivf",
		"internal/av1/testvector/testdata/profiles/palette-inter-edge-256x144.ivf",
		"testdata/realcorpus/streams/aomgood_mtbike_720p_1500k_c6.ivf",
		"testdata/realcorpus/streams/aomgood_parkjoy_360p_1000k_c4.ivf",
		"testdata/realcorpus/streams/aomgood_johnny_360p_600k_c2.ivf",
		"testdata/realcorpus/streams/svtra_johnny_720p_crf35_p8.ivf",
		"testdata/realcorpus/streams/svtra_oldtown_720p_crf35_p8_fgs8.ivf",
	}
	for _, clip := range clips {
		want, wantFrames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(1))
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		for _, band := range []int{8, 64} {
			decoderFramePipelineTailHook = poisonRowsTailHook(band)
			got, frames, err := decodeFrameThreadDigestErr(t, clip, WithWorkers(4), WithMaxFrameDelay(3))
			decoderFramePipelineTailHook = nil
			if err != nil {
				t.Fatalf("%s band=%d: %v", clip, band, err)
			}
			if frames != wantFrames || string(got) != string(want) {
				t.Fatalf("%s band=%d: a frame read reference rows before they were published (%d frames, want %d)", clip, band, frames, wantFrames)
			}
		}
	}
}
