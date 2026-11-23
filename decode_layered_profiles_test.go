package goav1

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"testing"
)

// layeredUnsupportedProfileClips are the vendored clips NewLayeredDecoder
// refuses: its layer pools hold one coded size per spatial layer, so streams
// that change the coded size from frame to frame do not fit.
var layeredUnsupportedProfileClips = map[string]string{
	"profile0-420-8bit-resize-random-192x108.ivf": "layer pool full",
}

// TestLayeredDecoderRejectsSuperResolution pins the layered decoder's other
// limit: it does not apply super-resolution, and says so instead of returning
// frames at the coded width. The clip that also resizes every frame is already
// refused when the decoder is constructed.
func TestLayeredDecoderRejectsSuperResolution(t *testing.T) {
	for _, clip := range superResProfileClips {
		t.Run(clip.file, func(t *testing.T) {
			ivf, err := os.ReadFile(profileClipPath(clip.file))
			if err != nil {
				t.Fatalf("read clip: %v", err)
			}
			it, err := NewIVFIterator(ivf)
			if err != nil {
				t.Fatalf("NewIVFIterator: %v", err)
			}
			var payloads [][]byte
			for {
				frame, ok, err := it.Next()
				if err != nil {
					t.Fatalf("ivf next: %v", err)
				}
				if !ok {
					break
				}
				payloads = append(payloads, frame.Payload)
			}
			dec, err := NewLayeredDecoder(payloads)
			if err != nil {
				if strings.Contains(clip.file, "resize-random") && strings.Contains(err.Error(), "layer pool full") {
					return
				}
				t.Fatalf("NewLayeredDecoder: %v", err)
			}
			defer dec.Close()
			for {
				_, ok, err := dec.DecodeNext()
				if err != nil {
					if !strings.Contains(err.Error(), "unsupported postfilter") {
						t.Fatalf("DecodeNext: %v, want an unsupported postfilter error", err)
					}
					return
				}
				if !ok {
					t.Fatal("decoded a super-resolution clip without an error")
				}
			}
		})
	}
}

// TestLayeredDecoderProfileClipsMatchGolden decodes the vendored profile clips
// through NewLayeredDecoder and checks every frame against the same goldens as
// NewDecoder. The layered decoder shares reference buffers across spatial
// layers and keeps their entropy contexts and motion vectors per buffer; these
// libaom and SVT-AV1 streams use reference structures and tools (temporal
// motion vectors, several primary reference frames) that goav1's own encoder
// does not produce.
func TestLayeredDecoderProfileClipsMatchGolden(t *testing.T) {
	for _, workers := range []int{1, 4} {
		for _, clip := range profileClips {
			t.Run(fmt.Sprintf("workers=%d/%s", workers, clip.file), func(t *testing.T) {
				ivf, err := os.ReadFile(profileClipPath(clip.file))
				if err != nil {
					t.Fatalf("read clip: %v", err)
				}
				it, err := NewIVFIterator(ivf)
				if err != nil {
					t.Fatalf("NewIVFIterator: %v", err)
				}
				var payloads [][]byte
				for {
					frame, ok, err := it.Next()
					if err != nil {
						t.Fatalf("ivf next: %v", err)
					}
					if !ok {
						break
					}
					payloads = append(payloads, frame.Payload)
				}

				want, unsupported := layeredUnsupportedProfileClips[clip.file]
				dec, err := NewLayeredDecoder(payloads, WithWorkers(workers))
				if unsupported {
					if err == nil {
						dec.Close()
						t.Fatalf("NewLayeredDecoder accepted a clip listed as unsupported (%s)", want)
					}
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("NewLayeredDecoder: %v, want %q", err, want)
					}
					return
				}
				if err != nil {
					t.Fatalf("NewLayeredDecoder: %v", err)
				}
				defer dec.Close()

				var got []string
				for {
					frames, ok, err := dec.DecodeNext()
					if err != nil {
						t.Fatalf("DecodeNext after %d frames: %v", len(got), err)
					}
					if !ok {
						break
					}
					for _, f := range frames {
						sum := frameMD5Visible(f)
						got = append(got, hex.EncodeToString(sum[:]))
					}
				}
				if len(got) != len(clip.frameMD5Hex) {
					t.Fatalf("decoded %d frames, want %d", len(got), len(clip.frameMD5Hex))
				}
				for i, md5 := range clip.frameMD5Hex {
					if got[i] != md5 {
						t.Fatalf("frame %d md5 got=%s want=%s", i, got[i], md5)
					}
				}
			})
		}
	}
}
