package goav1

import (
	"encoding/hex"
	"errors"
	"fmt"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
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

// Droppable layer frames refresh no reference slot, so their surfaces are
// owned by nothing once shown; sixty frames overflow a pool that leaks them
// (the public decoder previously died at frame fourteen).
func TestLongL1T2DecodeNoPoolLeak(t *testing.T) {
	const w, h, n = 320, 192, 60
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 500_000, Framerate: 30, TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	f := I420Frame{
		Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
	for i := 0; i < n; i++ {
		for j := range f.Y {
			f.Y[j] = byte(40 + (j+i*7)%160)
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatal(err)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}
	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", frames, err)
		}
		if !ok {
			break
		}
		frames += len(batch)
	}
	if frames != n {
		t.Fatalf("decoded %d frames, want %d", frames, n)
	}
}

func TestPublicDecoderFrameLayerPoolProviderAndReleaser(t *testing.T) {
	layerPool := newPublicDecoderLayerPool(t, 2, 2)
	layer0 := FrameFormat{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := FrameFormat{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	id0, want0, err := layerPool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, want1, err := layerPool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}

	adapter := internaldecoder.NewFrameLayerPool(&layerPool)
	var provider internaldecoder.FrameSurfaceProvider = adapter
	var releaser internaldecoder.FrameSurfaceReleaser = adapter
	if got, err := provider.FrameSurface(id0); err != nil || got != want0 {
		t.Fatalf("FrameSurface(layer0)=%p err=%v want %p", got, err, want0)
	}
	if got, err := provider.FrameSurface(id1); err != nil || got != want1 {
		t.Fatalf("FrameSurface(layer1)=%p err=%v want %p", got, err, want1)
	}

	gotRefs := make([]*Frame, 2)
	count, err := internaldecoder.ResolveFrameReferencesWithProvider(provider, []int{id0, id1}, gotRefs)
	if err != nil {
		t.Fatalf("ResolveDecoderFrameReferencesWithProvider: %v", err)
	}
	if count != 2 || gotRefs[0] != want0 || gotRefs[1] != want1 {
		t.Fatalf("count=%d refs=%p,%p want %p,%p", count, gotRefs[0], gotRefs[1], want0, want1)
	}

	if err := releaser.ReleaseFrameSurfaces([]int{id0, id1}); err != nil {
		t.Fatalf("ReleaseFrameSurfaces: %v", err)
	}
	if _, err := layerPool.Frame(id0); err == nil {
		t.Fatalf("layer0 surface was not released")
	}
	if _, err := layerPool.Frame(id1); err == nil {
		t.Fatalf("layer1 surface was not released")
	}
}

func TestPublicDecoderFrameLayerPoolNilRejected(t *testing.T) {
	adapter := internaldecoder.NewFrameLayerPool(nil)
	if _, err := adapter.FrameSurface(0); !errors.Is(err, internaldecoder.ErrInvalidSurfaceReference) {
		t.Fatalf("FrameSurface err=%v want %v", err, internaldecoder.ErrInvalidSurfaceReference)
	}
	if err := adapter.ReleaseFrameSurfaces([]int{0}); !errors.Is(err, internaldecoder.ErrInvalidSurfaceReference) {
		t.Fatalf("ReleaseFrameSurfaces err=%v want %v", err, internaldecoder.ErrInvalidSurfaceReference)
	}
}

func newPublicDecoderLayerPool(t *testing.T, layers int, surfacesPerLayer int) internalframe.LayerPool {
	t.Helper()
	var backings [][]byte
	factory := internalframe.LayerFactoryFunc(func(format FrameFormat) (internalframe.Pool, error) {
		_, backingSize, err := framePoolRequiredSize(format, surfacesPerLayer)
		if err != nil {
			return internalframe.Pool{}, err
		}
		backing := make([]byte, backingSize)
		pool, err := internalframe.BindPool(backing, format,
			make([]Frame, surfacesPerLayer),
			make([]int, surfacesPerLayer),
			make([]bool, surfacesPerLayer))
		if err != nil {
			return internalframe.Pool{}, err
		}
		backings = append(backings, backing)
		return pool, nil
	})
	layerPool, err := internalframe.BindLayerPool(
		make([]internalframe.Pool, layers),
		make([]FrameFormat, layers),
		make([]bool, layers),
		256,
		factory,
	)
	if err != nil {
		t.Fatal(err)
	}
	return layerPool
}
