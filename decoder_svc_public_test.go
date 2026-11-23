package goav1

import (
	"errors"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
)

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
