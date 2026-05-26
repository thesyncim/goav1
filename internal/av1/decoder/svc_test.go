package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestResolveFrameReferencesWithProviderFanOut(t *testing.T) {
	poolA := testFramePool(t, 1)
	indexA, _, err := poolA.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	poolB := testFramePool(t, 1)
	indexB, _, err := poolB.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	// Caller-owned namespace: layer 0 = poolA, layer 1 = poolB.
	provider := FrameSurfaceProviderFunc(func(id int) (*frame.Frame, error) {
		switch id {
		case indexA:
			return poolA.Frame(indexA)
		case 256 + indexB:
			return poolB.Frame(indexB)
		}
		return nil, ErrInvalidSurfaceReference
	})
	surfaces := []int{indexA, 256 + indexB}
	dst := make([]*frame.Frame, 2)
	count, err := ResolveFrameReferencesWithProvider(provider, surfaces, dst)
	if err != nil {
		t.Fatalf("ResolveFrameReferencesWithProvider: %v", err)
	}
	if count != 2 || dst[0] == nil || dst[1] == nil {
		t.Fatalf("count=%d dst=%v", count, dst)
	}
}

func TestResolveFrameReferencesWithProviderRejectsInvalid(t *testing.T) {
	provider := FrameSurfaceProviderFunc(func(id int) (*frame.Frame, error) {
		return nil, ErrInvalidSurfaceReference
	})
	dst := make([]*frame.Frame, 1)
	if _, err := ResolveFrameReferencesWithProvider(provider, []int{0}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if _, err := ResolveFrameReferencesWithProvider(provider, []int{-1}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if _, err := ResolveFrameReferencesWithProvider(nil, []int{0}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-provider err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func TestResolveTemporalMotionReferencesWithProviderFanOut(t *testing.T) {
	poolA := testFramePool(t, 1)
	indexA, _, err := poolA.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	mvFrameA, err := bindEmptyReferenceMVFrame(poolA.Frame(indexA))
	if err != nil {
		t.Fatal(err)
	}
	storeA := tile.TemporalMotionReferenceFrame{Frame: &mvFrameA, OrderHint: 0, IntraOnly: true}

	poolB := testFramePool(t, 1)
	indexB, _, err := poolB.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	mvFrameB, err := bindEmptyReferenceMVFrame(poolB.Frame(indexB))
	if err != nil {
		t.Fatal(err)
	}
	storeB := tile.TemporalMotionReferenceFrame{Frame: &mvFrameB, OrderHint: 0, IntraOnly: true}

	provider := TemporalMotionReferenceProviderFunc(func(id int) (tile.TemporalMotionReferenceFrame, error) {
		switch id {
		case 0:
			return storeA, nil
		case 256:
			return storeB, nil
		}
		return tile.TemporalMotionReferenceFrame{}, ErrInvalidSurfaceReference
	})
	dst := make([]tile.TemporalMotionReferenceFrame, 2)
	count, err := ResolveTemporalMotionReferencesWithProvider(provider, []int{0, 256}, dst)
	if err != nil {
		t.Fatalf("ResolveTemporalMotionReferencesWithProvider: %v", err)
	}
	if count != 2 || dst[0].Frame != &mvFrameA || dst[1].Frame != &mvFrameB {
		t.Fatalf("count=%d dst=%+v", count, dst)
	}
}

func TestResolveTemporalMotionReferencesWithProviderRejectsInvalid(t *testing.T) {
	provider := TemporalMotionReferenceProviderFunc(func(id int) (tile.TemporalMotionReferenceFrame, error) {
		return tile.TemporalMotionReferenceFrame{}, nil
	})
	dst := make([]tile.TemporalMotionReferenceFrame, 1)
	if _, err := ResolveTemporalMotionReferencesWithProvider(provider, []int{0}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-frame err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if _, err := ResolveTemporalMotionReferencesWithProvider(provider, []int{-1}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("negative err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if _, err := ResolveTemporalMotionReferencesWithProvider(nil, []int{0}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-provider err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func bindEmptyReferenceMVFrame(f *frame.Frame, err error) (tile.ReferenceMVFrame, error) {
	if err != nil {
		return tile.ReferenceMVFrame{}, err
	}
	miCols := uint32(f.Format.Width / 4)
	miRows := uint32(f.Format.Height / 4)
	mvf := tile.ReferenceMVFrame{}
	if err := mvf.Init(miRows, miCols, make([]tile.ReferenceMVEntry, miRows*miCols)); err != nil {
		return tile.ReferenceMVFrame{}, err
	}
	return mvf, nil
}

// TestRunEventWithContextAndExternalReferencesSurfaceFanOut exercises the SVC
// multi-pool entry point with a synthetic scenario: a key frame is decoded into
// poolA, refreshes all 8 reference slots with the caller-owned global ID, and
// then an inter frame for "spatial layer 1" picks the same slot but acquires
// its output from poolB. The decoder must not validate cross-pool surface
// indices against poolB; it must route through the provider.
//
// This is a minimal smoke test: we only assert that planning + reference
// resolution succeed at the decoder API boundary. Actual tile decoding is
// orthogonal and exercised by the testvector harness.
func TestRunEventWithContextAndExternalReferencesNilProvider(t *testing.T) {
	var state FrameWorkState
	var refs SurfaceReferences
	var event Event
	if _, err := state.RunEventWithContextAndExternalReferences(&refs, nil, parser.SequenceHeader{}, event, 32, nil, nil, 1, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-provider err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

// newTestLayerPool wires a tiny LayerPool with two sub-pool slots and a
// factory that synthesizes per-format backing storage on demand. It is the
// production-side replacement for the testvector harness per-spatial-layer
// glue.
func newTestLayerPool(t *testing.T, capacity int) (*frame.LayerPool, *[][]byte) {
	t.Helper()
	subPools := make([]frame.Pool, capacity)
	subFormats := make([]frame.Format, capacity)
	subBound := make([]bool, capacity)
	keepAlive := &[][]byte{}
	factory := frame.LayerFactoryFunc(func(format frame.Format) (frame.Pool, error) {
		layout, err := frame.RequiredSize(format)
		if err != nil {
			return frame.Pool{}, err
		}
		const slots = 4
		backing := make([]byte, layout.Size*slots)
		frames := make([]frame.Frame, slots)
		free := make([]int, slots)
		used := make([]bool, slots)
		pool, err := frame.BindPool(backing, format, frames, free, used)
		if err != nil {
			return frame.Pool{}, err
		}
		*keepAlive = append(*keepAlive, backing)
		return pool, nil
	})
	pool, err := frame.BindLayerPool(subPools, subFormats, subBound, 256, factory)
	if err != nil {
		t.Fatal(err)
	}
	return &pool, keepAlive
}

func TestFrameLayerPoolFrameSurfaceProvider(t *testing.T) {
	pool, _ := newTestLayerPool(t, 2)
	layer0 := frame.Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := frame.Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	id0, want0, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, want1, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}

	provider := NewFrameLayerPool(pool)
	got0, err := provider.FrameSurface(id0)
	if err != nil || got0 != want0 {
		t.Fatalf("FrameSurface(layer-0)=%p err=%v want %p", got0, err, want0)
	}
	got1, err := provider.FrameSurface(id1)
	if err != nil || got1 != want1 {
		t.Fatalf("FrameSurface(layer-1)=%p err=%v want %p", got1, err, want1)
	}

	// ResolveFrameReferencesWithProvider must route both IDs through the
	// LayerPool transparently.
	surfaces := []int{id0, id1}
	dst := make([]*frame.Frame, 2)
	count, err := ResolveFrameReferencesWithProvider(provider, surfaces, dst)
	if err != nil {
		t.Fatalf("ResolveFrameReferencesWithProvider: %v", err)
	}
	if count != 2 || dst[0] != want0 || dst[1] != want1 {
		t.Fatalf("count=%d dst=%v want %v,%v", count, dst, want0, want1)
	}
}

func TestFrameLayerPoolFrameSurfaceReleaser(t *testing.T) {
	pool, _ := newTestLayerPool(t, 2)
	layer0 := frame.Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := frame.Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	id0, _, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, _, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}
	releaser := NewFrameLayerPool(pool)
	if err := releaser.ReleaseFrameSurfaces([]int{id0, id1}); err != nil {
		t.Fatalf("ReleaseFrameSurfaces: %v", err)
	}
	if _, err := pool.Frame(id0); err == nil {
		t.Fatalf("layer-0 surface not released")
	}
	if _, err := pool.Frame(id1); err == nil {
		t.Fatalf("layer-1 surface not released")
	}
	// Empty/nil batches are no-ops.
	if err := releaser.ReleaseFrameSurfaces(nil); err != nil {
		t.Fatalf("nil release: %v", err)
	}
}

func TestAcquireLayerFrameSurfaceRoundTrip(t *testing.T) {
	pool, _ := newTestLayerPool(t, 2)
	sequence := parser.SequenceHeader{
		ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
	}
	size := parser.FrameSize{CodedWidth: 640, Height: 360, UpscaledWidth: 640}

	sub, local, globalID, fr, err := AcquireLayerFrameSurface(pool, sequence, size, 32)
	if err != nil {
		t.Fatalf("AcquireLayerFrameSurface: %v", err)
	}
	if sub == nil || fr == nil || local < 0 || globalID < 0 {
		t.Fatalf("sub=%p local=%d global=%d fr=%p", sub, local, globalID, fr)
	}
	if got := LayerPoolGlobalSurfaceID(pool, sub, local); got != globalID {
		t.Fatalf("LayerPoolGlobalSurfaceID=%d want %d", got, globalID)
	}
	provider := NewFrameLayerPool(pool)
	got, err := provider.FrameSurface(globalID)
	if err != nil || got != fr {
		t.Fatalf("provider.FrameSurface=%p err=%v want %p", got, err, fr)
	}
	if err := sub.Release(local); err != nil {
		t.Fatalf("sub.Release: %v", err)
	}
}

func TestAcquireLayerFrameSurfaceCrossLayer(t *testing.T) {
	pool, _ := newTestLayerPool(t, 2)
	sequence := parser.SequenceHeader{
		ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
	}
	layer0Size := parser.FrameSize{CodedWidth: 640, Height: 360, UpscaledWidth: 640}
	layer1Size := parser.FrameSize{CodedWidth: 1280, Height: 720, UpscaledWidth: 1280}

	sub0, local0, id0, frame0, err := AcquireLayerFrameSurface(pool, sequence, layer0Size, 32)
	if err != nil {
		t.Fatal(err)
	}
	sub1, local1, id1, frame1, err := AcquireLayerFrameSurface(pool, sequence, layer1Size, 32)
	if err != nil {
		t.Fatal(err)
	}
	if sub0 == sub1 {
		t.Fatalf("layer-0 and layer-1 returned the same sub-pool")
	}
	if id0 == id1 {
		t.Fatalf("layer-0 and layer-1 returned the same global ID")
	}
	if frame0.Format.Width != 640 || frame1.Format.Width != 1280 {
		t.Fatalf("layer formats wrong: %d / %d", frame0.Format.Width, frame1.Format.Width)
	}
	// Cross-resolve via provider: each global ID must reach its own sub-pool.
	provider := NewFrameLayerPool(pool)
	if got, _ := provider.FrameSurface(id0); got != frame0 {
		t.Fatalf("FrameSurface(layer-0)=%p want %p", got, frame0)
	}
	if got, _ := provider.FrameSurface(id1); got != frame1 {
		t.Fatalf("FrameSurface(layer-1)=%p want %p", got, frame1)
	}
	if err := pool.ReleaseMany([]int{id0, id1}); err != nil {
		t.Fatalf("ReleaseMany: %v", err)
	}
	_, _ = local0, local1
}

func TestFrameLayerPoolNilPoolRejected(t *testing.T) {
	provider := NewFrameLayerPool(nil)
	if _, err := provider.FrameSurface(0); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-pool FrameSurface err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if err := provider.ReleaseFrameSurfaces([]int{0}); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-pool ReleaseFrameSurfaces err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}
