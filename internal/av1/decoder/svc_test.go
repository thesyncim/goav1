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
