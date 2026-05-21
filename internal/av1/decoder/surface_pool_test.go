package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFinishFrameSurfaceReleasesPool(t *testing.T) {
	pool := testFramePool(t, 3)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := FinishFrameSurface(&refs, &pool, finalFrameEvent(0xff), index1, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != index0 {
		t.Fatalf("count=%d releases[0]=%d want %d", count, releases[0], index0)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	if _, err := pool.Frame(index0); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("released frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index1 {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, index1)
	}
}

func TestFinishFrameSurfaceRejectsPoolReleaseAtomically(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}

	_, err = FinishFrameSurface(&refs, &pool, finalFrameEvent(0xff), index1, releases[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("FinishFrameSurface err=%v want %v", err, frame.ErrInvalidSlot)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index0 {
		t.Fatalf("slot=%d ok=%v want unchanged %d", slot, ok, index0)
	}
	if pool.Available() != 1 {
		t.Fatalf("available=%d want 1", pool.Available())
	}
}

func TestShowExistingFrameSurfaceReleasesPool(t *testing.T) {
	pool := testFramePool(t, 3)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, index0, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(1<<1, index1, releases[:]); err != nil {
		t.Fatal(err)
	}

	surface, count, err := ShowExistingFrameSurface(&refs, &pool, Event{
		Kind:          EventExistingFrame,
		FrameHeader:   parser.FrameHeaderPrefix{ShowExistingFrame: true, ExistingFrameIdx: 1},
		ExistingFrame: parser.ReferenceFrame{FrameType: parser.FrameTypeKey},
	}, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if surface != index1 || count != 1 || releases[0] != index0 {
		t.Fatalf("surface=%d count=%d release=%d want %d,1,%d", surface, count, releases[0], index1, index0)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	for i := 0; i < parser.RefFrames; i++ {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != index1 {
			t.Fatalf("slot[%d]=%d ok=%v want %d", i, slot, ok, index1)
		}
	}
}

func TestSurfacePoolHelpersAllocs(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	event := finalFrameEvent(0xff)

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		index0, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		index1, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
			t.Fatal(err)
		}
		if _, err := FinishFrameSurface(&refs, &pool, event, index1, releases[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("surface pool helpers allocated: %f", allocs)
	}
}

func BenchmarkFinishFrameSurface(b *testing.B) {
	pool := benchmarkFramePool(b, 2)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	event := finalFrameEvent(0xff)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		index0, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		index1, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
			b.Fatal(err)
		}
		if _, err := FinishFrameSurface(&refs, &pool, event, index1, releases[:]); err != nil {
			b.Fatal(err)
		}
	}
}

func finalFrameEvent(refresh uint8) Event {
	return Event{
		Kind: EventTileGroup,
		FrameSize: parser.FrameSize{
			RefreshFrameFlags: refresh,
		},
		TileGroup: parser.TileGroup{Final: true},
	}
}

func testFramePool(t *testing.T, count int) frame.Pool {
	t.Helper()
	pool, err := makeFramePool(count)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func benchmarkFramePool(b *testing.B, count int) frame.Pool {
	b.Helper()
	pool, err := makeFramePool(count)
	if err != nil {
		b.Fatal(err)
	}
	return pool
}

func makeFramePool(count int) (frame.Pool, error) {
	format := frame.Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := frame.RequiredSize(format)
	if err != nil {
		return frame.Pool{}, err
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	return frame.BindPool(backing, format, frames, free, used)
}
