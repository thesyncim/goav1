package frame

import (
	"errors"
	"testing"
)

// layerPoolFixture owns caller-side storage for a small LayerPool and its
// on-demand sub-pools. backings retains the per-format byte slices so that GC
// cannot reclaim them while a sub-pool is in use.
type layerPoolFixture struct {
	subPools     []Pool
	subFormats   []Format
	subBound     []bool
	backings     [][]byte
	framesArenas [][]Frame
	freeArenas   [][]int
	usedArenas   [][]bool
	allocs       int
}

func (f *layerPoolFixture) factory(format Format) (Pool, error) {
	layout, err := RequiredSize(format)
	if err != nil {
		return Pool{}, err
	}
	count := 4
	backing := make([]byte, layout.Size*count)
	frames := make([]Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := BindPool(backing, format, frames, free, used)
	if err != nil {
		return Pool{}, err
	}
	f.backings = append(f.backings, backing)
	f.framesArenas = append(f.framesArenas, frames)
	f.freeArenas = append(f.freeArenas, free)
	f.usedArenas = append(f.usedArenas, used)
	f.allocs++
	return pool, nil
}

func newLayerPoolFixture(capacity int) (*layerPoolFixture, LayerPool, error) {
	f := &layerPoolFixture{
		subPools:   make([]Pool, capacity),
		subFormats: make([]Format, capacity),
		subBound:   make([]bool, capacity),
	}
	pool, err := BindLayerPool(f.subPools, f.subFormats, f.subBound, 256, LayerFactoryFunc(f.factory))
	if err != nil {
		return nil, LayerPool{}, err
	}
	return f, pool, nil
}

func TestLayerPoolSingleResolution(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	format := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	id0, frame0, err := pool.Acquire(format)
	if err != nil {
		t.Fatalf("Acquire layer-0 #0: %v", err)
	}
	if id0 < 0 || frame0 == nil {
		t.Fatalf("id=%d frame=%p", id0, frame0)
	}
	if pool.Len() != 1 || fixture.allocs != 1 {
		t.Fatalf("len=%d allocs=%d after first Acquire", pool.Len(), fixture.allocs)
	}

	id1, frame1, err := pool.Acquire(format)
	if err != nil {
		t.Fatalf("Acquire layer-0 #1: %v", err)
	}
	if id1 == id0 || frame1 == nil || frame1 == frame0 {
		t.Fatalf("id1=%d id0=%d frame1=%p frame0=%p", id1, id0, frame1, frame0)
	}
	if pool.Len() != 1 || fixture.allocs != 1 {
		t.Fatalf("second Acquire bound a new sub-pool: len=%d allocs=%d", pool.Len(), fixture.allocs)
	}

	got0, err := pool.Frame(id0)
	if err != nil || got0 != frame0 {
		t.Fatalf("Frame(id0)=%p err=%v want %p", got0, err, frame0)
	}
	got1, err := pool.Frame(id1)
	if err != nil || got1 != frame1 {
		t.Fatalf("Frame(id1)=%p err=%v want %p", got1, err, frame1)
	}

	if err := pool.Release(id0); err != nil {
		t.Fatalf("Release id0: %v", err)
	}
	if err := pool.Release(id1); err != nil {
		t.Fatalf("Release id1: %v", err)
	}
	if _, err := pool.Frame(id0); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Frame(released id0) err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestLayerPoolTwoSpatialLayers(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	layer0 := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layer1 := Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	id0, frame0, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatalf("Acquire layer-0: %v", err)
	}
	if pool.Len() != 1 {
		t.Fatalf("Len after layer-0 Acquire=%d want 1", pool.Len())
	}

	id1, frame1, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatalf("Acquire layer-1: %v", err)
	}
	if pool.Len() != 2 || fixture.allocs != 2 {
		t.Fatalf("Len=%d allocs=%d after layer-1 Acquire", pool.Len(), fixture.allocs)
	}
	if frame0.Format.Width == frame1.Format.Width {
		t.Fatalf("layer-0 and layer-1 share width; got %d / %d", frame0.Format.Width, frame1.Format.Width)
	}

	// Global IDs must encode disjoint sub-pool slots so cross-pool resolution
	// works.
	got0, err := pool.Frame(id0)
	if err != nil || got0 != frame0 {
		t.Fatalf("Frame(layer-0 id)=%p err=%v want %p", got0, err, frame0)
	}
	got1, err := pool.Frame(id1)
	if err != nil || got1 != frame1 {
		t.Fatalf("Frame(layer-1 id)=%p err=%v want %p", got1, err, frame1)
	}
	if got0.Format != layer0 || got1.Format != layer1 {
		t.Fatalf("formats: layer-0=%+v layer-1=%+v", got0.Format, got1.Format)
	}

	// Repeated Acquire of layer-0 must reuse the same sub-pool.
	id0Again, frame0Again, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatalf("Acquire layer-0 again: %v", err)
	}
	if id0Again == id0 || id0Again == id1 {
		t.Fatalf("id0Again=%d collides with prior ids", id0Again)
	}
	if frame0Again == nil || frame0Again == frame0 {
		t.Fatalf("frame0Again did not advance: %p / %p", frame0Again, frame0)
	}
	if pool.Len() != 2 || fixture.allocs != 2 {
		t.Fatalf("layer-0 reuse bound a new sub-pool: len=%d allocs=%d", pool.Len(), fixture.allocs)
	}

	if err := pool.ReleaseMany([]int{id0, id1, id0Again}); err != nil {
		t.Fatalf("ReleaseMany: %v", err)
	}
}

func TestLayerPoolAcquireReleasePingPong(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	layer0 := Format{Width: 320, Height: 180, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	// Warm both sub-pools so that the steady-state acquire/release loop does
	// not call the factory.
	first0, _, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	first1, _, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(first0); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(first1); err != nil {
		t.Fatal(err)
	}
	warmAllocs := fixture.allocs

	// 64 ping-pong iterations: each acquires from both layers, swaps which is
	// released first, and verifies cross-pool resolution holds throughout.
	for iter := 0; iter < 64; iter++ {
		id0, f0, err := pool.Acquire(layer0)
		if err != nil {
			t.Fatalf("iter=%d Acquire layer-0: %v", iter, err)
		}
		id1, f1, err := pool.Acquire(layer1)
		if err != nil {
			t.Fatalf("iter=%d Acquire layer-1: %v", iter, err)
		}
		if got, _ := pool.Frame(id0); got != f0 {
			t.Fatalf("iter=%d Frame(id0)=%p want %p", iter, got, f0)
		}
		if got, _ := pool.Frame(id1); got != f1 {
			t.Fatalf("iter=%d Frame(id1)=%p want %p", iter, got, f1)
		}
		if iter&1 == 0 {
			if err := pool.Release(id0); err != nil {
				t.Fatalf("iter=%d Release id0: %v", iter, err)
			}
			if err := pool.Release(id1); err != nil {
				t.Fatalf("iter=%d Release id1: %v", iter, err)
			}
		} else {
			if err := pool.Release(id1); err != nil {
				t.Fatalf("iter=%d Release id1: %v", iter, err)
			}
			if err := pool.Release(id0); err != nil {
				t.Fatalf("iter=%d Release id0: %v", iter, err)
			}
		}
	}
	if fixture.allocs != warmAllocs {
		t.Fatalf("steady-state allocs grew: warm=%d final=%d", warmAllocs, fixture.allocs)
	}
}

func TestLayerPoolAcquireZeroAllocSteadyState(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	layer0 := Format{Width: 320, Height: 180, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	// Warm both sub-pools.
	id0, _, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, _, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(id0); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(id1); err != nil {
		t.Fatal(err)
	}
	if fixture.allocs != 2 {
		t.Fatalf("expected 2 factory calls during warm-up, got %d", fixture.allocs)
	}

	allocs := testing.AllocsPerRun(200, func() {
		id, _, err := pool.Acquire(layer0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Frame(id); err != nil {
			t.Fatal(err)
		}
		if err := pool.Release(id); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("steady-state Acquire/Release allocated: %f", allocs)
	}
}

func TestLayerPoolReleaseManyRejectsAtomically(t *testing.T) {
	_, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	id0, _, err := pool.Acquire(format)
	if err != nil {
		t.Fatal(err)
	}
	id1, _, err := pool.Acquire(format)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.ReleaseMany([]int{id0, id0}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("duplicate err=%v want %v", err, ErrInvalidSlot)
	}
	// Both IDs must remain acquired after atomic rejection.
	if _, err := pool.Frame(id0); err != nil {
		t.Fatalf("id0 lost after duplicate rejection: %v", err)
	}
	if _, err := pool.Frame(id1); err != nil {
		t.Fatalf("id1 lost after duplicate rejection: %v", err)
	}

	if err := pool.ReleaseMany([]int{id0, -1}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("negative id err=%v want %v", err, ErrInvalidSlot)
	}
	if _, err := pool.Frame(id0); err != nil {
		t.Fatalf("id0 lost after invalid rejection: %v", err)
	}
	if err := pool.ReleaseMany([]int{id0, id1}); err != nil {
		t.Fatalf("valid ReleaseMany: %v", err)
	}
}

func TestLayerPoolRejectsInvalidGlobalID(t *testing.T) {
	_, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Frame(-1); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Frame(-1) err=%v want %v", err, ErrInvalidSlot)
	}
	if err := pool.Release(-1); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Release(-1) err=%v want %v", err, ErrInvalidSlot)
	}
	if _, err := pool.Frame(1024); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Frame(unbound sub-pool) err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestLayerPoolFactoryFormatMismatch(t *testing.T) {
	wrong := Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	subPools := make([]Pool, 1)
	subFormats := make([]Format, 1)
	subBound := make([]bool, 1)
	pool, err := BindLayerPool(subPools, subFormats, subBound, 256, LayerFactoryFunc(func(format Format) (Pool, error) {
		layout, err := RequiredSize(wrong)
		if err != nil {
			return Pool{}, err
		}
		backing := make([]byte, layout.Size*1)
		frames := make([]Frame, 1)
		free := make([]int, 1)
		used := make([]bool, 1)
		return BindPool(backing, wrong, frames, free, used)
	}))
	if err != nil {
		t.Fatal(err)
	}
	requested := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	if _, _, err := pool.Acquire(requested); !errors.Is(err, ErrInvalidLayerFactory) {
		t.Fatalf("Acquire mismatched factory err=%v want %v", err, ErrInvalidLayerFactory)
	}
}

func TestLayerPoolFull(t *testing.T) {
	_, pool, err := newLayerPoolFixture(1)
	if err != nil {
		t.Fatal(err)
	}
	a := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	b := Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	if _, _, err := pool.Acquire(a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := pool.Acquire(b); !errors.Is(err, ErrLayerPoolFull) {
		t.Fatalf("second distinct format err=%v want %v", err, ErrLayerPoolFull)
	}
}

func TestLayerPoolReset(t *testing.T) {
	_, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	a := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	b := Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	if _, _, err := pool.Acquire(a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := pool.Acquire(b); err != nil {
		t.Fatal(err)
	}
	pool.Reset()
	sub, err := pool.SubPool(a)
	if err != nil {
		t.Fatal(err)
	}
	if sub.Available() != sub.Cap() {
		t.Fatalf("layer-a not reset: avail=%d cap=%d", sub.Available(), sub.Cap())
	}
}

func TestBindLayerPoolRejectsBadStorage(t *testing.T) {
	pools := make([]Pool, 2)
	formats := make([]Format, 1)
	bound := make([]bool, 2)
	factory := LayerFactoryFunc(func(format Format) (Pool, error) { return Pool{}, nil })
	if _, err := BindLayerPool(pools, formats, bound, 256, factory); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("mismatched lens err=%v want %v", err, ErrInvalidLayerPool)
	}
	if _, err := BindLayerPool(nil, nil, nil, 256, factory); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("empty storage err=%v want %v", err, ErrInvalidLayerPool)
	}
	if _, err := BindLayerPool([]Pool{{}}, []Format{{}}, []bool{false}, 0, factory); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("zero stride err=%v want %v", err, ErrInvalidLayerPool)
	}
	if _, err := BindLayerPool([]Pool{{}}, []Format{{}}, []bool{false}, 256, nil); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("nil factory err=%v want %v", err, ErrInvalidLayerPool)
	}
}
