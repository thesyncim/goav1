package frame

import (
	"errors"
	"testing"
)

func TestBindPoolAcquireRelease(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	if pool.Cap() != 2 || pool.Available() != 2 {
		t.Fatalf("pool cap=%d available=%d", pool.Cap(), pool.Available())
	}

	index0, frame0, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index0 != 0 || frame0 == nil {
		t.Fatalf("first acquire index=%d frame=%p", index0, frame0)
	}
	frame0.Y.Pix[0] = 0x33
	if backing[0] != 0x33 {
		t.Fatalf("frame 0 did not bind backing")
	}

	index1, frame1, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index1 != 1 || frame1 == nil {
		t.Fatalf("second acquire index=%d frame=%p", index1, frame1)
	}
	frame1.Y.Pix[0] = 0x44
	if backing[layout.Size] != 0x44 {
		t.Fatalf("frame 1 did not bind backing")
	}

	_, _, err = pool.Acquire()
	if !errors.Is(err, ErrPoolEmpty) {
		t.Fatalf("third acquire err=%v want %v", err, ErrPoolEmpty)
	}

	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}
	index0Again, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index0Again != index0 {
		t.Fatalf("reacquire index=%d want %d", index0Again, index0)
	}
	if err := pool.Release(index0Again); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0Again); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("double release err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestBindPoolRejectsBadStorage(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [1]int
	var used [2]bool

	_, err = BindPool(backing, format, frames[:], free[:], used[:])
	if !errors.Is(err, ErrInvalidPool) {
		t.Fatalf("BindPool short free err=%v want %v", err, ErrInvalidPool)
	}

	var fullFree [2]int
	_, err = BindPool(backing[:layout.Size*2-1], format, frames[:], fullFree[:], used[:])
	if !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("BindPool short backing err=%v want %v", err, ErrShortBuffer)
	}
}

func TestFramePoolReset(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	pool.Reset()
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	index, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 {
		t.Fatalf("index=%d want 0", index)
	}
}

func TestFramePoolAllocs(t *testing.T) {
	format := Format{Width: 128, Height: 72, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		index, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("pool acquire/release allocated: %f", allocs)
	}
}

func BenchmarkFramePoolAcquireRelease(b *testing.B) {
	format := Format{Width: 1920, Height: 1080, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	backing := make([]byte, layout.Size*4)
	var frames [4]Frame
	var free [4]int
	var used [4]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		index, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			b.Fatal(err)
		}
	}
}
