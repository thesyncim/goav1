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

func TestFramePoolFrame(t *testing.T) {
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
	if _, err := pool.Frame(0); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("free Frame err=%v want %v", err, ErrInvalidSlot)
	}

	index, acquired, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	acquired.Y.Pix[0] = 0x55

	retained, err := pool.Frame(index)
	if err != nil {
		t.Fatal(err)
	}
	if retained != acquired || retained.Y.Pix[0] != 0x55 {
		t.Fatalf("retained=%p acquired=%p value=%x", retained, acquired, retained.Y.Pix[0])
	}

	if err := pool.Release(index); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Frame(index); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("released Frame err=%v want %v", err, ErrInvalidSlot)
	}
	if _, err := pool.Frame(-1); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("negative Frame err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestFramePoolFormatAndLayout(t *testing.T) {
	// SBSizeLog2 defaults to 6 (64x64) after normalization; the pool stores the
	// normalized format, so the comparison below uses the normalized value.
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, SBSizeLog2: 6, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	poolFormat, err := pool.Format()
	if err != nil {
		t.Fatal(err)
	}
	if poolFormat != format {
		t.Fatalf("format=%+v want %+v", poolFormat, format)
	}
	poolLayout, err := pool.Layout()
	if err != nil {
		t.Fatal(err)
	}
	if poolLayout != layout {
		t.Fatalf("layout=%+v want %+v", poolLayout, layout)
	}
}

func TestFramePoolAcquireFormat(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	index, acquired, err := pool.AcquireFormat(format)
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || acquired == nil {
		t.Fatalf("index=%d frame=%p", index, acquired)
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestFramePoolAcquireFormatRejectsMismatch(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = pool.AcquireFormat(Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	if !errors.Is(err, ErrInvalidFormat) {
		t.Fatalf("AcquireFormat err=%v want %v", err, ErrInvalidFormat)
	}
	if pool.Available() != 1 {
		t.Fatalf("mismatch consumed a slot, available=%d", pool.Available())
	}
}

func TestFramePoolReleaseMany(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*3)
	var frames [3]Frame
	var free [3]int
	var used [3]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index2, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	if err := pool.ReleaseMany([]int{index0, index2}); err != nil {
		t.Fatal(err)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	reacquired2, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if reacquired2 != index2 {
		t.Fatalf("first reacquire=%d want %d", reacquired2, index2)
	}
	reacquired0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if reacquired0 != index0 {
		t.Fatalf("second reacquire=%d want %d", reacquired0, index0)
	}

	if err := pool.ReleaseMany([]int{reacquired2, reacquired0, index1}); err != nil {
		t.Fatal(err)
	}
	if pool.Available() != 3 {
		t.Fatalf("available=%d want 3", pool.Available())
	}
	if err := pool.ReleaseMany(nil); err != nil {
		t.Fatal(err)
	}
}

func TestFramePoolReleaseManyRejectsAtomically(t *testing.T) {
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
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	if err := pool.ReleaseMany([]int{index0, index0}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("duplicate release err=%v want %v", err, ErrInvalidSlot)
	}
	if pool.Available() != 0 {
		t.Fatalf("duplicate changed available=%d", pool.Available())
	}
	if _, err := pool.Frame(index0); err != nil {
		t.Fatalf("duplicate released index0: %v", err)
	}

	if err := pool.ReleaseMany([]int{index0, 99}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("invalid release err=%v want %v", err, ErrInvalidSlot)
	}
	if pool.Available() != 0 {
		t.Fatalf("invalid changed available=%d", pool.Available())
	}
	if _, err := pool.Frame(index1); err != nil {
		t.Fatalf("invalid released index1: %v", err)
	}

	if err := pool.ReleaseMany([]int{index0, index1}); err != nil {
		t.Fatal(err)
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
		index0, _, err := pool.AcquireFormat(format)
		if err != nil {
			t.Fatal(err)
		}
		index1, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Frame(index0); err != nil {
			t.Fatal(err)
		}
		var batch [2]int
		batch[0] = index0
		batch[1] = index1
		if err := pool.ReleaseMany(batch[:]); err != nil {
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
	for b.Loop() {
		index, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFramePoolAcquireFormatRelease(b *testing.B) {
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
	for b.Loop() {
		index, _, err := pool.AcquireFormat(format)
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFramePoolReleaseMany(b *testing.B) {
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

	var batch [2]int
	b.ReportAllocs()
	for b.Loop() {
		index0, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		index1, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		batch[0] = index0
		batch[1] = index1
		if err := pool.ReleaseMany(batch[:]); err != nil {
			b.Fatal(err)
		}
	}
}
