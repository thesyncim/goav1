package memory

import (
	"errors"
	"testing"
)

func TestScratchAllocRewind(t *testing.T) {
	var backing [16]byte
	s := NewScratch(backing[:])

	a, err := s.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 4 || s.Len() != 4 {
		t.Fatalf("len=%d scratch=%d", len(a), s.Len())
	}

	mark := s.Mark()
	_, err = s.AllocAligned(4, 8)
	if err != nil {
		t.Fatal(err)
	}
	if s.Len() != 12 {
		t.Fatalf("aligned len=%d want 12", s.Len())
	}

	s.Rewind(mark)
	if s.Len() != 4 {
		t.Fatalf("rewound len=%d want 4", s.Len())
	}

	s.Reset()
	if s.Len() != 0 {
		t.Fatalf("reset len=%d want 0", s.Len())
	}
}

func TestScratchTooSmall(t *testing.T) {
	var backing [2]byte
	s := NewScratch(backing[:])
	_, err := s.Alloc(3)
	if !errors.Is(err, ErrScratchTooSmall) {
		t.Fatalf("Alloc err=%v want %v", err, ErrScratchTooSmall)
	}
}

func TestScratchAllocs(t *testing.T) {
	var backing [64]byte
	allocs := testing.AllocsPerRun(1000, func() {
		s := NewScratch(backing[:])
		_, err := s.AllocAligned(16, 16)
		if err != nil {
			t.Fatal(err)
		}
		s.Reset()
	})
	if allocs != 0 {
		t.Fatalf("Scratch allocated: %f", allocs)
	}
}
