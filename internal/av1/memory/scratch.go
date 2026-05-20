package memory

import "errors"

var ErrScratchTooSmall = errors.New("memory: scratch buffer too small")

// Scratch is a simple bump allocator over caller-owned memory.
//
// Scratch never allocates. It is intended for temporary per-frame and
// per-packet buffers whose lifetime is controlled by Reset or Rewind.
type Scratch struct {
	buf []byte
	off int
}

func NewScratch(buf []byte) Scratch {
	return Scratch{buf: buf}
}

func (s *Scratch) Reset() {
	s.off = 0
}

func (s *Scratch) Len() int {
	return s.off
}

func (s *Scratch) Cap() int {
	return len(s.buf)
}

func (s *Scratch) Mark() int {
	return s.off
}

func (s *Scratch) Rewind(mark int) {
	if mark < 0 {
		mark = 0
	}
	if mark > s.off {
		mark = s.off
	}
	s.off = mark
}

func (s *Scratch) Alloc(n int) ([]byte, error) {
	if n < 0 || len(s.buf)-s.off < n {
		return nil, ErrScratchTooSmall
	}
	start := s.off
	s.off += n
	return s.buf[start:s.off], nil
}

func (s *Scratch) AllocAligned(n int, align int) ([]byte, error) {
	if align <= 1 {
		return s.Alloc(n)
	}
	padding := (align - (s.off % align)) % align
	if n < 0 || len(s.buf)-s.off < padding+n {
		return nil, ErrScratchTooSmall
	}
	s.off += padding
	return s.Alloc(n)
}
