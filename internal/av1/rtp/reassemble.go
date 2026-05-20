package rtp

// FragmentReassembler rebuilds a single fragmented OBU into a caller-owned
// buffer. It does not allocate or retain payload byte slices.
type FragmentReassembler struct {
	active bool
	n      int
}

func (r *FragmentReassembler) Reset() {
	r.active = false
	r.n = 0
}

// Push copies elem into dst and returns a complete OBU when available.
func (r *FragmentReassembler) Push(dst []byte, elem Element) (n int, complete bool, err error) {
	if elem.ContinuesPrevious {
		if !r.active {
			return 0, false, ErrUnexpectedContinuation
		}
	} else if r.active {
		return 0, false, ErrFragmentInterrupted
	}

	if len(dst)-r.n < len(elem.Data) {
		return 0, false, ErrShortBuffer
	}
	r.n += copy(dst[r.n:], elem.Data)

	if elem.ContinuesNext {
		r.active = true
		return r.n, false, nil
	}

	n = r.n
	r.Reset()
	return n, true, nil
}
