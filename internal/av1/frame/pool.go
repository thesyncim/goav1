package frame

// Pool is a deterministic LIFO pool over caller-owned frame storage.
type Pool struct {
	frames []Frame
	free   []int
	used   []bool
	top    int
}

// BindPool binds count frame slots over backing. The caller owns every buffer:
// backing holds pixels, frames holds descriptors, free is the reusable stack,
// and used tracks checked release.
func BindPool(backing []byte, format Format, frames []Frame, free []int, used []bool) (Pool, error) {
	count := len(frames)
	if count == 0 || len(free) < count || len(used) < count {
		return Pool{}, ErrInvalidPool
	}

	layout, err := RequiredSize(format)
	if err != nil {
		return Pool{}, err
	}
	needed, ok := checkedMul(layout.Size, count)
	if !ok {
		return Pool{}, ErrInvalidFormat
	}
	if len(backing) < needed {
		return Pool{}, ErrShortBuffer
	}

	p := Pool{
		frames: frames[:count],
		free:   free[:count],
		used:   used[:count],
		top:    count,
	}
	for i := range count {
		start := i * layout.Size
		end := start + layout.Size
		frame, err := Bind(backing[start:end], format)
		if err != nil {
			return Pool{}, err
		}
		p.frames[i] = frame
		p.free[i] = count - 1 - i
		p.used[i] = false
	}
	return p, nil
}

func (p *Pool) Cap() int {
	if p == nil {
		return 0
	}
	return len(p.frames)
}

func (p *Pool) Available() int {
	if p == nil {
		return 0
	}
	return p.top
}

func (p *Pool) Format() (Format, error) {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top > len(p.frames) {
		return Format{}, ErrInvalidPool
	}
	return p.frames[0].Format, nil
}

func (p *Pool) Layout() (Layout, error) {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top > len(p.frames) {
		return Layout{}, ErrInvalidPool
	}
	return p.frames[0].Layout, nil
}

func (p *Pool) Frame(index int) (*Frame, error) {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top > len(p.frames) {
		return nil, ErrInvalidPool
	}
	if index < 0 || index >= len(p.frames) || !p.used[index] {
		return nil, ErrInvalidSlot
	}
	return &p.frames[index], nil
}

func (p *Pool) Acquire() (int, *Frame, error) {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top > len(p.frames) {
		return -1, nil, ErrInvalidPool
	}
	if p.top == 0 {
		return -1, nil, ErrPoolEmpty
	}

	p.top--
	index := p.free[p.top]
	if index < 0 || index >= len(p.frames) || p.used[index] {
		return -1, nil, ErrInvalidPool
	}
	p.used[index] = true
	return index, &p.frames[index], nil
}

func (p *Pool) AcquireFormat(format Format) (int, *Frame, error) {
	format, err := normalizeFormat(format)
	if err != nil {
		return -1, nil, err
	}
	poolFormat, err := p.Format()
	if err != nil {
		return -1, nil, err
	}
	if poolFormat != format {
		return -1, nil, ErrInvalidFormat
	}
	return p.Acquire()
}

func (p *Pool) Release(index int) error {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top >= len(p.frames)+1 {
		return ErrInvalidPool
	}
	if index < 0 || index >= len(p.frames) || !p.used[index] {
		return ErrInvalidSlot
	}
	if p.top >= len(p.frames) {
		return ErrInvalidPool
	}
	p.used[index] = false
	p.free[p.top] = index
	p.top++
	return nil
}

// ReleaseMany returns a caller-owned batch of frame slots to the pool. The
// batch is validated before the pool is modified, so invalid or duplicate
// indices leave ownership unchanged.
func (p *Pool) ReleaseMany(indices []int) error {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top > len(p.frames) {
		return ErrInvalidPool
	}
	if len(indices) == 0 {
		return nil
	}

	for i := range indices {
		index := indices[i]
		if index < 0 || index >= len(p.frames) || !p.used[index] {
			return ErrInvalidSlot
		}
		for j := range i {
			if indices[j] == index {
				return ErrInvalidSlot
			}
		}
	}

	for i := range indices {
		index := indices[i]
		p.used[index] = false
		p.free[p.top] = index
		p.top++
	}
	return nil
}

func (p *Pool) Reset() {
	if p == nil {
		return
	}
	count := len(p.frames)
	if len(p.free) < count || len(p.used) < count {
		p.top = 0
		return
	}
	for i := range count {
		p.free[i] = count - 1 - i
		p.used[i] = false
	}
	p.top = count
}
