// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

// Pool is a deterministic LIFO pool over caller-owned frame storage. Each slot
// is sized for the pool format (normally the sequence's maximum frame size);
// AcquireFormat binds a slot to any frame size that fits, as dav1d allocates
// each picture at its own frame size.
type Pool struct {
	frames  []Frame
	free    []int
	used    []bool
	top     int
	backing []byte
	layout  Layout
	format  Format
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
		frames:  frames[:count],
		free:    free[:count],
		used:    used[:count],
		top:     count,
		backing: backing[:needed],
		layout:  layout,
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
	p.format = p.frames[0].Format
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
	return p.format, nil
}

func (p *Pool) Layout() (Layout, error) {
	if p == nil || len(p.frames) == 0 || len(p.free) < len(p.frames) || len(p.used) < len(p.frames) || p.top < 0 || p.top > len(p.frames) {
		return Layout{}, ErrInvalidPool
	}
	return p.layout, nil
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
	return p.acquire(p.format)
}

func (p *Pool) acquire(format Format) (int, *Frame, error) {
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
	if p.frames[index].Format != format {
		// Rebind the free slot to this frame's size. Live frames keep their
		// descriptors; only a slot leaving the free list changes shape.
		start := index * p.layout.Size
		frame, err := Bind(p.backing[start:start+p.layout.Size], format)
		if err != nil {
			p.top++
			return -1, nil, err
		}
		p.frames[index] = frame
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
	if !poolFormatFits(poolFormat, p.layout, format) {
		return -1, nil, ErrInvalidFormat
	}
	return p.acquire(format)
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

// poolFormatFits reports whether format can be bound into a slot of the pool
// format: the same sample format and superblock alignment, and a layout no
// larger than the slot.
func poolFormatFits(pool Format, slot Layout, format Format) bool {
	if format == pool {
		return true
	}
	if format.BitDepth != pool.BitDepth || format.MonoChrome != pool.MonoChrome ||
		format.SubsamplingX != pool.SubsamplingX || format.SubsamplingY != pool.SubsamplingY ||
		format.SBSizeLog2 != pool.SBSizeLog2 || format.Align != pool.Align {
		return false
	}
	layout, err := RequiredSize(format)
	return err == nil && layout.Size <= slot.Size
}
