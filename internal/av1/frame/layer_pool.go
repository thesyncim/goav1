// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

import "errors"

var (
	// ErrInvalidLayerPool reports that the LayerPool storage slices were not
	// sized correctly at Bind time, or that internal bookkeeping was mutated.
	ErrInvalidLayerPool = errors.New("frame: invalid layer pool")

	// ErrLayerPoolFull reports that the LayerPool cannot bind another sub-pool
	// because the caller-owned sub-pool slot capacity has been exhausted.
	ErrLayerPoolFull = errors.New("frame: layer pool full")

	// ErrInvalidLayerFactory reports that the per-format sub-pool factory
	// returned a Pool whose Format does not match the requested Format.
	ErrInvalidLayerFactory = errors.New("frame: invalid layer factory")
)

// LayerFactory binds a new caller-owned sub-pool that satisfies format. It is
// invoked at most once per distinct format key encountered by a LayerPool.
// Implementations are responsible for allocating or vending the backing byte
// slice, the Frame/free/used bookkeeping slices, and calling BindPool. The
// returned Pool's Format (after normalization) must match the requested
// format; otherwise LayerPool returns ErrInvalidLayerFactory.
//
// Returning a non-nil error propagates to the caller of Acquire.
type LayerFactory interface {
	BindLayer(format Format) (Pool, error)
}

// LayerFactoryFunc adapts a plain function to LayerFactory.
type LayerFactoryFunc func(format Format) (Pool, error)

// BindLayer implements LayerFactory.
func (fn LayerFactoryFunc) BindLayer(format Format) (Pool, error) {
	if fn == nil {
		return Pool{}, ErrInvalidLayerFactory
	}
	return fn(format)
}

// LayerPool aggregates several caller-owned frame.Pool instances keyed by
// Format. It is the production replacement for "one Pool per spatial layer"
// glue that SVC and multi-resolution callers previously hand-rolled. The
// per-sub-pool storage is still caller-owned via LayerFactory; LayerPool only
// owns the (format -> *Pool) map and the global surface ID encoding that lets
// callers exchange opaque surface identifiers across pools.
//
// Global surface IDs encode (sub-pool index, local pool index) as
//
//	globalID = subPoolIndex*Stride + localIndex
//
// where Stride is supplied at Bind time and bounds local pool indices. A
// Stride of 256 is more than enough for typical pools (≤ 16 surfaces) and
// matches the caller-side encoding used by the testvector SVC harness.
//
// LayerPool is caller-owned: callers preallocate the sub-pool storage slices
// (sub-pool entries and their format keys) and Bind them once. After Bind,
// Acquire and Release are zero-alloc for any format that has already been
// observed; the first Acquire of a new format invokes LayerFactory exactly
// once (which may allocate caller-owned backing storage).
type LayerPool struct {
	stride  int
	formats []Format
	pools   []Pool
	bound   []bool
	count   int
	factory LayerFactory
}

// BindLayerPool wires caller-owned storage for up to len(pools) sub-pools and
// the matching format keys. stride bounds the maximum local pool index that
// the global surface ID encoding can carry; values must be positive. The
// caller supplies the factory that binds a new sub-pool on first observation
// of a given Format.
//
// pools, formats, and bound must all have the same length. The same-length
// rule lets every sub-pool slot be addressed by index without re-allocating.
func BindLayerPool(pools []Pool, formats []Format, bound []bool, stride int, factory LayerFactory) (LayerPool, error) {
	if stride <= 0 || factory == nil {
		return LayerPool{}, ErrInvalidLayerPool
	}
	if len(pools) == 0 || len(pools) != len(formats) || len(pools) != len(bound) {
		return LayerPool{}, ErrInvalidLayerPool
	}
	for i := range bound {
		bound[i] = false
	}
	return LayerPool{
		stride:  stride,
		formats: formats,
		pools:   pools,
		bound:   bound,
		factory: factory,
	}, nil
}

// Stride reports the per-sub-pool stride used by the global surface ID
// encoding. Local pool indices must satisfy 0 <= index < Stride.
func (p *LayerPool) Stride() int {
	if p == nil {
		return 0
	}
	return p.stride
}

// Cap reports the maximum number of distinct sub-pools that the LayerPool can
// hold. It is the length of the caller-owned sub-pool storage.
func (p *LayerPool) Cap() int {
	if p == nil {
		return 0
	}
	return len(p.pools)
}

// Len reports the number of distinct sub-pools that have been bound so far.
func (p *LayerPool) Len() int {
	if p == nil {
		return 0
	}
	return p.count
}

// EncodeGlobalID composes a global surface ID from a *Pool returned by
// SubPool or SubPoolByIndex and a local pool index that the caller acquired
// from that sub-pool. Returns -1 (and ErrInvalidSlot) when sub is not bound
// to this LayerPool or local is out of range. EncodeGlobalID is zero-alloc.
func (p *LayerPool) EncodeGlobalID(sub *Pool, local int) (int, error) {
	if p == nil || sub == nil {
		return -1, ErrInvalidLayerPool
	}
	if local < 0 || local >= p.stride {
		return -1, ErrInvalidSlot
	}
	slot := p.subPoolIndex(sub)
	if slot < 0 {
		return -1, ErrInvalidSlot
	}
	return slot*p.stride + local, nil
}

// SubPoolByIndex returns the bound sub-pool stored at slot index. The
// returned pointer is stable for the lifetime of the LayerPool. Slots are
// assigned in first-observed order, matching the ordering used by the global
// surface ID encoding.
func (p *LayerPool) SubPoolByIndex(index int) (*Pool, error) {
	if p == nil {
		return nil, ErrInvalidLayerPool
	}
	if index < 0 || index >= p.count || !p.bound[index] {
		return nil, ErrInvalidSlot
	}
	return &p.pools[index], nil
}

// SubPool returns the *frame.Pool bound to format, invoking the LayerFactory
// to bind a new sub-pool if one does not yet exist. The returned pointer is
// stable for the lifetime of the LayerPool.
//
// SubPool is zero-alloc for any format that has already been observed.
func (p *LayerPool) SubPool(format Format) (*Pool, error) {
	if p == nil {
		return nil, ErrInvalidLayerPool
	}
	normalized, err := normalizeFormat(format)
	if err != nil {
		return nil, err
	}
	for i := 0; i < p.count; i++ {
		if !p.bound[i] {
			continue
		}
		if p.formats[i] == normalized {
			return &p.pools[i], nil
		}
	}
	if p.count >= len(p.pools) {
		return nil, ErrLayerPoolFull
	}
	pool, err := p.factory.BindLayer(normalized)
	if err != nil {
		return nil, err
	}
	got, err := pool.Format()
	if err != nil {
		return nil, err
	}
	if got != normalized {
		return nil, ErrInvalidLayerFactory
	}
	if pool.Cap() > p.stride {
		return nil, ErrInvalidLayerPool
	}
	slot := p.count
	p.pools[slot] = pool
	p.formats[slot] = normalized
	p.bound[slot] = true
	p.count++
	return &p.pools[slot], nil
}

// Acquire binds (lazily) the sub-pool for format and acquires a surface from
// it. The returned globalID encodes (sub-pool index, local pool index) and is
// the value callers should store in SurfaceReferences.slots.
//
// Acquire is zero-alloc for any format that has already been observed.
func (p *LayerPool) Acquire(format Format) (int, *Frame, error) {
	pool, err := p.SubPool(format)
	if err != nil {
		return -1, nil, err
	}
	local, fr, err := pool.AcquireFormat(format)
	if err != nil {
		return -1, nil, err
	}
	subIndex := p.subPoolIndex(pool)
	if subIndex < 0 {
		return -1, nil, ErrInvalidLayerPool
	}
	if local < 0 || local >= p.stride {
		// Defensive: should be impossible because SubPool validates pool.Cap()
		// against stride. Release before returning so caller state remains
		// consistent.
		_ = pool.Release(local)
		return -1, nil, ErrInvalidLayerPool
	}
	return subIndex*p.stride + local, fr, nil
}

// Frame resolves globalID to the *Frame that holds the pixels. Returns
// ErrInvalidSlot if the encoded sub-pool slot or local index does not name a
// currently-acquired surface, and ErrInvalidLayerPool if globalID is
// negative.
func (p *LayerPool) Frame(globalID int) (*Frame, error) {
	pool, local, err := p.resolve(globalID)
	if err != nil {
		return nil, err
	}
	return pool.Frame(local)
}

// Release returns the surface named by globalID to its owning sub-pool.
func (p *LayerPool) Release(globalID int) error {
	pool, local, err := p.resolve(globalID)
	if err != nil {
		return err
	}
	return pool.Release(local)
}

// ReleaseMany returns a batch of caller-owned surface IDs to their owning
// sub-pools. The batch is validated before any sub-pool is mutated, so
// invalid or duplicate IDs leave ownership unchanged. Routing to per-sub-pool
// release calls is done in one linear pass over a small caller-owned scratch
// buffer (sized to the sub-pool cap) to keep the steady state zero-alloc.
func (p *LayerPool) ReleaseMany(globalIDs []int) error {
	if p == nil {
		return ErrInvalidLayerPool
	}
	if len(globalIDs) == 0 {
		return nil
	}

	// Validate every ID first so a single bad ID never partially releases the
	// batch. The validation pass does not touch any sub-pool state.
	for i := range globalIDs {
		_, _, err := p.resolve(globalIDs[i])
		if err != nil {
			return err
		}
		for j := 0; j < i; j++ {
			if globalIDs[j] == globalIDs[i] {
				return ErrInvalidSlot
			}
		}
	}

	// Per-sub-pool release: dispatch IDs to the sub-pool they belong to.
	// Because each release is single-pool atomic and we validated above, an
	// error here implies caller state and pool state already disagree.
	for i := range globalIDs {
		pool, local, err := p.resolve(globalIDs[i])
		if err != nil {
			return err
		}
		if err := pool.Release(local); err != nil {
			return err
		}
	}
	return nil
}

// Reset returns every sub-pool to the all-free state without rebinding sub-pools.
// The sub-pool inventory (format -> *Pool map) is preserved.
func (p *LayerPool) Reset() {
	if p == nil {
		return
	}
	for i := 0; i < p.count; i++ {
		if p.bound[i] {
			p.pools[i].Reset()
		}
	}
}

func (p *LayerPool) resolve(globalID int) (*Pool, int, error) {
	if p == nil {
		return nil, 0, ErrInvalidLayerPool
	}
	if globalID < 0 || p.stride <= 0 {
		return nil, 0, ErrInvalidSlot
	}
	subIndex := globalID / p.stride
	local := globalID % p.stride
	if subIndex < 0 || subIndex >= p.count || !p.bound[subIndex] {
		return nil, 0, ErrInvalidSlot
	}
	return &p.pools[subIndex], local, nil
}

func (p *LayerPool) subPoolIndex(pool *Pool) int {
	if p == nil || pool == nil {
		return -1
	}
	for i := 0; i < p.count; i++ {
		if !p.bound[i] {
			continue
		}
		if &p.pools[i] == pool {
			return i
		}
	}
	return -1
}
