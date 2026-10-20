package threading

import (
	"sync"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// copyCDFStorage is *dst = *src as a single memmove.
// FrameWorkTileResidualCDFStorage holds only fixed-size CDF arrays and no
// pointers (TestCDFStorageIsPointerFree), so a byte copy is exact. TinyGo
// lowers the plain assignment of this ~56 KB struct through a heap
// temporary, which made every frame allocate several of them.
func copyCDFStorage(dst, src *FrameWorkTileResidualCDFStorage) {
	n := unsafe.Sizeof(*dst)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(dst)), n), unsafe.Slice((*byte)(unsafe.Pointer(src)), n))
}

// defaultCDFStorage caches the default CDF storage per coefficient
// q-context; every other table in the storage is independent of the
// quantizer, so one entry per context covers all base q indexes.
var defaultCDFStorage [tile.CoeffQContexts]struct {
	once    sync.Once
	storage FrameWorkTileResidualCDFStorage
	err     error
}

func defaultTileResidualCDFStorage(baseQIndex uint8) (*FrameWorkTileResidualCDFStorage, error) {
	d := &defaultCDFStorage[tile.CoeffQContext(baseQIndex)]
	d.once.Do(func() { d.err = d.storage.buildDefault(baseQIndex) })
	return &d.storage, d.err
}
