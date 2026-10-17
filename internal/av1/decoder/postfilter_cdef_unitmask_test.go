package decoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// TestCDEFUnitFilterMaskMatchesPerBlock checks the cached per-unit mask path
// against frameWorkCDEFBlockPositionsFiltered for every unit of random skip
// maps, including units overhanging the map and partial (edge) unit extents.
func TestCDEFUnitFilterMaskMatchesPerBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for iter := 0; iter < 200; iter++ {
		miCols := 1 + rng.Intn(70)
		miRows := 1 + rng.Intn(70)
		recs := make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows)
		density := rng.Intn(4)
		for i := range recs {
			recs[i].Valid = rng.Intn(20) != 0
			recs[i].SkipTransform = rng.Intn(4) < density
		}
		skipMap := &FrameWorkLoopFilterMap{Records: recs, Stride: uint16(miCols), Rows: uint16(miRows)}
		cols := (miCols*4 + 63) / 64
		rows := (miRows*4 + 63) / 64
		var cache frameWorkCDEFUnitMaskCache
		if !cache.reset(skipMap, 0, rows, cols) {
			t.Fatal("cache rejected a small band")
		}
		var a, b [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
		for ur := 0; ur < rows; ur++ {
			for uc := 0; uc < cols; uc++ {
				for _, dims := range [][4]int{{64, 64, 8, 8}, {32, 32, 4, 4}, {64, 32, 8, 4}, {1 + rng.Intn(64), 1 + rng.Intn(64), 8, 8}, {1 + rng.Intn(32), 1 + rng.Intn(32), 4, 4}} {
					want := frameWorkCDEFBlockPositionsFiltered(a[:], dims[0], dims[1], dims[2], dims[3], skipMap, ur, uc)
					got := frameWorkCDEFBlockPositionsFromMask(b[:], cache.mask(ur, uc), dims[0], dims[1], dims[2], dims[3])
					if len(got) != len(want) {
						t.Fatalf("iter %d unit (%d,%d) dims %v: %d blocks, want %d", iter, ur, uc, dims, len(got), len(want))
					}
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("iter %d unit (%d,%d) dims %v: block %d = %+v, want %+v", iter, ur, uc, dims, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}
