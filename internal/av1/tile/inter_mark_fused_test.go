package tile

import (
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestMarkInterMotionAndFiltersMatchesSeparateUpdates(t *testing.T) {
	result := InterMotionResult{
		MV:         [2]motion.Vector{{Row: 7, Col: -11}},
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		Mode:       InterModeResult{Mode: InterModeNearMV},
	}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	for size := BlockSize128x128; size < blockSizeCount; size++ {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("size %d has no dimensions", size)
		}
		x4 := MaxBlockModeSlots - int(dims.W4)
		y4 := MaxBlockModeSlots - int(dims.H4)

		var separate BlockModeContext
		separate.AboveIntra[x4] = 1
		separate.LeftIntra[y4] = 1
		separate.GridMotionValid[y4][x4] = 1
		separate.GridInterpValid[y4][x4] = 1
		separate.GridBlockSizeVisited[y4][x4] = 1
		fused := separate

		if err := separate.MarkInterMotion(size, x4, y4, result, true); err != nil {
			t.Fatalf("size %d separate motion: %v", size, err)
		}
		if err := separate.MarkInterFilters(size, x4, y4, result.References, filters); err != nil {
			t.Fatalf("size %d separate filters: %v", size, err)
		}
		if err := fused.markInterMotionAndFilters(size, x4, y4, result, true, filters); err != nil {
			t.Fatalf("size %d fused: %v", size, err)
		}
		if !reflect.DeepEqual(&fused, &separate) {
			t.Fatalf("size %d fused inter state differs from separate motion/filter updates", size)
		}
	}
}
