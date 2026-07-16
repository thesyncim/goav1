package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func setGridRecordCellForTest(c *BlockModeContext, x4, y4 int, record blockModeGridRecord) {
	c.gridOwners[y4][x4] = gridRecordOwner(x4, y4)
	c.gridRecords[y4][x4] = record
}

func gridRecordCellForTest(c *BlockModeContext, x4, y4 int) (blockModeGridRecord, bool) {
	record, ok := c.gridRecordAt(x4, y4)
	if !ok {
		return blockModeGridRecord{}, false
	}
	return *record, true
}

func TestGridRecordOwnerRectangleMatchesEveryBlockSize(t *testing.T) {
	motionResult := InterMotionResult{
		MV:         [2]motion.Vector{{Row: 7, Col: -11}},
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		Mode:       InterModeResult{Mode: InterModeNewMV},
	}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	for size := BlockSize128x128; size < blockSizeCount; size++ {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("size %d has no dimensions", size)
		}
		x4 := MaxBlockModeSlots - int(dims.W4)
		y4 := MaxBlockModeSlots - int(dims.H4)
		var ctx BlockModeContext
		ctx.markGridInterMotionAndFilters(size, x4, y4, motionResult, filters, dims)
		wantOwner := gridRecordOwner(x4, y4)
		for y := range MaxBlockModeSlots {
			for x := range MaxBlockModeSlots {
				inside := x >= x4 && y >= y4
				if !inside {
					if ctx.gridOwners[y][x] != 0 {
						t.Fatalf("size %d owner leaked to (%d,%d)", size, x, y)
					}
					continue
				}
				if ctx.gridOwners[y][x] != wantOwner {
					t.Fatalf("size %d owner[%d][%d]=%d want %d", size, y, x, ctx.gridOwners[y][x], wantOwner)
				}
				record, ok := ctx.gridRecordAt(x, y)
				if !ok || record.Motion != motionResult || record.Filters != filters || record.Size != size ||
					record.Flags != gridRecordMotionValid|gridRecordInterpValid|gridRecordSizeVisited {
					t.Fatalf("size %d record[%d][%d]=%+v ok=%v", size, y, x, record, ok)
				}
			}
		}
	}
}
