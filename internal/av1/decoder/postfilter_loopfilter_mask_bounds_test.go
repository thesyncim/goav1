package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
)

func TestFrameWorkFillLoopFilterLevelsPreservesOtherComponents(t *testing.T) {
	cache := [][4]uint8{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}
	frameWorkFillLoopFilterLevels(cache, 1, 2, 0, 1, 21, 22)
	want := [][4]uint8{{1, 2, 3, 4}, {21, 22, 7, 8}, {21, 22, 11, 12}, {13, 14, 15, 16}}
	for i := range want {
		if cache[i] != want[i] {
			t.Fatalf("cell %d=%v, want %v", i, cache[i], want[i])
		}
	}
}

func TestFrameWorkLoopFilterMaskCellWidthMatchesCheckedBounds(t *testing.T) {
	widths := [...]uint8{4, 6, 8, 14}
	for posWidth := int32(4); posWidth <= 68; posWidth += 4 {
		for posHeight := int32(4); posHeight <= 68; posHeight += 4 {
			for bufWidth := posWidth; bufWidth <= posWidth+4; bufWidth += 4 {
				for bufHeight := posHeight; bufHeight <= posHeight+4; bufHeight += 4 {
					bounds := frameWorkLoopFilterBounds{
						posWidth:  posWidth,
						posHeight: posHeight,
						bufWidth:  bufWidth,
						bufHeight: bufHeight,
					}
					for y4 := 0; y4 <= int(bounds.bufHeight>>2)+1; y4++ {
						for x4 := 0; x4 <= int(bounds.bufWidth>>2)+1; x4++ {
							for edge := loopfilter.EdgeVertical; edge <= loopfilter.EdgeHorizontal; edge++ {
								for _, width := range widths {
									wantWidth, wantOK := uint8(0), false
									length, err := frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, 1)
									if err != nil {
										t.Fatal(err)
									}
									if length > 0 {
										wantWidth, wantOK, err = frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, 1, width)
										if err != nil {
											t.Fatal(err)
										}
									}
									gotWidth, gotOK := frameWorkLoopFilterMaskCellWidth(bounds, edge, x4, y4, width)
									if gotWidth != wantWidth || gotOK != wantOK {
										t.Fatalf("bounds=%+v edge=%d cell=(%d,%d) width=%d: got (%d,%v), want (%d,%v)", bounds, edge, x4, y4, width, gotWidth, gotOK, wantWidth, wantOK)
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
