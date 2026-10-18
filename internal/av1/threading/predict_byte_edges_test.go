package threading

import (
	"bytes"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFreshByteDirectionalEdgesIgnorePoisonOutsideConsumedSpans(t *testing.T) {
	for _, size := range []int{4, 8, 16, 32, 64} {
		for _, angle := range []int{45, 67, 90, 113, 135, 157, 180, 203, 225} {
			for available := 0; available < 4; available++ {
				for _, filter := range []bool{false, true} {
					var scratch FrameWorkIntraPredictionScratch
					for i := range scratch.Above {
						scratch.Above[i] = 0xffff
						scratch.Left[i] = 0xffff
					}
					for i := range scratch.Edge {
						scratch.Edge[i] = 0xffff
					}
					block := tile.BlockVisit{HaveTop: available&1 != 0, HaveLeft: available&2 != 0}
					x, y := 0, 0
					if block.HaveLeft {
						x = 1
					}
					if block.HaveTop {
						y = 1
					}
					plane := frame.Plane{Pix: make([]byte, 96*96), Stride: 96, Width: 96, Height: 96}
					for i := range plane.Pix {
						plane.Pix[i] = byte(i*79 + 3)
					}
					edges, err := frameWorkDirectionalPredictionEdges(plane, 1, 8, x, y, size, size, angle, block, &scratch, filter, false, true, true, x+size-1, y+size-1)
					if err != nil {
						t.Fatalf("gather size%d angle%d available%d filter%v: %v", size, angle, available, filter, err)
					}
					want, got := plane, plane
					want.Pix = slices.Clone(plane.Pix)
					got.Pix = slices.Clone(plane.Pix)
					// The general entry scans every consumed sample and therefore
					// rejects any poison left reachable by the fresh-gather route.
					we := prediction.PredictDirectionalIntraPlaneBlockWithExtent(want, 1, 8, x, y, size-1, size-1, size, size, angle, edges)
					ge := prediction.PredictDirectionalIntraPlaneBlockWithExtentBounded(got, 1, 8, x, y, size-1, size-1, size, size, angle, edges, true)
					if we != nil || ge != nil || !bytes.Equal(want.Pix, got.Pix) {
						t.Fatalf("size%d angle%d available%d filter%v: %v/%v or pixels differ", size, angle, available, filter, we, ge)
					}
				}
			}
		}
	}
}

func TestDirectDirectionalEdgeFilterStillRejectsUnboundedSamples(t *testing.T) {
	var scratch FrameWorkIntraPredictionScratch
	for i := range scratch.Above {
		scratch.Above[i] = 0xffff
		scratch.Left[i] = 0xffff
	}
	edges := prediction.DirectionalEdges{Above: scratch.Above[:], Left: scratch.Left[:], AboveOrigin: frameWorkDirectionalEdgeOrigin, LeftOrigin: frameWorkDirectionalEdgeOrigin}
	if err := frameWorkApplyDirectionalIntraEdgeFilter(8, 16, 16, 135, tile.BlockVisit{HaveTop: true, HaveLeft: true}, false, &edges, &scratch); err == nil {
		t.Fatal("direct helper accepted unbounded caller-supplied samples")
	}
}
