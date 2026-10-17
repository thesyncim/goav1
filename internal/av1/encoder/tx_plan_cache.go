package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// realtimeInterTXPlanEntry is one precomputed realtime TX plan with its leaves
// in walkLeaf order.
type realtimeInterTXPlanEntry struct {
	plan   realtimeInterTXPlan
	leaves int
	dx, dy [64]uint8
}

// realtimeInterTXPlans caches every (block size, leaf size) plan: a plan is a
// pure function of the pair, and building one per block, then walking its
// tree for every leaf visit, is per-block work otherwise.
var realtimeInterTXPlans = func() (t [int(tile.BlockSize4x4) + 1][2]*realtimeInterTXPlanEntry) {
	for b := range t {
		block := tile.BlockSize(b)
		for li, leaf := range [2]int{8, 16} {
			plan, err := realtimeInterTXPlanBuild(block, leaf)
			if err != nil {
				continue
			}
			maxY, err := tile.MaxTransformSize(block, parser.ColorConfig{}, 0)
			if err != nil {
				continue
			}
			e := &realtimeInterTXPlanEntry{plan: plan}
			i := 0
			if err := plan.walkLeaf(maxY, 0, 0, 0, 0, 0, &i, func(i, dx, dy int) error {
				if i >= len(e.dx) || dx > 255 || dy > 255 {
					return tile.ErrInvalidDecodeState
				}
				e.dx[i], e.dy[i] = uint8(dx), uint8(dy)
				return nil
			}); err != nil {
				continue
			}
			e.leaves = i
			t[b][li] = e
		}
	}
	return t
}()

func realtimeInterTXPlanCached(block tile.BlockSize, leafSize int) *realtimeInterTXPlanEntry {
	if int(block) >= len(realtimeInterTXPlans) || (leafSize != 8 && leafSize != 16) {
		return nil
	}
	return realtimeInterTXPlans[block][leafSize>>4]
}
