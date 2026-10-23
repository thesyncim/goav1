// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"math/rand"
	"testing"
)

// TestFrameThreadMotionRowsAggregatesTileGrid checks the multi-tile MV_REF
// progress against a brute-force reference: the published row must be the
// largest R such that every 8x8 row below R is saved in every tile column, for
// tile grids with one or several tile rows/columns and non-uniform spacing.
func TestFrameThreadMotionRowsAggregatesTileGrid(t *testing.T) {
	type grid struct {
		name     string
		rowStart []int32 // tile row starts in 8x8 rows, plus the frame end
		cols     int
	}
	grids := []grid{
		{"1x4", []int32{0, 34}, 4},
		{"4x1", []int32{0, 9, 18, 27, 34}, 1},
		{"4x2", []int32{0, 18, 34}, 4},
		{"nonuniform-3x3", []int32{0, 4, 24, 34}, 3},
		{"nonuniform-2x2", []int32{0, 30, 34}, 2},
	}
	rng := rand.New(rand.NewSource(1))
	for _, g := range grids {
		t.Run(g.name, func(t *testing.T) {
			var tiles []FrameThreadMotionTile
			for r := 0; r+1 < len(g.rowStart); r++ {
				for c := 0; c < g.cols; c++ {
					tiles = append(tiles, FrameThreadMotionTile{rows: g.rowStart[r], end: g.rowStart[r+1], used: true})
				}
			}
			frameEnd := g.rowStart[len(g.rowStart)-1]
			for step := 0; step < 2000; step++ {
				got := frameThreadMotionRows(tiles)
				// Reference: row y is saved when the tile covering it in
				// every column has rows > y.
				want := frameEnd
				for y := int32(0); y < frameEnd && want == frameEnd; y++ {
					for i := range tiles {
						if y >= tiles[i].rows && y < tiles[i].end && y >= startOf(tiles[i], g.rowStart) {
							want = y
							break
						}
					}
				}
				if want == frameEnd {
					want = FrameProgressDone
				}
				if got != want {
					t.Fatalf("step %d: got %d want %d tiles %+v", step, got, want, tiles)
				}
				if got == FrameProgressDone {
					break
				}
				// Advance a random unfinished tile by one SB row (2 8x8 rows).
				var open []int
				for i := range tiles {
					if tiles[i].rows < tiles[i].end {
						open = append(open, i)
					}
				}
				i := open[rng.Intn(len(open))]
				tiles[i].rows = min(tiles[i].rows+2, tiles[i].end+1)
			}
		})
	}
}

func startOf(tile FrameThreadMotionTile, rowStart []int32) int32 {
	for i := 0; i+1 < len(rowStart); i++ {
		if rowStart[i+1] == tile.end {
			return rowStart[i]
		}
	}
	return 0
}
