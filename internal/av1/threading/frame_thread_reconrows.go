// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameThreadReconRows is the reconstruction progress of a frame-threaded
// frame, in luma rows: dav1d's per-tile ts->progress[0] of the
// DAV1D_TASK_TYPE_TILE_RECONSTRUCTION tasks, which the frame's post-filter
// task for superblock row sby waits on across every tile column
// (src/thread_task.c, the filter task's recon_progress check). Progress holds
// the largest row R such that every row above R is reconstructed in every
// tile, FrameProgressDone once every tile finished, or FrameProgressError once
// a reconstruction lane failed.
//
// Rows finish in order within a tile: the last superblock of SB row r+1
// waits for the whole of row r (reconstructRow's top-right gate).
type FrameThreadReconRows struct {
	Progress FrameProgress

	mu        sync.Mutex
	tiles     []frameThreadReconTile
	published int32
}

type frameThreadReconTile struct {
	rows int32
	end  int32
	used bool
}

// bindTiles registers every job's luma-row span before any lane runs.
func (r *FrameThreadReconRows) bindTiles(base FrameWorkBatch, jobs []tile.Job) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	maxTile := 0
	for i := range jobs {
		maxTile = max(maxTile, int(jobs[i].Tile))
	}
	if cap(r.tiles) <= maxTile {
		r.tiles = make([]frameThreadReconTile, maxTile+1)
	}
	r.tiles = r.tiles[:maxTile+1]
	clear(r.tiles)
	ctx := base
	ctx.Jobs = jobs
	ctx.geomCache = nil
	for i := range jobs {
		region, err := ctx.JobRegion(i)
		if err != nil {
			return err
		}
		t := &r.tiles[jobs[i].Tile]
		t.rows = int32(region.MIRowStart) * 4
		t.end = int32(region.MIRowEnd) * 4
		t.used = true
	}
	return nil
}

// Reset marks no row reconstructed; the owner calls it before the frame's
// post-filter can start following the progress.
func (r *FrameThreadReconRows) Reset() {
	r.mu.Lock()
	r.published = 0
	clear(r.tiles)
	r.Progress.Reset()
	r.mu.Unlock()
}

// finishRows records that tile t reconstructed its rows up to rowEnd (luma)
// and publishes the frame's new contiguous row count when it grew.
func (r *FrameThreadReconRows) finishRows(t int, rowEnd int32) {
	r.mu.Lock()
	if t < 0 || t >= len(r.tiles) || !r.tiles[t].used {
		r.mu.Unlock()
		return
	}
	tt := &r.tiles[t]
	tt.rows = max(tt.rows, min(rowEnd, tt.end))
	v := FrameProgressDone
	for i := range r.tiles {
		x := &r.tiles[i]
		if x.used && x.rows < x.end && x.rows < v {
			v = x.rows
		}
	}
	if v > r.published && r.published != FrameProgressError {
		r.published = v
		r.Progress.Publish(v)
	}
	r.mu.Unlock()
}

// Fail ends the progress with FrameProgressError unless every row finished:
// a reconstruction lane failed, or the entropy pass failed and the lanes were
// cancelled.
func (r *FrameThreadReconRows) Fail() {
	r.mu.Lock()
	if r.published != FrameProgressDone {
		r.published = FrameProgressError
		r.Progress.Publish(FrameProgressError)
	}
	r.mu.Unlock()
}
