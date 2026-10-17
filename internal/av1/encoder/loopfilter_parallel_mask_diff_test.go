package encoder

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// loopfilter_parallel_mask_diff_test.go is the fast differential gate for the
// encoder's parallel mask-driven single-tile loop-filter apply
// (loopFilterApplier.applyParallelMasks). Its core assertion is that the parallel
// banded mask apply is byte-identical to the serial single-shot mask apply
// (applySerialMasks) -- which the existing loopfilter_mask_diff_test.go already
// pins byte-identical to the edge-list applySerial -- so any divergence introduced
// by the band fan-out / direction barrier shows up as a changed pixel. It sweeps
// several region-row counts (frame heights) so the luma fan-out spans 1..5 bands,
// and at frame sizes the dense-grid edge-list planner still fits it also runs the
// full four-way check (serial edge-list, parallel edge-list, serial mask, parallel
// mask all agree). It runs in milliseconds; run under -race to prove the fan-out
// has no data races on the shared masks / level cache / reconstruction planes.

// lfParallelMaskRun deblocks copies of the same reconstruction with the serial
// mask apply (ground truth) and the parallel mask apply, and asserts byte-identical
// planes. When fourWay is set it additionally runs the serial and parallel
// edge-list applies and asserts the whole family agrees (only valid at frame sizes
// whose fully dense grid stays within the edge-list planner's edge-counter cap).
func lfParallelMaskRun(t *testing.T, w, h, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult, lf parser.LoopFilterParams, fourWay bool) {
	t.Helper()
	var a loopFilterApplier
	if err := a.init(w, h); err != nil {
		t.Fatalf("init: %v", err)
	}
	defer a.close()
	if err := a.reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	lfDiffFillGrid(a.filtMap, bw4, bh4, size, tree)

	src := lfDiffFrame(w, h)
	serialMasks := lfDiffCopy(src)
	parMasks := lfDiffCopy(src)

	if err := a.applySerialMasks(&serialMasks, lf); err != nil {
		t.Fatalf("applySerialMasks: %v", err)
	}
	if err := a.applyParallelMasks(&parMasks, lf); err != nil {
		t.Fatalf("applyParallelMasks: %v", err)
	}

	// Guard against a silently empty apply masking a false pass.
	changed := false
	for i := range src.Y {
		if serialMasks.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("degenerate: serial mask apply changed no luma pixels")
	}
	lfDiffAssertEqual(t, serialMasks, parMasks)

	if !fourWay {
		return
	}
	serial := lfDiffCopy(src)
	parEdge := lfDiffCopy(src)
	if err := a.applySerial(&serial, lf); err != nil {
		t.Fatalf("applySerial: %v", err)
	}
	if err := a.apply(&parEdge, lf); err != nil {
		t.Fatalf("apply (parallel edge-list): %v", err)
	}
	lfDiffAssertEqual(t, serial, serialMasks)
	lfDiffAssertEqual(t, serial, parEdge)
	lfDiffAssertEqual(t, parEdge, parMasks)
}

func TestLoopFilterParallelMaskDiffUniform16x16(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	tree := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	// A 128x128 mask region covers 128px, so h=128,256,384,640 => 1,2,3,5 luma
	// region-row bands. fourWay only where the dense edge-list planner fits (<=256).
	for _, c := range []struct {
		w, h    int
		fourWay bool
	}{{128, 128, true}, {192, 256, true}, {256, 384, false}, {192, 640, false}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			lfParallelMaskRun(t, c.w, c.h, 4, 4, tile.BlockSize16x16, tree, lf, c.fourWay)
		})
	}
}

func TestLoopFilterParallelMaskDiff8x8(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, LevelU: 24, LevelV: 22, Sharpness: 0}
	tree := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	for _, c := range []struct {
		w, h    int
		fourWay bool
	}{{128, 96, true}, {256, 256, true}, {192, 512, false}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			lfParallelMaskRun(t, c.w, c.h, 2, 2, tile.BlockSize8x8, tree, lf, c.fourWay)
		})
	}
}

func TestLoopFilterParallelMaskDiff32x32(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{34, 30}, LevelU: 28, LevelV: 26, Sharpness: 2}
	tree := tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	for _, c := range []struct {
		w, h    int
		fourWay bool
	}{{192, 128, true}, {256, 256, true}, {256, 640, false}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			lfParallelMaskRun(t, c.w, c.h, 8, 8, tile.BlockSize32x32, tree, lf, c.fourWay)
		})
	}
}

// TestLoopFilterParallelMaskDiffChromaGated checks the chroma gating: a frame with
// a zero chroma base level must deblock luma only, and the parallel mask apply must
// skip the chroma planes exactly as the serial apply does (chroma == source).
func TestLoopFilterParallelMaskDiffChromaGated(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{24, 24}, LevelU: 0, LevelV: 0, Sharpness: 1}
	tree := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	lfParallelMaskRun(t, 256, 256, 4, 4, tile.BlockSize16x16, tree, lf, true)
}

func TestLoopFilterParallelMaskDiffHorizontalMixedWidthBoundary(t *testing.T) {
	for _, w := range []int{128, 136, 384} {
		t.Run(fmt.Sprintf("width-%d", w), func(t *testing.T) {
			const h = 256
			lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, Sharpness: 0}
			wideTree := tile.TransformTreeResult{Y: tile.TransformSize8x16, UV: tile.TransformSize4x4, HasUV: true}
			narrowTree := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}

			var a loopFilterApplier
			if err := a.init(w, h); err != nil {
				t.Fatalf("init: %v", err)
			}
			defer a.close()
			if err := a.reset(); err != nil {
				t.Fatalf("reset: %v", err)
			}

			for col := 0; col < int(a.filtMap.Stride); col += 4 {
				wideCols := min(2, int(a.filtMap.Stride)-col)
				for row := 0; row < int(a.filtMap.Rows); row += 4 {
					lfTilePutBlock(a.filtMap, col, row, wideCols, 4, tile.BlockSize8x16, wideTree, false)
				}
				narrowCol := col + wideCols
				if narrowCol >= int(a.filtMap.Stride) {
					continue
				}
				narrowCols := min(2, int(a.filtMap.Stride)-narrowCol)
				for row := 0; row < int(a.filtMap.Rows); row += 2 {
					lfTilePutBlock(a.filtMap, narrowCol, row, narrowCols, 2, tile.BlockSize8x8, narrowTree, false)
				}
			}

			src := lfDiffFrame(w, h)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					switch {
					case x%16 < 8 && y >= 128:
						src.Y[y*w+x] = 101
					case x%16 >= 8 && y >= 120:
						src.Y[y*w+x] = 108
					default:
						src.Y[y*w+x] = 100
					}
				}
			}
			serialMasks := lfDiffCopy(src)
			parallelMasks := lfDiffCopy(src)

			for iteration := range 100 {
				copy(serialMasks.Y, src.Y)
				copy(serialMasks.U, src.U)
				copy(serialMasks.V, src.V)
				copy(parallelMasks.Y, src.Y)
				copy(parallelMasks.U, src.U)
				copy(parallelMasks.V, src.V)

				if err := a.applySerialMasks(&serialMasks, lf); err != nil {
					t.Fatalf("iteration %d applySerialMasks: %v", iteration, err)
				}
				if got := serialMasks.Y[122*w+8]; got == 108 {
					t.Fatalf("iteration %d serial narrow q2=%d was not filtered", iteration, got)
				}
				if err := a.applyParallelMasks(&parallelMasks, lf); err != nil {
					t.Fatalf("iteration %d applyParallelMasks: %v", iteration, err)
				}
				lfDiffAssertEqual(t, serialMasks, parallelMasks)
			}
		})
	}
}

var _ = loopfilter.PlaneY
