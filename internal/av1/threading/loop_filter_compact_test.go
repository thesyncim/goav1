package threading

import (
	"encoding/binary"
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func compactTestHandle(cols, rows int, layout lfmask.Layout) *FrameWorkLoopFilterMasks {
	sb128w, sb128h, count := FrameWorkLoopFilterMaskShape(cols, rows)
	return &FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, count),
		LevelCache: make([][4]uint8, cols*rows),
		Cols:       cols,
		Rows:       rows,
		SB128W:     sb128w,
		SB128H:     sb128h,
		Layout:     layout,
		HasChroma:  !layout.Mono,
	}
}

func compactTestTransformForSize(size tile.BlockSize) tile.TransformSize {
	switch size {
	case tile.BlockSize128x128, tile.BlockSize128x64, tile.BlockSize64x128, tile.BlockSize64x64:
		return tile.TransformSize64x64
	case tile.BlockSize64x32, tile.BlockSize32x64, tile.BlockSize32x32:
		return tile.TransformSize32x32
	case tile.BlockSize64x16, tile.BlockSize16x64, tile.BlockSize32x16, tile.BlockSize16x32, tile.BlockSize16x16:
		return tile.TransformSize16x16
	case tile.BlockSize32x8, tile.BlockSize8x32, tile.BlockSize16x8, tile.BlockSize8x16:
		return tile.TransformSize8x8
	case tile.BlockSize16x4:
		return tile.TransformSize16x4
	case tile.BlockSize4x16:
		return tile.TransformSize4x16
	case tile.BlockSize8x4:
		return tile.TransformSize8x4
	case tile.BlockSize4x8:
		return tile.TransformSize4x8
	case tile.BlockSize8x8:
		return tile.TransformSize8x8
	default:
		return tile.TransformSize4x4
	}
}

func compactTestLegalBlocks(cols, rows, sbLog2 int) []orderTestBlock {
	var blocks []orderTestBlock
	appendBlock := func(x, y, w, h int, size tile.BlockSize, pattern int) {
		if x >= cols || y >= rows {
			return
		}
		if x+w > cols {
			w = cols - x
		}
		if y+h > rows {
			h = rows - y
		}
		ytx := compactTestTransformForSize(size)
		// Keep some transform variation within legal rectangles, including the
		// 8x4/4x8 shared-chroma siblings below.
		if pattern&1 != 0 && (size == tile.BlockSize8x4 || size == tile.BlockSize4x8) {
			ytx = tile.TransformSize4x4
		}
		blocks = append(blocks, orderTestBlock{
			bx: x, by: y, bw4: w, bh4: h, size: size,
			tree:  tile.TransformTreeResult{Y: ytx, UV: tile.TransformSize4x4, HasUV: true},
			intra: pattern%3 == 0,
		})
	}
	var walk func(x, y, size int)
	walk = func(x, y, size int) {
		if x >= cols || y >= rows {
			return
		}
		if size == 2 {
			choice := (x/2 + 3*(y/2)) % 4
			switch choice {
			case 0: // PartitionNone.
				appendBlock(x, y, 2, 2, tile.BlockSize8x8, choice)
			case 1: // PartitionH, with a smaller transform in the bottom sibling.
				appendBlock(x, y, 2, 1, tile.BlockSize8x4, 0)
				appendBlock(x, y+1, 2, 1, tile.BlockSize8x4, 1)
			case 2: // PartitionV, with a smaller transform in the right sibling.
				appendBlock(x, y, 1, 2, tile.BlockSize4x8, 0)
				appendBlock(x+1, y, 1, 2, tile.BlockSize4x8, 1)
			default: // PartitionSplit into four 4x4 leaves.
				appendBlock(x, y, 1, 1, tile.BlockSize4x4, 0)
				appendBlock(x+1, y, 1, 1, tile.BlockSize4x4, 1)
				appendBlock(x, y+1, 1, 1, tile.BlockSize4x4, 2)
				appendBlock(x+1, y+1, 1, 1, tile.BlockSize4x4, 3)
			}
			return
		}
		// At the 16x16 MI level, use legal H4/V4 partitions to include
		// 16x4 and 4x16 leaves; at larger levels, vary unsplit blocks and
		// recursive quad splits. Every recursive descent is TL, TR, BL, BR.
		if size == 4 {
			switch (x/4 + 3*(y/4)) % 6 {
			case 0: // PartitionH4.
				for dy := 0; dy < 4; dy++ {
					appendBlock(x, y+dy, 4, 1, tile.BlockSize16x4, dy)
				}
				return
			case 1: // PartitionV4.
				for dx := 0; dx < 4; dx++ {
					appendBlock(x+dx, y, 1, 4, tile.BlockSize4x16, dx)
				}
				return
			case 2:
				appendBlock(x, y, 4, 2, tile.BlockSize16x8, 0)
				appendBlock(x, y+2, 4, 2, tile.BlockSize16x8, 1)
				return
			case 3:
				appendBlock(x, y, 2, 4, tile.BlockSize8x16, 0)
				appendBlock(x+2, y, 2, 4, tile.BlockSize8x16, 1)
				return
			}
		}
		if size > 4 && (x+3*y+size)%5 == 0 {
			blockSize := tile.BlockSize16x16
			switch size {
			case 8:
				blockSize = tile.BlockSize32x32
			case 16:
				blockSize = tile.BlockSize64x64
			case 32:
				blockSize = tile.BlockSize128x128
			}
			appendBlock(x, y, size, size, blockSize, x+y)
			return
		}
		half := size / 2
		walk(x, y, half)
		walk(x+half, y, half)
		walk(x, y+half, half)
		walk(x+half, y+half, half)
	}
	sbMI := 1 << uint(sbLog2)
	for y := 0; y < rows; y += sbMI {
		for x := 0; x < cols; x += sbMI {
			walk(x, y, sbMI)
		}
	}
	return blocks
}

func compactTestVisit(b orderTestBlock, cols, rows, index int, skip bool) tile.BlockLoopVisit {
	endX, endY := b.bx+b.bw4, b.by+b.bh4
	if endX > cols {
		endX = cols
	}
	if endY > rows {
		endY = rows
	}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: uint16(b.bx), MIRow: uint16(b.by),
			MIColEnd: uint16(endX), MIRowEnd: uint16(endY),
			X4: uint8(b.bx), Y4: uint8(b.by),
			Size: b.size, VisibleW4: uint8(endX - b.bx), VisibleH4: uint8(endY - b.by),
		},
		SegmentID:         uint8(index % parser.MaxSegments),
		Prefix:            tile.BlockModeResult{SkipTransform: skip},
		Coefficients:      tile.BlockCoeffResult{Tree: b.tree},
		CoefficientsValid: true,
	}
	if b.intra {
		visit.Prediction = tile.BlockPredictionModeResult{Valid: true, Intra: true}
	} else {
		mode := tile.InterModeNewMV
		if index%3 == 0 {
			mode = tile.InterModeGlobalMV
		}
		visit.Prediction = tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref: [2]tile.ReferenceFrame{tile.ReferenceFrame(index % int(tile.ReferenceFrameAltref+1)), tile.ReferenceFrameNone},
			},
			InterModeValid: true,
			InterMode:      tile.InterModeResult{Mode: mode},
		}
	}
	return visit
}

func compactTestDecodeState(index int) tile.DecodeState {
	return tile.DecodeState{
		DeltaLFFromBase: int8(index%11 - 5),
		DeltaLF: [tile.FrameLoopFilterCount]int8{
			int8(index%13 - 6), int8(index%9 - 4), int8(index%7 - 3), int8(index%5 - 2),
		},
	}
}

func compactTestExpectedNoSkip(records FrameWorkLoopFilterMap, cols, rows int) []byte {
	unitCols, unitRows := (cols+15)>>4, (rows+15)>>4
	got := make([]byte, unitCols*unitRows*8)
	stride := int(records.Stride)
	for miRow := 0; miRow < rows; miRow++ {
		for miCol := 0; miCol < cols; miCol++ {
			record := records.Records[miRow*stride+miCol]
			if !record.Valid || record.SkipTransform {
				continue
			}
			x, y := miCol>>1, miRow>>1
			off := ((y>>3)*unitCols+(x>>3))*8 + (y & 7)
			got[off] |= 1 << uint(x&7)
		}
	}
	validCols, validRows := cols>>1, rows>>1
	for y := 0; y < unitRows*8; y++ {
		for x := 0; x < unitCols*8; x++ {
			if x < validCols && y < validRows {
				continue
			}
			off := ((y>>3)*unitCols+(x>>3))*8 + (y & 7)
			got[off] |= 1 << uint(x&7)
		}
	}
	return got
}

func compactTestExpectedLevels(t *testing.T, records FrameWorkLoopFilterMap, cols, rows int, layout lfmask.Layout, hasChroma bool, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams) [][4]uint8 {
	t.Helper()
	levels := make([][4]uint8, cols*rows)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; {
			rec := records.Records[row*int(records.Stride)+col]
			if !rec.Valid {
				col++
				continue
			}
			nextCol := int(rec.Block.MIColEnd)
			if nextCol <= col || nextCol > cols {
				t.Fatalf("invalid next block column at (%d,%d): %d", col, row, nextCol)
			}
			if int(rec.Block.MICol) != col || int(rec.Block.MIRow) != row {
				col = nextCol
				continue
			}
			resolve := func(plane loopfilter.Plane, edge loopfilter.Edge) uint8 {
				blockDelta, err := loopfilter.SelectDelta(delta, rec.DeltaLFFromBase, rec.DeltaLF, plane, edge)
				if err != nil {
					t.Fatalf("select delta (%d,%d plane=%d edge=%d): %v", col, row, plane, edge, err)
				}
				level, err := loopfilter.ResolveLevel(params, seg, loopfilter.LevelRequest{
					Plane: plane, Edge: edge, SegmentID: rec.SegmentID,
					RefFrame: rec.RefFrame, Mode: rec.Mode, DeltaLF: blockDelta,
				})
				if err != nil {
					t.Fatalf("resolve level (%d,%d plane=%d edge=%d): %v", col, row, plane, edge, err)
				}
				return level
			}
			dims, ok := rec.Block.Size.Dimensions()
			if !ok {
				t.Fatalf("invalid block size at (%d,%d): %d", col, row, rec.Block.Size)
			}
			bx, by := col, row
			bw4 := min(cols-bx, int(dims.W4))
			bh4 := min(rows-by, int(dims.H4))
			yVert := resolve(loopfilter.PlaneY, loopfilter.EdgeVertical)
			yHorz := resolve(loopfilter.PlaneY, loopfilter.EdgeHorizontal)
			for y := 0; y < bh4; y++ {
				for x := 0; x < bw4; x++ {
					cell := &levels[(by+y)*cols+bx+x]
					cell[0], cell[1] = yVert, yHorz
				}
			}
			if hasChroma {
				cbCols := (cols + layout.SSHor) >> layout.SSHor
				cbRows := (rows + layout.SSVer) >> layout.SSVer
				cbw4 := min(cbCols-(bx>>layout.SSHor), (int(dims.W4)+layout.SSHor)>>layout.SSHor)
				cbh4 := min(cbRows-(by>>layout.SSVer), (int(dims.H4)+layout.SSVer)>>layout.SSVer)
				uLevel := resolve(loopfilter.PlaneU, loopfilter.EdgeVertical)
				vLevel := resolve(loopfilter.PlaneV, loopfilter.EdgeVertical)
				cbx, cby := bx>>layout.SSHor, by>>layout.SSVer
				for y := 0; y < cbh4; y++ {
					for x := 0; x < cbw4; x++ {
						cell := &levels[(cby+y)*cols+cbx+x]
						cell[2], cell[3] = uLevel, vLevel
					}
				}
			}
			col = nextCol
		}
	}
	return levels
}

func TestCompactLoopFilterMatchesMapRasterAndCDEFNoSkip(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY: [2]uint8{31, 42}, LevelU: 29, LevelV: 37,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{-7, 3, 5, -2, 4, -3, 6, 1},
			Mode: [2]int8{-2, 5},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i] = parser.SegmentData{
			DeltaLFYV: int16(i*17 - 50), DeltaLFYH: int16(45 - i*13),
			DeltaLFU: int16(i*9 - 25), DeltaLFV: int16(30 - i*7),
		}
	}
	delta := parser.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true}

	cases := []struct {
		name   string
		cols   int
		rows   int
		sbLog2 int
		layout lfmask.Layout
		mono   bool
		blocks func(int, int, int) []orderTestBlock
	}{
		{name: "64sb-420-partial-crop", cols: 30, rows: 30, sbLog2: 4, layout: lfmask.I420(), blocks: compactTestLegalBlocks},
		{name: "128sb-422", cols: 32, rows: 32, sbLog2: 5, layout: lfmask.I422(), blocks: compactTestLegalBlocks},
		{name: "128sb-444", cols: 32, rows: 30, sbLog2: 5, layout: lfmask.I444(), blocks: compactTestLegalBlocks},
		{name: "shared-chroma-420-sub8-partitions", cols: 16, rows: 16, sbLog2: 4, layout: lfmask.I420(), blocks: compactTestLegalBlocks},
		{name: "mono-sub8-partitions", cols: 16, rows: 16, sbLog2: 4, layout: lfmask.Layout{Mono: true}, mono: true, blocks: compactTestLegalBlocks},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			blocks := tc.blocks(tc.cols, tc.rows, tc.sbLog2)
			compactMasks := compactTestHandle(tc.cols, tc.rows, tc.layout)
			state := &FrameWorkLoopFilterCompactState{}
			unitCols, unitRows := (tc.cols+15)>>4, (tc.rows+15)>>4
			noSkip := make([]byte, unitCols*unitRows*8)
			if err := BindFrameWorkLoopFilterCompactState(compactMasks, state, noSkip, params, seg, delta, tc.mono); err != nil {
				t.Fatalf("bind compact state: %v", err)
			}
			if !FrameWorkLoopFilterCompactActiveShape(compactMasks, state, tc.cols, tc.rows) ||
				FrameWorkLoopFilterCompactReady(compactMasks, state, tc.cols, tc.rows) {
				t.Fatal("new compact binding must be active with exact shape but not ready")
			}

			mapHandle := FrameWorkLoopFilterMap{
				Records: make([]FrameWorkLoopFilterBlockRecord, tc.cols*tc.rows),
				Stride:  uint16(tc.cols), Rows: uint16(tc.rows),
			}
			var compactScratch frameWorkLoopFilterMaskScratch
			compactScratch.reset(compactMasks)
			for i, block := range blocks {
				skip := i%5 == 0
				if block.size != tile.BlockSize4x4 && i%3 == 1 {
					block.intra = false
				}
				if tc.mono {
					block.tree.HasUV = false
				}
				visit := compactTestVisit(block, tc.cols, tc.rows, i, skip)
				decodeState := compactTestDecodeState(i)
				if err := mapHandle.MarkBlockPtr(&visit, &decodeState); err != nil {
					t.Fatalf("map mark block %d: %v", i, err)
				}
				if err := compactMasks.buildCompact(state, &compactScratch, &visit, &decodeState, tc.sbLog2); err != nil {
					t.Fatalf("compact block %d: %v", i, err)
				}
			}

			mapStats, err := mapHandle.CoverageStats(tc.cols, tc.rows)
			if err != nil || mapStats.Cells != int32(tc.cols*tc.rows) || mapStats.Blocks != int32(len(blocks)) || mapStats.Missing != 0 {
				t.Fatalf("map coverage=%+v err=%v blocks=%d", mapStats, err, len(blocks))
			}
			refMasks := compactTestHandle(tc.cols, tc.rows, tc.layout)
			var rasterScratch FrameWorkLoopFilterMaskBuildScratch
			if err := refMasks.BuildFromMap(&rasterScratch, mapHandle, tc.sbLog2); err != nil {
				t.Fatalf("raster mask build: %v", err)
			}
			if !reflect.DeepEqual(compactMasks.MaskBits(), refMasks.MaskBits()) {
				t.Fatal("compact edge masks differ from independent record-map raster build")
			}

			if want := compactTestExpectedLevels(t, mapHandle, tc.cols, tc.rows, tc.layout, !tc.mono, params, seg, delta); !reflect.DeepEqual(compactMasks.LevelCache, want) {
				for i := range want {
					if compactMasks.LevelCache[i] != want[i] {
						t.Fatalf("level cell %d: compact=%v independent=%v", i, compactMasks.LevelCache[i], want[i])
					}
				}
			}
			if err := CompleteFrameWorkLoopFilterCompactState(state); err != nil {
				t.Fatalf("complete full compact frame: %v cells=%d blocks=%d", err, state.cells, state.blocks)
			}
			cells, blockCount, ok := FrameWorkLoopFilterCompactCoverage(compactMasks, state, tc.cols, tc.rows)
			if !ok || cells != int32(tc.cols*tc.rows) || blockCount != int32(len(blocks)) {
				t.Fatalf("compact coverage=(%d,%d,%v) want=(%d,%d,true)", cells, blockCount, ok, tc.cols*tc.rows, len(blocks))
			}
			wantNoSkip := compactTestExpectedNoSkip(mapHandle, tc.cols, tc.rows)
			if !reflect.DeepEqual(state.noskip, wantNoSkip) {
				t.Fatalf("compact no-skip bytes differ: got=%x want=%x", state.noskip, wantNoSkip)
			}
			for unitRow := 0; unitRow < unitRows; unitRow++ {
				for unitCol := 0; unitCol < unitCols; unitCol++ {
					got, ok := FrameWorkLoopFilterCompactUnitMask(compactMasks, state, unitRow, unitCol)
					off := (unitRow*unitCols + unitCol) * 8
					want := binary.LittleEndian.Uint64(wantNoSkip[off : off+8])
					if !ok || got != want {
						t.Fatalf("unit mask (%d,%d)=(%#x,%v) want %#x", unitCol, unitRow, got, ok, want)
					}
				}
			}
		})
	}
}

func TestCompactLoopFilterReadinessRequiresCompleteValidWalk(t *testing.T) {
	const cols, rows = 8, 8
	h := compactTestHandle(cols, rows, lfmask.I420())
	state := &FrameWorkLoopFilterCompactState{ready: true, cells: 16, blocks: 1}
	noSkip := make([]byte, 8)
	for i := range noSkip {
		noSkip[i] = 0xff
	}
	if err := BindFrameWorkLoopFilterCompactState(h, state, noSkip, parser.LoopFilterParams{LevelY: [2]uint8{1, 1}}, parser.SegmentationParams{}, parser.DeltaParams{}, false); err != nil {
		t.Fatalf("rebind compact state: %v", err)
	}
	if state.ready || state.cells != 0 || state.blocks != 0 || !FrameWorkLoopFilterCompactActive(state) {
		t.Fatalf("bind did not reset compact state: %+v", *state)
	}
	if err := CompleteFrameWorkLoopFilterCompactState(state); err == nil {
		t.Fatal("incomplete frame unexpectedly became ready")
	}
	if FrameWorkLoopFilterCompactReady(h, state, cols, rows) {
		t.Fatal("incomplete frame reported ready")
	}

	bad := tile.BlockLoopVisit{Block: tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: uint16(cols + 1), MIRowEnd: 2}}
	var scratch frameWorkLoopFilterMaskScratch
	scratch.reset(h)
	if err := h.buildCompact(state, &scratch, &bad, &tile.DecodeState{}, 4); err == nil {
		t.Fatal("out-of-frame block unexpectedly accepted")
	}
	if state.cells != 0 || state.blocks != 0 || state.ready {
		t.Fatalf("rejected block mutated readiness/coverage: %+v", *state)
	}
	InvalidateFrameWorkLoopFilterCompactState(state)
	if !FrameWorkLoopFilterCompactStateActive(state) || state.ready {
		t.Fatal("invalidate should leave active marker but clear readiness")
	}
	DeactivateFrameWorkLoopFilterCompactState(state)
	if FrameWorkLoopFilterCompactStateActive(state) || state.ready || state.noskip != nil {
		t.Fatal("deactivate should clear compact binding")
	}
}
