package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// loop_filter_masks_order_test.go pins the neighbour-derived BuildFromMap
// byte-for-byte against the decode-time carried-context build fed in true
// Z-order, for a staircase partition around a superblock boundary. This is
// content-independent: it compares the raw edge bitmasks, so a filter-width
// class divergence fails even when the deblocked pixels would coincide.

type orderTestBlock struct {
	bx, by, bw4, bh4 int
	size             tile.BlockSize
	tree             tile.TransformTreeResult
	intra            bool
}

func TestFrameWorkLoopFilterLevelCacheShape(t *testing.T) {
	tests := []struct {
		name             string
		cols, rows       int
		layout           lfmask.Layout
		hasChroma        bool
		uvCols, uvRows   int
		packedLevelCells int
	}{
		{"720p-420", 320, 180, lfmask.Layout{SSHor: 1, SSVer: 1}, true, 160, 90, 36000},
		{"720p-422", 320, 180, lfmask.Layout{SSHor: 1}, true, 160, 180, 43200},
		{"720p-444", 320, 180, lfmask.Layout{}, true, 320, 180, 57600},
		{"720p-mono", 320, 180, lfmask.Layout{Mono: true}, false, 0, 0, 28800},
		{"odd-420", 3, 5, lfmask.Layout{SSHor: 1, SSVer: 1}, true, 2, 3, 11},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			uvCols, uvRows, length, err := FrameWorkLoopFilterLevelCacheShape(tt.cols, tt.rows, tt.layout, tt.hasChroma)
			if err != nil {
				t.Fatal(err)
			}
			if uvCols != tt.uvCols || uvRows != tt.uvRows || length != tt.packedLevelCells {
				t.Fatalf("shape=(%d,%d,%d), want (%d,%d,%d)", uvCols, uvRows, length, tt.uvCols, tt.uvRows, tt.packedLevelCells)
			}
		})
	}
}

func newOrderTestMasks(t *testing.T, cols, rows int) (*FrameWorkLoopFilterMasks, FrameWorkLoopFilterMap) {
	t.Helper()
	sb128w, sb128h, maskCount := FrameWorkLoopFilterMaskShape(cols, rows)
	h := &FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, maskCount),
		LevelCache: make([][4]uint8, cols*rows),
		Cols:       cols, Rows: rows, SB128W: sb128w, SB128H: sb128h,
		Layout:    lfmask.Layout{SSHor: 1, SSVer: 1},
		HasChroma: true,
	}
	m := FrameWorkLoopFilterMap{
		Records: make([]FrameWorkLoopFilterBlockRecord, cols*rows),
		Stride:  uint16(cols), Rows: uint16(rows),
	}
	return h, m
}

func (b orderTestBlock) put(m FrameWorkLoopFilterMap) {
	stride := int(m.Stride)
	rec := FrameWorkLoopFilterBlockRecord{
		Valid: true, TransformTree: b.tree, Intra: b.intra,
		Block: FrameWorkLoopFilterBlock{
			MICol: uint16(b.bx), MIRow: uint16(b.by),
			MIColEnd: uint16(b.bx + b.bw4), MIRowEnd: uint16(b.by + b.bh4),
			X4: uint8(b.bx), Y4: uint8(b.by), Size: b.size,
			VisibleW4: uint8(b.bw4), VisibleH4: uint8(b.bh4),
		},
	}
	for r := b.by; r < b.by+b.bh4; r++ {
		for c := b.bx; c < b.bx+b.bw4; c++ {
			m.Records[r*stride+c] = rec
		}
	}
}

// zOrderStaircase returns blocks in true single-tile decode Z-order (superblocks
// raster, blocks Z within each SB) for a staircase around the SB boundary at
// col 16: the left SB's right column is 8x8s, the right SB is 16x16s.
func zOrderStaircase(cols, rows int) []orderTestBlock {
	tree8 := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	tree16 := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	tree32 := tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	var out []orderTestBlock
	// Superblocks are 64px = 16 MI. Iterate SB rows then SB cols (raster).
	for sbY := 0; sbY < rows; sbY += 16 {
		for sbX := 0; sbX < cols; sbX += 16 {
			bh := 16
			if sbY+bh > rows {
				bh = rows - sbY
			}
			bw := 16
			if sbX+bw > cols {
				bw = cols - sbX
			}
			switch sbX {
			case 0:
				// Left SB: 8x8 columns for cols 8..15, 32x32 for cols 0..7.
				for row := sbY; row < sbY+bh; row += 8 {
					out = append(out, orderTestBlock{0, row, 8, 8, tile.BlockSize32x32, tree32, true})
				}
				// 8x8s in Z-order within the right half column band.
				for row := sbY; row < sbY+bh; row += 2 {
					for col := 8; col < 16; col += 2 {
						out = append(out, orderTestBlock{col, row, 2, 2, tile.BlockSize8x8, tree8, true})
					}
				}
			case 16:
				// Right SB: 16x16 blocks in Z (raster within SB is fine for equal size).
				for row := sbY; row < sbY+bh; row += 4 {
					for col := 16; col < 32 && col < cols; col += 4 {
						out = append(out, orderTestBlock{col, row, 4, 4, tile.BlockSize16x16, tree16, true})
					}
				}
			default:
				for row := sbY; row < sbY+bh; row += 8 {
					for col := sbX; col < sbX+bw; col += 8 {
						out = append(out, orderTestBlock{col, row, 8, 8, tile.BlockSize32x32, tree32, true})
					}
				}
			}
		}
	}
	return out
}

func TestBuildFromMapMatchesZOrderStaircase(t *testing.T) {
	const cols, rows = 32, 32 // 128x128
	blocks := zOrderStaircase(cols, rows)

	// Reference: decode-time carried-context build fed in true Z-order.
	ref, _ := newOrderTestMasks(t, cols, rows)
	ob := make([]OrderedBlock, len(blocks))
	for i, b := range blocks {
		ob[i] = OrderedBlock{BX: b.bx, BY: b.by, Size: b.size, Tree: b.tree, Intra: b.intra}
	}
	ref.BuildFromOrderedBlocks(ob, 4)

	// Under test: neighbour-derived raster build from the record map.
	got, m := newOrderTestMasks(t, cols, rows)
	for _, b := range blocks {
		b.put(m)
	}
	var scratch FrameWorkLoopFilterMaskBuildScratch
	if err := got.BuildFromMap(&scratch, m, 4); err != nil {
		t.Fatalf("BuildFromMap: %v", err)
	}

	rb := ref.MaskBits()
	gb := got.MaskBits()
	if len(rb) != len(gb) {
		t.Fatalf("mask count mismatch %d vs %d", len(rb), len(gb))
	}
	for i := range rb {
		if rb[i] != gb[i] {
			// Find the first differing luma vertical-edge cell for a useful report.
			for pos := 0; pos < 32; pos++ {
				for str := 0; str < 3; str++ {
					for half := 0; half < 2; half++ {
						if rb[i].Y[0][pos][str][half] != gb[i].Y[0][pos][str][half] {
							t.Fatalf("region %d luma vert edge pos=%d strength=%d half=%d: ref=%#x mask=%#x",
								i, pos, str, half, rb[i].Y[0][pos][str][half], gb[i].Y[0][pos][str][half])
						}
					}
				}
			}
			t.Fatalf("region %d mask differs (non-luma-vert)", i)
		}
	}
}
