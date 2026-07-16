package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// loopfilter_tile_diff_test.go is the CI-portable reproduction of the mask-build
// order bug the RealC recon oracle exposed: raster-by-origin order does not
// reproduce dav1d's Z-order left/above tx-context when a taller block on the
// right of a superblock boundary has a smaller origin row than the shorter
// block on its left (the "staircase"). The mask build must derive neighbour
// context from the shared record map (== what the edge-list sweep reads), so the
// two agree byte-for-byte. These synthetic maps engineer the staircase and a
// variable-transform neighbour so the boundary vertical edge width depends on the
// left neighbour, and assert sweep == mask.

func lfTilePutBlock(m threading.FrameWorkLoopFilterMap, col, row, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult, intra bool) {
	stride := int(m.Stride)
	rec := threading.FrameWorkLoopFilterBlockRecord{
		Valid:         true,
		TransformTree: tree,
		RefFrame:      1,
		Intra:         intra,
		Mode:          loopfilter.ModeDeltaClassZero,
		Block: threading.FrameWorkLoopFilterBlock{
			MICol: uint16(col), MIRow: uint16(row),
			MIColEnd: uint16(col + bw4), MIRowEnd: uint16(row + bh4),
			X4: uint8(col), Y4: uint8(row), Size: size,
			VisibleW4: uint8(bw4), VisibleH4: uint8(bh4),
		},
	}
	if intra {
		rec.RefFrame = 0
	}
	for r := row; r < row+bh4; r++ {
		for c := col; c < col+bw4; c++ {
			m.Records[r*stride+c] = rec
		}
	}
}

func lfTileFirstDiff(t *testing.T, a, b SourceFrame420) bool {
	t.Helper()
	for _, p := range []struct {
		name         string
		pa, pb       []byte
		stride, w, h int
	}{
		{"Y", a.Y, b.Y, a.YStride, a.Width, a.Height},
		{"U", a.U, b.U, a.ChromaStride, a.Width / 2, a.Height / 2},
		{"V", a.V, b.V, a.ChromaStride, a.Width / 2, a.Height / 2},
	} {
		for y := 0; y < p.h; y++ {
			for x := 0; x < p.w; x++ {
				if p.pa[y*p.stride+x] != p.pb[y*p.stride+x] {
					t.Logf("plane %s first diff at x=%d y=%d: sweep=%d mask=%d", p.name, x, y, p.pa[y*p.stride+x], p.pb[y*p.stride+x])
					return true
				}
			}
		}
	}
	return false
}

// lfTileCompareSweepMask fills the record map via fill, then deblocks copies of
// the same reconstruction with the edge-list sweep (ground truth) and the mask
// apply and asserts byte-identical planes.
func lfTileCompareSweepMask(t *testing.T, w, h int, lf parser.LoopFilterParams, fill func(m threading.FrameWorkLoopFilterMap)) {
	t.Helper()
	var a loopFilterApplier
	if err := a.init(w, h); err != nil {
		t.Fatalf("init: %v", err)
	}
	defer a.close()
	if err := a.reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	fill(a.filtMap)

	src := lfDiffFrame(w, h)
	sweep := lfDiffCopy(src)
	masks := lfDiffCopy(src)
	if err := a.applySerial(&sweep, lf); err != nil {
		t.Fatalf("applySerial: %v", err)
	}
	if err := a.applySerialMasks(&masks, lf); err != nil {
		t.Fatalf("applySerialMasks: %v", err)
	}
	changed := false
	for i := range src.Y {
		if sweep.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("degenerate: sweep apply changed no luma pixels")
	}
	if lfTileFirstDiff(t, sweep, masks) {
		lfDiffAssertEqual(t, sweep, masks)
	}
}

// TestLoopFilterMaskDiffStaircaseSBBoundary is an end-to-end positive check of
// the staircase configuration (a 16x16 column right of a 64px superblock
// boundary abutting shorter 8x8s whose origin rows exceed it): the mask apply
// must equal the edge-list sweep. The content-independent bit-level proof of the
// order fix lives in threading TestBuildFromMapMatchesZOrderStaircase.
func TestLoopFilterMaskDiffStaircaseSBBoundary(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{34, 30}, LevelU: 28, LevelV: 26, Sharpness: 1}
	tree8 := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	tree16 := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	tree32 := tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	const w, h = 128, 128
	lfTileCompareSweepMask(t, w, h, lf, func(m threading.FrameWorkLoopFilterMap) {
		stride := int(m.Stride)
		rows := int(m.Rows)
		// Left superblock [cols 0..15]: 32x32 to the left, then a column of 8x8s
		// filling the two cell-columns just left of the SB boundary (col 16). The
		// 8x8 origin rows (0,2,4,...) exceed the 16x16 right-block origin rows on
		// odd 16x16 rows, so raster-by-origin order reads a stale class-2 left
		// context (the reset value), while the true 8x8 right tx is class 1.
		for row := 0; row < rows; row += 8 {
			for col := 0; col < 8; col += 8 {
				lfTilePutBlock(m, col, row, 8, 8, tile.BlockSize32x32, tree32, true)
			}
		}
		for row := 0; row < rows; row += 2 {
			for col := 8; col < 16; col += 2 {
				lfTilePutBlock(m, col, row, 2, 2, tile.BlockSize8x8, tree8, true)
			}
		}
		// Right superblock [cols 16..31]: 16x16 blocks (origin rows 0,4,8,...),
		// whose left edge at rows 2,3,6,7,... abuts an 8x8 whose origin row is
		// LARGER, so raster order has not yet processed it.
		for row := 0; row < rows; row += 4 {
			for col := 16; col < 32; col += 4 {
				lfTilePutBlock(m, col, row, 4, 4, tile.BlockSize16x16, tree16, true)
			}
		}
		// Far right: 32x32 context.
		for row := 0; row < rows; row += 8 {
			for col := 32; col < stride; col += 8 {
				lfTilePutBlock(m, col, row, 8, 8, tile.BlockSize32x32, tree32, true)
			}
		}
	})
}

// TestLoopFilterMaskDiffVariableTxNeighbor engineers a variable-transform inter
// block whose right column carries mixed tx sizes, abutting a fixed-tx block, so
// the boundary edge width derivation depends on decomposing the variable
// neighbour's right-edge tx per row.
func TestLoopFilterMaskDiffVariableTxNeighbor(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{40, 40}, LevelU: 32, LevelV: 30, Sharpness: 0}
	// A 16x16 inter block coded with a split transform tree: depth-0 split of the
	// 16x16 into 8x8 sub-transforms. Split[0] bit for yOff=0,xOff=0.
	varTree := tile.TransformTreeResult{
		Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true,
		Variable: true, Split: [2]uint16{0xffff, 0xffff},
	}
	fixed16 := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	const w, h = 128, 128
	lfTileCompareSweepMask(t, w, h, lf, func(m threading.FrameWorkLoopFilterMap) {
		stride := int(m.Stride)
		rows := int(m.Rows)
		// Alternate 16x16 blocks between a split (variable) tree and a fixed
		// 16x16 tx, so each vertical block boundary derives its width from a
		// variable neighbour's right-column tx on one side.
		for row := 0; row < rows; row += 4 {
			for col := 0; col < stride; col += 4 {
				tree := fixed16
				if (col/4+row/4)%2 == 0 {
					tree = varTree
				}
				lfTilePutBlock(m, col, row, 4, 4, tile.BlockSize16x16, tree, false)
			}
		}
	})
}
