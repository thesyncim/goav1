package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// loopfilter_mask_diff_test.go is the fast differential gate for the encoder's
// mask-driven single-tile loop-filter apply. It fills a loop-filter record map,
// then deblocks two independent copies of the same reconstruction -- (a) the
// edge-list applySerial, (b) the mask-driven applySerialMasks -- and asserts the
// filtered planes are byte-identical. It runs in milliseconds and is the
// iteration gate for routing the single-tile serial callers onto masks.

// lfDiffFillPlane writes smoothly varying content so the deblock filter engages
// and any filter-width/level divergence changes pixels (a flat plane would make
// every filter a no-op and hide mask-class bugs).
func lfDiffFillPlane(pix []byte, stride, w, h, salt int) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 128 + ((x*3 + y*5 + salt*11) % 13) - 6
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			pix[y*stride+x] = byte(v)
		}
	}
}

func lfDiffFrame(w, h int) SourceFrame420 {
	cw, ch := w/2, h/2
	f := SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	lfDiffFillPlane(f.Y, w, w, h, 0)
	lfDiffFillPlane(f.U, cw, cw, ch, 1)
	lfDiffFillPlane(f.V, cw, cw, ch, 2)
	return f
}

func lfDiffCopy(src SourceFrame420) SourceFrame420 {
	dst := src
	dst.Y = append([]byte(nil), src.Y...)
	dst.U = append([]byte(nil), src.U...)
	dst.V = append([]byte(nil), src.V...)
	return dst
}

func lfDiffAssertEqual(t *testing.T, a, b SourceFrame420) {
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
					t.Fatalf("plane %s differs at x=%d y=%d: edge-list=%d mask=%d",
						p.name, x, y, p.pa[y*p.stride+x], p.pb[y*p.stride+x])
				}
			}
		}
	}
}

// lfDiffFillGrid tiles the record map with inter blocks of the given MI size and
// transform tree, in raster (== single-tile decode) order.
func lfDiffFillGrid(m threading.FrameWorkLoopFilterMap, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult) {
	stride := int(m.Stride)
	rows := int(m.Rows)
	for row := 0; row < rows; row += bh4 {
		for col := 0; col < stride; col += bw4 {
			rec := threading.FrameWorkLoopFilterBlockRecord{
				Valid:         true,
				TransformTree: tree,
				RefFrame:      1,
				Mode:          loopfilter.ModeDeltaClassZero,
				Block: threading.FrameWorkLoopFilterBlock{
					MICol: uint16(col), MIRow: uint16(row),
					MIColEnd: uint16(col + bw4), MIRowEnd: uint16(row + bh4),
					X4: uint8(col), Y4: uint8(row), Size: size,
					VisibleW4: uint8(bw4), VisibleH4: uint8(bh4),
				},
			}
			for r := row; r < row+bh4; r++ {
				for c := col; c < col+bw4; c++ {
					m.Records[r*stride+c] = rec
				}
			}
		}
	}
}

func lfDiffRun(t *testing.T, w, h, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult, lf parser.LoopFilterParams) {
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
	edgeList := lfDiffCopy(src)
	masks := lfDiffCopy(src)

	if err := a.applySerial(&edgeList, lf); err != nil {
		t.Fatalf("applySerial: %v", err)
	}
	if err := a.applySerialMasks(&masks, lf); err != nil {
		t.Fatalf("applySerialMasks: %v", err)
	}
	// Guard against a silently empty apply masking a false pass.
	changed := false
	for i := range src.Y {
		if edgeList.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("degenerate: edge-list apply changed no luma pixels")
	}
	lfDiffAssertEqual(t, edgeList, masks)
}

func TestLoopFilterMaskDiffUniform16x16(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	lfDiffRun(t, 128, 128, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}, lf)
}

func TestLoopFilterMaskDiff8x8(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, LevelU: 24, LevelV: 22, Sharpness: 0}
	lfDiffRun(t, 128, 96, 2, 2, tile.BlockSize8x8,
		tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}, lf)
}

func TestLoopFilterMaskDiff32x32(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{34, 30}, LevelU: 28, LevelV: 26, Sharpness: 2}
	lfDiffRun(t, 192, 128, 8, 8, tile.BlockSize32x32,
		tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}, lf)
}
