package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type txBlockGeometry struct {
	size        transform.Size
	tx          tile.TransformSize
	coeffSize   transform.Size
	width       int
	height      int
	coeffWidth  int
	coeffHeight int
	coeffCount  int
	sampleCount int
	txScale     uint8
}

func txBlockGeometryForTransformSize(size tile.TransformSize) (txBlockGeometry, error) {
	if int(size) < len(txBlockGeometries) {
		return txBlockGeometries[size], nil
	}
	return txBlockGeometry{}, transform.ErrInvalidTransform
}

func txBlockGeometryForDimensions(width, height int) (txBlockGeometry, error) {
	if lw, lh := log2Exact(width), log2Exact(height); lw >= 2 && lw <= 6 && lh >= 2 && lh <= 6 {
		if tx := tileTransformSizes[lw-2][lh-2]; tx != 255 {
			return txBlockGeometries[tx], nil
		}
	}
	return txBlockGeometry{}, transform.ErrInvalidTransform
}

// txBlockGeometries resolves every transform size's geometry once.
var txBlockGeometries = func() (g [int(tile.TransformSize64x16) + 1]txBlockGeometry) {
	for s := range g {
		txSize, err := tile.TransformSize(s).TransformSize()
		if err != nil {
			panic(err)
		}
		if g[s], err = txBlockGeometryForSize(txSize); err != nil {
			panic(err)
		}
	}
	return g
}()

// tileTransformSizes maps (log2 width - 2, log2 height - 2) to the tile
// transform size; invalid shapes hold tile.TransformSize(255).
var tileTransformSizes = func() (t [5][5]tile.TransformSize) {
	for i := range t {
		for j := range t[i] {
			t[i][j] = 255
		}
	}
	for s := tile.TransformSize4x4; s <= tile.TransformSize64x16; s++ {
		dims, ok := s.Dimensions()
		if !ok {
			continue
		}
		t[dims.Log2W][dims.Log2H] = s
	}
	return t
}()

func txBlockGeometryForSize(size transform.Size) (txBlockGeometry, error) {
	coeffSize, err := transform.ScanSize(size)
	if err != nil {
		return txBlockGeometry{}, err
	}
	txScale, err := quantize.TransformScale(int(size.Width), int(size.Height))
	if err != nil {
		return txBlockGeometry{}, err
	}
	coeffWidth := int(coeffSize.Width)
	coeffHeight := int(coeffSize.Height)
	width := int(size.Width)
	height := int(size.Height)
	tx := tile.TransformSize(255)
	if lw, lh := log2Exact(width), log2Exact(height); lw >= 2 && lw <= 6 && lh >= 2 && lh <= 6 {
		tx = tileTransformSizes[lw-2][lh-2]
	}
	if tx == 255 {
		return txBlockGeometry{}, transform.ErrInvalidTransform
	}
	return txBlockGeometry{
		size:        size,
		tx:          tx,
		coeffSize:   coeffSize,
		width:       width,
		height:      height,
		coeffWidth:  coeffWidth,
		coeffHeight: coeffHeight,
		coeffCount:  coeffWidth * coeffHeight,
		sampleCount: width * height,
		txScale:     txScale,
	}, nil
}

// log2Exact returns log2(n) for a power of two n, else -1.
func log2Exact(n int) int {
	if n <= 0 || n&(n-1) != 0 {
		return -1
	}
	l := 0
	for n > 1 {
		n >>= 1
		l++
	}
	return l
}

// Square geometries resolved once for hot paths.
var (
	txBlockGeometry8x8   = mustTxBlockGeometry(8)
	txBlockGeometry16x16 = mustTxBlockGeometry(16)
	txBlockGeometry32x32 = mustTxBlockGeometry(32)
)

func mustTxBlockGeometry(n int) txBlockGeometry {
	g, err := txBlockGeometryForDimensions(n, n)
	if err != nil {
		panic(err)
	}
	return g
}

func (g txBlockGeometry) matchesDimensions(width, height int) bool {
	return g.width == width && g.height == height
}
