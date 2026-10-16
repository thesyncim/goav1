package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type txBlockGeometry struct {
	size        transform.Size
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
	txSize, err := size.TransformSize()
	if err != nil {
		return txBlockGeometry{}, err
	}
	return txBlockGeometryForSize(txSize)
}

func txBlockGeometryForDimensions(width, height int) (txBlockGeometry, error) {
	txSize, err := transform.SizeFromDimensions(width, height)
	if err != nil {
		return txBlockGeometry{}, err
	}
	return txBlockGeometryForSize(txSize)
}

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
	return txBlockGeometry{
		size:        size,
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
