package encoder

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestPredictIntoScaledReferenceMatchesMotionConvolver(t *testing.T) {
	tests := []struct {
		name                         string
		refWidth, refHeight          int
		curWidth, curHeight          int
		px, py, blockWidth, blockHgt int
		mv                           motion.Vector
		subsamplingX, subsamplingY   bool
	}{
		{
			name:       "luma upsample reference",
			refWidth:   64,
			refHeight:  48,
			curWidth:   128,
			curHeight:  96,
			px:         80,
			py:         40,
			blockWidth: 16,
			blockHgt:   16,
			mv:         motion.Vector{Col: 5, Row: -3},
		},
		{
			name:         "chroma upsample reference",
			refWidth:     32,
			refHeight:    24,
			curWidth:     64,
			curHeight:    48,
			px:           30,
			py:           18,
			blockWidth:   8,
			blockHgt:     8,
			mv:           motion.Vector{Col: 5, Row: -3},
			subsamplingX: true,
			subsamplingY: true,
		},
		{
			name:       "luma downscale right edge",
			refWidth:   128,
			refHeight:  96,
			curWidth:   64,
			curHeight:  48,
			px:         56,
			py:         32,
			blockWidth: 8,
			blockHgt:   16,
			mv:         motion.Vector{Col: -4, Row: 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ref := scaledPredictionPattern(tt.refWidth, tt.refHeight)
			got := make([]byte, tt.blockWidth*tt.blockHgt)
			var scratch motion.ScaledConvolveScratch
			if err := predictIntoScaled(got, ref, tt.refWidth, tt.refWidth, tt.refHeight, tt.curWidth, tt.curHeight,
				tt.px, tt.py, tt.blockWidth, tt.blockHgt, tt.mv, tt.subsamplingX, tt.subsamplingY, &scratch); err != nil {
				t.Fatalf("predictIntoScaled: %v", err)
			}

			want := make([]byte, tt.blockWidth*tt.blockHgt)
			if err := directScaledPredict(want, ref, tt.refWidth, tt.refHeight, tt.curWidth, tt.curHeight,
				tt.px, tt.py, tt.blockWidth, tt.blockHgt, tt.mv, tt.subsamplingX, tt.subsamplingY); err != nil {
				t.Fatalf("direct scaled predict: %v", err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("prediction mismatch\n got %v\nwant %v", got, want)
			}
		})
	}
}

func TestPredictIntoScaledReferenceIdentityMatchesSameSizePredictor(t *testing.T) {
	const (
		width      = 64
		height     = 48
		px         = 24
		py         = 16
		blockWidth = 16
		blockHgt   = 8
	)
	ref := scaledPredictionPattern(width, height)
	mv := motion.Vector{Col: 5, Row: -3}

	got := make([]byte, blockWidth*blockHgt)
	var scratch motion.ScaledConvolveScratch
	if err := predictIntoScaled(got, ref, width, width, height, width, height, px, py, blockWidth, blockHgt, mv, false, false, &scratch); err != nil {
		t.Fatalf("predictIntoScaled: %v", err)
	}

	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, false, false)
	if err != nil {
		t.Fatalf("ReferenceOriginSubsampled: %v", err)
	}
	want := make([]byte, blockWidth*blockHgt)
	dstPlane := frame.Plane{Pix: want, Stride: blockWidth, Width: blockWidth, Height: blockHgt}
	refPlane := frame.Plane{Pix: ref, Stride: width, Width: width, Height: height}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstPlane, refPlane, 1, 8, 0, 0,
		refX, refY, blockWidth, blockHgt, subX, subY, motion.RegularFilters); err != nil {
		t.Fatalf("same-size predictor: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("identity prediction mismatch\n got %v\nwant %v", got, want)
	}
}

func TestPredictIntoScaledReferenceRejectsInvalidScale(t *testing.T) {
	ref := scaledPredictionPattern(4, 4)
	err := predictIntoScaled(make([]byte, 64), ref, 4, 4, 4, 128, 128, 0, 0, 8, 8, motion.Vector{}, false, false, nil)
	if !errors.Is(err, motion.ErrInvalidMotion) {
		t.Fatalf("err=%v, want ErrInvalidMotion", err)
	}
}

func scaledPredictionPattern(width, height int) []byte {
	pix := make([]byte, width*height)
	for y := range height {
		for x := range width {
			pix[y*width+x] = byte(17*x + 29*y + 3*x*y + 19)
		}
	}
	return pix
}

func directScaledPredict(dst, ref []byte, refWidth, refHeight, curWidth, curHeight, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool) error {
	sf, err := motion.NewScaleFactors(refWidth, refHeight, curWidth, curHeight)
	if err != nil {
		return err
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(px, py, mv, ssX, ssY)
	if err != nil {
		return err
	}
	filters := motion.RegularFilters
	xTable, err := motion.SubpelKernelTableFor(filters.X, bw)
	if err != nil {
		return err
	}
	yTable, err := motion.SubpelKernelTableFor(filters.Y, bh)
	if err != nil {
		return err
	}
	dstPlane := frame.Plane{Pix: dst, Stride: bw, Width: bw, Height: bh}
	refPlane := frame.Plane{Pix: ref, Stride: refWidth, Width: refWidth, Height: refHeight}
	var scratch motion.ScaledConvolveScratch
	return motion.ConvolveScale2D8ClampedWithScratch(dstPlane, refPlane, 0, 0, bw, bh, startX, xStep, startY, yStep, xTable, yTable, &scratch)
}
