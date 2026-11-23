package goav1

import (
	"errors"
	"testing"

	internalprediction "github.com/thesyncim/goav1/internal/av1/prediction"
)

func TestPublicPredictIntraPlaneBlock(t *testing.T) {
	plane := publicPredictionPlane(6, 5, 1, 8)
	for i := range plane.Pix {
		plane.Pix[i] = 0xee
	}
	edges := internalprediction.IntraEdges{
		Above:          []uint16{10, 20, 30},
		Left:           []uint16{50, 70},
		AboveAvailable: true,
		LeftAvailable:  true,
	}
	if err := internalprediction.PredictIntraPlaneBlock(plane, 1, 8, 2, 1, 3, 2, internalprediction.IntraModeDC, edges); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = 36
			}
			if got := getPublicFrameSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictIntraPlaneBlockHighBitDepth(t *testing.T) {
	plane := publicPredictionPlane(4, 3, 2, 4)
	edges := internalprediction.IntraEdges{
		Above:          []uint16{100, 200, 300, 400},
		AboveAvailable: true,
	}
	if err := internalprediction.PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 4, 3, internalprediction.IntraModeVertical, edges); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			want := uint16((x + 1) * 100)
			if got := getPublicFrameSample(plane, 2, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictDirectionalIntraPlaneBlock(t *testing.T) {
	plane := publicPredictionPlane(4, 3, 1, 4)
	above := []uint16{7, 11, 13, 17}
	if err := internalprediction.PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, 4, 3, 90, internalprediction.DirectionalEdges{
		Above:       above,
		AboveOrigin: 0,
	}); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			if got := getPublicFrameSample(plane, 1, x, y); got != above[x] {
				t.Fatalf("vertical sample(%d,%d)=%d want %d", x, y, got, above[x])
			}
		}
	}
	if dx, dy := internalprediction.DirectionalDX(45), internalprediction.DirectionalDY(225); dx <= 1 || dy <= 1 {
		t.Fatalf("directional derivatives dx=%d dy=%d", dx, dy)
	}
}

func TestPublicPredictFilterIntraAndEdges(t *testing.T) {
	edge := []uint16{10, 20, 30, 40}
	scratch := make([]uint16, len(edge))
	if err := internalprediction.FilterIntraEdgeBounded(edge, scratch, 1, 8, false); err != nil {
		t.Fatal(err)
	}
	wantEdge := []uint16{10, 20, 30, 38}
	for i := range wantEdge {
		if edge[i] != wantEdge[i] {
			t.Fatalf("edge[%d]=%d want %d", i, edge[i], wantEdge[i])
		}
	}

	above := []uint16{77, 10, 20, 30, 40}
	left := []uint16{99, 50, 60, 70, 80}
	if err := internalprediction.FilterIntraEdgeCorner(above, 1, left, 1, 8); err != nil {
		t.Fatal(err)
	}
	if above[0] != left[0] || above[0] != 48 {
		t.Fatalf("corner above=%d left=%d want 48", above[0], left[0])
	}

	plane := publicPredictionPlane(4, 4, 1, 4)
	edges := internalprediction.IntraEdges{
		Above:              []uint16{10, 20, 30, 40},
		Left:               []uint16{50, 60, 70, 80},
		AboveLeft:          31,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	if err := internalprediction.PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, 4, 4, 4, 4, internalprediction.FilterIntraModeDC, edges, false); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(plane, 1, 0, 0); got == 0 {
		t.Fatalf("filter intra left top sample was not written")
	}
	if strength := internalprediction.IntraEdgeFilterStrength(8, 8, 40, false); strength != 1 {
		t.Fatalf("strength=%d want 1", strength)
	}
	if !internalprediction.UseIntraEdgeUpsample(8, 8, 20, false) {
		t.Fatalf("expected intra edge upsample for small directional block")
	}
}

func TestPublicCFLPredictionHelpers(t *testing.T) {
	luma := []uint8{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
		130, 140, 150, 160,
	}
	recon := make([]uint16, internalprediction.CFLBufSquare)
	if err := internalprediction.SubsampleLuma8ToQ3(recon, luma, 4, 4, 4, true, true); err != nil {
		t.Fatal(err)
	}
	if recon[0] != 280 || recon[1] != 440 || recon[internalprediction.CFLBufLine] != 920 || recon[internalprediction.CFLBufLine+1] != 1080 {
		t.Fatalf("unexpected cfl recon values %d %d %d %d", recon[0], recon[1], recon[internalprediction.CFLBufLine], recon[internalprediction.CFLBufLine+1])
	}
	ac := make([]int16, internalprediction.CFLBufSquare)
	if err := internalprediction.SubtractCFLAverage(recon, ac, 4, 4); err != nil {
		t.Fatal(err)
	}
	alpha, err := internalprediction.CFLAlphaQ3(0x21, 6, internalprediction.CFLPredU)
	if err != nil {
		t.Fatal(err)
	}
	if alpha != 3 {
		t.Fatalf("alpha=%d want 3", alpha)
	}

	plane := publicPredictionPlane(4, 4, 1, 4)
	fillPublicReconstructPlane(plane, 1, 100)
	if err := internalprediction.PredictCFLPlaneBlock(plane, 1, 8, 0, 0, 4, 4, ac, alpha); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(plane, 1, 0, 0); got == 100 {
		t.Fatalf("cfl prediction did not adjust chroma sample")
	}
}

func TestPublicPredictionRejectsInvalid(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := internalprediction.PredictIntraPlaneBlock(plane, 1, 10, 0, 0, 1, 1, internalprediction.IntraModeDC, internalprediction.IntraEdges{}); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("bitdepth mismatch err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
	if err := internalprediction.PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, 1, 1, 0, internalprediction.DirectionalEdges{}); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("directional angle err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
	if err := internalprediction.FilterIntraEdgeBounded([]uint16{1, 2}, nil, 1, 8, false); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("filter edge scratch err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
	if _, err := internalprediction.CFLAlphaQ3(0, 99, internalprediction.CFLPredU); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("cfl alpha err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
}

func TestPublicPredictionAllocs(t *testing.T) {
	plane := publicPredictionPlane(16, 16, 1, 16)
	edges := internalprediction.IntraEdges{
		Above:              make([]uint16, 16),
		Left:               make([]uint16, 16),
		AboveLeft:          91,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	for i := range 16 {
		edges.Above[i] = 90
		edges.Left[i] = 92
	}
	edge := make([]uint16, 16)
	filterScratch := make([]uint16, 16)
	recon := make([]uint16, internalprediction.CFLBufSquare)
	ac := make([]int16, internalprediction.CFLBufSquare)

	allocs := testing.AllocsPerRun(1000, func() {
		if err := internalprediction.PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 16, 16, internalprediction.IntraModeDC, edges); err != nil {
			t.Fatalf("PredictIntraPlaneBlock err=%v", err)
		}
		if err := internalprediction.PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, 16, 16, 16, 16, internalprediction.FilterIntraModeDC, edges, false); err != nil {
			t.Fatalf("PredictFilterIntraPlaneBlock err=%v", err)
		}
		copy(edge, edges.Above)
		if err := internalprediction.FilterIntraEdgeBounded(edge, filterScratch, 1, 8, false); err != nil {
			t.Fatalf("FilterIntraEdge err=%v", err)
		}
		if err := internalprediction.SubtractCFLAverage(recon, ac, 4, 4); err != nil {
			t.Fatalf("SubtractCFLAverage err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicPredictionPlane(width int, height int, bytesPerSample int, strideSamples int) FramePlane {
	return FramePlane{
		Pix:    make([]byte, strideSamples*height*bytesPerSample),
		Stride: strideSamples * bytesPerSample,
		Width:  width,
		Height: height,
	}
}
