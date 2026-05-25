package threading

import (
	"errors"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkBatchPredictBlockLumaIntraDC(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != 30 {
				t.Fatalf("sample(%d,%d)=%d want 30", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaFilterIntra(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(20+x))
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16(40+y))
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 33)

	edges := testFrameWorkIntraEdges(output, output.Y, 16, 16, 16, 16)
	wantPix := make([]byte, 16*16)
	want := frame.Plane{Pix: wantPix, Stride: 16, Width: 16, Height: 16}
	if err := prediction.PredictFilterIntraPlaneBlock(want, 1, 8, 0, 0, 16, 16, prediction.FilterIntraModePaeth, edges); err != nil {
		t.Fatal(err)
	}

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.FilterIntraValid = true
	visit.Prediction.FilterIntraMode = tile.FilterIntraModePaeth
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, 16+y)
			if want := uint16(wantPix[y*16+x]); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", 16+x, 16+y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraStaticModes(t *testing.T) {
	for _, tt := range []struct {
		name string
		mode tile.IntraMode
		want func(x int, y int) uint16
	}{
		{
			name: "vertical",
			mode: tile.IntraModeVertical,
			want: func(x int, _ int) uint16 { return uint16(200 + x - 16) },
		},
		{
			name: "horizontal",
			mode: tile.IntraModeHorizontal,
			want: func(_ int, y int) uint16 { return uint16(300 + y - 16) },
		},
		{
			name: "paeth",
			mode: tile.IntraModePaeth,
			want: func(_ int, y int) uint16 { return uint16(300 + y - 16) },
		},
		{
			name: "smooth",
			mode: tile.IntraModeSmooth,
			want: nil,
		},
		{
			name: "smooth-vertical",
			mode: tile.IntraModeSmoothVertical,
			want: nil,
		},
		{
			name: "smooth-horizontal",
			mode: tile.IntraModeSmoothHorizontal,
			want: nil,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
			ctx := testIntraPredictionBatch(output)
			for x := 16; x < 32; x++ {
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(200+x-16))
			}
			for y := 16; y < 32; y++ {
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16(300+y-16))
			}
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 180)

			var scratch FrameWorkIntraPredictionScratch
			visit := testIntraPredictionVisit(tt.mode)
			if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
				t.Fatal(err)
			}
			for y := 16; y < 32; y++ {
				for x := 16; x < 32; x++ {
					got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
					if got > 1023 {
						t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
					}
					if tt.want != nil {
						if want := tt.want(x, y); got != want {
							t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
						}
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalKnownVectors(t *testing.T) {
	tests := []struct {
		name string
		mode tile.IntraMode
		seed func(*frame.Frame)
		want []uint16
	}{
		{
			name: "d45",
			mode: tile.IntraModeD45,
			seed: func(output *frame.Frame) {
				for i, sample := range []uint16{10, 20, 30, 40, 50, 60, 70, 80} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+i, 15, sample)
				}
			},
			want: []uint16{
				20, 30, 40, 50,
				30, 40, 50, 60,
				40, 50, 60, 70,
				50, 60, 70, 80,
			},
		},
		{
			name: "d135",
			mode: tile.IntraModeD135,
			seed: func(output *frame.Frame) {
				for i, sample := range []uint16{21, 31, 41, 51} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+i, 15, sample)
				}
				for i, sample := range []uint16{101, 111, 121, 131} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 16+i, sample)
				}
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 11)
			},
			want: []uint16{
				11, 21, 31, 41,
				101, 11, 21, 31,
				111, 101, 11, 21,
				121, 111, 101, 11,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
			ctx := testIntraPredictionBatch(output)
			tt.seed(output)

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, testIntraPrediction4x4Visit(tt.mode), &scratch); err != nil {
				t.Fatal(err)
			}
			got := make([]uint16, 0, 16)
			for y := 16; y < 20; y++ {
				for x := 16; x < 20; x++ {
					got = append(got, frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y))
				}
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("prediction=%v want %v", got, tt.want)
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalModes(t *testing.T) {
	for _, tt := range []struct {
		name string
		mode tile.IntraMode
	}{
		{name: "d45", mode: tile.IntraModeD45},
		{name: "d67", mode: tile.IntraModeD67},
		{name: "d113", mode: tile.IntraModeD113},
		{name: "d135", mode: tile.IntraModeD135},
		{name: "d157", mode: tile.IntraModeD157},
		{name: "d203", mode: tile.IntraModeD203},
	} {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 96, Height: 96, BitDepth: 10, Align: 128})
			ctx := testIntraPredictionBatch(output)
			seedFrameWorkDirectionalEdges(output, 0x3ff, 19, 37, 101)

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, testIntraPredictionVisit(tt.mode), &scratch); err != nil {
				t.Fatal(err)
			}
			for y := 16; y < 32; y++ {
				for x := 16; x < 32; x++ {
					if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got > 0x3ff {
						t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalIntraEdgeUpsample(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence.EnableIntraEdgeFilter = true
	leftSamples := []uint16{97, 100, 103, 107, 109, 111, 112, 113}
	for y, sample := range leftSamples {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, sample)
	}
	visit := testIntraPrediction4x4Visit(tile.IntraModeD203)
	visit.Block.MIRow = 0
	visit.Block.MIRowEnd = 1
	visit.Block.Y4 = 0
	visit.Block.HaveTop = false
	visit.Block.HaveLeft = true

	left := make([]uint16, frameWorkDirectionalEdgeSamples)
	origin := frameWorkDirectionalEdgeOrigin
	left[origin-1] = leftSamples[0]
	copy(left[origin:], leftSamples)
	scratch := make([]uint16, frameWorkIntraEdgeScratchSamples)
	if err := prediction.UpsampleIntraEdge(left, origin, 8, scratch, 8); err != nil {
		t.Fatal(err)
	}
	wantPix := make([]byte, 16)
	want := frame.Plane{Pix: wantPix, Stride: 4, Width: 4, Height: 4}
	edges := prediction.DirectionalEdges{
		Left:         left,
		LeftOrigin:   origin,
		UpsampleLeft: true,
	}
	if err := prediction.PredictDirectionalIntraPlaneBlock(want, 1, 8, 0, 0, 4, 4, 203, edges); err != nil {
		t.Fatal(err)
	}

	var predScratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &predScratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, y)
			if want := uint16(wantPix[y*4+x]); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", 16+x, y, got, want)
			}
		}
	}
}

func TestFrameWorkLumaTransformDirectionalExtendedEdgesMatchesLibaomCases(t *testing.T) {
	tests := []struct {
		name           string
		block          tile.BlockVisit
		miColEnd       uint32
		miRowEnd       uint32
		absX           int
		absY           int
		width          int
		height         int
		wantTopRight   bool
		wantBottomLeft bool
	}{
		{
			name: "d203 right-half 8x4 cannot use bottom-left",
			block: tile.BlockVisit{
				MICol: 56, MIRow: 0, MIColEnd: 58, MIRowEnd: 1,
				Size: tile.BlockSize8x4, VisibleW4: 2, VisibleH4: 1, HaveLeft: true,
			},
			absX: 228, absY: 0, width: 4, height: 4,
		},
		{
			name: "d203 left-half 8x4 table allows bottom-left",
			block: tile.BlockVisit{
				MICol: 56, MIRow: 0, MIColEnd: 58, MIRowEnd: 1,
				Size: tile.BlockSize8x4, VisibleW4: 2, VisibleH4: 1, HaveLeft: true,
			},
			absX: 224, absY: 0, width: 4, height: 4, wantBottomLeft: true,
		},
		{
			name: "d203 4x8 table denies bottom-left",
			block: tile.BlockVisit{
				MICol: 66, MIRow: 0, MIColEnd: 67, MIRowEnd: 2,
				Size: tile.BlockSize4x8, VisibleW4: 1, VisibleH4: 2, HaveLeft: true,
			},
			absX: 264, absY: 4, width: 4, height: 4,
		},
		{
			name: "d45 right-half 8x8 cannot use top-right",
			block: tile.BlockVisit{
				MICol: 74, MIRow: 2, MIColEnd: 76, MIRowEnd: 4,
				Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2, HaveTop: true, HaveLeft: true,
			},
			absX: 300, absY: 8, width: 4, height: 4,
		},
		{
			name: "d45 left-half 8x8 has internal top-right",
			block: tile.BlockVisit{
				MICol: 74, MIRow: 2, MIColEnd: 76, MIRowEnd: 4,
				Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2, HaveTop: true, HaveLeft: true,
			},
			absX: 296, absY: 8, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "quantizer00 d67 4x4 table allows top-right",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			miColEnd: 88, miRowEnd: 72,
			absX: 32, absY: 8, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "quantizer00 d67 4x4 denies top-right at tile edge",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			miColEnd: 9, miRowEnd: 72,
			absX: 32, absY: 8, width: 4, height: 4, wantBottomLeft: true,
		},
		{
			name: "quantizer00 internal transform top edge allows top-right",
			block: func() tile.BlockVisit {
				parent := tile.BlockVisit{
					MICol: 24, MIRow: 0, MIColEnd: 28, MIRowEnd: 2,
					X4: 24, Y4: 0, Partition: tile.PartitionTBottomSplit, Size: tile.BlockSize16x8,
					VisibleW4: 4, VisibleH4: 2, HaveLeft: true,
				}
				return frameWorkPredictionTransformEdgeBlock(parent, parent.X4, parent.Y4, 24, 1)
			}(),
			miColEnd: 88, miRowEnd: 72,
			absX: 96, absY: 4, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			miColEnd := tt.miColEnd
			if miColEnd == 0 {
				miColEnd = 88
			}
			miRowEnd := tt.miRowEnd
			if miRowEnd == 0 {
				miRowEnd = 72
			}
			gotTopRight, gotBottomLeft := frameWorkLumaDirectionalExtendedEdges(tt.block, 32, miColEnd, miRowEnd, tt.absX, tt.absY, tt.width, tt.height)
			if gotTopRight != tt.wantTopRight || gotBottomLeft != tt.wantBottomLeft {
				t.Fatalf("topRight=%v bottomLeft=%v want %v/%v", gotTopRight, gotBottomLeft, tt.wantTopRight, tt.wantBottomLeft)
			}
		})
	}
}

func TestFrameWorkDirectionalAvailabilityTablesCoverBlockSizes(t *testing.T) {
	sizes := []tile.BlockSize{
		tile.BlockSize4x4, tile.BlockSize4x8, tile.BlockSize8x4, tile.BlockSize8x8,
		tile.BlockSize8x16, tile.BlockSize16x8, tile.BlockSize16x16,
		tile.BlockSize16x32, tile.BlockSize32x16, tile.BlockSize32x32,
		tile.BlockSize32x64, tile.BlockSize64x32, tile.BlockSize64x64,
		tile.BlockSize64x128, tile.BlockSize128x64, tile.BlockSize128x128,
		tile.BlockSize4x16, tile.BlockSize16x4, tile.BlockSize8x32,
		tile.BlockSize32x8, tile.BlockSize16x64, tile.BlockSize64x16,
	}
	for _, size := range sizes {
		if table := frameWorkTopRightAvailabilityTable(tile.PartitionNone, size); len(table) == 0 {
			t.Fatalf("missing top-right table for size %d", size)
		}
		if table := frameWorkBottomLeftAvailabilityTable(tile.PartitionNone, size); len(table) == 0 {
			t.Fatalf("missing bottom-left table for size %d", size)
		}
	}
	if table := frameWorkTopRightAvailabilityTable(tile.PartitionTLeftSplit, tile.BlockSize8x8); len(table) == 0 || table[2] != 0 {
		t.Fatalf("mixed-vertical top-right table not selected: %v", table)
	}
	if table := frameWorkBottomLeftAvailabilityTable(tile.PartitionTRightSplit, tile.BlockSize8x8); len(table) == 0 || table[0] != 254 {
		t.Fatalf("mixed-vertical bottom-left table not selected: %v", table)
	}
}

func TestFrameWorkChromaDirectionalExtendedEdgesMatchesLibaomCases(t *testing.T) {
	tests := []struct {
		name           string
		block          tile.BlockVisit
		originX        int
		originY        int
		absX           int
		absY           int
		width          int
		height         int
		wantTopRight   bool
		wantBottomLeft bool
	}{
		{
			name: "quantizer00 horizontal right-half denies bottom-left",
			block: func() tile.BlockVisit {
				parent := tile.BlockVisit{
					MICol: 4, MIRow: 0, MIColEnd: 8, MIRowEnd: 2,
					X4: 4, Y4: 0, Partition: tile.PartitionH, Size: tile.BlockSize16x8,
					VisibleW4: 4, VisibleH4: 2, HaveLeft: true,
				}
				return frameWorkPredictionTransformEdgeBlock(parent, parent.X4>>1, parent.Y4>>1, 3, 0)
			}(),
			originX: 8, originY: 0,
			absX: 12, absY: 0, width: 4, height: 4,
		},
		{
			name: "quantizer00 dc next block has left-edge bottom-left",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 1, MIColEnd: 10, MIRowEnd: 2,
				X4: 8, Y4: 1, Partition: tile.PartitionH, Size: tile.BlockSize8x4,
				VisibleW4: 2, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 0,
			absX: 16, absY: 0, width: 4, height: 4,
			wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "small 4x4 420 scales to 8x8 for table lookup",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 4,
			absX: 16, absY: 4, width: 4, height: 4,
			wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "rightmost 420 chroma tx denies top-right at tile edge",
			block: tile.BlockVisit{
				MICol: 84, MIRow: 8, MIColEnd: 88, MIRowEnd: 12,
				Partition: tile.PartitionNone, Size: tile.BlockSize16x16,
				VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true,
			},
			originX: 168, originY: 16,
			absX: 172, absY: 16, width: 4, height: 4,
		},
		{
			name: "bottom 420 chroma tx denies bottom-left at tile edge",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 68, MIColEnd: 12, MIRowEnd: 72,
				Partition: tile.PartitionNone, Size: tile.BlockSize16x16,
				VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 136,
			absX: 16, absY: 140, width: 4, height: 4,
			wantTopRight: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTopRight, gotBottomLeft := frameWorkChromaDirectionalExtendedEdges(tt.block, 32, 88, 72, tt.originX, tt.originY, tt.absX, tt.absY, tt.width, tt.height, true, true)
			if gotTopRight != tt.wantTopRight || gotBottomLeft != tt.wantBottomLeft {
				t.Fatalf("topRight=%v bottomLeft=%v want %v/%v", gotTopRight, gotBottomLeft, tt.wantTopRight, tt.wantBottomLeft)
			}
		})
	}
}

func TestFrameWorkChromaAvailabilityBlockSizeMatchesLibaom(t *testing.T) {
	tests := []struct {
		name         string
		size         tile.BlockSize
		subsamplingX bool
		subsamplingY bool
		want         tile.BlockSize
	}{
		{name: "4x4 420", size: tile.BlockSize4x4, subsamplingX: true, subsamplingY: true, want: tile.BlockSize8x8},
		{name: "4x4 422", size: tile.BlockSize4x4, subsamplingX: true, want: tile.BlockSize8x4},
		{name: "4x4 440", size: tile.BlockSize4x4, subsamplingY: true, want: tile.BlockSize4x8},
		{name: "4x8 422", size: tile.BlockSize4x8, subsamplingX: true, want: tile.BlockSize8x8},
		{name: "8x4 440", size: tile.BlockSize8x4, subsamplingY: true, want: tile.BlockSize8x8},
		{name: "4x16 420", size: tile.BlockSize4x16, subsamplingX: true, subsamplingY: true, want: tile.BlockSize8x16},
		{name: "16x4 420", size: tile.BlockSize16x4, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x8},
		{name: "16x16 420 unchanged", size: tile.BlockSize16x16, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x16},
		{name: "16x8 420 unchanged", size: tile.BlockSize16x8, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := frameWorkChromaAvailabilityBlockSize(tt.size, tt.subsamplingX, tt.subsamplingY); got != tt.want {
				t.Fatalf("size=%d want %d", got, tt.want)
			}
		})
	}
	got := frameWorkChromaAvailabilityBlockSize(tile.BlockSize16x16, true, true)
	plane, err := tile.PlaneBlockSize(tile.BlockSize16x16, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got == plane {
		t.Fatalf("availability size used residual plane size %d", plane)
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraBoundaryEdges(t *testing.T) {
	tests := []struct {
		name  string
		mode  tile.IntraMode
		seed  func(*frame.Frame)
		visit func() tile.BlockLoopVisit
		want  uint16
	}{
		{
			name: "vertical-missing-top-uses-left",
			mode: tile.IntraModeVertical,
			seed: func(output *frame.Frame) {
				for y := 0; y < 16; y++ {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 70)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeVertical)
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				return visit
			},
			want: 70,
		},
		{
			name: "directional-missing-top-uses-left",
			mode: tile.IntraModeD45,
			seed: func(output *frame.Frame) {
				for y := 0; y < 16; y++ {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 81)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeD45)
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				return visit
			},
			want: 81,
		},
		{
			name: "horizontal-missing-left-uses-top",
			mode: tile.IntraModeHorizontal,
			seed: func(output *frame.Frame) {
				for x := 0; x < 16; x++ {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 93)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeHorizontal)
				visit.Block.MICol = 0
				visit.Block.MIColEnd = 4
				visit.Block.X4 = 0
				visit.Block.HaveLeft = false
				return visit
			},
			want: 93,
		},
		{
			name: "vertical-missing-both-uses-mid-minus-one",
			mode: tile.IntraModeVertical,
			seed: func(*frame.Frame) {},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeVertical)
				visit.Block.MICol = 0
				visit.Block.MIColEnd = 4
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.X4 = 0
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				visit.Block.HaveLeft = false
				return visit
			},
			want: 127,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
			ctx := testIntraPredictionBatch(output)
			tt.seed(output)
			visit := tt.visit()

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
				t.Fatal(err)
			}
			x, y, err := frameWorkBlockLumaPosition(visit.Block)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < 16; row++ {
				for col := 0; col < 16; col++ {
					if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x+col, y+row); got != tt.want {
						t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, got, tt.want)
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaInterFullpel(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Col: 8, Row: -8})
	if err := ctx.PredictBlockLumaInter(0, visit); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
			want := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x+1, y-1)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterFullpelExtendsReferenceEdges(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	visit := testInterPredictionVisit(motion.Vector{Col: -16, Row: 16})
	visit.Block = tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4,
		X4: 0, Y4: 0, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
	}
	if err := ctx.PredictBlockInter(0, visit, nil); err != nil {
		t.Fatal(err)
	}

	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
			want := frameWorkTestSampleClamped(reference.Y, output.Layout.BytesPerSample, x-2, y+2)
			if got != want {
				t.Fatalf("y sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			wantU := frameWorkTestSampleClamped(reference.U, output.Layout.BytesPerSample, x-1, y+1)
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != wantU {
				t.Fatalf("u sample(%d,%d)=%d want %d", x, y, got, wantU)
			}
			wantV := frameWorkTestSampleClamped(reference.V, output.Layout.BytesPerSample, x-1, y+1)
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != wantV {
				t.Fatalf("v sample(%d,%d)=%d want %d", x, y, got, wantV)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaInterFractionalMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: 5}
	filters := motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterInvalidWarpFallsBackToTranslation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: 5}
	filters := motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	visit.Prediction.WarpedMotionInvalid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterWarpIdentityConstant(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 0xab)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := output.Y.Pix[y*output.Y.Stride+x]; got != 0xab {
				t.Fatalf("sample(%d,%d)=%d want 0xab", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterWarpSmallChromaFallsBackToRegular(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapRegular}
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Block.MIColEnd = 6
	visit.Block.MIRowEnd = 6
	visit.Block.Size = tile.BlockSize8x8
	visit.Block.VisibleW4 = 2
	visit.Block.VisibleH4 = 2
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[0] = 1 << 16
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, filters); err != nil {
		t.Fatal(err)
	}

	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 4, 4, motion.Vector{}, true, true, filters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 4, 4, motion.Vector{}, true, true, filters)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 4, 4)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 4, 4)
}

func TestFrameWorkBatchPredictBlockLumaInterHighBitDepthFractionalMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 4, Row: 6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterOBMCLeftMatchesLibaomMask(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(reference, 0xff, 17)
	ctx := testInterPredictionBatch(output, reference)

	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	visit.Prediction.OverlappableNeighbors.LeftCount = 1
	visit.Prediction.OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{{Col: 8}},
		},
		InterpFilters:      motion.RegularFilters,
		InterpFiltersValid: true,
	}

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}

	mask, ok := frameWorkOBMCMask(8)
	if !ok {
		t.Fatal("missing 8-wide obmc mask")
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			base := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y)
			want := base
			if x < 24 {
				neighbor := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x+1, y)
				want = frameWorkBlendA64(uint16(mask[x-16]), base, neighbor)
			}
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterOBMCChromaSubsampledMatchesLibaomMasks(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	visit := testOBMCInterPredictionVisit(motion.Vector{}, motion.Vector{Row: 16}, motion.Vector{Col: 16})
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}

	mask4, ok := frameWorkOBMCMask(4)
	if !ok {
		t.Fatal("missing 4-wide obmc mask")
	}
	for _, tt := range []struct {
		plane frame.Plane
		ref   frame.Plane
	}{
		{plane: output.U, ref: reference.U},
		{plane: output.V, ref: reference.V},
	} {
		for y := 8; y < 16; y++ {
			for x := 8; x < 16; x++ {
				want := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x, y)
				if y < 12 {
					neighbor := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x, y+1)
					want = frameWorkBlendA64(uint16(mask4[y-8]), want, neighbor)
				}
				if x < 12 {
					neighbor := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x+1, y)
					want = frameWorkBlendA64(uint16(mask4[x-8]), want, neighbor)
				}
				if got := frameWorkTestSample(tt.plane, output.Layout.BytesPerSample, x, y); got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
				}
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterChromaSubsampledMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 128})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: -5}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, filters); err != nil {
		t.Fatal(err)
	}

	wantY := testFrameWorkMotionPredictionPlane(t, reference.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv, filters)
	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, filters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, filters)
	assertFrameWorkPlaneBlockEqualAt(t, output.Y, 16, 16, wantY, 0, 0, output.Layout.BytesPerSample, 16, 16)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 8, 8)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 8, 8)
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageMatchesMotion(t *testing.T) {
	tests := []struct {
		name    string
		format  frame.Format
		max     uint16
		mv0     motion.Vector
		mv1     motion.Vector
		filters motion.InterpFilters
	}{
		{
			name:    "lowbd",
			format:  frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64},
			max:     0xff,
			mv0:     motion.Vector{Col: 3, Row: 5},
			mv1:     motion.Vector{Col: -1, Row: 6},
			filters: motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth},
		},
		{
			name:    "highbd-dist-wtd",
			format:  frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128},
			max:     0x3ff,
			mv0:     motion.Vector{Col: 4, Row: 6},
			mv1:     motion.Vector{Col: -1, Row: 3},
			filters: motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, tt.format)
			last := testBatchFrame(t, tt.format)
			bwd := testBatchFrame(t, tt.format)
			fillFrameWorkInterReference(last, tt.max)
			fillFrameWorkInterReferenceVariant(bwd, tt.max, 37)
			ctx := testCompoundInterPredictionBatch(output, last, bwd)
			compoundType := tile.CompoundTypeAverage
			if tt.name == "highbd-dist-wtd" {
				compoundType = tile.CompoundTypeDistWtd
			}
			visit := testCompoundInterPredictionVisit(tt.mv0, tt.mv1, compoundType)

			var scratch FrameWorkInterPredictionScratch
			if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, tt.filters); err != nil {
				t.Fatal(err)
			}

			first := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv0, tt.filters)
			second := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv1, tt.filters)
			fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
			if err != nil {
				t.Fatal(err)
			}
			assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, 16, 16, 16, 16, fwdOffset, bckOffset)
		})
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundChromaSubsampledMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 19)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 73)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDistWtd)

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstU := testFrameWorkMotionPredictionPlaneSubsampled(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkMotionPredictionPlaneSubsampled(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
	if err != nil {
		t.Fatal(err)
	}
	assertFrameWorkCompoundBlendEqual(t, output.U, firstU, secondU, output.Layout.BytesPerSample, 8, 8, 8, 8, fwdOffset, bckOffset)
	assertFrameWorkCompoundBlendEqual(t, output.V, firstV, secondV, output.Layout.BytesPerSample, 8, 8, 8, 8, fwdOffset, bckOffset)
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundDiffWtdMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0x3ff, 29)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 113)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38Inv

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, filters)
	second := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, filters)
	mask := testFrameWorkDiffWtdMask(t, first, second, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, tile.DiffWtdMaskType38Inv)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, first, second, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockInterCompoundDiffWtdChromaSubsampledMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 17)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 191)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 5, Row: -3}
	mv1 := motion.Vector{Col: -1, Row: 7}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstY := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, filters)
	secondY := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, filters)
	mask := testFrameWorkDiffWtdMask(t, firstY, secondY, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, tile.DiffWtdMaskType38)
	firstU := testFrameWorkMotionPredictionPlaneSubsampled(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkMotionPredictionPlaneSubsampled(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, firstY, secondY, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
	assertFrameWorkMaskedCompoundEqual(t, output.U, firstU, secondU, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
	assertFrameWorkMaskedCompoundEqual(t, output.V, firstV, secondV, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
}

func TestFrameWorkBuildWedgeMaskMatchesLibaomSamples(t *testing.T) {
	tests := []struct {
		name       string
		size       tile.BlockSize
		index      uint8
		sign       bool
		width      int
		height     int
		samples    [][2]int
		wantSample []byte
	}{
		{
			name:       "8x8 oblique27 neg",
			size:       tile.BlockSize8x8,
			index:      0,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {7, 7}},
			wantSample: []byte{64, 0},
		},
		{
			name:       "8x8 oblique27 positive",
			size:       tile.BlockSize8x8,
			index:      0,
			sign:       true,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {7, 7}},
			wantSample: []byte{0, 64},
		},
		{
			name:       "8x8 vertical neg",
			size:       tile.BlockSize8x8,
			index:      6,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {1, 0}, {2, 0}, {7, 0}},
			wantSample: []byte{57, 43, 21, 0},
		},
		{
			name:       "32x8 oblique63 positive",
			size:       tile.BlockSize32x8,
			index:      12,
			width:      32,
			height:     8,
			samples:    [][2]int{{0, 0}, {31, 7}},
			wantSample: []byte{0, 64},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := make([]byte, tt.width*tt.height)
			if err := frameWorkBuildWedgeMask(mask, tt.width, tt.size, tt.index, tt.sign); err != nil {
				t.Fatal(err)
			}
			for i, sample := range tt.samples {
				got := mask[sample[1]*tt.width+sample[0]]
				if got != tt.wantSample[i] {
					t.Fatalf("mask(%d,%d)=%d want %d", sample[0], sample[1], got, tt.wantSample[i])
				}
			}
		})
	}
}

func TestFrameWorkBuildWedgeMaskSignComplementsLibaom(t *testing.T) {
	sizes := []tile.BlockSize{
		tile.BlockSize8x8,
		tile.BlockSize8x16,
		tile.BlockSize16x8,
		tile.BlockSize16x16,
		tile.BlockSize8x32,
		tile.BlockSize32x8,
		tile.BlockSize16x32,
		tile.BlockSize32x16,
		tile.BlockSize32x32,
	}
	var mask [32 * 32]byte
	var complement [32 * 32]byte
	for _, size := range sizes {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("missing dimensions for %v", size)
		}
		width := int(dims.W4) * 4
		height := int(dims.H4) * 4
		for index := uint8(0); index < tile.MaxWedgeTypes; index++ {
			if err := frameWorkBuildWedgeMask(mask[:], width, size, index, false); err != nil {
				t.Fatalf("size=%v index=%d neg err=%v", size, index, err)
			}
			if err := frameWorkBuildWedgeMask(complement[:], width, size, index, true); err != nil {
				t.Fatalf("size=%v index=%d pos err=%v", size, index, err)
			}
			for row := 0; row < height; row++ {
				for col := 0; col < width; col++ {
					got := int(mask[row*width+col]) + int(complement[row*width+col])
					if got != frameWorkWedgeMaxAlpha {
						t.Fatalf("size=%v index=%d mask(%d,%d) sum=%d want %d", size, index, col, row, got, frameWorkWedgeMaxAlpha)
					}
				}
			}
		}
	}
	if err := frameWorkBuildWedgeMask(mask[:], 4, tile.BlockSize4x4, 0, false); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("unsupported wedge err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBuildWedgeMaskAllocs(t *testing.T) {
	var mask [16 * 16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		if err := frameWorkBuildWedgeMask(mask[:], 16, tile.BlockSize16x16, 0, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("frameWorkBuildWedgeMask allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundWedgeMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0x3ff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 149)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 0
	visit.Prediction.CompoundBlend.WedgeSign = false

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, filters)
	second := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, filters)
	mask := make([]byte, 16*16)
	if err := frameWorkBuildWedgeMask(mask, 16, visit.Block.Size, visit.Prediction.CompoundBlend.WedgeIndex, visit.Prediction.CompoundBlend.WedgeSign); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkMaskedCompoundEqual(t, output.Y, first, second, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockInterCompoundWedgeChromaSubsampledMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 43)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 211)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 5, Row: -3}
	mv1 := motion.Vector{Col: -1, Row: 7}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 6
	visit.Prediction.CompoundBlend.WedgeSign = false

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstY := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, filters)
	secondY := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, filters)
	mask := make([]byte, 16*16)
	if err := frameWorkBuildWedgeMask(mask, 16, visit.Block.Size, visit.Prediction.CompoundBlend.WedgeIndex, visit.Prediction.CompoundBlend.WedgeSign); err != nil {
		t.Fatal(err)
	}
	firstU := testFrameWorkMotionPredictionPlaneSubsampled(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkMotionPredictionPlaneSubsampled(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, firstY, secondY, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
	assertFrameWorkMaskedCompoundEqual(t, output.U, firstU, secondU, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
	assertFrameWorkMaskedCompoundEqual(t, output.V, firstV, secondV, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
}

func TestFrameWorkCompoundDistanceWeightedOffsetsMatchLibaom(t *testing.T) {
	tests := []struct {
		name string
		cur  uint32
		ref0 uint32
		ref1 uint32
		want [2]int
	}{
		{name: "equal distance", cur: 8, ref0: 4, ref1: 12, want: [2]int{7, 9}},
		{name: "forward nearer", cur: 8, ref0: 2, ref1: 10, want: [2]int{4, 12}},
		{name: "backward nearer", cur: 8, ref0: 6, ref1: 15, want: [2]int{13, 3}},
		{name: "zero distance", cur: 8, ref0: 8, ref1: 12, want: [2]int{13, 3}},
		{name: "wraparound", cur: 1, ref0: 14, ref1: 4, want: [2]int{7, 9}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fwd, bck, err := frameWorkDistanceWeightedCompoundOffsets(4, tt.cur, tt.ref0, tt.ref1)
			if err != nil {
				t.Fatal(err)
			}
			if got := [2]int{fwd, bck}; got != tt.want {
				t.Fatalf("offsets=%v want %v", got, tt.want)
			}
		})
	}
	if _, _, err := frameWorkDistanceWeightedCompoundOffsets(0, 0, 0, 0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad bits err=%v want %v", err, ErrInvalidBatch)
	}
	if _, _, err := frameWorkDistanceWeightedCompoundOffsets(4, 16, 0, 1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("out of range hint err=%v want %v", err, ErrInvalidBatch)
	}
}

func FuzzFrameWorkDistanceWeightedCompoundOffsets(f *testing.F) {
	f.Add(uint8(4), uint32(8), uint32(4), uint32(12))
	f.Add(uint8(4), uint32(8), uint32(2), uint32(10))
	f.Add(uint8(4), uint32(1), uint32(14), uint32(4))
	f.Add(uint8(5), uint32(16), uint32(3), uint32(28))

	f.Fuzz(func(t *testing.T, bits uint8, cur uint32, ref0 uint32, ref1 uint32) {
		bits = bits%8 + 1
		limit := uint32(1) << bits
		cur %= limit
		ref0 %= limit
		ref1 %= limit
		fwd, bck, err := frameWorkDistanceWeightedCompoundOffsets(bits, cur, ref0, ref1)
		if err != nil {
			t.Fatalf("frameWorkDistanceWeightedCompoundOffsets err=%v", err)
		}
		if fwd < 0 || bck < 0 || fwd+bck != 16 {
			t.Fatalf("offsets=%d,%d", fwd, bck)
		}
	})
}

func TestFrameWorkBatchPredictBlockLumaDispatchesCompoundAverage(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(last, 0xff)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 91)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 8, Row: 0}, motion.Vector{Col: -8, Row: 0}, tile.CompoundTypeAverage)

	var interScratch FrameWorkInterPredictionScratch
	scratch := FrameWorkPredictionScratch{Inter: &interScratch}
	if err := ctx.PredictBlockLuma(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	first := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[0], motion.RegularFilters)
	second := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[1], motion.RegularFilters)
	assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, 16, 16, 16, 16, 8, 8)
}

func TestFrameWorkBatchPredictBlockDispatchesCompletePlanes(t *testing.T) {
	mono := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	monoCtx := testIntraPredictionBatch(mono)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, 15, y, 50)
	}
	var predictionScratch FrameWorkPredictionScratch
	if err := monoCtx.PredictBlock(0, testIntraPredictionVisit(tile.IntraModeDC), &predictionScratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("mono intra sample=%d want 30", got)
	}

	colorIntra := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	colorIntraCtx := testIntraPredictionBatch(colorIntra)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, 15, y, 50)
	}
	for x := 8; x < 16; x++ {
		setFrameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, x, 7, 20)
		setFrameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, x, 7, 30)
	}
	for y := 8; y < 16; y++ {
		setFrameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, 7, y, 60)
		setFrameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, 7, y, 70)
	}
	colorVisit := testIntraPredictionVisit(tile.IntraModeDC)
	colorVisit.Prediction.ChromaMode = tile.ChromaIntraModeDC
	colorVisit.Prediction.ChromaModeValid = true
	if err := colorIntraCtx.PredictBlock(0, colorVisit, &predictionScratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("color intra y sample=%d want 30", got)
	}
	if got := frameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, 8, 8); got != 40 {
		t.Fatalf("color intra u sample=%d want 40", got)
	}
	if got := frameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, 8, 8); got != 50 {
		t.Fatalf("color intra v sample=%d want 50", got)
	}

	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: -5}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlock(0, visit, nil); err != nil {
		t.Fatal(err)
	}

	wantY := testFrameWorkMotionPredictionPlane(t, reference.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv, motion.RegularFilters)
	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, motion.RegularFilters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, motion.RegularFilters)
	assertFrameWorkPlaneBlockEqualAt(t, output.Y, 16, 16, wantY, 0, 0, output.Layout.BytesPerSample, 16, 16)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 8, 8)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 8, 8)
}

func TestFrameWorkBatchPredictBlockIntraSubsampledChromaUsesPlaneEdges(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for y := 0; y < 16; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 3, y, 64)
	}

	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 1, MIRow: 0, MIColEnd: 2, MIRowEnd: 4,
		X4: 1, Y4: 0, Size: tile.BlockSize4x16, VisibleW4: 1, VisibleH4: 4,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeVertical
	visit.Prediction.ChromaModeValid = true

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 4; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 127 {
				t.Fatalf("u sample(%d,%d)=%d want 127", x, y, got)
			}
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 127 {
				t.Fatalf("v sample(%d,%d)=%d want 127", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockIntraCoeffClippedChromaSmooth(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 352, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	})
	ctx.Jobs = []tile.Job{{SBCols: 3, SBRows: 2}}
	for y := 0; y < 32; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 159, y, 77)
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 159, y, 91)
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 64, MIRow: 0, MIColEnd: 88, MIRowEnd: 32,
		X4: 0, Y4: 0, Size: tile.BlockSize128x128, VisibleW4: 24, VisibleH4: 32,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeSmooth
	visit.Prediction.ChromaModeValid = true
	block := tile.BlockCoeffBlock{
		Plane: 1,
		Block: tile.TransformBlock{X4: 8, Y4: 0, Size: tile.TransformSize32x32, VisibleW4: 4, VisibleH4: 8},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 32; y++ {
		for x := 160; x < 176; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 77 {
				t.Fatalf("u sample(%d,%d)=%d want 77", x, y, got)
			}
		}
	}
	block.Plane = 2
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 32; y++ {
		for x := 160; x < 176; x++ {
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 91 {
				t.Fatalf("v sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockIntraClippedChromaSmooth(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 352, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	})
	ctx.Jobs = []tile.Job{{SBCols: 3, SBRows: 2}}
	for y := 0; y < 64; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 127, y, 77)
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 127, y, 91)
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 64, MIRow: 0, MIColEnd: 88, MIRowEnd: 32,
		X4: 0, Y4: 0, Size: tile.BlockSize128x128, VisibleW4: 24, VisibleH4: 32,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeSmooth
	visit.Prediction.ChromaModeValid = true

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 64; y++ {
		for x := 128; x < 176; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 77 {
				t.Fatalf("u sample(%d,%d)=%d want 77", x, y, got)
			}
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 91 {
				t.Fatalf("v sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockRejectsIncompletePlaneSupport(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing chroma mode err=%v want %v", err, ErrInvalidBatch)
	}
	cfl := testIntraPredictionVisit(tile.IntraModeDC)
	cfl.Prediction.ChromaMode = tile.ChromaIntraModeCFL
	cfl.Prediction.ChromaModeValid = true
	cfl.Prediction.CFLAlphaValid = true
	if err := ctx.PredictBlock(0, cfl, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("cfl intra err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlock(0, tile.BlockLoopVisit{}, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing prediction err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLMatchesPrimitives(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	want := testBatchFrame(t, output.Format)
	visit := testCFLPredictionVisit()
	seedFrameWorkCFLPredictionFrame(output)
	seedFrameWorkCFLPredictionFrame(want)

	ctx := testIntraPredictionBatch(output)
	var scratch FrameWorkCFLPredictionScratch
	if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}

	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneU)
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneV)
	assertFrameWorkPlaneBlockEqual(t, output.U, want.U, output.Layout.BytesPerSample, 8, 8, 8, 8)
	assertFrameWorkPlaneBlockEqual(t, output.V, want.V, output.Layout.BytesPerSample, 8, 8, 8, 8)
}

func TestFrameWorkSubsampleLumaCFLQ3MatchesPrimitives(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		subX     bool
		subY     bool
	}{
		{name: "lowbd-420", bitDepth: 8, subX: true, subY: true},
		{name: "highbd-444", bitDepth: 10},
		{name: "highbd-422", bitDepth: 12, subX: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format := frame.Format{Width: 64, Height: 64, BitDepth: tt.bitDepth, SubsamplingX: tt.subX, SubsamplingY: tt.subY, Align: 128}
			output := testBatchFrame(t, format)
			bytesPerSample := output.Layout.BytesPerSample
			const x = 16
			const y = 16
			const width = 16
			const height = 16
			stride := width + 5
			max := int((1 << tt.bitDepth) - 1)
			var got [prediction.CFLBufSquare]uint16
			var want [prediction.CFLBufSquare]uint16
			if bytesPerSample == 1 {
				src := make([]uint8, stride*height)
				for row := 0; row < height; row++ {
					for col := 0; col < width; col++ {
						value := uint8((17 + row*13 + col*19) & max)
						src[row*stride+col] = value
						setFrameWorkTestSample(output.Y, bytesPerSample, x+col, y+row, uint16(value))
					}
				}
				if err := prediction.SubsampleLuma8ToQ3(want[:], src, stride, width, height, tt.subX, tt.subY); err != nil {
					t.Fatal(err)
				}
			} else {
				src := make([]uint16, stride*height)
				for row := 0; row < height; row++ {
					for col := 0; col < width; col++ {
						value := uint16((257 + row*83 + col*41) & max)
						src[row*stride+col] = value
						setFrameWorkTestSample(output.Y, bytesPerSample, x+col, y+row, value)
					}
				}
				if err := prediction.SubsampleLuma16ToQ3(want[:], src, stride, width, height, tt.subX, tt.subY, tt.bitDepth); err != nil {
					t.Fatal(err)
				}
			}
			if err := frameWorkSubsampleLumaCFLQ3(got[:], output.Y, bytesPerSample, tt.bitDepth, x, y, width, height, tt.subX, tt.subY); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got[:], want[:]) {
				t.Fatalf("subsample mismatch")
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	valid := testCFLPredictionVisit()
	var scratch FrameWorkCFLPredictionScratch
	if err := ctx.PredictBlockChromaCFL(0, valid, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil scratch err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockChromaCFL(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("non-cfl err=%v want %v", err, ErrInvalidBatch)
	}
	noAlpha := valid
	noAlpha.Prediction.CFLAlphaValid = false
	if err := ctx.PredictBlockChromaCFL(0, noAlpha, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing alpha err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	seedFrameWorkCFLPredictionFrame(output)
	ctx := testIntraPredictionBatch(output)
	visit := testCFLPredictionVisit()
	var scratch FrameWorkCFLPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockChromaCFL allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	var scratch FrameWorkPredictionScratch
	intra := testIntraPredictionVisit(tile.IntraModeDC)
	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlock(0, intra, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlock(0, inter, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlock allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	valid := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch

	tests := []struct {
		name    string
		ctx     FrameWorkBatch
		visit   tile.BlockLoopVisit
		scratch *FrameWorkInterPredictionScratch
	}{
		{name: "nil scratch", ctx: ctx, visit: valid},
		{name: "missing blend", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.CompoundBlendValid = false
			return visit
		}(), scratch: &scratch},
		{name: "wedge", ctx: ctx, visit: testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeWedge), scratch: &scratch},
		{name: "dist wtd without order hint", ctx: func() FrameWorkBatch {
			next := ctx
			next.Sequence.EnableOrderHint = false
			return next
		}(), visit: testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeDistWtd), scratch: &scratch},
		{name: "single ref", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
			visit.Prediction.InterReferences = refs
			visit.Prediction.InterMotion.References = refs
			return visit
		}(), scratch: &scratch},
		{name: "inter intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterIntraValid = true
			visit.Prediction.InterIntra.Enabled = true
			return visit
		}(), scratch: &scratch},
		{name: "non translation motion mode", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.MotionModeValid = true
			visit.Prediction.MotionMode = tile.MotionModeOBMC
			return visit
		}(), scratch: &scratch},
		{name: "missing second reference frame", ctx: func() FrameWorkBatch {
			next := ctx
			next.References = next.References[:int(FrameWorkReferenceBwd)]
			return next
		}(), visit: valid, scratch: &scratch},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInterCompoundAverage(0, tt.visit, tt.scratch); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundRejectsInvalidDiffWtdMask(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 11)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 97)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType(99)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(last, 0x3ff)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 17)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 4, Row: 6}, motion.Vector{Col: -1, Row: 3}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterCompoundAverageWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaInterCompoundAverage allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundDiffWtdAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 157)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 3, Row: 5}, motion.Vector{Col: -1, Row: 7}, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter diff-wtd allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundWedgeAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 157)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 3, Row: 5}, motion.Vector{Col: -1, Row: 7}, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 0
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter wedge allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterOBMCAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testOBMCInterPredictionVisit(motion.Vector{}, motion.Vector{Row: 16}, motion.Vector{Col: 16})
	var scratch FrameWorkInterPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter OBMC allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	ctx := testInterPredictionBatch(output, reference)
	valid := testInterPredictionVisit(motion.Vector{})

	tests := []struct {
		name  string
		ctx   FrameWorkBatch
		visit tile.BlockLoopVisit
	}{
		{name: "missing prediction", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Valid = false
			return visit
		}()},
		{name: "intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intra = true
			return visit
		}()},
		{name: "missing motion", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterMotionValid = false
			return visit
		}()},
		{name: "compound", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterMotion.References.Compound = true
			visit.Prediction.InterMotion.References.Ref[1] = tile.ReferenceFrameBWD
			return visit
		}()},
		{name: "non translation motion mode", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.MotionModeValid = true
			visit.Prediction.MotionMode = tile.MotionModeWarp
			return visit
		}()},
		{name: "inter intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterIntraValid = true
			visit.Prediction.InterIntra.Enabled = true
			return visit
		}()},
		{name: "missing reference frame", ctx: func() FrameWorkBatch {
			next := ctx
			next.References = nil
			return next
		}(), visit: valid},
		{name: "switchable filter not decoded", ctx: func() FrameWorkBatch {
			next := ctx
			next.TileInfo.InterpolationFilter = parser.InterpolationSwitchable
			return next
		}(), visit: valid},
		{name: "outside job", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Block.MICol = 16
			visit.Block.MIColEnd = 20
			return visit
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInter(0, tt.visit); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
	if err := ctx.PredictBlockLumaInterWithFilters(0, valid, motion.InterpFilters{X: motion.InterpFilter(99)}); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad filters err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockInterRejectsInterIntraBeforeMutation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	output.Y.Pix[0] = 0x44
	output.U.Pix[0] = 0x55
	output.V.Pix[0] = 0x66

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.InterIntraValid = true
	visit.Prediction.InterIntra.Enabled = true
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("PredictBlockInterWithFilters err=%v want %v", err, ErrInvalidBatch)
	}
	if output.Y.Pix[0] != 0x44 || output.U.Pix[0] != 0x55 || output.V.Pix[0] != 0x66 {
		t.Fatalf("output mutated y=%#x u=%#x v=%#x", output.Y.Pix[0], output.U.Pix[0], output.V.Pix[0])
	}
}

func TestFrameWorkBatchPredictBlockLumaInterAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Col: 4, Row: 6})
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaInter allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaDispatchesIntraAndInter(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlockLuma(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("intra dispatch sample=%d want 30", got)
	}

	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	if err := ctx.PredictBlockLuma(0, inter, nil); err != nil {
		t.Fatal(err)
	}
	got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16)
	want := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, 17, 16)
	if got != want {
		t.Fatalf("inter dispatch sample=%d want %d", got, want)
	}
}

func TestFrameWorkBatchPredictBlockLumaRejectsInvalidDispatch(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	ctx := testInterPredictionBatch(output, reference)
	if err := ctx.PredictBlockLuma(0, tile.BlockLoopVisit{}, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing prediction err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockLuma(0, testIntraPredictionVisit(tile.IntraModeDC), nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil intra scratch err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockLuma(0, func() tile.BlockLoopVisit {
		visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
		return visit
	}(), nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil compound scratch err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockLumaAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	var scratch FrameWorkPredictionScratch
	intra := testIntraPredictionVisit(tile.IntraModeDC)
	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLuma(0, intra, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlockLuma(0, inter, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLuma allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	valid := testIntraPredictionVisit(tile.IntraModeDC)
	var scratch FrameWorkIntraPredictionScratch

	tests := []struct {
		name    string
		ctx     FrameWorkBatch
		visit   tile.BlockLoopVisit
		scratch *FrameWorkIntraPredictionScratch
	}{
		{name: "nil scratch", ctx: ctx, visit: valid},
		{name: "missing prediction", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Valid = false
			return visit
		}(), scratch: &scratch},
		{name: "inter", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intra = false
			return visit
		}(), scratch: &scratch},
		{name: "outside job", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Block.MICol = 16
			visit.Block.MIColEnd = 20
			return visit
		}(), scratch: &scratch},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaIntra(0, tt.visit, tt.scratch); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 91)

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	directional := testIntraPredictionVisit(tile.IntraModeD135)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlockLumaIntra(0, directional, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaIntra allocated: %f", allocs)
	}
}

func FuzzFrameWorkBatchPredictBlockLumaIntra(f *testing.F) {
	f.Add(uint8(0), uint16(90), uint16(92), uint16(88))
	f.Add(uint8(1), uint16(12), uint16(200), uint16(40))
	f.Add(uint8(6), uint16(511), uint16(700), uint16(512))

	modes := [...]tile.IntraMode{
		tile.IntraModeDC,
		tile.IntraModeVertical,
		tile.IntraModeHorizontal,
		tile.IntraModeSmooth,
		tile.IntraModeSmoothVertical,
		tile.IntraModeSmoothHorizontal,
		tile.IntraModePaeth,
		tile.IntraModeD45,
		tile.IntraModeD67,
		tile.IntraModeD113,
		tile.IntraModeD135,
		tile.IntraModeD157,
		tile.IntraModeD203,
	}
	f.Fuzz(func(t *testing.T, rawMode uint8, above uint16, left uint16, aboveLeft uint16) {
		output := testBatchFrame(t, frame.Format{Width: 96, Height: 96, BitDepth: 10, Align: 128})
		ctx := testIntraPredictionBatch(output)
		above &= 0x3ff
		left &= 0x3ff
		aboveLeft &= 0x3ff
		seedFrameWorkDirectionalEdges(output, 0x3ff, above, left, aboveLeft)

		var scratch FrameWorkIntraPredictionScratch
		visit := testIntraPredictionVisit(modes[int(rawMode)%len(modes)])
		if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
			t.Fatalf("PredictBlockLumaIntra err=%v", err)
		}
		for y := 16; y < 32; y++ {
			for x := 16; x < 32; x++ {
				if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got > 0x3ff {
					t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
				}
			}
		}
	})
}

func testCFLPredictionVisit() tile.BlockLoopVisit {
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.ChromaMode = tile.ChromaIntraModeCFL
	visit.Prediction.ChromaModeValid = true
	visit.Prediction.CFLAlpha = tile.CFLAlphaResult{Index: 0x21, JointSign: 7}
	visit.Prediction.CFLAlphaValid = true
	return visit
}

func seedFrameWorkCFLPredictionFrame(output *frame.Frame) {
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			value := uint16(35 + ((x-16)*7+(y-16)*11)%180)
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y, value)
		}
	}
	for x := 8; x < 16; x++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, x, 7, uint16(50+x))
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, x, 7, uint16(70+x))
	}
	for y := 8; y < 16; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 7, y, uint16(90+y))
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 7, y, uint16(110+y))
	}
	setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 7, 7, 80)
	setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 7, 7, 100)
}

func testPredictFrameWorkCFLWant(t *testing.T, output *frame.Frame, visit tile.BlockLoopVisit, plane FrameWorkPlane) {
	t.Helper()
	var dst frame.Plane
	var predType prediction.CFLPredType
	switch plane {
	case FrameWorkPlaneU:
		dst = output.U
		predType = prediction.CFLPredU
	case FrameWorkPlaneV:
		dst = output.V
		predType = prediction.CFLPredV
	default:
		t.Fatalf("bad plane=%d", plane)
	}
	recon := make([]uint16, prediction.CFLBufSquare)
	ac := make([]int16, prediction.CFLBufSquare)
	offset := 16*output.Y.Stride + 16
	if err := prediction.SubsampleLuma8ToQ3(recon, output.Y.Pix[offset:], output.Y.Stride, 16, 16, true, true); err != nil {
		t.Fatal(err)
	}
	if err := prediction.SubtractCFLAverage(recon, ac, 8, 8); err != nil {
		t.Fatal(err)
	}
	edges := testFrameWorkIntraEdges(output, dst, 8, 8, 8, 8)
	if err := prediction.PredictIntraPlaneBlock(dst, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, prediction.IntraModeDC, edges); err != nil {
		t.Fatal(err)
	}
	alphaQ3, err := prediction.CFLAlphaQ3(visit.Prediction.CFLAlpha.Index, visit.Prediction.CFLAlpha.JointSign, predType)
	if err != nil {
		t.Fatal(err)
	}
	if err := prediction.PredictCFLPlaneBlock(dst, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, ac, alphaQ3); err != nil {
		t.Fatal(err)
	}
}

func testFrameWorkIntraEdges(output *frame.Frame, plane frame.Plane, x int, y int, width int, height int) prediction.IntraEdges {
	above := make([]uint16, width)
	left := make([]uint16, height)
	for col := 0; col < width; col++ {
		above[col] = frameWorkTestSample(plane, output.Layout.BytesPerSample, x+col, y-1)
	}
	for row := 0; row < height; row++ {
		left[row] = frameWorkTestSample(plane, output.Layout.BytesPerSample, x-1, y+row)
	}
	return prediction.IntraEdges{
		Above:              above,
		Left:               left,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeft:          frameWorkTestSample(plane, output.Layout.BytesPerSample, x-1, y-1),
		AboveLeftAvailable: true,
	}
}

func testIntraPredictionBatch(output *frame.Frame) FrameWorkBatch {
	return FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
}

func testInterPredictionBatch(output *frame.Frame, reference *frame.Frame) FrameWorkBatch {
	return FrameWorkBatch{
		Output:     output,
		References: []*frame.Frame{reference},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
			TileInfo:  parser.TileInfo{InterpolationFilter: parser.InterpolationEightTap},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
}

func testCompoundInterPredictionBatch(output *frame.Frame, last *frame.Frame, bwd *frame.Frame) FrameWorkBatch {
	ctx := testInterPredictionBatch(output, last)
	refs := make([]*frame.Frame, int(FrameWorkReferenceBwd)+1)
	refs[int(FrameWorkReferenceLast)] = last
	refs[int(FrameWorkReferenceBwd)] = bwd
	ctx.References = refs
	ctx.Sequence.EnableOrderHint = true
	ctx.Sequence.OrderHintBits = 4
	ctx.FrameHeader.OrderHint = 8
	ctx.ReferenceOrderHints[tile.ReferenceFrameLast] = 4
	ctx.ReferenceOrderHints[tile.ReferenceFrameBWD] = 12
	return ctx
}

func testIntraPredictionVisit(mode tile.IntraMode) tile.BlockLoopVisit {
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: mode,
		},
	}
}

func testInterPredictionVisit(mv motion.Vector) tile.BlockLoopVisit {
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			Intra:                false,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMotion: tile.InterMotionResult{
				References: refs,
				MV:         [2]motion.Vector{mv},
			},
			InterMotionValid: true,
		},
	}
}

func testOBMCInterPredictionVisit(baseMV motion.Vector, aboveMV motion.Vector, leftMV motion.Vector) tile.BlockLoopVisit {
	visit := testInterPredictionVisit(baseMV)
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	visit.Prediction.OverlappableNeighbors.AboveCount = 1
	visit.Prediction.OverlappableNeighbors.Above[0] = tile.OverlappableNeighbor{
		RelX4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{aboveMV},
		},
		InterpFilters:      motion.RegularFilters,
		InterpFiltersValid: true,
	}
	visit.Prediction.OverlappableNeighbors.LeftCount = 1
	visit.Prediction.OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{leftMV},
		},
		InterpFilters:      motion.RegularFilters,
		InterpFiltersValid: true,
	}
	return visit
}

func testCompoundInterPredictionVisit(mv0 motion.Vector, mv1 motion.Vector, compoundType tile.CompoundType) tile.BlockLoopVisit {
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameBWD}, Compound: true}
	compoundIndex := uint8(0)
	if compoundType == tile.CompoundTypeAverage {
		compoundIndex = 1
	}
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			Intra:                false,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMode: tile.InterModeResult{
				Compound:     true,
				CompoundMode: tile.CompoundInterModeNearestNearest,
			},
			InterModeValid: true,
			InterMotion: tile.InterMotionResult{
				References: refs,
				Mode: tile.InterModeResult{
					Compound:     true,
					CompoundMode: tile.CompoundInterModeNearestNearest,
				},
				MV: [2]motion.Vector{mv0, mv1},
			},
			InterMotionValid: true,
			MotionMode:       tile.MotionModeTranslation,
			MotionModeValid:  true,
			CompoundBlend: tile.CompoundBlendResult{
				Type:          compoundType,
				CompoundIndex: compoundIndex,
			},
			CompoundBlendValid: true,
		},
	}
}

func testIntraPrediction4x4Visit(mode tile.IntraMode) tile.BlockLoopVisit {
	visit := testIntraPredictionVisit(mode)
	visit.Block.MIColEnd = 5
	visit.Block.MIRowEnd = 5
	visit.Block.Size = tile.BlockSize4x4
	visit.Block.VisibleW4 = 1
	visit.Block.VisibleH4 = 1
	return visit
}

func frameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	sample, ok := frameWorkLoadSample(plane, bytesPerSample, x, y)
	if !ok {
		panic("bad test sample")
	}
	return sample
}

func frameWorkTestSampleClamped(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	if x < 0 {
		x = 0
	} else if x >= plane.Width {
		x = plane.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= plane.Height {
		y = plane.Height - 1
	}
	return frameWorkTestSample(plane, bytesPerSample, x, y)
}

func assertFrameWorkPlaneBlockEqual(t *testing.T, got frame.Plane, want frame.Plane, bytesPerSample int, x int, y int, width int, height int) {
	t.Helper()
	assertFrameWorkPlaneBlockEqualAt(t, got, x, y, want, x, y, bytesPerSample, width, height)
}

func assertFrameWorkPlaneBlockEqualAt(t *testing.T, got frame.Plane, gotX int, gotY int, want frame.Plane, wantX int, wantY int, bytesPerSample int, width int, height int) {
	t.Helper()
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			g := frameWorkTestSample(got, bytesPerSample, gotX+col, gotY+row)
			w := frameWorkTestSample(want, bytesPerSample, wantX+col, wantY+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", gotX+col, gotY+row, g, w)
			}
		}
	}
}

func assertFrameWorkCompoundBlendEqual(t *testing.T, got frame.Plane, first frame.Plane, second frame.Plane, bytesPerSample int, x int, y int, width int, height int, fwdOffset int, bckOffset int) {
	t.Helper()
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			a := frameWorkTestSample(first, bytesPerSample, col, row)
			b := frameWorkTestSample(second, bytesPerSample, col, row)
			want := uint16((uint32(a)*uint32(fwdOffset) + uint32(b)*uint32(bckOffset) + 8) >> 4)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, want)
			}
		}
	}
}

func assertFrameWorkMaskedCompoundEqual(t *testing.T, got frame.Plane, first frame.Plane, second frame.Plane, mask []byte, maskStride int, subX bool, subY bool, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int) {
	t.Helper()
	max := uint16((1 << bitDepth) - 1)
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			a := frameWorkTestSample(first, bytesPerSample, col, row)
			b := frameWorkTestSample(second, bytesPerSample, col, row)
			if a > max || b > max {
				t.Fatalf("input sample(%d,%d)=%d/%d exceeds max %d", col, row, a, b, max)
			}
			m := testFrameWorkBlendMaskSample(t, mask, maskStride, row, col, subX, subY)
			want := uint16((uint32(m)*uint32(a) + uint32(64-m)*uint32(b) + 32) >> 6)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != want {
				t.Fatalf("sample(%d,%d)=%d want %d mask=%d", x+col, y+row, g, want, m)
			}
		}
	}
}

func setFrameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func testFrameWorkMotionPredictionPlane(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, filters motion.InterpFilters) frame.Plane {
	t.Helper()
	stride := width * bytesPerSample
	dst := frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
	refX, refY, subX, subY, err := motion.ReferenceOrigin(dstX, dstY, mv)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, reference, bytesPerSample, bitDepth, 0, 0, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return dst
}

func testFrameWorkMotionPredictionPlaneSubsampled(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) frame.Plane {
	t.Helper()
	stride := width * bytesPerSample
	dst := frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, reference, bytesPerSample, bitDepth, 0, 0, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return dst
}

func testFrameWorkDiffWtdMask(t *testing.T, first frame.Plane, second frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int, maskType tile.DiffWtdMaskType) []byte {
	t.Helper()
	mask := make([]byte, width*height)
	shift := uint8(0)
	if bitDepth > 8 {
		shift = bitDepth - 8
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			a := frameWorkTestSample(first, bytesPerSample, x, y)
			b := frameWorkTestSample(second, bytesPerSample, x, y)
			var diff uint16
			if a > b {
				diff = a - b
			} else {
				diff = b - a
			}
			diff >>= shift
			m := 38 + int(diff)/16
			if m > 64 {
				m = 64
			}
			if maskType == tile.DiffWtdMaskType38Inv {
				m = 64 - m
			}
			mask[y*width+x] = byte(m)
		}
	}
	return mask
}

func testFrameWorkBlendMaskSample(t *testing.T, mask []byte, stride int, row int, col int, subX bool, subY bool) uint8 {
	t.Helper()
	var m int
	switch {
	case !subX && !subY:
		m = int(mask[row*stride+col])
	case subX && subY:
		m = (int(mask[(2*row)*stride+2*col]) +
			int(mask[(2*row+1)*stride+2*col]) +
			int(mask[(2*row)*stride+2*col+1]) +
			int(mask[(2*row+1)*stride+2*col+1]) + 2) >> 2
	case subX:
		m = (int(mask[row*stride+2*col]) + int(mask[row*stride+2*col+1]) + 1) >> 1
	default:
		m = (int(mask[(2*row)*stride+col]) + int(mask[(2*row+1)*stride+col]) + 1) >> 1
	}
	if m < 0 || m > 64 {
		t.Fatalf("mask sample=%d outside A64 range", m)
	}
	return uint8(m)
}

func fillFrameWorkInterReference(reference *frame.Frame, max uint16) {
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((x*x+3*y*y+17*x+11*y)&int(max)))
		}
	}
}

func fillFrameWorkInterReferenceAllPlanes(reference *frame.Frame, max uint16) {
	fillFrameWorkInterReference(reference, max)
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, uint16((5*x*x+7*y*y+13*x+19*y+23)&int(max)))
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, uint16((11*x*x+3*y*y+29*x+31*y+37)&int(max)))
		}
	}
}

func fillFrameWorkInterReferenceVariant(reference *frame.Frame, max uint16, seed uint16) {
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((29*x+31*y+int(seed))&int(max)))
		}
	}
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, uint16((17*x+23*y+int(seed)*3)&int(max)))
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, uint16((41*x+43*y+int(seed)*5)&int(max)))
		}
	}
}

func seedFrameWorkDirectionalEdges(output *frame.Frame, max uint16, above uint16, left uint16, aboveLeft uint16) {
	for x := 0; x < output.Y.Width; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16((int(above)+x)&int(max)))
	}
	for y := 0; y < output.Y.Height; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16((int(left)+y)&int(max)))
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, aboveLeft&max)
}
