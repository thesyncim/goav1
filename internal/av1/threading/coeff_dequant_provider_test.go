package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkBlockDequantIsKeyed(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization.UDCDelta = 3
	batch.Quantization.UACDelta = -5
	batch.Quantization.VDCDelta = -7
	batch.Quantization.VACDelta = 11
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	check := func(segmentID uint8) {
		t.Helper()
		dq := c.blockDequant(segmentID)
		if dq == nil {
			t.Fatalf("segment %d: no dequant factors", segmentID)
		}
		_, dqMax := quantize.DequantCoefficientBoundsTrusted(c.batch.Sequence.ColorConfig.BitDepth)
		for plane, p := range []FrameWorkPlane{FrameWorkPlaneY, FrameWorkPlaneU, FrameWorkPlaneV} {
			want, lossless, err := c.batch.BlockQuantizer(state.CurrentBaseQIdx, segmentID, p)
			if err != nil || lossless {
				t.Fatalf("plane %d oracle quantizer q=%+v lossless=%v err=%v", plane, want, lossless, err)
			}
			if got := dq[plane]; got.DC != uint32(want.DC) || got.AC != uint32(want.AC) || got.Max != uint32(dqMax) {
				t.Fatalf("segment %d plane %d factors=%+v want DC/AC %d/%d max %d", segmentID, plane, got, want.DC, want.AC, dqMax)
			}
		}
	}
	check(0)
	state.CurrentBaseQIdx++
	check(0)
	c.batch.Segmentation.Enabled = true
	c.batch.Segmentation.Data.Segments[1].DeltaQ = -11
	check(1)
}

func TestFrameWorkBlockDequantLossless(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization = parser.QuantizationParams{UsingQMatrix: true, QMatrixLevelY: 3, QMatrixLevelU: 3, QMatrixLevelV: 3}
	state.CurrentBaseQIdx = 0
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	dq := c.blockDequant(0)
	if dq == nil {
		t.Fatal("lossless block: no dequant factors")
	}
	for plane, p := range dq {
		if p.DC != 4 || p.AC != 4 || p.QM != nil {
			t.Fatalf("lossless plane %d factors %+v, want DC/AC 4/4 without matrices", plane, p)
		}
	}
}

func TestFrameWorkBlockDequantSelectsQMatrices(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization.UsingQMatrix = true
	batch.Quantization.QMatrixLevelY = 3
	batch.Quantization.QMatrixLevelU = quantize.QMLevels - 1
	batch.Quantization.QMatrixLevelV = 7
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	dq := c.blockDequant(0)
	if dq == nil {
		t.Fatal("quantizer-matrix block: no dequant factors")
	}
	want := [3]*tile.CoefQMatrices{tile.QMatrices(3, false), nil, tile.QMatrices(7, true)}
	for plane := range dq {
		if dq[plane].QM != want[plane] {
			t.Fatalf("plane %d matrices %p, want %p", plane, dq[plane].QM, want[plane])
		}
	}
	if want[0] == nil || want[2] == nil || want[0] == want[2] {
		t.Fatalf("matrix selection %p %p", want[0], want[2])
	}
}
