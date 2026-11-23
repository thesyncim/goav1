package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/quantize"
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
		dq := c.blockDequant(segmentID, false)
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

func TestFrameWorkBlockDequantRejectsDequantizingPaths(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	if dq := c.blockDequant(0, true); dq != nil {
		t.Fatalf("lossless block received dequant factors %+v", *dq)
	}
	c.batch.Quantization.UsingQMatrix = true
	if dq := c.blockDequant(0, false); dq != nil {
		t.Fatalf("quantizer-matrix block received dequant factors %+v", *dq)
	}
}
