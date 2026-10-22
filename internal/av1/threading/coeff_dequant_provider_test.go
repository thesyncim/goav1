package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkCoeffDequantProviderIsLazyAndKeyed(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization.UDCDelta = 3
	batch.Quantization.UACDelta = -5
	batch.Quantization.VDCDelta = -7
	batch.Quantization.VACDelta = 11
	req.Transforms = nil
	req.UseDefaultTransforms = true
	reconState := frameWorkReconState{batch: batch, int32Scratch: req.Int32Scratch}
	c := frameWorkTileResidualLoopController{
		batch: batch,
		state: state,
		req:   req,
		recon: reconState,
	}
	c.reconHot = &c.recon
	visit := tile.BlockLoopVisit{SegmentID: 0}
	target := c.bindCoeffDequantTarget(&visit, false)
	if target == nil || target.Provider != &c || len(target.Dst) == 0 || &target.Dst[0] != &reconState.int32Scratch[0] {
		t.Fatalf("target=%+v was not bound for the eligible default path", target)
	}
	if target.PreparedPlanes != 0 {
		t.Fatalf("binding eagerly prepared plane factors: mask=%03b", target.PreparedPlanes)
	}

	if !c.PrepareCoeffDequantPlane(0) {
		t.Fatal("prepare luma factors failed")
	}
	wantY, lossless, err := batch.BlockQuantizer(state.CurrentBaseQIdx, visit.SegmentID, FrameWorkPlaneY)
	if err != nil || lossless {
		t.Fatalf("oracle quantizer q=%+v lossless=%v err=%v", wantY, lossless, err)
	}
	if got := target.Factors[0]; got != [2]int32{wantY.DC, wantY.AC} {
		t.Fatalf("luma factors=%v want DC/AC %d/%d", got, wantY.DC, wantY.AC)
	}
	if target.PreparedPlanes != 1 {
		t.Fatalf("prepared planes=%03b want only Y", target.PreparedPlanes)
	}
	if !c.PrepareCoeffDequantPlane(0) || target.PreparedPlanes != 1 {
		t.Fatalf("repeated Y prepare changed state: mask=%03b", target.PreparedPlanes)
	}
	for _, plane := range []uint8{uint8(quantize.PlaneU), uint8(quantize.PlaneV)} {
		if !c.PrepareCoeffDequantPlane(plane) {
			t.Fatalf("prepare plane %d factors failed", plane)
		}
		qPlane := FrameWorkPlaneU
		if plane == uint8(quantize.PlaneV) {
			qPlane = FrameWorkPlaneV
		}
		want, lossless, err := batch.BlockQuantizer(state.CurrentBaseQIdx, visit.SegmentID, qPlane)
		if err != nil || lossless {
			t.Fatalf("plane %d oracle quantizer q=%+v lossless=%v err=%v", plane, want, lossless, err)
		}
		if got := target.Factors[plane]; got != [2]int32{want.DC, want.AC} {
			t.Fatalf("plane %d factors=%v want DC/AC %d/%d", plane, got, want.DC, want.AC)
		}
	}
	if target.PreparedPlanes != 0b111 {
		t.Fatalf("prepared planes=%03b want Y/U/V after explicit lazy requests", target.PreparedPlanes)
	}

	state.CurrentBaseQIdx++
	visit.SegmentID = 0
	target = c.bindCoeffDequantTarget(&visit, false)
	if target == nil || target.PreparedPlanes != 0 || target.Factors != ([3][2]int32{}) {
		t.Fatalf("qindex change retained stale factors: target=%+v", target)
	}
	if !c.PrepareCoeffDequantPlane(0) {
		t.Fatal("prepare factors after qindex change failed")
	}

	c.batch.Segmentation.Enabled = true
	c.batch.Segmentation.Data.Segments[1].DeltaQ = -11
	visit.SegmentID = 1
	target = c.bindCoeffDequantTarget(&visit, false)
	if target == nil || target.PreparedPlanes != 0 {
		t.Fatalf("segment change retained stale plane mask: target=%+v", target)
	}
	if !c.PrepareCoeffDequantPlane(0) {
		t.Fatal("prepare factors after segment change failed")
	}
	wantSegmentY, lossless, err := c.batch.BlockQuantizer(state.CurrentBaseQIdx, visit.SegmentID, FrameWorkPlaneY)
	if err != nil || lossless {
		t.Fatalf("segment oracle quantizer q=%+v lossless=%v err=%v", wantSegmentY, lossless, err)
	}
	if got := target.Factors[0]; got != [2]int32{wantSegmentY.DC, wantSegmentY.AC} {
		t.Fatalf("segment factors=%v want DC/AC %d/%d", got, wantSegmentY.DC, wantSegmentY.AC)
	}
	if target.PreparedPlanes != 1 {
		t.Fatalf("segment prepared planes=%03b want only Y", target.PreparedPlanes)
	}
	if c.PrepareCoeffDequantPlane(3) {
		t.Fatal("invalid plane index was accepted")
	}

	// A fresh controller is created per tile job. The raw q/segment key can
	// repeat while frame quantization deltas change, so no prepared plane bits
	// or factors may carry across that full reset.
	batch2 := c.batch
	batch2.Quantization.YDCDelta += 23
	recon2 := frameWorkReconState{batch: batch2, int32Scratch: req.Int32Scratch}
	c2 := frameWorkTileResidualLoopController{
		batch: batch2,
		state: state,
		req:   req,
		recon: recon2,
	}
	c2.reconHot = &c2.recon
	visit.SegmentID = 1
	target2 := c2.bindCoeffDequantTarget(&visit, false)
	if target2 == nil || target2.PreparedPlanes != 0 {
		t.Fatalf("new-job target inherited factors: target=%+v", target2)
	}
	if !c2.PrepareCoeffDequantPlane(uint8(quantize.PlaneY)) {
		t.Fatal("new-job prepare failed")
	}
	wantFrameY, lossless, err := batch2.BlockQuantizer(state.CurrentBaseQIdx, visit.SegmentID, FrameWorkPlaneY)
	if err != nil || lossless {
		t.Fatalf("new-job oracle quantizer q=%+v lossless=%v err=%v", wantFrameY, lossless, err)
	}
	if got := target2.Factors[0]; got != [2]int32{wantFrameY.DC, wantFrameY.AC} {
		t.Fatalf("new-job factors=%v want DC/AC %d/%d", got, wantFrameY.DC, wantFrameY.AC)
	}

	c2.dequantTarget.Ready = true
	c2.dequantTarget.ActiveRows = 7
	if _, err := c2.SelectBlockCoeffRequestPtr(nil); err == nil {
		t.Fatal("nil visit unexpectedly succeeded")
	}
	if c2.dequantTarget.Ready || c2.dequantTarget.ActiveRows != 0 {
		t.Fatalf("early Select error retained ready target: %+v", c2.dequantTarget)
	}
}

func TestFrameWorkCoeffDequantTargetRejectsFallbackOwnershipPaths(t *testing.T) {
	for _, name := range []string{"visitor", "custom-predictor", "custom-transforms", "qmatrix", "lossless", "skip-transform"} {
		t.Run(name, func(t *testing.T) {
			batch, state, _, _, req := testFrameWorkResidualDriver(t)
			req.Transforms = nil
			req.UseDefaultTransforms = true
			reconState := frameWorkReconState{batch: batch, int32Scratch: req.Int32Scratch}
			c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req, recon: reconState}
			c.reconHot = &c.recon
			visit := tile.BlockLoopVisit{SegmentID: 0}
			lossless := false
			switch name {
			case "visitor":
				c.userCoeffVisitor = func(tile.BlockLoopVisit, tile.BlockCoeffBlock) error { return nil }
			case "custom-predictor":
				c.req.Predict = func(tile.BlockLoopVisit) error { return nil }
			case "custom-transforms":
				c.req.Transforms = func(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
					return FrameWorkBlockTransforms{}, nil
				}
			case "qmatrix":
				c.batch.Quantization.UsingQMatrix = true
			case "lossless":
				lossless = true
			case "skip-transform":
				visit.Prefix.SkipTransform = true
			}
			if target := c.bindCoeffDequantTarget(&visit, lossless); target != nil {
				t.Fatalf("unsupported ownership route received target: %+v", target)
			}
		})
	}
}

// TestFrameWorkCoeffDequantTargetDeferredUsesArena checks that deferred
// reconstruction binds the frame-thread cf arena rather than int32Scratch.
func TestFrameWorkCoeffDequantTargetDeferredUsesArena(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	req.Transforms = nil
	req.UseDefaultTransforms = true
	var scratch FrameWorkTileResidualScratch
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req, scratch: &scratch,
		recon: frameWorkReconState{batch: batch, int32Scratch: req.Int32Scratch}, deferReconstruction: true}
	c.reconHot = &c.recon
	target := c.bindCoeffDequantTarget(&tile.BlockLoopVisit{}, false)
	if target == nil || target.Arena != &scratch.cfArena || target.Dst != nil {
		t.Fatalf("deferred target = %+v, want the scratch cf arena", target)
	}
}
