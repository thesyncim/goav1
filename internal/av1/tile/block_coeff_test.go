package tile

import (
	"testing"
)

func TestBlockQIndexUsesSegmentAdjustedQIndex(t *testing.T) {
	for _, tt := range []struct {
		name          string
		current       uint8
		segmentDeltaQ int16
		deltasZero    bool
		wantQIndex    uint8
		wantLossless  bool
	}{
		{name: "segment_delta_drives_qindex_to_zero", current: 64, segmentDeltaQ: -64, deltasZero: true, wantQIndex: 0, wantLossless: true},
		{name: "zero_qindex_with_quant_deltas_is_not_lossless", current: 64, segmentDeltaQ: -64, wantQIndex: 0},
		{name: "segment_delta_lifts_qindex_above_zero", current: 0, segmentDeltaQ: 32, deltasZero: true, wantQIndex: 32},
		{name: "segment_delta_keeps_qindex", current: 64, deltasZero: true, wantQIndex: 64},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var req BlockLoopRequest
			req.Segmentation.Enabled = true
			req.Segmentation.Data.Segments[1].DeltaQ = tt.segmentDeltaQ
			req.QuantDeltasZero = tt.deltasZero
			q, lossless, ok := blockQIndex(&req, tt.current, 1)
			if !ok || q != tt.wantQIndex || lossless != tt.wantLossless {
				t.Fatalf("qindex=%d lossless=%v ok=%v want %d %v", q, lossless, ok, tt.wantQIndex, tt.wantLossless)
			}
		})
	}
	var req BlockLoopRequest
	if _, _, ok := blockQIndex(&req, 10, 1); ok {
		t.Fatal("nonzero segment without segmentation accepted")
	}
}

func TestFilterIntraTransformModeMatchesDav1d(t *testing.T) {
	for mode, want := range map[FilterIntraMode]IntraMode{
		FilterIntraModeDC:         IntraModeDC,
		FilterIntraModeVertical:   IntraModeVertical,
		FilterIntraModeHorizontal: IntraModeHorizontal,
		FilterIntraModeD157:       IntraModeD157,
		FilterIntraModePaeth:      IntraModeDC,
	} {
		if got := filterIntraTransformMode(mode); got != want {
			t.Fatalf("filter intra mode %d: transform mode=%d want %d", mode, got, want)
		}
	}
}
