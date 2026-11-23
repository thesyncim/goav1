package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestEOBMultiContextForClassMatchesLibaom(t *testing.T) {
	tests := []struct {
		name  string
		class transform.Class
		want  uint8
	}{
		{name: "2d", class: transform.Class2D, want: 0},
		{name: "horizontal", class: transform.ClassHoriz, want: 1},
		{name: "vertical", class: transform.ClassVert, want: 1},
	}
	for _, tt := range tests {
		if got := eobMultiContextForClass(tt.class); got != tt.want {
			t.Fatalf("%s context=%d want %d", tt.name, got, tt.want)
		}
	}
}

func TestLumaCoeffScratchMatchesAdjustedLibaomScanBounds(t *testing.T) {
	var scratch LumaCoeffTreeScratch
	coeffs, scan, levels, err := scratch.coeffBuffers(TransformSize64x64, transform.Class2D)
	if err != nil {
		t.Fatal(err)
	}
	if len(coeffs) != 1024 || len(scan) != 1024 {
		t.Fatalf("scan buffers coeffs=%d scan=%d want adjusted 32x32", len(coeffs), len(scan))
	}
	if len(levels) != (32+txPadHorizontal)*(32+txPadHorizontal) {
		t.Fatalf("levels len=%d want adjusted padded 32x32", len(levels))
	}
}

func BenchmarkLumaCoeffTreeScratchClearCoeffDirty(b *testing.B) {
	var scratch LumaCoeffTreeScratch
	for i := range scratch.Coeffs {
		scratch.Coeffs[i] = int16(i + 1)
	}
	const dirtyLen = 128
	for i := 0; i < dirtyLen; i++ {
		scratch.InverseScan[i] = int16((i * 17) & (maxCoeffScanLen - 1))
	}
	b.ReportAllocs()
	for b.Loop() {
		scratch.coeffDirtyLen = dirtyLen
		scratch.clearCoeffDirty()
	}
}

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
