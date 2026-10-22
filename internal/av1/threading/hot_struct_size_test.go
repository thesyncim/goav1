package threading

import (
	"testing"
	"unsafe"
)

const hotStructKiB = 1024

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Batch", size: unsafe.Sizeof(Batch{}), max: 20},
		// The wavefront lane channel (8 bytes) lets tiles reuse the pool's
		// persistent lane goroutines instead of spawning them per frame.
		// DeferredTemporalReferences (8 bytes) hands the tile loops the
		// references they project per superblock row (dav1d load_tmvs).
		{name: "FrameWorkBatch", size: unsafe.Sizeof(FrameWorkBatch{}), max: 1624},
		{name: "FrameWorkFrameContext", size: unsafe.Sizeof(FrameWorkFrameContext{}), max: 1152},
		{name: "FrameWorkPlaneRegion", size: unsafe.Sizeof(FrameWorkPlaneRegion{}), max: 72},
		{name: "FrameWorkTileResidualCDFStorage", size: unsafe.Sizeof(FrameWorkTileResidualCDFStorage{}), max: 56 * hotStructKiB},
		// The fused coefficient path adds one compact job-owned dequant target
		// to the controller; its independent 80-byte layout is guarded in tile.
		// It embeds FrameWorkBatch copies, so the 8-byte wavefront lane channel
		// appears here too.
		// The tile refmvs state carries dav1d's mfmv setup and rp_proj window.
		{name: "FrameWorkTileResidualScratch", size: unsafe.Sizeof(FrameWorkTileResidualScratch{}), max: 91264},
		{name: "FrameWorkJobRegion", size: unsafe.Sizeof(FrameWorkJobRegion{}), max: 36},
		{name: "FrameWorkCDEFIndexMap", size: unsafe.Sizeof(FrameWorkCDEFIndexMap{}), max: 56},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(FrameWorkLoopFilterBlockRecord{}), max: 34},
		{name: "FrameWorkLoopFilterMapStats", size: unsafe.Sizeof(FrameWorkLoopFilterMapStats{}), max: 12},
		{name: "FrameWorkTileResidualStats", size: unsafe.Sizeof(FrameWorkTileResidualStats{}), max: 136},
		// poolTask carries one RangeRunner interface plus the low/high range for
		// row-banded postfilter work. The band ordinal reuses Batch.Worker, so the
		// typed runner adds no space beyond the former function-plus-range fields.
		// It also embeds a FrameWorkBatch, hence the wavefront lane channel and
		// the deferred temporal-reference pointer.
		{name: "poolTask", size: unsafe.Sizeof(poolTask{}), max: 1736},
		{name: "frameWorkReconEvent", size: unsafe.Sizeof(frameWorkReconEvent{}), max: 8},
		{name: "frameWorkReconPaletteBinding", size: unsafe.Sizeof(frameWorkReconPaletteBinding{}), max: 12},
		{name: "frameWorkReconSB", size: unsafe.Sizeof(frameWorkReconSB{}), max: 12},
		// The wavefront is allocated once per scratch behind a pointer; it carries
		// the parse-side span cursor and the published row snapshot the lanes
		// read while the tile is still parsing.
		{name: "frameWorkReconWavefront", size: unsafe.Sizeof(frameWorkReconWavefront{}), max: 472},
		{name: "frameWorkBlockCoeffGeometry", size: unsafe.Sizeof(frameWorkBlockCoeffGeometry{}), max: 88},
		// The geometry cache also binds each job's inter-prediction output planes
		// once (dav1d binds f->cur per frame); it lives in per-worker scratch.
		{name: "frameWorkJobGeometryCache", size: unsafe.Sizeof(frameWorkJobGeometryCache{}), max: 664},
		{name: "frameWorkPredictionPlaneGeometry", size: unsafe.Sizeof(frameWorkPredictionPlaneGeometry{}), max: 152},
		{name: "frameWorkWedgeCode", size: unsafe.Sizeof(frameWorkWedgeCode{}), max: 3},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
