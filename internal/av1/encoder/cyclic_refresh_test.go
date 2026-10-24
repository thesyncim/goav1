package encoder

import (
	"bytes"
	"testing"
)

func testCRRateControl(bestQ, worstQ uint8) *svtRateControl {
	var rc svtRateControl
	rc.init(1_000_000, 30, bestQ, worstQ, 1280, 720, rcBuffers{})
	return &rc
}

func TestCyclicComputeDeltaQ(t *testing.T) {
	rc := testCRRateControl(1, 255)
	// A higher rate target lowers q; the floor is -60% of q.
	for _, q := range []int{40, 100, 180, 250} {
		d := cyclicComputeDeltaQ(rc, q, 1.75, false)
		if d >= 0 {
			t.Fatalf("q=%d rate ratio 1.75 delta %d, want negative", q, d)
		}
		if d < -crMaxQDeltaPerc*q/100 {
			t.Fatalf("q=%d delta %d below the -60%% floor", q, d)
		}
	}
	if d := cyclicComputeDeltaQ(rc, 120, 1.0, false); d != 0 {
		t.Fatalf("unit rate ratio delta %d, want 0", d)
	}
}

func TestCyclicRefreshWindowAndGates(t *testing.T) {
	var cr cyclicRefresh
	rc := testCRRateControl(20, 200)
	rc.avgFrameQIndex[svtFrameInter] = 150
	percent := 20 + rc.percentRefreshAdjustment
	const w, h = 1280, 720 // 20x12 superblocks
	f := svtCRFrame{flat: true, hierLevels: 2, avgFrameLowMotion: 80}
	grid, stats := cr.plan(rc, f, 150, w, h, nil)
	if grid == nil || !stats.apply {
		t.Fatal("cyclic refresh did not apply")
	}
	boosted := 0
	for i, q := range grid {
		switch {
		case q < 150:
			boosted++
			if i >= 240*percent/100 {
				t.Fatalf("superblock %d boosted outside the first window", i)
			}
		case q != 150:
			t.Fatalf("superblock %d q=%d, want base 150", i, q)
		}
	}
	if want := 240 * percent / 100; boosted != want || stats.seg1 != want {
		t.Fatalf("boosted %d superblocks (seg1 %d), want %d", boosted, stats.seg1, want)
	}
	if cr.sbEnd != 240*percent/100 {
		t.Fatalf("window end %d", cr.sbEnd)
	}
	// avg_frame_low_motion under 50 turns it off.
	f.avgFrameLowMotion = 30
	if g, _ := cr.plan(rc, f, 150, w, h, nil); g != nil {
		t.Fatal("cyclic refresh applied on high-motion content")
	}
	// Non-base layers of a layered structure do not refresh.
	f = svtCRFrame{hierLevels: 2, temporalLayer: 1, avgFrameLowMotion: 80}
	if g, _ := cr.plan(rc, f, 150, w, h, nil); g != nil {
		t.Fatal("cyclic refresh applied on a non-base layer")
	}
}

// encodeCyclicOracle encodes a single-layer CBR sequence (the configuration
// cyclic refresh runs in) with the fused walk, the serial split walk or the
// split walk on wavefront lanes.
func encodeCyclicOracle(t *testing.T, frames []SourceFrame420, w, h int, split bool, lanes int) ([][]byte, int) {
	t.Helper()
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 300_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	enc.forceSplit = split
	enc.wavefrontLanes = lanes
	enc.SetTileColumns(1)
	out := make([][]byte, 0, len(frames))
	deltaFrames := 0
	for i, f := range frames {
		data, _, err := enc.Encode(f, i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if enc.pc.st.sbQIndex != nil {
			deltaFrames++
		}
		out = append(out, append([]byte(nil), data...))
	}
	return out, deltaFrames
}

func TestCyclicRefreshSplitMatchesFused(t *testing.T) {
	const w, h = 320, 192
	frames := syntheticMotionFrames(10, w, h)
	fused, deltaFrames := encodeCyclicOracle(t, frames, w, h, false, 0)
	if deltaFrames == 0 {
		t.Fatal("no frame signalled delta_q")
	}
	for _, cfg := range []struct {
		name  string
		lanes int
	}{{"split", 0}, {"wavefront2", 2}, {"wavefront3", 3}} {
		t.Run(cfg.name, func(t *testing.T) {
			got, _ := encodeCyclicOracle(t, frames, w, h, true, cfg.lanes)
			for i := range fused {
				if !bytes.Equal(fused[i], got[i]) {
					t.Fatalf("frame %d differs from fused (%d vs %d bytes)", i, len(fused[i]), len(got[i]))
				}
			}
		})
	}
}
