package encoder

import "testing"

func TestRateControlTemporalLayerPerFrameBitsMatchesLibaomRTC(t *testing.T) {
	const targetBPS = 1_000_000
	const fps = 30
	twoLayer := [...]int{40_000, 26_667}
	for temporalID, want := range twoLayer {
		got := rateControlTemporalLayerPerFrameBits(targetBPS, fps, 2, uint8(temporalID), 0)
		if got != want {
			t.Fatalf("L1T2 temporal layer %d per-frame bits=%d want %d", temporalID, got, want)
		}
	}
	threeLayer := [...]int{66_667, 26_667, 20_000}
	for temporalID, want := range threeLayer {
		got := rateControlTemporalLayerPerFrameBits(targetBPS, fps, 3, uint8(temporalID), 0)
		if got != want {
			t.Fatalf("L1T3 temporal layer %d per-frame bits=%d want %d", temporalID, got, want)
		}
	}
}

type rcControllerSnapshot struct {
	qIndex              uint8
	buffer              int
	recent              [2]int
	temporalQ           [WebRTCMaxTemporalLayers]uint8
	temporalBuffer      [WebRTCMaxTemporalLayers]int
	temporalRecentBits  [WebRTCMaxTemporalLayers][2]int
	temporalPerFrameBit [WebRTCMaxTemporalLayers]int
	perFrameBits        int
}

func TestUpdateRateControlConfigPreservesControllerState(t *testing.T) {
	oldRC := RateControlConfig{
		TargetBitsPerSecond: 1_200_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}
	nextRC := RateControlConfig{
		TargetBitsPerSecond: 900_000,
		FramesPerSecond:     60,
		MinQIndex:           30,
		MaxQIndex:           180,
	}
	for _, tc := range []struct {
		name string
		run  func(*testing.T)
	}{
		{
			name: "i400",
			run: func(t *testing.T) {
				enc, err := NewMonochromeVideoEncoderCBR(64, 64, oldRC)
				if err != nil {
					t.Fatalf("NewMonochromeVideoEncoderCBR: %v", err)
				}
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatalf("SetTemporalLayers: %v", err)
				}
				seedVideoRCState(&enc.qIndex, &enc.rcBuffer, &enc.rcRecentBits, &enc.rcTemporalQ, &enc.rcTemporalBuffer, &enc.rcTemporalRecentBits)
				before := snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits)
				if err := enc.UpdateRateControlConfig(nextRC); err != nil {
					t.Fatalf("UpdateRateControlConfig: %v", err)
				}
				assertPreservedRCUpdate(t, before, snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits), nextRC, enc.temporalLayers)
			},
		},
		{
			name: "i400-10",
			run: func(t *testing.T) {
				enc, err := NewHighBitDepthMonochromeVideoEncoderCBR(64, 64, 10, oldRC)
				if err != nil {
					t.Fatalf("NewHighBitDepthMonochromeVideoEncoderCBR: %v", err)
				}
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatalf("SetTemporalLayers: %v", err)
				}
				seedVideoRCState(&enc.qIndex, &enc.rcBuffer, &enc.rcRecentBits, &enc.rcTemporalQ, &enc.rcTemporalBuffer, &enc.rcTemporalRecentBits)
				before := snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits)
				if err := enc.UpdateRateControlConfig(nextRC); err != nil {
					t.Fatalf("UpdateRateControlConfig: %v", err)
				}
				assertPreservedRCUpdate(t, before, snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits), nextRC, enc.temporalLayers)
			},
		},
		{
			name: "i420-10",
			run: func(t *testing.T) {
				enc, err := NewHighBitDepth420VideoEncoderCBR(64, 64, 10, oldRC)
				if err != nil {
					t.Fatalf("NewHighBitDepth420VideoEncoderCBR: %v", err)
				}
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatalf("SetTemporalLayers: %v", err)
				}
				seedVideoRCState(&enc.qIndex, &enc.rcBuffer, &enc.rcRecentBits, &enc.rcTemporalQ, &enc.rcTemporalBuffer, &enc.rcTemporalRecentBits)
				before := snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits)
				if err := enc.UpdateRateControlConfig(nextRC); err != nil {
					t.Fatalf("UpdateRateControlConfig: %v", err)
				}
				assertPreservedRCUpdate(t, before, snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits), nextRC, enc.temporalLayers)
			},
		},
	} {
		t.Run(tc.name, tc.run)
	}
}

func TestRateControlTemporalLayerQClampMatchesAdjustQCBR(t *testing.T) {
	qs := [WebRTCMaxTemporalLayers]uint8{80, 70, 60}
	for _, temporalID := range []uint8{1, 2} {
		got := rateControlTemporalLayerQIndex(qs, 20, 200, 3, temporalID)
		if got != 76 {
			t.Fatalf("temporal layer %d q=%d want TL0-4 clamp 76", temporalID, got)
		}
	}
	qs[0] = 22
	qs[1] = 1
	if got := rateControlTemporalLayerQIndex(qs, 20, 200, 2, 1); got != 20 {
		t.Fatalf("min-clamped temporal q=%d want 20", got)
	}
}

func seedVideoRCState(q *uint8, buffer *int, recent *[2]int, temporalQ *[WebRTCMaxTemporalLayers]uint8, temporalBuffer *[WebRTCMaxTemporalLayers]int, temporalRecent *[WebRTCMaxTemporalLayers][2]int) {
	*q = 92
	*buffer = -1234
	*recent = [2]int{111, 222}
	*temporalQ = [WebRTCMaxTemporalLayers]uint8{92, 96, 104}
	*temporalBuffer = [WebRTCMaxTemporalLayers]int{1400, -900, 300}
	*temporalRecent = [WebRTCMaxTemporalLayers][2]int{{11, 12}, {21, 22}, {31, 32}}
}

func snapshotVideoRC(q uint8, buffer int, recent [2]int, temporalQ [WebRTCMaxTemporalLayers]uint8, temporalBuffer [WebRTCMaxTemporalLayers]int, temporalRecent [WebRTCMaxTemporalLayers][2]int, temporalPerFrameBits [WebRTCMaxTemporalLayers]int, perFrameBits int) rcControllerSnapshot {
	return rcControllerSnapshot{
		qIndex:              q,
		buffer:              buffer,
		recent:              recent,
		temporalQ:           temporalQ,
		temporalBuffer:      temporalBuffer,
		temporalRecentBits:  temporalRecent,
		temporalPerFrameBit: temporalPerFrameBits,
		perFrameBits:        perFrameBits,
	}
}

func assertPreservedRCUpdate(t *testing.T, before rcControllerSnapshot, after rcControllerSnapshot, rc RateControlConfig, temporalLayers int) {
	t.Helper()
	wantPerFrame, err := rateControlPerFrameBits(rc)
	if err != nil {
		t.Fatalf("rateControlPerFrameBits: %v", err)
	}
	if after.qIndex != before.qIndex || after.buffer != before.buffer || after.recent != before.recent ||
		after.temporalQ != before.temporalQ || after.temporalBuffer != before.temporalBuffer ||
		after.temporalRecentBits != before.temporalRecentBits {
		t.Fatalf("controller state not preserved: before=%+v after=%+v", before, after)
	}
	if after.perFrameBits != wantPerFrame {
		t.Fatalf("per-frame bits=%d want %d", after.perFrameBits, wantPerFrame)
	}
	for temporalID := 0; temporalID < temporalLayers; temporalID++ {
		wantLayerBits := rateControlTemporalLayerPerFrameBits(rc.TargetBitsPerSecond, rc.FramesPerSecond, temporalLayers, uint8(temporalID), wantPerFrame)
		if got := after.temporalPerFrameBit[temporalID]; got != wantLayerBits {
			t.Fatalf("temporal layer %d per-frame bits=%d want %d", temporalID, got, wantLayerBits)
		}
	}
}

func TestUpdateRateControlConfigReinitializesSVTRateControl(t *testing.T) {
	// SVT-AV1 restarts its rate control on a rate change event
	// (seq_param_changed -> set_rc_buffer_sizes + svt_av1_rc_init).
	enc, err := NewVideoEncoderCBR(64, 64, RateControlConfig{
		TargetBitsPerSecond: 1_200_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoderCBR: %v", err)
	}
	enc.svtRC.bufferLevel = -1234
	enc.svtRC.rateCorrection[svtRCInterNormal] = 3
	next := RateControlConfig{TargetBitsPerSecond: 900_000, FramesPerSecond: 60, MinQIndex: 30, MaxQIndex: 180}
	if err := enc.UpdateRateControlConfig(next); err != nil {
		t.Fatalf("UpdateRateControlConfig: %v", err)
	}
	rc := &enc.svtRC
	if rc.avgFrameBandwidth != 15_000 || rc.bestQuality != 30 || rc.worstQuality != 180 {
		t.Fatalf("rc after update: bandwidth=%d q=[%d,%d]", rc.avgFrameBandwidth, rc.bestQuality, rc.worstQuality)
	}
	if rc.bufferLevel != rc.startingBufferLevel || rc.startingBufferLevel != 900_000*600/1000 {
		t.Fatalf("buffer %d starting %d", rc.bufferLevel, rc.startingBufferLevel)
	}
	if rc.rateCorrection[svtRCInterNormal] != 0.7 || enc.rcPerFrameBits != 15_000 {
		t.Fatalf("rate correction %v per-frame %d", rc.rateCorrection[svtRCInterNormal], enc.rcPerFrameBits)
	}
}

func TestSVTRateControlTargetsAndQ(t *testing.T) {
	var rc svtRateControl
	rc.init(1_000_000, 30, 16, 255, 1280, 720)
	// av1_calc_iframe_target_size_one_pass_cbr: the first key takes half
	// the starting buffer, clamped to max_intra_bitrate_pct.
	key := svtRCFrame{key: true, width: 1280, height: 720}
	rc.target(&key)
	if want := min(int(rc.startingBufferLevel/2), rc.avgFrameBandwidth*3); key.thisFrame != want {
		t.Fatalf("key target %d want %d", key.thisFrame, want)
	}
	// A buffer above optimal raises the inter target by at most 25%/2.
	rc.bufferLevel = rc.optimalBufferLevel * 2
	p := svtRCFrame{flat: true, hierLevels: 2, width: 1280, height: 720}
	rc.target(&p)
	if want := rc.avgFrameBandwidth + rc.avgFrameBandwidth*25/200; p.thisFrame != want {
		t.Fatalf("inter target %d want %d", p.thisFrame, want)
	}
	// More bits per macroblock never pick a higher q.
	lo := svtRCFrame{flat: true, hierLevels: 2, width: 1280, height: 720, thisFrame: 10_000}
	hi := lo
	hi.thisFrame = 200_000
	if qLo, qHi := rc.regulateQ(&lo, 16, 255), rc.regulateQ(&hi, 16, 255); qHi > qLo {
		t.Fatalf("q at 200k bits %d above q at 10k bits %d", qHi, qLo)
	}
	// Non-base layers never go below their reference's q.
	layered := svtRCFrame{hierLevels: 2, temporalLayer: 2, refValid: true, refBaseQ: 120, refLayer: 1, width: 1280, height: 720}
	rc.target(&layered)
	if q := rc.pickQ(&layered); q < 120 {
		t.Fatalf("TL2 q %d below the TL1 reference q 120", q)
	}
	// Feedback: an overshoot drains the buffer and raises the correction.
	before := rc.rateCorrection[svtRCInterNormal]
	buf := rc.bufferLevel
	rc.postencode(&p, 100, p.thisFrame*12, svtCRStats{})
	if rc.bufferLevel >= buf || rc.rateCorrection[svtRCInterNormal] <= before || rc.rc1Frame != -1 {
		t.Fatalf("overshoot feedback: buffer %d->%d rcf %v->%v rc1 %d", buf, rc.bufferLevel, before,
			rc.rateCorrection[svtRCInterNormal], rc.rc1Frame)
	}
}

func TestRCInterQWithoutRateControlKeepsFixedQ(t *testing.T) {
	enc := &VideoEncoder{qIndex: 90, temporalLayers: 3}
	for temporalID := uint8(0); temporalID < 3; temporalID++ {
		if got := enc.rcInterQ(temporalID, svtRefLast, 1, 64, 64, false); got != 90 {
			t.Fatalf("temporal layer %d q=%d want fixed CQP q 90", temporalID, got)
		}
	}
}

func TestSetQIndexDisablesRateControlState(t *testing.T) {
	enc := &VideoEncoder{qIndex: 120, rcEnabled: true, rcPerFrameBits: 1000, rcMinQ: 20, rcMaxQ: 200}
	enc.svtRC.enabled = true
	if err := enc.SetQIndex(37); err != nil {
		t.Fatalf("SetQIndex: %v", err)
	}
	if enc.rcEnabled || enc.qIndex != 37 || enc.rcPerFrameBits != 0 || enc.svtRC.enabled {
		t.Fatalf("encoder state after SetQIndex: rc=%v q=%d", enc.rcEnabled, enc.qIndex)
	}
	if err := enc.SetQIndex(0); err == nil {
		t.Fatal("SetQIndex(0) succeeded")
	}
	if enc.qIndex != 37 || enc.rcEnabled {
		t.Fatalf("invalid SetQIndex mutated state: rc=%v q=%d", enc.rcEnabled, enc.qIndex)
	}
}
