package encoder

import "testing"

func TestKeyframeQIndexCBR(t *testing.T) {
	enc := &VideoEncoder{
		qIndex:         200,
		rcEnabled:      true,
		rcPerFrameBits: 100_000,
		rcMinQ:         20,
		rcMaxQ:         200,
	}
	if got, want := enc.keyframeQIndex(), uint8(140); got != want {
		t.Fatalf("max-debt key q=%d want %d", got, want)
	}
	enc.rcBuffer = -24 * enc.rcPerFrameBits
	if got, want := enc.keyframeQIndex(), uint8(163); got != want {
		t.Fatalf("buffer-debt key q=%d want %d", got, want)
	}
	enc.rcBuffer = 0
	enc.qIndex = 110
	if got, want := enc.keyframeQIndex(), uint8(60); got != want {
		t.Fatalf("midrange key q=%d want %d", got, want)
	}
	enc.qIndex = 30
	if got, want := enc.keyframeQIndex(), uint8(20); got != want {
		t.Fatalf("min-clamped key q=%d want %d", got, want)
	}
}

func TestRateControlSurplusFrameLimit(t *testing.T) {
	enc := &VideoEncoder{}
	if got, want := enc.rcSurplusFrameLimit(), 8; got != want {
		t.Fatalf("flat surplus limit=%d want %d", got, want)
	}
	enc.temporalLayers = 3
	if got, want := enc.rcSurplusFrameLimit(), 8; got != want {
		t.Fatalf("L1T3 surplus limit=%d want %d", got, want)
	}
	enc.temporalLayers = 2
	if got, want := enc.rcSurplusFrameLimit(), 2; got != want {
		t.Fatalf("L1T2 surplus limit=%d want %d", got, want)
	}
}

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

func TestRateControlTemporalLayerStateUsesLayerBudgets(t *testing.T) {
	enc, err := NewVideoEncoderCBR(64, 64, RateControlConfig{
		TargetBitsPerSecond: 1_000_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoderCBR: %v", err)
	}
	if err := enc.SetTemporalLayers(3); err != nil {
		t.Fatalf("SetTemporalLayers: %v", err)
	}
	wantBits := [...]int{66_667, 26_667, 20_000}
	for i, want := range wantBits {
		if got := enc.rcTemporalPerFrameBits[i]; got != want {
			t.Fatalf("temporal layer %d per-frame bits=%d want %d", i, got, want)
		}
	}
	baseQ := enc.QIndex()
	enc.rcUpdate(wantBits[2]*2, 2)
	if got := enc.rcTemporalQ[2]; got <= baseQ {
		t.Fatalf("TL2 q=%d did not rise above base q=%d after overshoot", got, baseQ)
	}
	if got := enc.QIndex(); got != baseQ {
		t.Fatalf("base q changed after TL2 update: got %d want %d", got, baseQ)
	}
	if got := enc.layerQIndex(2); got != enc.rcTemporalQ[2] {
		t.Fatalf("TL2 layer q=%d want %d", got, enc.rcTemporalQ[2])
	}
}

func TestSetTemporalLayersSameCountPreservesRateControlState(t *testing.T) {
	enc, err := NewVideoEncoderCBR(64, 64, RateControlConfig{
		TargetBitsPerSecond: 1_000_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoderCBR: %v", err)
	}
	if err := enc.SetTemporalLayers(3); err != nil {
		t.Fatalf("SetTemporalLayers: %v", err)
	}
	enc.rcTemporalQ = [WebRTCMaxTemporalLayers]uint8{80, 84, 88}
	enc.rcTemporalBuffer = [WebRTCMaxTemporalLayers]int{1200, -800, 300}
	enc.rcTemporalRecentBits = [WebRTCMaxTemporalLayers][2]int{{11, 12}, {21, 22}, {31, 32}}
	keepQ := enc.rcTemporalQ
	keepBuffers := enc.rcTemporalBuffer
	keepRecent := enc.rcTemporalRecentBits

	if err := enc.SetTemporalLayers(3); err != nil {
		t.Fatalf("SetTemporalLayers same: %v", err)
	}
	if enc.rcTemporalQ != keepQ || enc.rcTemporalBuffer != keepBuffers || enc.rcTemporalRecentBits != keepRecent {
		t.Fatalf("same temporal layer reset rc state: q=%v buffers=%v recent=%v", enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits)
	}

	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatalf("SetTemporalLayers changed: %v", err)
	}
	if enc.rcTemporalBuffer != ([WebRTCMaxTemporalLayers]int{}) || enc.rcTemporalRecentBits != ([WebRTCMaxTemporalLayers][2]int{}) {
		t.Fatalf("changed temporal layer did not reset rc state: buffers=%v recent=%v", enc.rcTemporalBuffer, enc.rcTemporalRecentBits)
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
			name: "i420",
			run: func(t *testing.T) {
				enc, err := NewVideoEncoderCBR(64, 64, oldRC)
				if err != nil {
					t.Fatalf("NewVideoEncoderCBR: %v", err)
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

func TestLayerQIndexDoesNotApplyFixedTemporalOffsetWithoutRC(t *testing.T) {
	enc := &VideoEncoder{qIndex: 90, temporalLayers: 3}
	for temporalID := uint8(0); temporalID < 3; temporalID++ {
		if got := enc.layerQIndex(temporalID); got != 90 {
			t.Fatalf("temporal layer %d q=%d want fixed CQP q 90", temporalID, got)
		}
	}
}

func TestSetQIndexDisablesRateControlState(t *testing.T) {
	enc := &VideoEncoder{
		qIndex:         120,
		rcEnabled:      true,
		rcPerFrameBits: 1000,
		rcBuffer:       -500,
		rcRecentBits:   [2]int{300, 400},
		rcMinQ:         20,
		rcMaxQ:         200,
	}
	if err := enc.SetQIndex(37); err != nil {
		t.Fatalf("SetQIndex: %v", err)
	}
	if enc.rcEnabled || enc.qIndex != 37 || enc.rcPerFrameBits != 0 || enc.rcBuffer != 0 || enc.rcRecentBits != ([2]int{}) {
		t.Fatalf("encoder state after SetQIndex: %+v", enc)
	}
	if err := enc.SetQIndex(0); err == nil {
		t.Fatal("SetQIndex(0) succeeded")
	}
	if enc.qIndex != 37 || enc.rcEnabled {
		t.Fatalf("invalid SetQIndex mutated state: %+v", enc)
	}
}
