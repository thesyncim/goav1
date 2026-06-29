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
