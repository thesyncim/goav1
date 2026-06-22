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
