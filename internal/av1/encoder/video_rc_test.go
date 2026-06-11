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
	enc.qIndex = 110
	if got, want := enc.keyframeQIndex(), uint8(60); got != want {
		t.Fatalf("midrange key q=%d want %d", got, want)
	}
	enc.qIndex = 30
	if got, want := enc.keyframeQIndex(), uint8(20); got != want {
		t.Fatalf("min-clamped key q=%d want %d", got, want)
	}
}
