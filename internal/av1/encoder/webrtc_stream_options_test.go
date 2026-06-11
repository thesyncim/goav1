package encoder

import "testing"

func TestWebRTCStreamEncoderOptions(t *testing.T) {
	stream, err := NewWebRTCStreamLayers(192, 128, RateControlConfig{
		TargetBitsPerSecond: 500_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	stream.SetGoldenInterval(0)
	if got := stream.enc.goldenEvery; got != 0 {
		t.Fatalf("golden interval=%d want 0", got)
	}
	stream.SetTileColumns(4)
	if got := stream.enc.tileColsLog2; got != 2 {
		t.Fatalf("tileColsLog2=%d want 2", got)
	}
	if err := stream.Prewarm(); err != nil {
		t.Fatalf("prewarm: %v", err)
	}
	if stream.frameID != 0 || stream.groupPos != 0 || stream.enc.haveKey {
		t.Fatalf("prewarm advanced stream state: frameID=%d groupPos=%d haveKey=%v", stream.frameID, stream.groupPos, stream.enc.haveKey)
	}
}
