package encoder

import "testing"

func TestVideoEncoderDecisionStatsDisabledByDefault(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 90)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := enc.Encode(decisionStatsFrame(64, 64, 0), false); err != nil {
		t.Fatal(err)
	}
	if _, _, err := enc.Encode(decisionStatsFrame(64, 64, 1), false); err != nil {
		t.Fatal(err)
	}
	if got := enc.DecisionStats(); got != (EncoderDecisionStats{}) {
		t.Fatalf("disabled stats=%+v want zero", got)
	}
}

func TestVideoEncoderDecisionStatsAccumulatesKeyAndInter(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 90)
	if err != nil {
		t.Fatal(err)
	}
	enc.SetTileColumns(1)
	enc.SetDecisionStatsEnabled(true)
	if _, key, err := enc.Encode(decisionStatsFrame(64, 64, 0), false); err != nil || !key {
		t.Fatalf("key encode key=%v err=%v", key, err)
	}
	if _, key, err := enc.Encode(decisionStatsFrame(64, 64, 1), false); err != nil || key {
		t.Fatalf("inter encode key=%v err=%v", key, err)
	}
	stats := enc.DecisionStats()
	if stats.Frames != 2 || stats.Keyframes != 1 || stats.InterFrames != 1 {
		t.Fatalf("frame counts=%+v", stats)
	}
	if stats.Tiles != 2 {
		t.Fatalf("tiles=%d want 2", stats.Tiles)
	}
	if stats.PartitionDecisions == 0 || stats.Blocks == 0 {
		t.Fatalf("missing decision counts=%+v", stats)
	}
	if stats.IntraBlocks == 0 || stats.InterBlocks == 0 {
		t.Fatalf("block kind counts=%+v", stats)
	}
	if stats.CodedBlocks+stats.SkipBlocks != stats.Blocks {
		t.Fatalf("coded+skip=%d blocks=%d", stats.CodedBlocks+stats.SkipBlocks, stats.Blocks)
	}
	if stats.LumaTXBs == 0 || stats.TXTypes[EncoderDecisionTransformDCTDCT] == 0 {
		t.Fatalf("missing tx counts=%+v", stats)
	}
}

func TestVideoEncoderDecisionStatsReset(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 90)
	if err != nil {
		t.Fatal(err)
	}
	enc.SetDecisionStatsEnabled(true)
	if _, _, err := enc.Encode(decisionStatsFrame(64, 64, 0), false); err != nil {
		t.Fatal(err)
	}
	if enc.DecisionStats().Frames == 0 {
		t.Fatal("expected stats before reset")
	}
	enc.ResetDecisionStats()
	if got := enc.DecisionStats(); got != (EncoderDecisionStats{}) {
		t.Fatalf("reset stats=%+v want zero", got)
	}
}

func decisionStatsFrame(w, h, n int) SourceFrame420 {
	cw, ch := w/2, h/2
	f := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 48 + (x+y+n*3)%96
			if x >= 16+n && x < 32+n && y >= 20 && y < 36 {
				v = 220
			}
			f.Y[y*w+x] = byte(v)
		}
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 132
	}
	return f
}
