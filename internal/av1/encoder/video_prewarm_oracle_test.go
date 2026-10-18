package encoder

import (
	"bytes"
	"reflect"
	"testing"
)

// This body is frozen from main 10f4a6ae956e881e8d9e85a4797c55a441cd27c3.
// It is independent of the shorter candidate walk.
func frozenPrewarmOracle(e *VideoEncoder) error {
	color := e.parserColorConfig()
	cw := chromaWidthForColor(e.width, color)
	ch := chromaHeightForColor(e.height, color)
	src := SourceFrame420{
		Y:            make([]byte, e.width*e.height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      e.width,
		ChromaStride: cw,
		Width:        e.renderWidth,
		Height:       e.renderHeight,
	}
	savedQ := e.qIndex
	// Size the assembly buffers for real content up front; the throwaway
	// frame below is black and would leave them near-empty.
	if bound := e.width * e.height; cap(e.tuScratch) < bound {
		e.tuScratch = make([]byte, 0, bound)
		e.tuGroup = make([]byte, 0, bound)
	}
	if _, _, err := e.Encode(src, true); err != nil {
		return err
	}
	// Exercise every reconstruction buffer and the golden-refresh cadence so
	// delayed scratch growth happens before the first externally visible frame.
	frames := e.temporalLayers
	if frames < 2 {
		frames = 2
	} else {
		frames *= 2
	}
	if e.goldenEvery > 0 && frames < e.goldenEvery+1 {
		frames = e.goldenEvery + 1
	}
	for i := 0; i < frames; i++ {
		if _, _, err := e.Encode(src, false); err != nil {
			return err
		}
	}
	if err := e.joinFilter(); err != nil {
		return err
	}
	e.prewarmPipeline(src)
	e.haveKey = false
	e.haveCtx = false
	e.interpOnlyRegular = false
	e.haveCtxT1 = false
	e.frameIndex = 0
	e.lastTemporalID = 0
	e.qIndex = savedQ
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
	e.sinceGoldenFresh = 0
	e.hme.armed = false
	e.lastRecon = SourceFrame420{}
	return nil
}

func prewarmStateForOracle(e *VideoEncoder) any {
	return struct {
		HaveKey, HaveCtx, HaveCtxT1, InterpOnlyRegular, HMEArmed bool
		FrameIndex, ReconIdx, RCBuffer, SinceGolden, LowMotion   int
		LastTemporalID, QIndex                                   uint8
		RecentBits                                               [2]int
		TemporalQ                                                [WebRTCMaxTemporalLayers]uint8
		TemporalBuffer                                           [WebRTCMaxTemporalLayers]int
		TemporalRecentBits                                       [WebRTCMaxTemporalLayers][2]int
		Stats                                                    EncoderDecisionStats
	}{e.haveKey, e.haveCtx, e.haveCtxT1, e.interpOnlyRegular, e.hme.armed,
		e.frameIndex, e.reconIdx, e.rcBuffer, e.sinceGoldenFresh, e.avgFrameLowMotion,
		e.lastTemporalID, e.qIndex, e.rcRecentBits, e.rcTemporalQ, e.rcTemporalBuffer,
		e.rcTemporalRecentBits, e.decisionStats}
}

func equalPrewarmRecon(a, b SourceFrame420) bool {
	return a.Width == b.Width && a.Height == b.Height && a.YStride == b.YStride &&
		a.ChromaStride == b.ChromaStride && bytes.Equal(a.Y, b.Y) && bytes.Equal(a.U, b.U) && bytes.Equal(a.V, b.V)
}

func TestVideoEncoderPrewarmLegacyOracle(t *testing.T) {
	cases := []struct {
		name               string
		q                  uint8
		golden, layers     int
		cbr, reused, stats bool
	}{
		{name: "fresh_cbr_default_golden", q: 100, golden: 16, cbr: true},
		{name: "fresh_fixed_high_q", q: 200, golden: 16},
		{name: "fresh_even_recon_parity", q: 80, golden: 17},
		{name: "golden_disabled_fallback", q: 100, golden: 0, cbr: true},
		{name: "temporal_state_fallback", q: 100, golden: 16, layers: 2, cbr: true},
		{name: "reused_state_fallback", q: 100, golden: 16, cbr: true, reused: true},
		{name: "decision_stats_fallback", q: 100, golden: 16, cbr: true, stats: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			makeEnc := func() *VideoEncoder {
				e, err := NewVideoEncoder(192, 128, tc.q)
				if err != nil {
					t.Fatal(err)
				}
				e.SetMaxThreads(1)
				e.SetSceneCutKeyframes(false)
				e.SetGoldenInterval(tc.golden)
				if tc.cbr {
					if err = e.SetRateControlConfig(RateControlConfig{TargetBitsPerSecond: 500000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200}); err != nil {
						t.Fatal(err)
					}
				}
				if tc.layers > 1 {
					if err = e.SetTemporalLayers(tc.layers); err != nil {
						t.Fatal(err)
					}
				}
				e.SetDecisionStatsEnabled(tc.stats)
				t.Cleanup(func() { _ = e.Close() })
				return e
			}
			got, want := makeEnc(), makeEnc()
			if tc.reused {
				for n := 0; n < 3; n++ {
					f := lfDiffFrame(192, 128)
					for _, e := range []*VideoEncoder{got, want} {
						if _, _, err := e.Encode(f, false); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
			if err := got.Prewarm(); err != nil {
				t.Fatal(err)
			}
			if err := frozenPrewarmOracle(want); err != nil {
				t.Fatal(err)
			}
			gs, ws := prewarmStateForOracle(got), prewarmStateForOracle(want)
			if !reflect.DeepEqual(gs, ws) {
				t.Fatalf("retained state differs: got %+v; want %+v", gs, ws)
			}
			if !equalPrewarmRecon(got.Recon(), want.Recon()) {
				t.Fatal("returned reconstruction differs before first real frame")
			}
			for n := 0; n < 20; n++ {
				f := lfDiffFrame(192, 128)
				for y := 0; y < 128; y++ {
					for x := 0; x < 192; x++ {
						f.Y[y*192+x] = byte(40 + ((x+n*3)/5+y/3)%150)
					}
				}
				gb, gk, ge := got.Encode(f, n == 11)
				wb, wk, we := want.Encode(f, n == 11)
				if ge != nil || we != nil {
					t.Fatalf("frame %d errors %v/%v", n, ge, we)
				}
				if gk != wk || !bytes.Equal(gb, wb) {
					t.Fatalf("frame %d encoded bytes differ, sizes %d/%d", n, len(gb), len(wb))
				}
				if !equalPrewarmRecon(got.Recon(), want.Recon()) {
					t.Fatalf("frame %d reconstruction differs", n)
				}
				if !reflect.DeepEqual(prewarmStateForOracle(got), prewarmStateForOracle(want)) {
					t.Fatalf("frame %d retained state differs", n)
				}
			}
		})
	}
}

func TestVideoEncoderPrewarmFirstFramesAllocs(t *testing.T) {
	// AllocsPerRun calls the closure once before measuring. Each invocation uses
	// a different already-prewarmed encoder so the measured frames are still its
	// first externally visible keyframe and both P reconstruction buffers.
	encs := make([]*VideoEncoder, 4)
	for i := range encs {
		e, err := NewVideoEncoderCBR(192, 128, RateControlConfig{TargetBitsPerSecond: 500000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200})
		if err != nil {
			t.Fatal(err)
		}
		e.SetMaxThreads(1)
		e.SetSceneCutKeyframes(false)
		if err = e.Prewarm(); err != nil {
			t.Fatal(err)
		}
		encs[i] = e
		t.Cleanup(func() { _ = e.Close() })
	}
	f := lfDiffFrame(192, 128)
	i := 0
	if n := testing.AllocsPerRun(3, func() {
		e := encs[i]
		i++
		for frame := 0; frame < 3; frame++ {
			if _, _, err := e.Encode(f, false); err != nil {
				t.Fatal(err)
			}
		}
	}); n != 0 {
		t.Fatalf("first three frames allocated %g times", n)
	}
}
