package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestMotionModeCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs MotionModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "motion 8x8", cdf: &cdfs.MotionMode[BlockSize8x8], want: []uint16{25117, 8008, 0, 0}},
		{name: "motion 32x8", cdf: &cdfs.MotionMode[BlockSize32x8], want: []uint16{3969, 1378, 0, 0}},
		{name: "motion 128x128", cdf: &cdfs.MotionMode[BlockSize128x128], want: []uint16{261, 210, 0, 0}},
		{name: "obmc 8x8", cdf: &cdfs.OBMC[BlockSize8x8], want: []uint16{22331, 0, 0}},
		{name: "obmc 32x8", cdf: &cdfs.OBMC[BlockSize32x8], want: []uint16{9104, 0, 0}},
		{name: "obmc 128x128", cdf: &cdfs.OBMC[BlockSize128x128], want: []uint16{130, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}

	if _, err := cdfs.MotionModeCDF(BlockSize16x4); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("disallowed motion cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestLastMotionModeAllowedMatchesLibaom(t *testing.T) {
	base := MotionModeRequest{
		Size:                  BlockSize16x16,
		Mode:                  InterModeNewMV,
		SwitchableMotionMode:  true,
		AllowWarpedMotion:     true,
		OverlappableNeighbors: 1,
		NumProjRef:            1,
	}

	tests := []struct {
		name string
		req  MotionModeRequest
		want MotionMode
	}{
		{name: "warp", req: base, want: MotionModeWarp},
		{name: "switchable off", req: motionModeReq(base, func(r *MotionModeRequest) { r.SwitchableMotionMode = false }), want: MotionModeTranslation},
		{name: "skip mode", req: motionModeReq(base, func(r *MotionModeRequest) { r.SkipMode = true }), want: MotionModeTranslation},
		{name: "inter intra", req: motionModeReq(base, func(r *MotionModeRequest) { r.InterIntra = true }), want: MotionModeTranslation},
		{name: "no neighbors", req: motionModeReq(base, func(r *MotionModeRequest) { r.OverlappableNeighbors = 0 }), want: MotionModeTranslation},
		{name: "small block", req: motionModeReq(base, func(r *MotionModeRequest) { r.Size = BlockSize16x4 }), want: MotionModeTranslation},
		{name: "compound", req: motionModeReq(base, func(r *MotionModeRequest) { r.Compound = true }), want: MotionModeTranslation},
		{name: "warped globalmv", req: motionModeReq(base, func(r *MotionModeRequest) {
			r.Mode = InterModeGlobalMV
			r.GlobalMotionType = parser.GlobalMotionAffine
		}), want: MotionModeTranslation},
		{name: "global translation", req: motionModeReq(base, func(r *MotionModeRequest) {
			r.Mode = InterModeGlobalMV
			r.GlobalMotionType = parser.GlobalMotionTranslation
		}), want: MotionModeWarp},
		{name: "warp disabled", req: motionModeReq(base, func(r *MotionModeRequest) { r.AllowWarpedMotion = false }), want: MotionModeOBMC},
		{name: "force integer", req: motionModeReq(base, func(r *MotionModeRequest) { r.ForceIntegerMV = true }), want: MotionModeOBMC},
		{name: "scaled ref", req: motionModeReq(base, func(r *MotionModeRequest) { r.ScaledReference = true }), want: MotionModeOBMC},
		{name: "no projected refs", req: motionModeReq(base, func(r *MotionModeRequest) { r.NumProjRef = 0 }), want: MotionModeOBMC},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LastMotionModeAllowed(tt.req)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("allowed=%d want %d", got, tt.want)
			}
		})
	}
}

func TestReadMotionMode(t *testing.T) {
	var cdfs MotionModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	req := MotionModeRequest{
		Size:                  BlockSize16x16,
		Mode:                  InterModeNewMV,
		SwitchableMotionMode:  true,
		OverlappableNeighbors: 1,
	}
	mode, err := state.ReadMotionMode(&cdfs, req)
	if err != nil {
		t.Fatal(err)
	}
	if mode != MotionModeTranslation {
		t.Fatalf("obmc-only mode=%d want translation", mode)
	}
	if got := cdfs.OBMC[BlockSize16x16].Values()[2]; got != 1 {
		t.Fatalf("obmc count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	req.AllowWarpedMotion = true
	req.NumProjRef = 1
	mode, err = state.ReadMotionMode(&cdfs, req)
	if err != nil {
		t.Fatal(err)
	}
	if mode != MotionModeTranslation {
		t.Fatalf("warp-capable mode=%d want translation", mode)
	}
	if got := cdfs.MotionMode[BlockSize16x16].Values()[3]; got != 1 {
		t.Fatalf("motion count=%d want 1", got)
	}

	before := cdfs.MotionMode[BlockSize16x16]
	mode, err = state.ReadMotionMode(&cdfs, motionModeReq(req, func(r *MotionModeRequest) {
		r.SwitchableMotionMode = false
	}))
	if err != nil {
		t.Fatal(err)
	}
	if mode != MotionModeTranslation || cdfs.MotionMode[BlockSize16x16] != before {
		t.Fatalf("forced translation mode=%d cdf changed=%v", mode, cdfs.MotionMode[BlockSize16x16] != before)
	}
}

func TestMotionModeRejectsInvalidInputs(t *testing.T) {
	var cdfs MotionModeCDFs
	if _, err := cdfs.MotionModeCDF(BlockSize8x8); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.OBMCCDF(blockSizeCount); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}

	valid := MotionModeRequest{
		Size:                  BlockSize16x16,
		Mode:                  InterModeNewMV,
		SwitchableMotionMode:  true,
		OverlappableNeighbors: 1,
	}
	if _, err := LastMotionModeAllowed(motionModeReq(valid, func(r *MotionModeRequest) { r.Size = blockSizeCount })); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad size err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := LastMotionModeAllowed(motionModeReq(valid, func(r *MotionModeRequest) { r.Mode = interModeCount })); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := LastMotionModeAllowed(motionModeReq(valid, func(r *MotionModeRequest) { r.GlobalMotionType = parser.GlobalMotionAffine + 1 })); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad gm err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := LastMotionModeAllowed(motionModeReq(valid, func(r *MotionModeRequest) { r.OverlappableNeighbors = -1 })); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad neighbors err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := LastMotionModeAllowed(motionModeReq(valid, func(r *MotionModeRequest) { r.NumProjRef = -1 })); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad proj refs err=%v want %v", err, ErrInvalidDecodeState)
	}

	var nilState *DecodeState
	if _, err := nilState.ReadMotionMode(&cdfs, valid); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.ReadMotionMode(nil, valid); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil cdfs err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestMotionModeAllocs(t *testing.T) {
	var cdfs MotionModeCDFs
	var state DecodeState
	payload := []byte{0x00}
	req := MotionModeRequest{
		Size:                  BlockSize16x16,
		Mode:                  InterModeNewMV,
		SwitchableMotionMode:  true,
		AllowWarpedMotion:     true,
		OverlappableNeighbors: 1,
		NumProjRef:            1,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadMotionMode(&cdfs, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("motion mode decode allocated: %f", allocs)
	}
}

func FuzzReadMotionMode(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(InterModeNewMV), true, true, uint8(1), uint8(1), false, false)
	f.Add([]byte{0xff}, uint8(BlockSize8x8), uint8(InterModeGlobalMV), true, true, uint8(2), uint8(1), false, false)
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize16x4), uint8(InterModeNearMV), false, false, uint8(0), uint8(0), true, true)

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawMode uint8, switchable bool, allowWarp bool, rawGM uint8, rawCount uint8, forceInteger bool, scaled bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		var cdfs MotionModeCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		req := MotionModeRequest{
			Size:                  BlockSize(rawSize % uint8(blockSizeCount)),
			Mode:                  InterMode(rawMode % uint8(interModeCount)),
			SwitchableMotionMode:  switchable,
			AllowWarpedMotion:     allowWarp,
			ForceIntegerMV:        forceInteger,
			GlobalMotionType:      parser.GlobalMotionType(rawGM % 4),
			ScaledReference:       scaled,
			OverlappableNeighbors: int(rawCount & 3),
			NumProjRef:            int((rawCount >> 2) & 3),
		}
		mode, err := state.ReadMotionMode(&cdfs, req)
		if err != nil {
			t.Fatalf("ReadMotionMode err=%v req=%+v", err, req)
		}
		if !mode.Valid() {
			t.Fatalf("mode=%d", mode)
		}
	})
}

func motionModeReq(req MotionModeRequest, mutate func(*MotionModeRequest)) MotionModeRequest {
	mutate(&req)
	return req
}
