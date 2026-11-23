package goav1

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"

	"github.com/thesyncim/goav1/internal/av1/obu"
)

var publicRTCEncoderHotPathSink int

func TestPublicEncoderWebRTCScalabilityModeCatalogue(t *testing.T) {
	want := []string{
		"L1T1", "L1T2", "L1T3",
		"L2T1", "L2T1h", "L2T1_KEY",
		"L2T2", "L2T2h", "L2T2_KEY", "L2T2_KEY_SHIFT",
		"L2T3", "L2T3h", "L2T3_KEY",
		"L3T1", "L3T1h", "L3T1_KEY",
		"L3T2", "L3T2h", "L3T2_KEY",
		"L3T3", "L3T3h", "L3T3_KEY",
		"S2T1", "S2T1h", "S2T2", "S2T2h", "S2T3", "S2T3h",
		"S3T1", "S3T1h", "S3T2", "S3T2h", "S3T3", "S3T3h",
	}
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) != len(want) {
		t.Fatalf("EncoderWebRTCScalabilityModes len=%d want %d", len(modes), len(want))
	}
	prefixed := internalencoder.AppendWebRTCScalabilityModes([]EncoderScalabilityMode{internalencoder.ScalabilityModeL3T3})
	if len(prefixed) != len(want)+1 || prefixed[0] != internalencoder.ScalabilityModeL3T3 {
		t.Fatalf("AppendEncoderWebRTCScalabilityModes prefix len=%d first=%s", len(prefixed), prefixed[0])
	}
	for i, name := range want {
		mode := modes[i]
		if prefixed[i+1] != mode {
			t.Fatalf("prefixed mode %d=%s want %s", i+1, prefixed[i+1], mode)
		}
		parsed, ok := internalencoder.ParseScalabilityMode(name)
		if !ok || parsed != mode {
			t.Fatalf("ParseEncoderScalabilityMode(%q)=%s,%v want %s,true", name, parsed, ok, mode)
		}
		if got := mode.String(); got != name {
			t.Fatalf("mode %d String()=%q want %q", i, got, name)
		}
		tIndex := strings.IndexByte(name, 'T')
		if len(name) < 4 || tIndex < 2 || tIndex+1 >= len(name) {
			t.Fatalf("bad expected scalability name %q", name)
		}
		wantSpatial := uint8(name[1] - '0')
		wantTemporal := uint8(name[tIndex+1] - '0')
		spatial, temporal, key, ok := mode.Layers()
		if !ok || spatial != wantSpatial || temporal != wantTemporal ||
			key != strings.Contains(name, "_KEY") {
			t.Fatalf("%q Layers()=%d,%d,%v,%v want %d,%d,%v,true",
				name, spatial, temporal, key, ok,
				wantSpatial, wantTemporal, strings.Contains(name, "_KEY"))
		}
		if mode.IsSimulcast() != strings.HasPrefix(name, "S") ||
			mode.UsesSmallResolutionStep() != strings.Contains(name, "h") ||
			mode.UsesSharedReferenceSlots() != (wantSpatial > 1 && !strings.HasPrefix(name, "S")) ||
			mode.UsesKeyFrameInterLayerDependency() != strings.Contains(name, "_KEY") ||
			mode.UsesKeyFrameInterLayerDependencyShift() != strings.Contains(name, "_KEY_SHIFT") {
			t.Fatalf("%q flags simulcast=%v small=%v shared=%v key=%v shift=%v",
				name,
				mode.IsSimulcast(),
				mode.UsesSmallResolutionStep(),
				mode.UsesSharedReferenceSlots(),
				mode.UsesKeyFrameInterLayerDependency(),
				mode.UsesKeyFrameInterLayerDependencyShift())
		}
	}
	if _, ok := internalencoder.ParseScalabilityMode("L4T4"); ok {
		t.Fatal("ParseEncoderScalabilityMode accepted invalid L4T4")
	}
}

func TestPublicEncoderScalabilityModeCatalogueIncludesExplicitModes(t *testing.T) {
	want := []string{
		"L1T1", "L1T2", "L1T3",
		"L2T1", "L2T1h", "L2T1_KEY",
		"L2T2", "L2T2h", "L2T2_KEY", "L2T2_KEY_SHIFT",
		"L2T3", "L2T3h", "L2T3_KEY", "L2T3_KEY_SHIFT",
		"L3T1", "L3T1h", "L3T1_KEY",
		"L3T2", "L3T2h", "L3T2_KEY", "L3T2_KEY_SHIFT",
		"L3T3", "L3T3h", "L3T3_KEY", "L3T3_KEY_SHIFT",
		"S2T1", "S2T1h", "S2T2", "S2T2h", "S2T3", "S2T3h",
		"S3T1", "S3T1h", "S3T2", "S3T2h", "S3T3", "S3T3h",
	}
	modes := internalencoder.AppendScalabilityModes(nil)
	if len(modes) != len(want) {
		t.Fatalf("EncoderScalabilityModes len=%d want %d", len(modes), len(want))
	}
	prefixed := internalencoder.AppendScalabilityModes([]EncoderScalabilityMode{internalencoder.ScalabilityModeL3T3})
	if len(prefixed) != len(want)+1 || prefixed[0] != internalencoder.ScalabilityModeL3T3 {
		t.Fatalf("AppendEncoderScalabilityModes prefix len=%d first=%s", len(prefixed), prefixed[0])
	}
	for i, name := range want {
		mode := modes[i]
		if prefixed[i+1] != mode {
			t.Fatalf("prefixed mode %d=%s want %s", i+1, prefixed[i+1], mode)
		}
		parsed, ok := internalencoder.ParseScalabilityMode(name)
		if !ok || parsed != mode {
			t.Fatalf("ParseEncoderScalabilityMode(%q)=%s,%v want %s,true", name, parsed, ok, mode)
		}
		if got := mode.String(); got != name {
			t.Fatalf("mode %d String()=%q want %q", i, got, name)
		}
	}

	pinnedLibWebRTCOnly := make(map[EncoderScalabilityMode]bool)
	for _, mode := range EncoderWebRTCScalabilityModes() {
		pinnedLibWebRTCOnly[mode] = true
	}
	webRTCSVC := make(map[EncoderScalabilityMode]bool)
	for _, mode := range internalencoder.AppendWebRTCSVCScalabilityModes(nil) {
		webRTCSVC[mode] = true
	}
	for _, mode := range []EncoderScalabilityMode{
		EncoderScalabilityModeL2T3_KEY_SHIFT,
		EncoderScalabilityModeL3T2_KEY_SHIFT,
		EncoderScalabilityModeL3T3_KEY_SHIFT,
	} {
		if !containsEncoderScalabilityMode(modes, mode) {
			t.Fatalf("EncoderScalabilityModes omitted explicit mode %s", mode)
		}
		if pinnedLibWebRTCOnly[mode] {
			t.Fatalf("%s unexpectedly exported by EncoderWebRTCScalabilityModes", mode)
		}
		if !webRTCSVC[mode] {
			t.Fatalf("EncoderWebRTCSVCScalabilityModes omitted W3C mode %s", mode)
		}
		if err := internalencoder.ValidateWebRTCActiveScalabilityModes([]EncoderScalabilityMode{mode}); err != nil {
			t.Fatalf("ValidateEncoderWebRTCActiveScalabilityModes(%s): %v", mode, err)
		}
	}
}

func TestPublicEncoderWebRTCSVCScalabilityModeCatalogueIncludesCompleteW3CList(t *testing.T) {
	want := []string{
		"L1T1", "L1T2", "L1T3",
		"L2T1", "L2T1h", "L2T1_KEY",
		"L2T2", "L2T2h", "L2T2_KEY", "L2T2_KEY_SHIFT",
		"L2T3", "L2T3h", "L2T3_KEY", "L2T3_KEY_SHIFT",
		"L3T1", "L3T1h", "L3T1_KEY",
		"L3T2", "L3T2h", "L3T2_KEY", "L3T2_KEY_SHIFT",
		"L3T3", "L3T3h", "L3T3_KEY", "L3T3_KEY_SHIFT",
		"S2T1", "S2T1h", "S2T2", "S2T2h", "S2T3", "S2T3h",
		"S3T1", "S3T1h", "S3T2", "S3T2h", "S3T3", "S3T3h",
	}
	modes := internalencoder.AppendWebRTCSVCScalabilityModes(nil)
	if len(modes) != len(want) {
		t.Fatalf("EncoderWebRTCSVCScalabilityModes len=%d want %d", len(modes), len(want))
	}
	prefixed := internalencoder.AppendWebRTCSVCScalabilityModes([]EncoderScalabilityMode{internalencoder.ScalabilityModeL3T3})
	if len(prefixed) != len(want)+1 || prefixed[0] != internalencoder.ScalabilityModeL3T3 {
		t.Fatalf("AppendEncoderWebRTCSVCScalabilityModes prefix len=%d first=%s", len(prefixed), prefixed[0])
	}
	seen := make(map[EncoderScalabilityMode]bool, len(modes))
	for i, name := range want {
		mode := modes[i]
		if prefixed[i+1] != mode {
			t.Fatalf("prefixed mode %d=%s want %s", i+1, prefixed[i+1], mode)
		}
		parsed, ok := internalencoder.ParseScalabilityMode(name)
		if !ok || parsed != mode {
			t.Fatalf("ParseEncoderScalabilityMode(%q)=%s,%v want %s,true", name, parsed, ok, mode)
		}
		if got := mode.String(); got != name {
			t.Fatalf("mode %d String()=%q want %q", i, got, name)
		}
		if seen[mode] {
			t.Fatalf("duplicate mode %s", mode)
		}
		seen[mode] = true
	}
	modes[0] = internalencoder.ScalabilityModeS3T3h
	if got := internalencoder.AppendWebRTCSVCScalabilityModes(nil)[0]; got != EncoderScalabilityModeL1T1 {
		t.Fatalf("EncoderWebRTCSVCScalabilityModes aliased caller mutation: %s", got)
	}
}

func containsEncoderScalabilityMode(modes []EncoderScalabilityMode, want EncoderScalabilityMode) bool {
	for _, mode := range modes {
		if mode == want {
			return true
		}
	}
	return false
}

// TestPublicVideoEncoderRoundTrip drives the public encoding surface end to
// end: CBR with two temporal layers, a mid-stream forced keyframe, and decode
// through the public Decoder with every frame bit-exact against the encoder
// reconstruction.
func TestPublicVideoEncoderRoundTrip(t *testing.T) {
	const w, h = 320, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(60 + rng.Intn(80))
	}
	makeFrame := func(n int) I420Frame {
		f := I420Frame{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		for y := 20 + n*3; y < 52+n*3 && y < h; y++ {
			for x := 16 + n*5; x < 48+n*5 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 400_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	const frames = 12
	var tus [][]byte
	type snap struct {
		y, u, v []byte
	}
	var recons []snap
	keyAt := map[int]bool{}
	for i := range frames {
		out, err := enc.Encode(makeFrame(i), i == 6) // force a mid-stream keyframe
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if out.Keyframe {
			keyAt[i] = true
		}
		if i >= 1 && i < 6 && i%2 == 1 && out.TemporalID != 1 {
			t.Fatalf("frame %d temporal id %d, want 1", i, out.TemporalID)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
		r := enc.Reconstruction()
		recons = append(recons, snap{
			y: append([]byte(nil), r.Y...),
			u: append([]byte(nil), r.U...),
			v: append([]byte(nil), r.V...),
		})
	}
	if !keyAt[0] || !keyAt[6] {
		t.Fatalf("keyframes at %v, want frames 0 and 6", keyAt)
	}
	if enc.QIndex() == 0 {
		t.Fatal("rate controller reported qindex 0")
	}

	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			for y := range h {
				for x := range w {
					if f.Y.Pix[y*f.Y.Stride+x] != recons[i].y[y*w+x] {
						t.Fatalf("frame %d Y mismatch at (%d,%d)", i, x, y)
					}
				}
			}
			for y := range ch {
				for x := range cw {
					if f.U.Pix[y*f.U.Stride+x] != recons[i].u[y*cw+x] || f.V.Pix[y*f.V.Stride+x] != recons[i].v[y*cw+x] {
						t.Fatalf("frame %d chroma mismatch at (%d,%d)", i, x, y)
					}
				}
			}
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}

// TestPublicVideoEncoderHardCutStaysInter pins SVT-AV1 rtc behaviour: a hard
// content cut is coded as an inter frame (SVT forces scene-change detection
// off), so only the first frame is a keyframe.
func TestPublicVideoEncoderHardCutStaysInter(t *testing.T) {
	const w, h = 320, 192
	sceneA := make([]byte, (w+64)*h)
	sceneB := make([]byte, w*h)
	rngA := rand.New(rand.NewSource(21))
	rngB := rand.New(rand.NewSource(22))
	for y := range h {
		for x := 0; x < w+64; x++ {
			sceneA[y*(w+64)+x] = uint8(60 + (x/5+y/7)%60 + rngA.Intn(40))
		}
	}
	for i := range sceneB {
		sceneB[i] = uint8(140 + (i/9)%50 + rngB.Intn(50))
	}
	makeFrame := func(idx int) I420Frame {
		f := I420Frame{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		if idx < 6 {
			off := idx * 6
			for y := range h {
				copy(f.Y[y*w:(y+1)*w], sceneA[y*(w+64)+off:])
			}
		} else {
			copy(f.Y, sceneB)
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 128
		}
		return f
	}
	enc, err := NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, QIndex: 110})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	keys := make([]bool, 9)
	for i := range keys {
		out, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		keys[i] = out.Keyframe
	}
	if !keys[0] {
		t.Fatalf("keys=%v, want first frame key", keys)
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] {
			t.Fatalf("hard cut emitted keyframe at %d: %v", i, keys)
		}
	}
}

func TestPublicKeyframeEncoderRoundTripAnd1080pAllocs(t *testing.T) {
	const w, h = 1920, 1080
	enc, err := NewKeyframeEncoder(w, h, 80)
	if err != nil {
		t.Fatalf("NewKeyframeEncoder: %v", err)
	}
	defer enc.Close()

	frames := [...]I420Frame{
		publicRTCMatrixFrame(w, h, 0),
		publicRTCMatrixFrame(w, h, 1),
		publicRTCMatrixFrame(w, h, 2),
		publicRTCMatrixFrame(w, h, 3),
	}
	out, err := enc.Encode(frames[0])
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !out.Keyframe || out.TemporalID != 0 || len(out.Data) == 0 {
		t.Fatalf("encoded frame key=%v tid=%d len=%d", out.Keyframe, out.TemporalID, len(out.Data))
	}
	publicAssertI420TUDecodesToRecon(t, out.Data, enc.Reconstruction())
	if err := enc.SetQIndex(72); err != nil {
		t.Fatalf("SetQIndex: %v", err)
	}
	if out, err = enc.Encode(frames[1]); err != nil {
		t.Fatalf("Encode after SetQIndex: %v", err)
	} else if !out.Keyframe || len(out.Data) == 0 {
		t.Fatalf("encoded frame after SetQIndex key=%v len=%d", out.Keyframe, len(out.Data))
	}

	var allocErr error
	frameIndex := 0
	sum := 0
	allocs := testing.AllocsPerRun(5, func() {
		out, err := enc.Encode(frames[frameIndex&3])
		frameIndex++
		if err != nil {
			allocErr = err
			return
		}
		recon := enc.Reconstruction()
		if !out.Keyframe || len(out.Data) == 0 || len(recon.Y) == 0 {
			allocErr = fmt.Errorf("empty keyframe output")
			return
		}
		sum += len(out.Data) + len(recon.Y)
	})
	publicRTCEncoderHotPathSink = sum
	if allocErr != nil {
		t.Fatalf("Encode allocation run: %v", allocErr)
	}
	if allocs != 0 {
		t.Fatalf("KeyframeEncoder Encode 1080p allocations=%f want 0", allocs)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := enc.Encode(frames[2]); err == nil {
		t.Fatal("Encode after Close succeeded")
	}
}

func publicAssertI420TUDecodesToRecon(t *testing.T, tu []byte, recon I420Frame) {
	t.Helper()
	dec, err := NewDecoder([][]byte{append([]byte(nil), tu...)})
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("DecodeAll: %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("decoded %d frames, want 1", len(frames))
	}
	got := frames[0]
	for y := 0; y < recon.Height; y++ {
		gotRow := got.Y.Pix[y*got.Y.Stride : y*got.Y.Stride+recon.Width]
		wantRow := recon.Y[y*recon.YStride : y*recon.YStride+recon.Width]
		if !bytes.Equal(gotRow, wantRow) {
			t.Fatalf("decoded Y row %d mismatch", y)
		}
	}
	chromaW, chromaH := recon.Width/2, recon.Height/2
	for y := 0; y < chromaH; y++ {
		gotU := got.U.Pix[y*got.U.Stride : y*got.U.Stride+chromaW]
		gotV := got.V.Pix[y*got.V.Stride : y*got.V.Stride+chromaW]
		wantU := recon.U[y*recon.ChromaStride : y*recon.ChromaStride+chromaW]
		wantV := recon.V[y*recon.ChromaStride : y*recon.ChromaStride+chromaW]
		if !bytes.Equal(gotU, wantU) || !bytes.Equal(gotV, wantV) {
			t.Fatalf("decoded chroma row %d mismatch", y)
		}
	}
}

func TestPublicEncodeI400KeyframeNativeMonochrome(t *testing.T) {
	const w, h, stride = 96, 64, 112
	src := I400Frame{
		Y:       make([]byte, stride*h),
		YStride: stride,
		Width:   w,
		Height:  h,
	}
	for y := range h {
		for x := range w {
			src.Y[y*stride+x] = uint8((64 + x*3 + y*5 + (x*y)%17) & 0xff)
		}
	}

	for _, tc := range []struct {
		name     string
		qIndex   uint8
		lossless bool
	}{
		{name: "lossless", qIndex: 0, lossless: true},
		{name: "lossy", qIndex: 96},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tu, recon, err := EncodeI400Keyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI400Keyframe: %v", err)
			}
			if len(tu) == 0 {
				t.Fatal("empty temporal unit")
			}
			seq := publicFirstSequenceHeader(t, tu)
			if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != 8 {
				t.Fatalf("sequence color=%+v, want native 8-bit monochrome", seq.ColorConfig)
			}
			if recon.Width != w || recon.Height != h || recon.YStride < w {
				t.Fatalf("recon geometry=%dx%d stride=%d", recon.Width, recon.Height, recon.YStride)
			}
			if tc.lossless {
				for y := range h {
					got := recon.Y[y*recon.YStride : y*recon.YStride+w]
					want := src.Y[y*src.YStride : y*src.YStride+w]
					if !bytes.Equal(got, want) {
						t.Fatalf("lossless recon row %d differs", y)
					}
				}
			}

			dec, err := NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("DecodeAll: %v", err)
			}
			if len(frames) != 1 {
				t.Fatalf("decoded %d frames, want 1", len(frames))
			}
			frame := frames[0]
			if !frame.Format.MonoChrome || frame.Format.BitDepth != 8 {
				t.Fatalf("decoded format=%+v, want native 8-bit monochrome", frame.Format)
			}
			if frame.U.Width != 0 || frame.V.Width != 0 || len(frame.U.Pix) != 0 || len(frame.V.Pix) != 0 {
				t.Fatalf("decoded chroma U=%+v V=%+v, want absent", frame.U, frame.V)
			}
			for y := range h {
				got := frame.Y.Pix[y*frame.Y.Stride : y*frame.Y.Stride+w]
				want := recon.Y[y*recon.YStride : y*recon.YStride+w]
				if !bytes.Equal(got, want) {
					t.Fatalf("decoded luma row %d differs from reconstruction", y)
				}
			}
		})
	}
}

func TestPublicEncodeI400PFrameNativeMonochrome(t *testing.T) {
	const w, h, stride = 128, 96, 144
	src1 := I400Frame{
		Y:       make([]byte, stride*h),
		YStride: stride,
		Width:   w,
		Height:  h,
	}
	for y := range h {
		for x := range w {
			src1.Y[y*stride+x] = uint8((72 + x*2 + y*3 + (x*y)%19) & 0xff)
		}
	}
	src2 := I400Frame{
		Y:       make([]byte, stride*h),
		YStride: stride,
		Width:   w,
		Height:  h,
	}
	const shiftX, shiftY = 4, 2
	for y := range h {
		for x := range w {
			sx, sy := x-shiftX, y-shiftY
			if sx < 0 {
				sx = 0
			}
			if sy < 0 {
				sy = 0
			}
			src2.Y[y*stride+x] = uint8(min(255, int(src1.Y[sy*stride+sx])+1))
		}
	}

	keyTU, keyRecon, err := EncodeI400Keyframe(src1, 80)
	if err != nil {
		t.Fatalf("EncodeI400Keyframe: %v", err)
	}
	seq := publicFirstSequenceHeader(t, keyTU)
	if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != 8 {
		t.Fatalf("sequence color=%+v, want native 8-bit monochrome", seq.ColorConfig)
	}

	for _, tc := range []struct {
		name   string
		qIndex uint8
	}{
		{name: "lossless", qIndex: 0},
		{name: "lossy", qIndex: 80},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pTU, pRecon, err := EncodeI400PFrame(src2, keyRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI400PFrame: %v", err)
			}
			if len(pTU) == 0 {
				t.Fatal("empty P temporal unit")
			}
			if tc.qIndex == 0 && !bytes.Equal(pRecon.Y, src2.Y) {
				t.Fatal("lossless P-frame reconstruction differs from source")
			}
			dec, err := NewDecoder([][]byte{keyTU, pTU})
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("DecodeAll: %v", err)
			}
			if len(frames) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(frames))
			}
			for i, frame := range frames {
				if !frame.Format.MonoChrome || frame.Format.BitDepth != 8 {
					t.Fatalf("decoded frame %d format=%+v, want native 8-bit monochrome", i, frame.Format)
				}
				if frame.U.Width != 0 || frame.V.Width != 0 || len(frame.U.Pix) != 0 || len(frame.V.Pix) != 0 {
					t.Fatalf("decoded frame %d chroma U=%+v V=%+v, want absent", i, frame.U, frame.V)
				}
			}
			for y := range h {
				got := frames[1].Y.Pix[y*frames[1].Y.Stride : y*frames[1].Y.Stride+w]
				want := pRecon.Y[y*pRecon.YStride : y*pRecon.YStride+w]
				if !bytes.Equal(got, want) {
					t.Fatalf("decoded P-frame luma row %d differs from reconstruction", y)
				}
			}
		})
	}
}

func TestPublicEncodeI400KeyframeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 128, 72
	src := I400Frame{
		Y:       make([]byte, w*h),
		YStride: w,
		Width:   w,
		Height:  h,
	}
	for y := range h {
		for x := range w {
			src.Y[y*w+x] = uint8((80 + x*2 + y*3 + (x^y)%23) & 0xff)
		}
	}
	for _, qIndex := range []uint8{0, 88} {
		t.Run(fmt.Sprintf("q%d", qIndex), func(t *testing.T) {
			tu, _, err := EncodeI400Keyframe(src, qIndex)
			if err != nil {
				t.Fatalf("EncodeI400Keyframe: %v", err)
			}
			ivf := appendPublicIVF(nil, w, h, 30, 1, []publicIVFFrame{{payload: tu}})
			assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, fmt.Sprintf("public-i400-keyframe-q%d", qIndex), ivf)
		})
	}
}

func TestPublicEncodeI400HighBitDepthLosslessKeyframeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 64, 64
	for _, bitDepth := range []uint8{10, 12} {
		t.Run(fmt.Sprintf("%dbit", bitDepth), func(t *testing.T) {
			maxSample := uint16((1 << bitDepth) - 1)
			src := I400HighBitDepthFrame{
				Y:        make([]uint16, w*h),
				YStride:  w,
				Width:    w,
				Height:   h,
				BitDepth: bitDepth,
			}
			for y := range h {
				for x := range w {
					src.Y[y*w+x] = uint16((37 + x*19 + y*23 + (x*y)%97) & int(maxSample))
				}
			}

			tu, recon, err := EncodeI400HighBitDepthLosslessKeyframe(src)
			if err != nil {
				t.Fatalf("EncodeI400HighBitDepthLosslessKeyframe: %v", err)
			}
			if recon.BitDepth != src.BitDepth || recon.Width != src.Width || recon.Height != src.Height ||
				recon.YStride != src.YStride || !publicUint16Equal(recon.Y, src.Y) {
				t.Fatalf("reconstruction mismatch: got=%+v src=%+v", recon, src)
			}
			seq := publicFirstSequenceHeader(t, tu)
			wantProfile := uint8(0)
			if bitDepth == 12 {
				wantProfile = 2
			}
			if seq.SeqProfile != wantProfile ||
				!seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != bitDepth ||
				!seq.ColorConfig.HighBitdepth ||
				(bitDepth == 12 && !seq.ColorConfig.TwelveBit) {
				t.Fatalf("sequence profile=%d color=%+v want profile=%d native %d-bit monochrome", seq.SeqProfile, seq.ColorConfig, wantProfile, bitDepth)
			}
			ivf := appendPublicIVF(nil, w, h, 30, 1, []publicIVFFrame{{payload: tu}})
			wantY := appendPublicI400HighBitDepthRaw(nil, src)
			assertPublicIVFHighBitDepthMonochromeMatchesReferenceDecoders(t, decoders, fmt.Sprintf("public-i400-%dbit-lossless", bitDepth), ivf, wantY, w, h, 2)
		})
	}
}

func TestPublicEncodeI400HighBitDepthKeyframeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 64, 64
	cases := []struct {
		bitDepth uint8
		qIndex   uint8
	}{
		{bitDepth: 10, qIndex: 0},
		{bitDepth: 10, qIndex: 72},
		{bitDepth: 12, qIndex: 104},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dbit-q%d", tc.bitDepth, tc.qIndex), func(t *testing.T) {
			maxSample := uint16((1 << tc.bitDepth) - 1)
			src := I400HighBitDepthFrame{
				Y:        make([]uint16, w*h),
				YStride:  w,
				Width:    w,
				Height:   h,
				BitDepth: tc.bitDepth,
			}
			for y := range h {
				for x := range w {
					src.Y[y*w+x] = uint16((73 + x*29 + y*17 + (x*y)%211) & int(maxSample))
				}
			}

			tu, recon, err := EncodeI400HighBitDepthKeyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI400HighBitDepthKeyframe: %v", err)
			}
			if recon.BitDepth != src.BitDepth || recon.Width != src.Width || recon.Height != src.Height ||
				recon.YStride != src.YStride {
				t.Fatalf("unexpected reconstruction: got=%+v src=%+v", recon, src)
			}
			if tc.qIndex == 0 && !publicUint16Equal(recon.Y, src.Y) {
				t.Fatalf("lossless reconstruction mismatch: got=%+v src=%+v", recon, src)
			}
			if tc.qIndex != 0 && publicUint16Equal(recon.Y, src.Y) {
				t.Fatalf("lossy reconstruction unexpectedly equals source: got=%+v src=%+v", recon, src)
			}
			seq := publicFirstSequenceHeader(t, tu)
			wantProfile := uint8(0)
			if tc.bitDepth == 12 {
				wantProfile = 2
			}
			if seq.SeqProfile != wantProfile ||
				!seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != tc.bitDepth ||
				!seq.ColorConfig.HighBitdepth ||
				(tc.bitDepth == 12 && !seq.ColorConfig.TwelveBit) {
				t.Fatalf("sequence profile=%d color=%+v want profile=%d native %d-bit monochrome", seq.SeqProfile, seq.ColorConfig, wantProfile, tc.bitDepth)
			}
			ivf := appendPublicIVF(nil, w, h, 30, 1, []publicIVFFrame{{payload: tu}})
			wantY := appendPublicI400HighBitDepthRaw(nil, recon)
			assertPublicIVFHighBitDepthMonochromeMatchesReferenceDecoders(t, decoders, fmt.Sprintf("public-i400-%dbit-q%d", tc.bitDepth, tc.qIndex), ivf, wantY, w, h, 2)
		})
	}
}

func TestPublicEncodeI420HighBitDepthKeyframeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 64, 64
	cases := []struct {
		bitDepth uint8
		qIndex   uint8
		profile  uint8
	}{
		{bitDepth: 10, qIndex: 0, profile: uint8(EncoderProfile0)},
		{bitDepth: 10, qIndex: 72, profile: uint8(EncoderProfile0)},
		{bitDepth: 12, qIndex: 0, profile: uint8(internalencoder.Profile2)},
		{bitDepth: 12, qIndex: 104, profile: uint8(internalencoder.Profile2)},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dbit-q%d", tc.bitDepth, tc.qIndex), func(t *testing.T) {
			src := publicI420HighBitDepthFrame(w, h, tc.bitDepth, int(tc.qIndex))
			tu, recon, err := EncodeI420HighBitDepthKeyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI420HighBitDepthKeyframe: %v", err)
			}
			if recon.BitDepth != src.BitDepth || recon.Width != src.Width || recon.Height != src.Height ||
				recon.YStride != src.YStride || recon.ChromaStride != src.ChromaStride {
				t.Fatalf("unexpected reconstruction: got=%+v src=%+v", recon, src)
			}
			if tc.qIndex == 0 && !publicI420HighBitDepthFrameEqual(recon, src) {
				t.Fatal("lossless high-bit-depth 4:2:0 reconstruction differs from source")
			}
			seq := publicFirstSequenceHeader(t, tu)
			if seq.SeqProfile != tc.profile ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != tc.bitDepth ||
				!seq.ColorConfig.HighBitdepth ||
				!seq.ColorConfig.SubsamplingX ||
				!seq.ColorConfig.SubsamplingY ||
				(tc.bitDepth == 12 && !seq.ColorConfig.TwelveBit) {
				t.Fatalf("sequence profile=%d color=%+v want profile=%d native %d-bit 4:2:0", seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth)
			}

			ivf := appendPublicIVF(nil, w, h, 30, 1, []publicIVFFrame{{payload: tu}})
			decoded, err := DecodeIVF(ivf)
			if err != nil {
				t.Fatalf("DecodeIVF: %v", err)
			}
			if len(decoded) != 1 ||
				decoded[0].Width != w ||
				decoded[0].Height != h ||
				decoded[0].ChromaWidth != w/2 ||
				decoded[0].ChromaHeight != h/2 ||
				decoded[0].BytesPerSample != 2 {
				t.Fatalf("decoded frame=%+v want one %dx%d 2-byte 4:2:0 frame", decoded, w, h)
			}
			wantYUV := appendPublicI420HighBitDepthRaw(nil, recon)
			if got := publicDecodedFramesRawYUV(decoded); !bytes.Equal(got, wantYUV) {
				offset := firstPublicByteDiff(got, wantYUV)
				t.Fatalf("local decoded high-bit-depth 4:2:0 differs first at byte %d", offset)
			}
			assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, fmt.Sprintf("public-i420-%dbit-q%d", tc.bitDepth, tc.qIndex), ivf, wantYUV, 1)
		})
	}
}

func TestPublicEncodeI400HighBitDepthPFrameReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 64, 64
	cases := []struct {
		bitDepth uint8
		qIndex   uint8
	}{
		{bitDepth: 10, qIndex: 0},
		{bitDepth: 10, qIndex: 32},
		{bitDepth: 12, qIndex: 0},
		{bitDepth: 12, qIndex: 48},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dbit-q%d", tc.bitDepth, tc.qIndex), func(t *testing.T) {
			maxSample := uint16((1 << tc.bitDepth) - 1)
			src1 := I400HighBitDepthFrame{
				Y:        make([]uint16, w*h),
				YStride:  w,
				Width:    w,
				Height:   h,
				BitDepth: tc.bitDepth,
			}
			for y := range h {
				for x := range w {
					src1.Y[y*w+x] = uint16((91 + x*17 + y*29 + (x*y)%181) & int(maxSample))
				}
			}
			keyTU, keyRecon, err := EncodeI400HighBitDepthKeyframe(src1, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI400HighBitDepthKeyframe: %v", err)
			}
			src2 := I400HighBitDepthFrame{
				Y:        make([]uint16, w*h),
				YStride:  w,
				Width:    w,
				Height:   h,
				BitDepth: tc.bitDepth,
			}
			for y := range h {
				for x := range w {
					base := int(keyRecon.Y[y*keyRecon.YStride+x])
					delta := ((x*3 + 2*y) % 65) - 32
					v := base + delta
					if v < 0 {
						v = 0
					} else if v > int(maxSample) {
						v = int(maxSample)
					}
					src2.Y[y*w+x] = uint16(v)
				}
			}

			pTU, pRecon, err := EncodeI400HighBitDepthPFrame(src2, keyRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI400HighBitDepthPFrame: %v", err)
			}
			if publicI400HighBitDepthFrameEqual(pRecon, keyRecon) {
				t.Fatal("P-frame reconstruction unexpectedly equals reference")
			}
			if tc.qIndex == 0 && !publicI400HighBitDepthFrameEqual(pRecon, src2) {
				t.Fatal("lossless high-bit-depth monochrome P-frame reconstruction differs from source")
			}
			ivf := appendPublicIVF(nil, w, h, 30, 1, []publicIVFFrame{
				{payload: keyTU},
				{timestamp: 1, payload: pTU},
			})
			wantY := appendPublicI400HighBitDepthRaw(nil, keyRecon)
			wantY = appendPublicI400HighBitDepthRaw(wantY, pRecon)
			assertPublicIVFHighBitDepthMonochromeMatchesReferenceDecoders(t, decoders, fmt.Sprintf("public-i400-%dbit-pframe-q%d", tc.bitDepth, tc.qIndex), ivf, wantY, w, h, 2)
		})
	}
}

func TestPublicEncodeI420HighBitDepthPFrameReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 64, 64
	cases := []struct {
		bitDepth uint8
		qIndex   uint8
	}{
		{bitDepth: 10, qIndex: 0},
		{bitDepth: 10, qIndex: 32},
		{bitDepth: 12, qIndex: 0},
		{bitDepth: 12, qIndex: 48},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dbit-q%d", tc.bitDepth, tc.qIndex), func(t *testing.T) {
			maxSample := uint16((1 << tc.bitDepth) - 1)
			src1 := publicI420HighBitDepthFrame(w, h, tc.bitDepth, int(tc.qIndex))
			keyTU, keyRecon, err := EncodeI420HighBitDepthKeyframe(src1, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI420HighBitDepthKeyframe: %v", err)
			}
			src2 := I420HighBitDepthFrame{
				Y:            make([]uint16, w*h),
				U:            make([]uint16, (w/2)*(h/2)),
				V:            make([]uint16, (w/2)*(h/2)),
				YStride:      w,
				ChromaStride: w / 2,
				Width:        w,
				Height:       h,
				BitDepth:     tc.bitDepth,
			}
			for y := range h {
				for x := range w {
					base := int(keyRecon.Y[y*keyRecon.YStride+x])
					src2.Y[y*w+x] = publicClampUint16(base+((x*5+y*3)%81-40), maxSample)
				}
			}
			for y := range h / 2 {
				for x := range w / 2 {
					off := y*src2.ChromaStride + x
					src2.U[off] = publicClampUint16(int(keyRecon.U[y*keyRecon.ChromaStride+x])+((x*7+y*5)%53-26), maxSample)
					src2.V[off] = publicClampUint16(int(keyRecon.V[y*keyRecon.ChromaStride+x])+((x*3+y*11)%59-29), maxSample)
				}
			}

			pTU, pRecon, err := EncodeI420HighBitDepthPFrame(src2, keyRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("EncodeI420HighBitDepthPFrame: %v", err)
			}
			if publicI420HighBitDepthFrameEqual(pRecon, keyRecon) {
				t.Fatal("P-frame reconstruction unexpectedly equals reference")
			}
			if tc.qIndex == 0 && !publicI420HighBitDepthFrameEqual(pRecon, src2) {
				t.Fatal("lossless high-bit-depth 4:2:0 P-frame reconstruction differs from source")
			}
			ivf := appendPublicIVF(nil, w, h, 30, 1, []publicIVFFrame{
				{payload: keyTU},
				{timestamp: 1, payload: pTU},
			})
			dec, err := NewDecoder([][]byte{keyTU, pTU})
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			defer dec.Close()
			decoded, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("DecodeAll: %v", err)
			}
			if len(decoded) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(decoded))
			}
			wantYUV := appendPublicI420HighBitDepthRaw(nil, keyRecon)
			wantYUV = appendPublicI420HighBitDepthRaw(wantYUV, pRecon)
			var gotYUV []byte
			for _, frame := range decoded {
				gotYUV = appendPublicFrameRawYUV(gotYUV, frame)
			}
			if !bytes.Equal(gotYUV, wantYUV) {
				offset := firstPublicByteDiff(gotYUV, wantYUV)
				t.Fatalf("local decoded high-bit-depth 4:2:0 key+P differs first at byte %d", offset)
			}
			assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, fmt.Sprintf("public-i420-%dbit-pframe-q%d", tc.bitDepth, tc.qIndex), ivf, wantYUV, 2)
		})
	}
}

func TestPublicEncodeI400HighBitDepthLosslessKeyframeRejectsInvalid(t *testing.T) {
	valid := I400HighBitDepthFrame{
		Y:        make([]uint16, 64*64),
		YStride:  64,
		Width:    64,
		Height:   64,
		BitDepth: 10,
	}
	if _, _, err := EncodeI400HighBitDepthLosslessKeyframe(I400HighBitDepthFrame{
		Y:        valid.Y,
		YStride:  valid.YStride,
		Width:    valid.Width,
		Height:   valid.Height,
		BitDepth: 9,
	}); err == nil {
		t.Fatal("EncodeI400HighBitDepthLosslessKeyframe accepted 9-bit input")
	}
	invalidSample := valid
	invalidSample.Y = append([]uint16(nil), valid.Y...)
	invalidSample.Y[17] = 1024
	if _, _, err := EncodeI400HighBitDepthLosslessKeyframe(invalidSample); err == nil {
		t.Fatal("EncodeI400HighBitDepthLosslessKeyframe accepted sample above 10-bit range")
	}
}

func TestPublicEncodeI420HighBitDepthKeyframeRejectsInvalid(t *testing.T) {
	valid := publicI420HighBitDepthFrame(64, 64, 10, 3)
	if _, _, err := EncodeI420HighBitDepthLosslessKeyframe(valid); err != nil {
		t.Fatalf("EncodeI420HighBitDepthLosslessKeyframe valid: %v", err)
	}
	bitDepthMismatch := valid
	bitDepthMismatch.BitDepth = 9
	if _, _, err := EncodeI420HighBitDepthLosslessKeyframe(bitDepthMismatch); err == nil {
		t.Fatal("EncodeI420HighBitDepthLosslessKeyframe accepted 9-bit input")
	}
	if _, _, err := EncodeI420HighBitDepthKeyframe(bitDepthMismatch, 72); err == nil {
		t.Fatal("EncodeI420HighBitDepthKeyframe accepted 9-bit input")
	}
	shortChroma := valid
	shortChroma.U = shortChroma.U[:len(shortChroma.U)-1]
	if _, _, err := EncodeI420HighBitDepthLosslessKeyframe(shortChroma); err == nil {
		t.Fatal("EncodeI420HighBitDepthLosslessKeyframe accepted short chroma plane")
	}
	if _, _, err := EncodeI420HighBitDepthKeyframe(shortChroma, 72); err == nil {
		t.Fatal("EncodeI420HighBitDepthKeyframe accepted short chroma plane")
	}
	invalidSample := valid
	invalidSample.V = append([]uint16(nil), valid.V...)
	invalidSample.V[7] = 1024
	if _, _, err := EncodeI420HighBitDepthLosslessKeyframe(invalidSample); err == nil {
		t.Fatal("EncodeI420HighBitDepthLosslessKeyframe accepted sample above 10-bit range")
	}
	if _, _, err := EncodeI420HighBitDepthKeyframe(invalidSample, 72); err == nil {
		t.Fatal("EncodeI420HighBitDepthKeyframe accepted sample above 10-bit range")
	}
}

func TestPublicEncodeI400HighBitDepthPFrameRejectsInvalid(t *testing.T) {
	valid := I400HighBitDepthFrame{
		Y:        make([]uint16, 64*64),
		YStride:  64,
		Width:    64,
		Height:   64,
		BitDepth: 10,
	}
	bitDepthMismatch := valid
	bitDepthMismatch.BitDepth = 12
	if _, _, err := EncodeI400HighBitDepthPFrame(valid, bitDepthMismatch, 72); err == nil {
		t.Fatal("EncodeI400HighBitDepthPFrame accepted mismatched bit depth")
	}
	dimMismatch := valid
	dimMismatch.Width = 32
	if _, _, err := EncodeI400HighBitDepthPFrame(valid, dimMismatch, 72); err == nil {
		t.Fatal("EncodeI400HighBitDepthPFrame accepted mismatched dimensions")
	}
}

func TestPublicEncodeI420HighBitDepthPFrameRejectsInvalid(t *testing.T) {
	valid := publicI420HighBitDepthFrame(64, 64, 10, 3)
	bitDepthMismatch := valid
	bitDepthMismatch.BitDepth = 12
	if _, _, err := EncodeI420HighBitDepthPFrame(valid, bitDepthMismatch, 72); err == nil {
		t.Fatal("EncodeI420HighBitDepthPFrame accepted mismatched bit depth")
	}
	dimMismatch := valid
	dimMismatch.Width = 32
	if _, _, err := EncodeI420HighBitDepthPFrame(valid, dimMismatch, 72); err == nil {
		t.Fatal("EncodeI420HighBitDepthPFrame accepted mismatched dimensions")
	}
	shortRef := valid
	shortRef.V = shortRef.V[:len(shortRef.V)-1]
	if _, _, err := EncodeI420HighBitDepthPFrame(valid, shortRef, 72); err == nil {
		t.Fatal("EncodeI420HighBitDepthPFrame accepted short reference chroma")
	}
}

func TestPublicVideoEncoderClose(t *testing.T) {
	const w, h = 192, 128
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 300_000, Framerate: 30,
		TileColumns: 2,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); err != nil {
		t.Fatalf("Encode before close: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 1), false); err == nil {
		t.Fatal("Encode after Close succeeded")
	}
}

// TestPublicRTCEncoder checks the WebRTC surface: every frame carries a
// dependency descriptor and the temporal units decode.
func TestPublicRTCEncoder(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	for i := range 6 {
		f := I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for j := range f.Y {
			f.Y[j] = uint8(40 + (j+i*7)%150)
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if len(out.DependencyDescriptor) == 0 {
			t.Fatalf("frame %d missing dependency descriptor", i)
		}
		if i == 0 && !out.Keyframe {
			t.Fatal("first frame not a keyframe")
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}
	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	n := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		n += len(batch)
	}
	if n != 6 {
		t.Fatalf("decoded %d frames, want 6", n)
	}
}

func TestPublicVideoEncoderInputFormatsMatchI420(t *testing.T) {
	const w, h = 192, 128
	cfg := VideoEncoderConfig{
		Width: w, Height: h,
		QIndex:         80,
		TemporalLayers: 2,
	}
	formats := []struct {
		name   string
		source func(int) I420Frame
		encode func(*VideoEncoder, I420Frame, bool) (EncodedFrame, error)
	}{
		{
			name: "I422",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				return enc.EncodeI422(publicI422FromI420(src), forceKey)
			},
		},
		{
			name: "I444",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				return enc.EncodeI444(publicI444FromI420(src), forceKey)
			},
		},
		{
			name: "I400",
			source: func(frame int) I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(w, h, frame))
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				return enc.EncodeI400(publicI400FromI420(src), forceKey)
			},
		},
		{
			name: "Frame420-8bit",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     8,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
		},
		{
			name: "Frame420-10bit",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
		},
		{
			name: "Frame422-10bit",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
		},
		{
			name: "Frame444-12bit",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:    src.Width,
					Height:   src.Height,
					BitDepth: 12,
					Align:    32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
		},
		{
			name: "Frame400-10bit",
			source: func(frame int) I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(w, h, frame))
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:      src.Width,
					Height:     src.Height,
					BitDepth:   10,
					MonoChrome: true,
					Align:      32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
		},
		{
			name: "NV12",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				return enc.EncodeNV12(publicNV12FromI420(src), forceKey)
			},
		},
		{
			name: "NV21",
			source: func(frame int) I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *VideoEncoder, src I420Frame, forceKey bool) (EncodedFrame, error) {
				return enc.EncodeNV21(publicNV21FromI420(src), forceKey)
			},
		},
	}
	for _, format := range formats {
		t.Run(format.name, func(t *testing.T) {
			i420Enc, err := NewVideoEncoder(cfg)
			if err != nil {
				t.Fatalf("NewVideoEncoder I420: %v", err)
			}
			defer i420Enc.Close()
			testEnc, err := NewVideoEncoder(cfg)
			if err != nil {
				t.Fatalf("NewVideoEncoder %s: %v", format.name, err)
			}
			defer testEnc.Close()

			for frame := 0; frame < 3; frame++ {
				i420 := format.source(frame)
				want, err := i420Enc.Encode(i420, false)
				if err != nil {
					t.Fatalf("I420 Encode(%d): %v", frame, err)
				}
				got, err := format.encode(testEnc, i420, false)
				if err != nil {
					t.Fatalf("%s Encode(%d): %v", format.name, frame, err)
				}
				if got.Keyframe != want.Keyframe || got.TemporalID != want.TemporalID || !bytes.Equal(got.Data, want.Data) {
					t.Fatalf("%s frame %d differs: got key=%v T%d %dB want key=%v T%d %dB", format.name, frame, got.Keyframe, got.TemporalID, len(got.Data), want.Keyframe, want.TemporalID, len(want.Data))
				}
			}
		})
	}
}

func TestPublicVideoEncoderRejectsUnsupportedFrameFormat(t *testing.T) {
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: 16, Height: 16,
		QIndex: 80,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer enc.Close()

	format := FrameFormat{
		Width:        16,
		Height:       16,
		BitDepth:     8,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatalf("FrameRequiredSize: %v", err)
	}
	frame, err := internalframe.Bind(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatalf("BindFrame: %v", err)
	}
	if _, err := enc.EncodeFrame(frame, false); !errors.Is(err, internalframe.ErrInvalidFormat) {
		t.Fatalf("EncodeFrame err=%v want %v", err, internalframe.ErrInvalidFormat)
	}
}

func TestPublicRTCEncoderInputFormatsEncodeAndPictureDecode(t *testing.T) {
	var referenceDecoders []publicReferenceAV1Decoder
	if os.Getenv("GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS") == "1" {
		referenceDecoders = publicReferenceAV1Decoders(t)
	}
	formats := []struct {
		name          string
		source        func(int, int, int) I420Frame
		encode        func(*RTCEncoder, I420Frame, bool) (RTCFrame, error)
		encodePicture func(*RTCEncoder, I420Frame, bool) (RTCPicture, error)
	}{
		{
			name: "I422",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				return enc.EncodeI422(publicI422FromI420(src), forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				return enc.EncodeI422Picture(publicI422FromI420(src), forceKey)
			},
		},
		{
			name: "I444",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				return enc.EncodeI444(publicI444FromI420(src), forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				return enc.EncodeI444Picture(publicI444FromI420(src), forceKey)
			},
		},
		{
			name: "I400",
			source: func(width int, height int, frame int) I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(width, height, frame))
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				return enc.EncodeI400(publicI400FromI420(src), forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				return enc.EncodeI400Picture(publicI400FromI420(src), forceKey)
			},
		},
		{
			name: "Frame420-8bit",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     8,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     8,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFramePicture(frame, forceKey)
			},
		},
		{
			name: "Frame420-10bit",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFramePicture(frame, forceKey)
			},
		},
		{
			name: "Frame422-10bit",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					Align:        32,
				})
				return enc.EncodeFramePicture(frame, forceKey)
			},
		},
		{
			name: "Frame444-12bit",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:    src.Width,
					Height:   src.Height,
					BitDepth: 12,
					Align:    32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:    src.Width,
					Height:   src.Height,
					BitDepth: 12,
					Align:    32,
				})
				return enc.EncodeFramePicture(frame, forceKey)
			},
		},
		{
			name: "Frame400-10bit",
			source: func(width int, height int, frame int) I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(width, height, frame))
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:      src.Width,
					Height:     src.Height,
					BitDepth:   10,
					MonoChrome: true,
					Align:      32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				frame := publicFrameFromI420(t, src, FrameFormat{
					Width:      src.Width,
					Height:     src.Height,
					BitDepth:   10,
					MonoChrome: true,
					Align:      32,
				})
				return enc.EncodeFramePicture(frame, forceKey)
			},
		},
		{
			name: "NV12",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				return enc.EncodeNV12(publicNV12FromI420(src), forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				return enc.EncodeNV12Picture(publicNV12FromI420(src), forceKey)
			},
		},
		{
			name: "NV21",
			source: func(width int, height int, frame int) I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCFrame, error) {
				return enc.EncodeNV21(publicNV21FromI420(src), forceKey)
			},
			encodePicture: func(enc *RTCEncoder, src I420Frame, forceKey bool) (RTCPicture, error) {
				return enc.EncodeNV21Picture(publicNV21FromI420(src), forceKey)
			},
		},
	}
	for _, format := range formats {
		t.Run(format.name+"/single-spatial", func(t *testing.T) {
			const w, h = 192, 128
			enc, err := NewRTCEncoder(VideoEncoderConfig{
				Width: w, Height: h,
				TargetBitrate: 250_000, Framerate: 30,
				TemporalLayers: 2,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer enc.Close()
			var tus [][]byte
			for frame := 0; frame < 3; frame++ {
				out, err := format.encode(enc, format.source(w, h, frame), false)
				if err != nil {
					t.Fatalf("%s Encode(%d): %v", format.name, frame, err)
				}
				if frame == 0 && (!out.Keyframe || len(out.DependencyDescriptor) == 0) {
					t.Fatalf("first %s RTC frame=%+v", format.name, out)
				}
				tus = append(tus, append([]byte(nil), out.Data...))
			}
			dec, err := NewDecoder(tus)
			if err != nil {
				t.Fatalf("NewDecoder: %v", err)
			}
			defer dec.Close()
			decoded := 0
			for {
				batch, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatalf("DecodeNext: %v", err)
				}
				if !ok {
					break
				}
				decoded += len(batch)
			}
			if decoded != len(tus) {
				t.Fatalf("decoded %d frames want %d", decoded, len(tus))
			}
		})

		t.Run(format.name+"/multi-spatial-rtp", func(t *testing.T) {
			const w, h = 640, 360
			cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeS2T2)
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 34
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 2; frame++ {
				picture, err := format.encodePicture(enc, format.source(w, h, frame), false)
				if err != nil {
					t.Fatalf("%s EncodePicture(%d): %v", format.name, frame, err)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0, &nextFrameID)
			}
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			if len(referenceDecoders) > 0 {
				assertPublicRTCRTPReferenceDecoders(t, referenceDecoders, "input-format-"+format.name, enc.Config(), layerTUs, orderedTUs)
			}
		})
	}
}

func TestPublicRTCEncoderI422NativeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	cfg.Profile = internalencoder.Profile2
	cfg.ColorConfigSet = true
	cfg.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
	}
	cfg.RateControl = EncoderRateControlCQP
	cfg.Quantizer = 28
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()
	if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); !errors.Is(err, internalencoder.ErrUnsupported) {
		t.Fatalf("Encode(I420) on native I422 err=%v want %v", err, internalencoder.ErrUnsupported)
	}

	frames := make([]publicIVFFrame, 0, 5)
	for frame := 0; frame < 4; frame++ {
		out, err := enc.EncodeI422(publicI422NativePattern(w, h, frame), false)
		if err != nil {
			t.Fatalf("EncodeI422(%d): %v", frame, err)
		}
		if frame == 0 {
			seq := publicFirstSequenceHeader(t, out.Data)
			if seq.SeqProfile != uint8(internalencoder.Profile2) ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != 8 ||
				!seq.ColorConfig.SubsamplingX ||
				seq.ColorConfig.SubsamplingY {
				t.Fatalf("sequence profile=%d color=%+v want native 8-bit 4:2:2", seq.SeqProfile, seq.ColorConfig)
			}
		}
		frames = append(frames, publicIVFFrame{
			timestamp: uint64(frame),
			payload:   append([]byte(nil), out.Data...),
		})
	}
	generic, err := enc.EncodeFrame(publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 4), FrameFormat{
		Width:        w,
		Height:       h,
		BitDepth:     8,
		SubsamplingX: true,
		Align:        32,
	}), false)
	if err != nil {
		t.Fatalf("EncodeFrame native I422: %v", err)
	}
	frames = append(frames, publicIVFFrame{
		timestamp: 4,
		payload:   append([]byte(nil), generic.Data...),
	})

	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	decoded, err := DecodeIVF(ivf)
	if err != nil {
		t.Fatalf("DecodeIVF: %v", err)
	}
	if len(decoded) != len(frames) {
		t.Fatalf("decoded %d frames want %d", len(decoded), len(frames))
	}
	for i, frame := range decoded {
		if frame.ChromaWidth != w/2 || frame.ChromaHeight != h || len(frame.U) != (w/2)*h || len(frame.V) != (w/2)*h {
			t.Fatalf("decoded frame %d chroma=%dx%d U=%d V=%d want %dx%d",
				i, frame.ChromaWidth, frame.ChromaHeight, len(frame.U), len(frame.V), w/2, h)
		}
	}
	assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, "public-rtc-i422-native", ivf)
}

func TestPublicRTCEncoderI422NativeMultiSpatialReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2},
		{name: "shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			w, h := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(w, h, scenario.mode)
			cfg.Profile = internalencoder.Profile2
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 32
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     8,
				SubsamplingX: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if enc.Config().Profile != internalencoder.Profile2 ||
				enc.Config().ColorConfig.MonoChrome ||
				!enc.Config().ColorConfig.SubsamplingX ||
				enc.Config().ColorConfig.SubsamplingY ||
				enc.Config().SpatialLayerCount < 2 {
				t.Fatalf("normalized config=%+v want native multi-spatial 4:2:2", enc.Config())
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI422Picture(publicI422NativePattern(w, h, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI422Picture(%d): %v", frame, err)
				}
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					if !picture.Frames[i].CodedKeyframe {
						continue
					}
					seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
					if seq.SeqProfile != uint8(internalencoder.Profile2) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != 8 ||
						!seq.ColorConfig.SubsamplingX ||
						seq.ColorConfig.SubsamplingY {
						t.Fatalf("picture %d frame %d profile=%d color=%+v want native 8-bit 4:2:2",
							frame, i, seq.SeqProfile, seq.ColorConfig)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i422-native-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI422NativeControlChurnReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	mode := EncoderScalabilityModeL2T3_KEY_SHIFT
	width, height := publicRTCMatrixGeometry(t, mode)
	baseKbps := publicRTCMatrixControlBitrateKbps(t, mode)
	fpsCycle := []EncoderRational{
		{Num: 15, Den: 1},
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 60, Den: 1},
	}
	configForStep := func(step int) EncoderConfig {
		cfg := publicRTCMatrixConfig(width, height, mode)
		cfg.Profile = internalencoder.Profile2
		cfg.ColorConfigSet = true
		cfg.ColorConfig = internalencoder.SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
		}
		cfg.MaxFramerate = fpsCycle[step%len(fpsCycle)]
		cfg.MaxThreads = int32(1 + step%3)
		cfg.Content = EncoderContentCamera
		if step%2 == 1 {
			cfg.Content = EncoderContentScreen
		}
		publicRTCApplyControlBitrates(&cfg, baseKbps+int32(step*97))
		if step%2 == 0 {
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = uint8(26 + step)
		} else {
			cfg.RateControl = EncoderRateControlCBR
			cfg.Quantizer = 0
		}
		return cfg
	}

	var enc *RTCEncoder
	var descriptorReceiver RTPDependencyDescriptorState
	var rtpReceiver RTPDependencyDescriptorState
	var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
	var orderedTUs [][]byte
	var sequence SequenceHeader
	nextFrameID := uint64(0)
	for step := 0; step < 6; step++ {
		cfg := configForStep(step)
		forceKey := step == 3
		wantKey := step == 0 || forceKey
		if enc == nil {
			var err error
			enc, err = NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
		} else {
			wantKey = wantKey || publicRTCSetConfigRequiresKey(t, enc.Config(), cfg)
			if err := enc.SetConfig(cfg); err != nil {
				t.Fatalf("step %d SetConfig: %v", step, err)
			}
		}
		assertPublicRTCConfigControls(t, enc.Config(), cfg)
		wantDuration, err := EncoderWebRTCRTPFrameDuration(enc.Config())
		if err != nil {
			t.Fatalf("step %d EncoderWebRTCRTPFrameDuration: %v", step, err)
		}
		if got, err := enc.RTPFrameDuration(); err != nil || got != wantDuration {
			t.Fatalf("step %d RTPFrameDuration=%+v err=%v want %+v", step, got, err, wantDuration)
		}

		picture, err := enc.EncodeI422Picture(publicI422NativePattern(width, height, step), forceKey)
		if err != nil {
			t.Fatalf("step %d EncodeI422Picture: %v", step, err)
		}
		if picture.Keyframe != wantKey {
			t.Fatalf("step %d key=%v want %v picture=%+v", step, picture.Keyframe, wantKey, picture)
		}
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, wantKey, &nextFrameID)
		for i := 0; i < picture.FrameNum; i++ {
			if !picture.Frames[i].CodedKeyframe {
				continue
			}
			assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("native-i422-control-step-%d-S%d", step, picture.Frames[i].SpatialID), picture.Frames[i].Data, &sequence, enc.Config().Content == EncoderContentScreen)
			seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
			if seq.SeqProfile != uint8(internalencoder.Profile2) ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != 8 ||
				!seq.ColorConfig.SubsamplingX ||
				seq.ColorConfig.SubsamplingY {
				t.Fatalf("step %d sequence profile=%d color=%+v want native 8-bit 4:2:2", step, seq.SeqProfile, seq.ColorConfig)
			}
		}
		appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
	}

	normalized := enc.Config()
	assertPublicRTCLayerStreamsDecode(t, normalized, layerTUs, orderedTUs)
	assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i422-native-control-churn", normalized, layerTUs, orderedTUs)
}

func TestPublicRTCEncoderI444NativeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	cfg.Profile = internalencoder.Profile1
	cfg.ColorConfigSet = true
	cfg.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth: 8,
	}
	cfg.RateControl = EncoderRateControlCQP
	cfg.Quantizer = 28
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()
	if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); !errors.Is(err, internalencoder.ErrUnsupported) {
		t.Fatalf("Encode(I420) on native I444 err=%v want %v", err, internalencoder.ErrUnsupported)
	}

	frames := make([]publicIVFFrame, 0, 4)
	for frame := 0; frame < 4; frame++ {
		out, err := enc.EncodeI444(publicI444NativePattern(w, h, frame), false)
		if err != nil {
			t.Fatalf("EncodeI444(%d): %v", frame, err)
		}
		if frame == 0 {
			seq := publicFirstSequenceHeader(t, out.Data)
			if seq.SeqProfile != uint8(internalencoder.Profile1) ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != 8 ||
				seq.ColorConfig.SubsamplingX ||
				seq.ColorConfig.SubsamplingY {
				t.Fatalf("sequence profile=%d color=%+v want native 8-bit 4:4:4", seq.SeqProfile, seq.ColorConfig)
			}
		}
		frames = append(frames, publicIVFFrame{
			timestamp: uint64(frame),
			payload:   append([]byte(nil), out.Data...),
		})
	}

	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	decoded, err := DecodeIVF(ivf)
	if err != nil {
		t.Fatalf("DecodeIVF: %v", err)
	}
	if len(decoded) != len(frames) {
		t.Fatalf("decoded %d frames want %d", len(decoded), len(frames))
	}
	for i, frame := range decoded {
		if frame.ChromaWidth != w || frame.ChromaHeight != h || len(frame.U) != w*h || len(frame.V) != w*h {
			t.Fatalf("decoded frame %d chroma=%dx%d U=%d V=%d want %dx%d",
				i, frame.ChromaWidth, frame.ChromaHeight, len(frame.U), len(frame.V), w, h)
		}
	}
	assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, "public-rtc-i444-native", ivf)
}

func TestPublicRTCEncoderI444NativeMultiSpatialReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2},
		{name: "shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			w, h := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(w, h, scenario.mode)
			cfg.Profile = internalencoder.Profile1
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 32
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth: 8,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if enc.Config().Profile != internalencoder.Profile1 ||
				enc.Config().ColorConfig.MonoChrome ||
				enc.Config().ColorConfig.SubsamplingX ||
				enc.Config().ColorConfig.SubsamplingY ||
				enc.Config().SpatialLayerCount < 2 {
				t.Fatalf("normalized config=%+v want native multi-spatial 4:4:4", enc.Config())
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI444Picture(publicI444NativePattern(w, h, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI444Picture(%d): %v", frame, err)
				}
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					if !picture.Frames[i].CodedKeyframe {
						continue
					}
					seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
					if seq.SeqProfile != uint8(internalencoder.Profile1) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != 8 ||
						seq.ColorConfig.SubsamplingX ||
						seq.ColorConfig.SubsamplingY {
						t.Fatalf("picture %d frame %d profile=%d color=%+v want native 8-bit 4:4:4",
							frame, i, seq.SeqProfile, seq.ColorConfig)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i444-native-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI444NativeControlChurnReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	mode := EncoderScalabilityModeL2T3_KEY_SHIFT
	width, height := publicRTCMatrixGeometry(t, mode)
	baseKbps := publicRTCMatrixControlBitrateKbps(t, mode)
	fpsCycle := []EncoderRational{
		{Num: 15, Den: 1},
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 60, Den: 1},
	}
	configForStep := func(step int) EncoderConfig {
		cfg := publicRTCMatrixConfig(width, height, mode)
		cfg.Profile = internalencoder.Profile1
		cfg.ColorConfigSet = true
		cfg.ColorConfig = internalencoder.SequenceColorConfig{BitDepth: 8}
		cfg.MaxFramerate = fpsCycle[step%len(fpsCycle)]
		cfg.MaxThreads = int32(1 + step%3)
		cfg.Content = EncoderContentCamera
		if step%2 == 1 {
			cfg.Content = EncoderContentScreen
		}
		publicRTCApplyControlBitrates(&cfg, baseKbps+int32(step*97))
		if step%2 == 0 {
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = uint8(26 + step)
		} else {
			cfg.RateControl = EncoderRateControlCBR
			cfg.Quantizer = 0
		}
		return cfg
	}

	var enc *RTCEncoder
	var descriptorReceiver RTPDependencyDescriptorState
	var rtpReceiver RTPDependencyDescriptorState
	var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
	var orderedTUs [][]byte
	var sequence SequenceHeader
	nextFrameID := uint64(0)
	for step := 0; step < 6; step++ {
		cfg := configForStep(step)
		forceKey := step == 3
		wantKey := step == 0 || forceKey
		if enc == nil {
			var err error
			enc, err = NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
		} else {
			wantKey = wantKey || publicRTCSetConfigRequiresKey(t, enc.Config(), cfg)
			if err := enc.SetConfig(cfg); err != nil {
				t.Fatalf("step %d SetConfig: %v", step, err)
			}
		}
		assertPublicRTCConfigControls(t, enc.Config(), cfg)
		wantDuration, err := EncoderWebRTCRTPFrameDuration(enc.Config())
		if err != nil {
			t.Fatalf("step %d EncoderWebRTCRTPFrameDuration: %v", step, err)
		}
		if got, err := enc.RTPFrameDuration(); err != nil || got != wantDuration {
			t.Fatalf("step %d RTPFrameDuration=%+v err=%v want %+v", step, got, err, wantDuration)
		}

		picture, err := enc.EncodeI444Picture(publicI444NativePattern(width, height, step), forceKey)
		if err != nil {
			t.Fatalf("step %d EncodeI444Picture: %v", step, err)
		}
		if picture.Keyframe != wantKey {
			t.Fatalf("step %d key=%v want %v picture=%+v", step, picture.Keyframe, wantKey, picture)
		}
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, wantKey, &nextFrameID)
		for i := 0; i < picture.FrameNum; i++ {
			if !picture.Frames[i].CodedKeyframe {
				continue
			}
			assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("native-i444-control-step-%d-S%d", step, picture.Frames[i].SpatialID), picture.Frames[i].Data, &sequence, enc.Config().Content == EncoderContentScreen)
			seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
			if seq.SeqProfile != uint8(internalencoder.Profile1) ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != 8 ||
				seq.ColorConfig.SubsamplingX ||
				seq.ColorConfig.SubsamplingY {
				t.Fatalf("step %d sequence profile=%d color=%+v want native 8-bit 4:4:4", step, seq.SeqProfile, seq.ColorConfig)
			}
		}
		appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
	}

	normalized := enc.Config()
	assertPublicRTCLayerStreamsDecode(t, normalized, layerTUs, orderedTUs)
	assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i444-native-control-churn", normalized, layerTUs, orderedTUs)
}

func TestPublicRTCEncoderI400NativeMonochrome(t *testing.T) {
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T3)
	cfg.ColorConfigSet = true
	cfg.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()
	if !enc.Config().ColorConfig.MonoChrome {
		t.Fatalf("normalized config is not monochrome: %+v", enc.Config().ColorConfig)
	}

	wantTID := []uint8{0, 2, 1, 2, 0, 2}
	var tus [][]byte
	for frame := 0; frame < len(wantTID); frame++ {
		if frame == 3 {
			control := enc.Config()
			control.MaxFramerate = EncoderRational{Num: 60, Den: 1}
			control.MinBitrateKbps += 15
			control.MaxBitrateKbps += 180
			control.TargetBitrateKbps += 90
			if err := enc.SetConfig(control); err != nil {
				t.Fatalf("SetConfig control: %v", err)
			}
		}
		src := publicI400FromI420(publicRTCMatrixFrame(w, h, frame))
		out, err := enc.EncodeI400(src, false)
		if err != nil {
			t.Fatalf("EncodeI400(%d): %v", frame, err)
		}
		if out.TemporalID != wantTID[frame] {
			t.Fatalf("frame %d temporal id=%d want %d", frame, out.TemporalID, wantTID[frame])
		}
		if frame == 0 {
			seq := publicFirstSequenceHeader(t, out.Data)
			if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != 8 {
				t.Fatalf("sequence color=%+v want native 8-bit monochrome", seq.ColorConfig)
			}
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}

	frame := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, len(wantTID)), FrameFormat{
		Width:      w,
		Height:     h,
		BitDepth:   10,
		MonoChrome: true,
		Align:      32,
	})
	generic, err := enc.EncodeFrame(frame, false)
	if err != nil {
		t.Fatalf("EncodeFrame monochrome: %v", err)
	}
	tus = append(tus, append([]byte(nil), generic.Data...))

	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	defer dec.Close()
	decoded := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range batch {
			if !frame.Format.MonoChrome || frame.Format.BitDepth != 8 || len(frame.U.Pix) != 0 || len(frame.V.Pix) != 0 {
				t.Fatalf("decoded frame format=%+v U=%d V=%d want native monochrome", frame.Format, len(frame.U.Pix), len(frame.V.Pix))
			}
		}
		decoded += len(batch)
	}
	if decoded != len(tus) {
		t.Fatalf("decoded %d frames want %d", decoded, len(tus))
	}
}

func TestPublicRTCEncoderI400NativeMonochromeGoldenInterval(t *testing.T) {
	const w, h = 192, 128
	rng := rand.New(rand.NewSource(29))
	base := make([]byte, w*h)
	for i := range base {
		base[i] = uint8(50 + rng.Intn(170))
	}
	mk := func(boxX int) I400Frame {
		f := I400Frame{
			Y: append([]byte(nil), base...), YStride: w, Width: w, Height: h,
		}
		if boxX >= 0 {
			for y := 32; y < 96; y++ {
				for x := boxX; x < boxX+64 && x < w; x++ {
					f.Y[y*w+x] = 235
				}
			}
		}
		return f
	}
	frames := []I400Frame{
		mk(-1),
		mk(64),
		mk(-1),
	}
	encode := func(goldenInterval int) [][]byte {
		cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T1)
		cfg.MinBitrateKbps = 2000
		cfg.MaxBitrateKbps = 2000
		cfg.TargetBitrateKbps = 2000
		cfg.ColorConfigSet = true
		cfg.ColorConfig = internalencoder.SequenceColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		}
		enc, err := NewRTCEncoderWithConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer enc.Close()
		enc.SetGoldenInterval(goldenInterval)
		var tus [][]byte
		for i, f := range frames {
			picture, err := enc.EncodeI400Picture(f, false)
			if err != nil {
				t.Fatalf("EncodeI400Picture(%d): %v", i, err)
			}
			if picture.FrameNum != 1 || picture.Frames[0].SpatialID != 0 || picture.Frames[0].TemporalID != 0 {
				t.Fatalf("picture %d=%+v", i, picture)
			}
			tus = append(tus, append([]byte(nil), picture.Frames[0].Data...))
		}
		return tus
	}
	lastOnly := encode(0)
	withGolden := encode(16)
	t.Logf("rtc monochrome reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
	if len(withGolden[2])*5 >= len(lastOnly[2])*4 {
		t.Fatalf("golden interval did not materially reduce monochrome reveal frame: last-only %dB, golden %dB", len(lastOnly[2]), len(withGolden[2]))
	}
	dec, err := NewDecoder(withGolden)
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	defer dec.Close()
	decoded := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range batch {
			if !frame.Format.MonoChrome || frame.Format.BitDepth != 8 {
				t.Fatalf("decoded frame format=%+v want native 8-bit monochrome", frame.Format)
			}
		}
		decoded += len(batch)
	}
	if decoded != len(withGolden) {
		t.Fatalf("decoded %d frames want %d", decoded, len(withGolden))
	}
}

func TestPublicRTCEncoderI400NativeMonochromeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	cfg.RateControl = EncoderRateControlCQP
	cfg.Quantizer = 42
	cfg.ColorConfigSet = true
	cfg.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	frames := make([]publicIVFFrame, 0, 5)
	for frame := 0; frame < cap(frames); frame++ {
		picture, err := enc.EncodeI400Picture(publicI400FromI420(publicRTCMatrixFrame(w, h, frame)), false)
		if err != nil {
			t.Fatalf("EncodeI400Picture(%d): %v", frame, err)
		}
		if picture.FrameNum != 1 || picture.Frames[0].SpatialID != 0 {
			t.Fatalf("picture %d=%+v", frame, picture)
		}
		frames = append(frames, publicIVFFrame{
			timestamp: uint64(frame),
			payload:   append([]byte(nil), picture.Frames[0].Data...),
		})
	}
	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	assertPublicIVFMonochromeMatchesReferenceDecoders(t, decoders, "public-rtc-i400-native-monochrome", ivf, w, h)
}

func TestPublicRTCEncoderI400NativeMonochromeMultiSpatialReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2},
		{name: "shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			w, h := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(w, h, scenario.mode)
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 39
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:   8,
				MonoChrome: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if !enc.Config().ColorConfig.MonoChrome || enc.Config().SpatialLayerCount < 2 {
				t.Fatalf("normalized config=%+v want native multi-spatial mono", enc.Config())
			}

			var receiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 5; frame++ {
				forceKey := frame == 3
				picture, err := enc.EncodeI400Picture(publicI400FromI420(publicRTCMatrixFrame(w, h, frame)), forceKey)
				if err != nil {
					t.Fatalf("EncodeI400Picture(%d): %v", frame, err)
				}
				if picture.FrameNum != int(enc.Config().SpatialLayerCount) {
					t.Fatalf("picture %d frames=%d want %d", frame, picture.FrameNum, enc.Config().SpatialLayerCount)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				for i := 0; i < picture.FrameNum; i++ {
					if !picture.Frames[i].CodedKeyframe {
						continue
					}
					seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
					if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != 8 {
						t.Fatalf("picture %d frame %d color=%+v want native monochrome", frame, i, seq.ColorConfig)
					}
				}
				appendPublicRTCPictureRTPData(t, &receiver, &layerTUs, &orderedTUs, picture)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPMonochromeReferenceDecoders(t, decoders, scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI400HighBitDepthNativeMonochromeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit", bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit", bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
			cfg.Profile = tc.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 42
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:   tc.bitDepth,
				MonoChrome: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if enc.Config().ColorConfig.BitDepth != tc.bitDepth || !enc.Config().ColorConfig.MonoChrome {
				t.Fatalf("normalized color=%+v want native %d-bit monochrome", enc.Config().ColorConfig, tc.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				picture, err := enc.EncodeI400HighBitDepthPicture(publicI400HighBitDepthFromI420(publicRTCMatrixFrame(w, h, frame), tc.bitDepth, frame), frame == 2)
				if err != nil {
					t.Fatalf("EncodeI400HighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != 1 || picture.Frames[0].SpatialID != 0 {
					t.Fatalf("picture %d=%+v", frame, picture)
				}
				if picture.Keyframe != (frame == 0 || frame == 2) {
					t.Fatalf("picture %d key=%v", frame, picture.Keyframe)
				}
				if picture.Frames[0].CodedKeyframe {
					seq := publicFirstSequenceHeader(t, picture.Frames[0].Data)
					if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != tc.bitDepth || seq.SeqProfile != uint8(tc.profile) {
						t.Fatalf("sequence profile=%d color=%+v want profile=%d %d-bit monochrome", seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || frame == 2, &nextFrameID)
			}
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPMonochromeReferenceDecoders(t, decoders, "public-rtc-i400-"+tc.name+"-native-monochrome", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI400HighBitDepthNativeMonochromeMultiSpatialReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name     string
		mode     EncoderScalabilityMode
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit-simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-simulcast-s3t2", mode: internalencoder.ScalabilityModeS3T2, bitDepth: 12, profile: internalencoder.Profile2},
		{name: "10bit-shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
		{name: "10bit-shared-svc-l3t2-key-shift", mode: EncoderScalabilityModeL3T2_KEY_SHIFT, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-shared-svc-l3t2-key-shift", mode: EncoderScalabilityModeL3T2_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			w, h := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(w, h, scenario.mode)
			cfg.Profile = scenario.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 39
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:   scenario.bitDepth,
				MonoChrome: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if enc.Config().ColorConfig.BitDepth != scenario.bitDepth || !enc.Config().ColorConfig.MonoChrome || enc.Config().SpatialLayerCount < 2 {
				t.Fatalf("normalized config=%+v want native multi-spatial %d-bit mono", enc.Config(), scenario.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI400HighBitDepthPicture(publicI400HighBitDepthFromI420(publicRTCMatrixFrame(w, h, frame), scenario.bitDepth, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI400HighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != int(enc.Config().SpatialLayerCount) {
					t.Fatalf("picture %d frames=%d want %d", frame, picture.FrameNum, enc.Config().SpatialLayerCount)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				for i := 0; i < picture.FrameNum; i++ {
					if !picture.Frames[i].CodedKeyframe {
						continue
					}
					seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
					if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != scenario.bitDepth || seq.SeqProfile != uint8(scenario.profile) {
						t.Fatalf("picture %d frame %d profile=%d color=%+v want profile=%d %d-bit monochrome",
							frame, i, seq.SeqProfile, seq.ColorConfig, scenario.profile, scenario.bitDepth)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPMonochromeReferenceDecoders(t, decoders, scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI400HighBitDepthNativeMonochromeControlChurnReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name     string
		mode     EncoderScalabilityMode
		reconfig EncoderScalabilityMode
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit-single-l1t2-to-l1t3", mode: EncoderScalabilityModeL1T2, reconfig: EncoderScalabilityModeL1T3, bitDepth: 10, profile: EncoderProfile0},
		{name: "10bit-simulcast-s2t2-to-s2t3", mode: internalencoder.ScalabilityModeS2T2, reconfig: internalencoder.ScalabilityModeS2T3, bitDepth: 10, profile: EncoderProfile0},
		{name: "10bit-shared-svc-l2t2-to-l2t3-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, reconfig: EncoderScalabilityModeL2T3_KEY_SHIFT, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-shared-svc-l3t2-to-l3t3-key-shift", mode: EncoderScalabilityModeL3T2_KEY_SHIFT, reconfig: EncoderScalabilityModeL3T3_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
	}
	for scenarioIndex, scenario := range scenarios {
		scenarioIndex, scenario := scenarioIndex, scenario
		t.Run(scenario.name, func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(width, height, scenario.mode)
			cfg.Profile = scenario.profile
			cfg.MaxFramerate = EncoderRational{Num: 30, Den: 1}
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:   scenario.bitDepth,
				MonoChrome: true,
			}
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(75*scenarioIndex))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader

			appendPicture := func(label string, frameIndex int, forceKey bool, wantKey bool, wantScreen bool) {
				t.Helper()
				picture, err := enc.EncodeI400HighBitDepthPicture(publicI400HighBitDepthFromI420(publicRTCMatrixFrame(width, height, frameIndex), scenario.bitDepth, frameIndex), forceKey)
				if err != nil {
					t.Fatalf("%s EncodeI400HighBitDepthPicture: %v", label, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					if frame.CodedKeyframe {
						seq := publicFirstSequenceHeader(t, frame.Data)
						if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != scenario.bitDepth || seq.SeqProfile != uint8(scenario.profile) {
							t.Fatalf("%s frame %d profile=%d color=%+v want profile=%d %d-bit monochrome",
								label, i, seq.SeqProfile, seq.ColorConfig, scenario.profile, scenario.bitDepth)
						}
					}
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &sharedSequence, wantScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], wantScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			appendPicture("initial key", 0, false, true, false)

			fpsBitrateChange := enc.Config()
			fpsBitrateChange.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
			publicRTCApplyControlBitrates(&fpsBitrateChange, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(150+scenarioIndex*31))
			if err := enc.SetConfig(fpsBitrateChange); err != nil {
				t.Fatalf("SetConfig fps/bitrate: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
			appendPicture("fps bitrate delta", 1, false, false, false)

			cqpChange := enc.Config()
			cqpChange.RateControl = EncoderRateControlCQP
			cqpChange.Quantizer = uint8(31 + scenarioIndex)
			cqpChange.Content = EncoderContentScreen
			if err := enc.SetConfig(cqpChange); err != nil {
				t.Fatalf("SetConfig CQP/screen: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
			appendPicture("CQP screen delta", 2, false, false, true)

			scalabilityChange := enc.Config()
			scalabilityChange.Scalability = scenario.reconfig
			scalabilityChange.MaxFramerate = EncoderRational{Num: 24, Den: 1}
			scalabilityChange.Quantizer = uint8(28 + scenarioIndex)
			wantReconfigKey := publicRTCSetConfigRequiresKey(t, enc.Config(), scalabilityChange)
			if err := enc.SetConfig(scalabilityChange); err != nil {
				t.Fatalf("SetConfig scalability: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
			if !wantReconfigKey {
				t.Fatalf("%s to %s unexpectedly did not require a key picture", scenario.mode, scenario.reconfig)
			}
			appendPicture("scalability screen key", 3, false, true, true)

			cbrChange := enc.Config()
			cbrChange.RateControl = EncoderRateControlCBR
			cbrChange.Quantizer = 0
			cbrChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
			cbrChange.Content = EncoderContentCamera
			publicRTCApplyControlBitrates(&cbrChange, publicRTCMatrixControlBitrateKbps(t, scenario.reconfig)+int32(220+scenarioIndex*41))
			if err := enc.SetConfig(cbrChange); err != nil {
				t.Fatalf("SetConfig CBR/camera: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
			appendPicture("CBR camera delta", 4, false, false, false)

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPMonochromeReferenceDecoders(t, decoders, scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI420HighBitDepthNativeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit", bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit", bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
			cfg.Profile = tc.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 42
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     tc.bitDepth,
				SubsamplingX: true,
				SubsamplingY: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); !errors.Is(err, internalencoder.ErrUnsupported) {
				t.Fatalf("Encode(I420) on native %d-bit I420 err=%v want %v", tc.bitDepth, err, internalencoder.ErrUnsupported)
			}
			if enc.Config().Profile != tc.profile ||
				enc.Config().ColorConfig.BitDepth != tc.bitDepth ||
				enc.Config().ColorConfig.MonoChrome ||
				!enc.Config().ColorConfig.SubsamplingX ||
				!enc.Config().ColorConfig.SubsamplingY {
				t.Fatalf("normalized color=%+v profile=%d want native profile=%d %d-bit 4:2:0",
					enc.Config().ColorConfig, enc.Config().Profile, tc.profile, tc.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI420HighBitDepthPicture(publicI420HighBitDepthFrame(w, h, tc.bitDepth, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI420HighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != 1 || picture.Frames[0].SpatialID != 0 {
					t.Fatalf("picture %d=%+v", frame, picture)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				if picture.Frames[0].CodedKeyframe {
					seq := publicFirstSequenceHeader(t, picture.Frames[0].Data)
					if seq.SeqProfile != uint8(tc.profile) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != tc.bitDepth ||
						!seq.ColorConfig.SubsamplingX ||
						!seq.ColorConfig.SubsamplingY {
						t.Fatalf("sequence profile=%d color=%+v want profile=%d %d-bit 4:2:0",
							seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
			}
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i420-"+tc.name+"-native", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI420HighBitDepthNativeTileColumnsReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit", bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit", bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 512, 288
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T1)
			cfg.Profile = tc.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 41
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     tc.bitDepth,
				SubsamplingX: true,
				SubsamplingY: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			enc.SetTileColumns(4)

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 3; frame++ {
				picture, err := enc.EncodeI420HighBitDepthPicture(publicI420HighBitDepthFrame(w, h, tc.bitDepth, frame), false)
				if err != nil {
					t.Fatalf("EncodeI420HighBitDepthPicture(%d): %v", frame, err)
				}
				if frame == 0 {
					if got := publicFirstFrameTileColumns(t, picture.Frames[0].Data); got != 4 {
						t.Fatalf("tile columns=%d want 4", got)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0, &nextFrameID)
			}
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i420-"+tc.name+"-native-tiles", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI420HighBitDepthNativeMultiSpatialReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name     string
		mode     EncoderScalabilityMode
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit-simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-simulcast-s3t2", mode: internalencoder.ScalabilityModeS3T2, bitDepth: 12, profile: internalencoder.Profile2},
		{name: "10bit-shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
		{name: "10bit-shared-svc-l3t2-key-shift", mode: EncoderScalabilityModeL3T2_KEY_SHIFT, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-shared-svc-l3t2-key-shift", mode: EncoderScalabilityModeL3T2_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			w, h := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(w, h, scenario.mode)
			cfg.Profile = scenario.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 39
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     scenario.bitDepth,
				SubsamplingX: true,
				SubsamplingY: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if enc.Config().Profile != scenario.profile ||
				enc.Config().ColorConfig.BitDepth != scenario.bitDepth ||
				enc.Config().ColorConfig.MonoChrome ||
				!enc.Config().ColorConfig.SubsamplingX ||
				!enc.Config().ColorConfig.SubsamplingY ||
				enc.Config().SpatialLayerCount < 2 {
				t.Fatalf("normalized config=%+v want native multi-spatial profile=%d %d-bit 4:2:0",
					enc.Config(), scenario.profile, scenario.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI420HighBitDepthPicture(publicI420HighBitDepthFrame(w, h, scenario.bitDepth, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI420HighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != int(enc.Config().SpatialLayerCount) {
					t.Fatalf("picture %d frames=%d want %d", frame, picture.FrameNum, enc.Config().SpatialLayerCount)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				for i := 0; i < picture.FrameNum; i++ {
					if !picture.Frames[i].CodedKeyframe {
						continue
					}
					seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
					if seq.SeqProfile != uint8(scenario.profile) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != scenario.bitDepth ||
						!seq.ColorConfig.SubsamplingX ||
						!seq.ColorConfig.SubsamplingY {
						t.Fatalf("picture %d frame %d profile=%d color=%+v want profile=%d %d-bit 4:2:0",
							frame, i, seq.SeqProfile, seq.ColorConfig, scenario.profile, scenario.bitDepth)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i420-native-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI420HighBitDepthNativeControlChurnReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name     string
		mode     EncoderScalabilityMode
		reconfig EncoderScalabilityMode
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit-single-l1t2-to-l1t3", mode: EncoderScalabilityModeL1T2, reconfig: EncoderScalabilityModeL1T3, bitDepth: 10, profile: EncoderProfile0},
		{name: "10bit-simulcast-s2t2-to-s2t3", mode: internalencoder.ScalabilityModeS2T2, reconfig: internalencoder.ScalabilityModeS2T3, bitDepth: 10, profile: EncoderProfile0},
		{name: "10bit-shared-svc-l2t2-to-l2t3-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, reconfig: EncoderScalabilityModeL2T3_KEY_SHIFT, bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-shared-svc-l3t2-to-l3t3-key-shift", mode: EncoderScalabilityModeL3T2_KEY_SHIFT, reconfig: EncoderScalabilityModeL3T3_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
	}
	for scenarioIndex, scenario := range scenarios {
		scenarioIndex, scenario := scenarioIndex, scenario
		t.Run(scenario.name, func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(width, height, scenario.mode)
			cfg.Profile = scenario.profile
			cfg.MaxFramerate = EncoderRational{Num: 30, Den: 1}
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     scenario.bitDepth,
				SubsamplingX: true,
				SubsamplingY: true,
			}
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(75*scenarioIndex))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader

			appendPicture := func(label string, frameIndex int, forceKey bool, wantKey bool, wantScreen bool) {
				t.Helper()
				picture, err := enc.EncodeI420HighBitDepthPicture(publicI420HighBitDepthFrame(width, height, scenario.bitDepth, frameIndex), forceKey)
				if err != nil {
					t.Fatalf("%s EncodeI420HighBitDepthPicture: %v", label, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					if frame.CodedKeyframe {
						seq := publicFirstSequenceHeader(t, frame.Data)
						if seq.SeqProfile != uint8(scenario.profile) ||
							seq.ColorConfig.MonoChrome ||
							seq.ColorConfig.BitDepth != scenario.bitDepth ||
							!seq.ColorConfig.SubsamplingX ||
							!seq.ColorConfig.SubsamplingY {
							t.Fatalf("%s frame %d profile=%d color=%+v want profile=%d %d-bit 4:2:0",
								label, i, seq.SeqProfile, seq.ColorConfig, scenario.profile, scenario.bitDepth)
						}
					}
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &sharedSequence, wantScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], wantScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			appendPicture("initial key", 0, false, true, false)

			fpsBitrateChange := enc.Config()
			fpsBitrateChange.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
			publicRTCApplyControlBitrates(&fpsBitrateChange, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(150+scenarioIndex*31))
			if err := enc.SetConfig(fpsBitrateChange); err != nil {
				t.Fatalf("SetConfig fps/bitrate: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
			appendPicture("fps bitrate delta", 1, false, false, false)

			cqpChange := enc.Config()
			cqpChange.RateControl = EncoderRateControlCQP
			cqpChange.Quantizer = uint8(31 + scenarioIndex)
			cqpChange.Content = EncoderContentScreen
			if err := enc.SetConfig(cqpChange); err != nil {
				t.Fatalf("SetConfig CQP/screen: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
			appendPicture("CQP screen delta", 2, false, false, true)

			scalabilityChange := enc.Config()
			scalabilityChange.Scalability = scenario.reconfig
			scalabilityChange.MaxFramerate = EncoderRational{Num: 24, Den: 1}
			scalabilityChange.Quantizer = uint8(28 + scenarioIndex)
			wantReconfigKey := publicRTCSetConfigRequiresKey(t, enc.Config(), scalabilityChange)
			if err := enc.SetConfig(scalabilityChange); err != nil {
				t.Fatalf("SetConfig scalability: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
			if !wantReconfigKey {
				t.Fatalf("%s to %s unexpectedly did not require a key picture", scenario.mode, scenario.reconfig)
			}
			appendPicture("scalability screen key", 3, false, true, true)

			cbrChange := enc.Config()
			cbrChange.RateControl = EncoderRateControlCBR
			cbrChange.Quantizer = 0
			cbrChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
			cbrChange.Content = EncoderContentCamera
			publicRTCApplyControlBitrates(&cbrChange, publicRTCMatrixControlBitrateKbps(t, scenario.reconfig)+int32(220+scenarioIndex*41))
			if err := enc.SetConfig(cbrChange); err != nil {
				t.Fatalf("SetConfig CBR/camera: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
			appendPicture("CBR camera delta", 4, false, false, false)

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i420-control-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI422HighBitDepthNativeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
	}{
		{name: "10bit", bitDepth: 10},
		{name: "12bit", bitDepth: 12},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
			cfg.Profile = internalencoder.Profile2
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 42
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     tc.bitDepth,
				SubsamplingX: true,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); !errors.Is(err, internalencoder.ErrUnsupported) {
				t.Fatalf("Encode(I420) on native %d-bit I422 err=%v want %v", tc.bitDepth, err, internalencoder.ErrUnsupported)
			}
			if enc.Config().Profile != internalencoder.Profile2 ||
				enc.Config().ColorConfig.BitDepth != tc.bitDepth ||
				enc.Config().ColorConfig.MonoChrome ||
				!enc.Config().ColorConfig.SubsamplingX ||
				enc.Config().ColorConfig.SubsamplingY {
				t.Fatalf("normalized color=%+v profile=%d want native profile-2 %d-bit 4:2:2",
					enc.Config().ColorConfig, enc.Config().Profile, tc.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI422HighBitDepthPicture(publicI422HighBitDepthFrame(w, h, tc.bitDepth, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI422HighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != 1 || picture.Frames[0].SpatialID != 0 {
					t.Fatalf("picture %d=%+v", frame, picture)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				if picture.Frames[0].CodedKeyframe {
					seq := publicFirstSequenceHeader(t, picture.Frames[0].Data)
					if seq.SeqProfile != uint8(internalencoder.Profile2) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != tc.bitDepth ||
						!seq.ColorConfig.SubsamplingX ||
						seq.ColorConfig.SubsamplingY {
						t.Fatalf("sequence profile=%d color=%+v want profile-2 %d-bit 4:2:2",
							seq.SeqProfile, seq.ColorConfig, tc.bitDepth)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
			}
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i422-"+tc.name+"-native", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderI444HighBitDepthNativeReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit", bitDepth: 10, profile: internalencoder.Profile1},
		{name: "12bit", bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
			cfg.Profile = tc.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 42
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{BitDepth: tc.bitDepth}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); !errors.Is(err, internalencoder.ErrUnsupported) {
				t.Fatalf("Encode(I420) on native %d-bit I444 err=%v want %v", tc.bitDepth, err, internalencoder.ErrUnsupported)
			}
			if enc.Config().Profile != tc.profile ||
				enc.Config().ColorConfig.BitDepth != tc.bitDepth ||
				enc.Config().ColorConfig.MonoChrome ||
				enc.Config().ColorConfig.SubsamplingX ||
				enc.Config().ColorConfig.SubsamplingY {
				t.Fatalf("normalized color=%+v profile=%d want native profile-%d %d-bit 4:4:4",
					enc.Config().ColorConfig, enc.Config().Profile, tc.profile, tc.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				picture, err := enc.EncodeI444HighBitDepthPicture(publicI444HighBitDepthFrame(w, h, tc.bitDepth, frame), forceKey)
				if err != nil {
					t.Fatalf("EncodeI444HighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != 1 || picture.Frames[0].SpatialID != 0 {
					t.Fatalf("picture %d=%+v", frame, picture)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				if picture.Frames[0].CodedKeyframe {
					seq := publicFirstSequenceHeader(t, picture.Frames[0].Data)
					if seq.SeqProfile != uint8(tc.profile) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != tc.bitDepth ||
						seq.ColorConfig.SubsamplingX ||
						seq.ColorConfig.SubsamplingY {
						t.Fatalf("sequence profile=%d color=%+v want profile-%d %d-bit 4:4:4",
							seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
			}
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i444-"+tc.name+"-native", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderHighBitDepthNon420ScalabilityModeCatalogueReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
		i444     bool
	}{
		{name: "i422-10bit", bitDepth: 10, profile: internalencoder.Profile2},
		{name: "i422-12bit", bitDepth: 12, profile: internalencoder.Profile2},
		{name: "i444-10bit", bitDepth: 10, profile: internalencoder.Profile1, i444: true},
		{name: "i444-12bit", bitDepth: 12, profile: internalencoder.Profile2, i444: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			for modeIndex, mode := range modes {
				modeIndex, mode := modeIndex, mode
				t.Run(mode.String(), func(t *testing.T) {
					width, height := publicRTCMatrixGeometry(t, mode)
					cfg := publicRTCMatrixConfig(width, height, mode)
					cfg.Profile = tc.profile
					cfg.RateControl = EncoderRateControlCQP
					cfg.Quantizer = uint8(36 + modeIndex%13)
					cfg.Content = EncoderContentCamera
					if modeIndex%3 == 1 {
						cfg.Content = EncoderContentScreen
					}
					cfg.ColorConfigSet = true
					cfg.ColorConfig = internalencoder.SequenceColorConfig{
						BitDepth:     tc.bitDepth,
						SubsamplingX: !tc.i444,
					}
					enc, err := NewRTCEncoderWithConfig(cfg)
					if err != nil {
						t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
					}
					defer enc.Close()
					if enc.Config().Profile != tc.profile ||
						enc.Config().ColorConfig.BitDepth != tc.bitDepth ||
						enc.Config().ColorConfig.MonoChrome ||
						enc.Config().ColorConfig.SubsamplingX != !tc.i444 ||
						enc.Config().ColorConfig.SubsamplingY {
						t.Fatalf("%s normalized config=%+v want profile=%d %d-bit i444=%v",
							mode, enc.Config(), tc.profile, tc.bitDepth, tc.i444)
					}

					var descriptorReceiver RTPDependencyDescriptorState
					var rtpReceiver RTPDependencyDescriptorState
					var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
					var orderedTUs [][]byte
					var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
					var sharedSequence SequenceHeader
					nextFrameID := uint64(0)

					for frameIndex := 0; frameIndex < 3; frameIndex++ {
						forceKey := frameIndex == 2
						var picture RTCPicture
						if tc.i444 {
							picture, err = enc.EncodeI444HighBitDepthPicture(publicI444HighBitDepthFrame(width, height, tc.bitDepth, frameIndex), forceKey)
						} else {
							picture, err = enc.EncodeI422HighBitDepthPicture(publicI422HighBitDepthFrame(width, height, tc.bitDepth, frameIndex), forceKey)
						}
						if err != nil {
							t.Fatalf("EncodeHighBitDepthPicture(%s frame %d): %v", mode, frameIndex, err)
						}
						wantKey := frameIndex == 0 || forceKey
						if picture.Keyframe != wantKey || picture.FrameNum != int(enc.Config().SpatialLayerCount) {
							t.Fatalf("%s frame %d picture key=%v frames=%d want key=%v frames=%d",
								mode, frameIndex, picture.Keyframe, picture.FrameNum, wantKey, enc.Config().SpatialLayerCount)
						}
						current := enc.Config()
						assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
						for i := 0; i < picture.FrameNum; i++ {
							frame := picture.Frames[i]
							if frame.CodedKeyframe {
								seq := publicFirstSequenceHeader(t, frame.Data)
								if seq.SeqProfile != uint8(tc.profile) ||
									seq.ColorConfig.MonoChrome ||
									seq.ColorConfig.BitDepth != tc.bitDepth ||
									seq.ColorConfig.SubsamplingX != !tc.i444 ||
									seq.ColorConfig.SubsamplingY {
									t.Fatalf("%s frame %d spatial %d profile=%d color=%+v want profile=%d %d-bit i444=%v",
										mode, frameIndex, frame.SpatialID, seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth, tc.i444)
								}
							}
							if publicRTCSharedReferenceSlotMode(current.Scalability) {
								assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s frame %d S%d", mode, frameIndex, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
								continue
							}
							assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s frame %d S%d", mode, frameIndex, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
						}
						appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
					}

					assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
					assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-highbd-non420-catalogue-"+tc.name+"-"+mode.String(), enc.Config(), layerTUs, orderedTUs)
				})
			}
		})
	}
}

func TestPublicRTCEncoderHighBitDepthNon420ControllerSettingsReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
		{Num: 15, Den: 1},
	}
	speedCycle := []int8{
		EncoderWebRTCMinEffortLevel,
		0,
		EncoderWebRTCMaxEffortLevel,
		1,
	}
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
		i444     bool
	}{
		{name: "i422-10bit", bitDepth: 10, profile: internalencoder.Profile2},
		{name: "i422-12bit", bitDepth: 12, profile: internalencoder.Profile2},
		{name: "i444-10bit", bitDepth: 10, profile: internalencoder.Profile1, i444: true},
		{name: "i444-12bit", bitDepth: 12, profile: internalencoder.Profile2, i444: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			for modeIndex, mode := range modes {
				modeIndex, mode := modeIndex, mode
				t.Run(mode.String(), func(t *testing.T) {
					width, height := publicRTCMatrixGeometry(t, mode)
					configForStep := func(step int) EncoderConfig {
						cfg := publicRTCMatrixConfig(width, height, mode)
						cfg.Profile = tc.profile
						cfg.ColorConfigSet = true
						cfg.ColorConfig = internalencoder.SequenceColorConfig{
							BitDepth:     tc.bitDepth,
							SubsamplingX: !tc.i444,
						}
						cfg.MaxFramerate = fpsCycle[(modeIndex+step)%len(fpsCycle)]
						cfg.Speed = speedCycle[(modeIndex+step)%len(speedCycle)]
						cfg.MaxThreads = int32(1 + (modeIndex+step)%3)
						cfg.KeyFrameInterval = int32(2 + (modeIndex+step)%3)
						cfg.Content = EncoderContentCamera
						publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(modeIndex*19+step*47))
						switch step % 3 {
						case 1:
							cfg.RateControl = EncoderRateControlCQP
							cfg.Quantizer = uint8(24 + (modeIndex+step)%35)
							cfg.Content = EncoderContentScreen
						case 2:
							cfg.Content = EncoderContentScreen
						}
						if cfg.RateControl == EncoderRateControlCBR {
							cfg.Quantizer = 0
						}
						return cfg
					}

					initial := configForStep(0)
					rtc, err := NewRTCEncoderWithConfig(initial)
					if err != nil {
						t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
					}
					defer rtc.Close()
					controller, err := newPublicRTCController(initial)
					if err != nil {
						t.Fatalf("controller(%s): %v", mode, err)
					}

					var descriptorReceiver RTPDependencyDescriptorState
					var rtpReceiver RTPDependencyDescriptorState
					var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
					var orderedTUs [][]byte
					var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
					var sharedSequence SequenceHeader
					nextFrameID := uint64(0)

					for step := 0; step < 5; step++ {
						cfg := configForStep(step)
						forceKey := step == 3
						if step > 0 {
							if err := rtc.SetConfig(cfg); err != nil {
								t.Fatalf("%s step %d RTC SetConfig: %v", mode, step, err)
							}
							if err := controller.SetConfig(cfg); err != nil {
								t.Fatalf("%s step %d controller SetConfig: %v", mode, step, err)
							}
						}
						assertPublicRTCConfigControls(t, rtc.Config(), cfg)
						assertPublicRTCConfigControls(t, controller.Config(), cfg)

						unit, err := controller.NextTemporalUnit(forceKey)
						if err != nil {
							t.Fatalf("%s step %d controller NextTemporalUnit: %v", mode, step, err)
						}
						controllerState := controller.State()
						var picture RTCPicture
						if tc.i444 {
							picture, err = rtc.EncodeI444HighBitDepthPicture(publicI444HighBitDepthFrame(width, height, tc.bitDepth, step), forceKey)
						} else {
							picture, err = rtc.EncodeI422HighBitDepthPicture(publicI422HighBitDepthFrame(width, height, tc.bitDepth, step), forceKey)
						}
						if err != nil {
							t.Fatalf("%s step %d EncodeHighBitDepthPicture: %v", mode, step, err)
						}
						frameNum := publicRTCUnitFrameNum(unit)
						if picture.FrameNum != int(frameNum) || picture.Keyframe != unit.Key {
							t.Fatalf("%s step %d picture frames=%d key=%v want frames=%d key=%v",
								mode, step, picture.FrameNum, picture.Keyframe, frameNum, unit.Key)
						}

						current := rtc.Config()
						assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, unit.Key, &nextFrameID)
						for frameIndex := 0; frameIndex < picture.FrameNum; frameIndex++ {
							frame := picture.Frames[frameIndex]
							control, _, err := publicRTCUnitFrameControl(unit, controllerState, uint8(frameIndex))
							if err != nil {
								t.Fatalf("%s step %d frame %d controller frame control: %v", mode, step, frameIndex, err)
							}
							settings := control.Settings
							info := control.GenericFrameInfo
							wantCodedKey := unit.Key && (current.Scalability.IsSimulcast() || settings.Type == internalencoder.FrameTypeKey)
							if frame.FrameID != info.FrameID ||
								frame.TemporalID != info.TemporalID ||
								frame.SpatialID != info.SpatialID ||
								frame.CodedKeyframe != wantCodedKey ||
								frame.Keyframe != unit.Key ||
								frame.LastFrameInPicture != (frameIndex+1 == picture.FrameNum) {
								t.Fatalf("%s step %d frame %d actual id=%d S%d/T%d key=%v codedKey=%v last=%v control=%+v settings=%+v want codedKey=%v",
									mode, step, frameIndex, frame.FrameID, frame.SpatialID, frame.TemporalID, frame.Keyframe, frame.CodedKeyframe, frame.LastFrameInPicture, info, settings, wantCodedKey)
							}
							if settings.RateControl != current.RateControl ||
								settings.Quantizer != current.Quantizer ||
								settings.EffortLevel != current.Speed {
								t.Fatalf("%s step %d frame %d settings=%+v config rate=%d q=%d speed=%d",
									mode, step, frameIndex, settings, current.RateControl, current.Quantizer, current.Speed)
							}
							if frame.CodedKeyframe {
								seq := publicFirstSequenceHeader(t, frame.Data)
								if seq.SeqProfile != uint8(tc.profile) ||
									seq.ColorConfig.MonoChrome ||
									seq.ColorConfig.BitDepth != tc.bitDepth ||
									seq.ColorConfig.SubsamplingX != !tc.i444 ||
									seq.ColorConfig.SubsamplingY {
									t.Fatalf("%s step %d frame %d profile=%d color=%+v want profile=%d %d-bit i444=%v",
										mode, step, frameIndex, seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth, tc.i444)
								}
							}
							if publicRTCSharedReferenceSlotMode(current.Scalability) {
								assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s step %d S%d", mode, step, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
								continue
							}
							assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s step %d S%d", mode, step, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
						}
						appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
					}

					assertPublicRTCLayerStreamsDecode(t, rtc.Config(), layerTUs, orderedTUs)
					assertPublicRTCRTPReferenceDecoders(t, decoders, "controller-settings-highbd-non420-"+tc.name+"-"+mode.String(), rtc.Config(), layerTUs, orderedTUs)
				})
			}
		})
	}
}

func TestPublicRTCEncoderHighBitDepthNon420NativeMultiSpatialReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name     string
		mode     EncoderScalabilityMode
		bitDepth uint8
		profile  internalencoder.Profile
		i444     bool
	}{
		{name: "i422-10bit-simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2, bitDepth: 10, profile: internalencoder.Profile2},
		{name: "i422-12bit-shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2},
		{name: "i444-10bit-simulcast-s2t2", mode: internalencoder.ScalabilityModeS2T2, bitDepth: 10, profile: internalencoder.Profile1, i444: true},
		{name: "i444-12bit-shared-svc-l2t2-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2, i444: true},
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			w, h := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(w, h, scenario.mode)
			cfg.Profile = scenario.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 39
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     scenario.bitDepth,
				SubsamplingX: !scenario.i444,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()
			if enc.Config().Profile != scenario.profile ||
				enc.Config().ColorConfig.BitDepth != scenario.bitDepth ||
				enc.Config().ColorConfig.MonoChrome ||
				enc.Config().ColorConfig.SubsamplingX != !scenario.i444 ||
				enc.Config().ColorConfig.SubsamplingY ||
				enc.Config().SpatialLayerCount < 2 {
				t.Fatalf("normalized config=%+v want native multi-spatial high-bit-depth non-4:2:0", enc.Config())
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 4; frame++ {
				forceKey := frame == 2
				var picture RTCPicture
				if scenario.i444 {
					picture, err = enc.EncodeI444HighBitDepthPicture(publicI444HighBitDepthFrame(w, h, scenario.bitDepth, frame), forceKey)
				} else {
					picture, err = enc.EncodeI422HighBitDepthPicture(publicI422HighBitDepthFrame(w, h, scenario.bitDepth, frame), forceKey)
				}
				if err != nil {
					t.Fatalf("EncodeHighBitDepthPicture(%d): %v", frame, err)
				}
				if picture.FrameNum != int(enc.Config().SpatialLayerCount) {
					t.Fatalf("picture %d frames=%d want %d", frame, picture.FrameNum, enc.Config().SpatialLayerCount)
				}
				if picture.Keyframe != (frame == 0 || forceKey) {
					t.Fatalf("picture %d key=%v force=%v", frame, picture.Keyframe, forceKey)
				}
				for i := 0; i < picture.FrameNum; i++ {
					if !picture.Frames[i].CodedKeyframe {
						continue
					}
					seq := publicFirstSequenceHeader(t, picture.Frames[i].Data)
					if seq.SeqProfile != uint8(scenario.profile) ||
						seq.ColorConfig.MonoChrome ||
						seq.ColorConfig.BitDepth != scenario.bitDepth ||
						seq.ColorConfig.SubsamplingX != !scenario.i444 ||
						seq.ColorConfig.SubsamplingY {
						t.Fatalf("picture %d frame %d profile=%d color=%+v want profile=%d %d-bit i444=%v",
							frame, i, seq.SeqProfile, seq.ColorConfig, scenario.profile, scenario.bitDepth, scenario.i444)
					}
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0 || forceKey, &nextFrameID)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-highbd-non420-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderHighBitDepthNon420ControlChurnReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name     string
		mode     EncoderScalabilityMode
		reconfig EncoderScalabilityMode
		bitDepth uint8
		profile  internalencoder.Profile
		i444     bool
	}{
		{name: "i422-10bit-l2t2-to-l2t3-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, reconfig: EncoderScalabilityModeL2T3_KEY_SHIFT, bitDepth: 10, profile: internalencoder.Profile2},
		{name: "i444-12bit-l2t2-to-l2t3-key-shift", mode: internalencoder.ScalabilityModeL2T2_KEY_SHIFT, reconfig: EncoderScalabilityModeL2T3_KEY_SHIFT, bitDepth: 12, profile: internalencoder.Profile2, i444: true},
	}
	for scenarioIndex, scenario := range scenarios {
		scenarioIndex, scenario := scenarioIndex, scenario
		t.Run(scenario.name, func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(width, height, scenario.mode)
			cfg.Profile = scenario.profile
			cfg.MaxFramerate = EncoderRational{Num: 30, Den: 1}
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     scenario.bitDepth,
				SubsamplingX: !scenario.i444,
			}
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(75*scenarioIndex))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader

			appendPicture := func(label string, frameIndex int, forceKey bool, wantKey bool, wantScreen bool) {
				t.Helper()
				var picture RTCPicture
				if scenario.i444 {
					picture, err = enc.EncodeI444HighBitDepthPicture(publicI444HighBitDepthFrame(width, height, scenario.bitDepth, frameIndex), forceKey)
				} else {
					picture, err = enc.EncodeI422HighBitDepthPicture(publicI422HighBitDepthFrame(width, height, scenario.bitDepth, frameIndex), forceKey)
				}
				if err != nil {
					t.Fatalf("%s EncodeHighBitDepthPicture: %v", label, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					if frame.CodedKeyframe {
						seq := publicFirstSequenceHeader(t, frame.Data)
						if seq.SeqProfile != uint8(scenario.profile) ||
							seq.ColorConfig.MonoChrome ||
							seq.ColorConfig.BitDepth != scenario.bitDepth ||
							seq.ColorConfig.SubsamplingX != !scenario.i444 ||
							seq.ColorConfig.SubsamplingY {
							t.Fatalf("%s frame %d profile=%d color=%+v want profile=%d %d-bit i444=%v",
								label, i, seq.SeqProfile, seq.ColorConfig, scenario.profile, scenario.bitDepth, scenario.i444)
						}
					}
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &sharedSequence, wantScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], wantScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			appendPicture("initial key", 0, false, true, false)

			fpsBitrateChange := enc.Config()
			fpsBitrateChange.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
			publicRTCApplyControlBitrates(&fpsBitrateChange, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(150+scenarioIndex*31))
			if err := enc.SetConfig(fpsBitrateChange); err != nil {
				t.Fatalf("SetConfig fps/bitrate: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
			appendPicture("fps bitrate delta", 1, false, false, false)

			cqpChange := enc.Config()
			cqpChange.RateControl = EncoderRateControlCQP
			cqpChange.Quantizer = uint8(31 + scenarioIndex)
			cqpChange.Content = EncoderContentScreen
			if err := enc.SetConfig(cqpChange); err != nil {
				t.Fatalf("SetConfig CQP/screen: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
			appendPicture("CQP screen delta", 2, false, false, true)

			scalabilityChange := enc.Config()
			scalabilityChange.Scalability = scenario.reconfig
			scalabilityChange.MaxFramerate = EncoderRational{Num: 24, Den: 1}
			scalabilityChange.Quantizer = uint8(28 + scenarioIndex)
			wantReconfigKey := publicRTCSetConfigRequiresKey(t, enc.Config(), scalabilityChange)
			if err := enc.SetConfig(scalabilityChange); err != nil {
				t.Fatalf("SetConfig scalability: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
			if !wantReconfigKey {
				t.Fatalf("%s to %s unexpectedly did not require a key picture", scenario.mode, scenario.reconfig)
			}
			appendPicture("scalability screen key", 3, false, true, true)

			cbrChange := enc.Config()
			cbrChange.RateControl = EncoderRateControlCBR
			cbrChange.Quantizer = 0
			cbrChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
			cbrChange.Content = EncoderContentCamera
			publicRTCApplyControlBitrates(&cbrChange, publicRTCMatrixControlBitrateKbps(t, scenario.reconfig)+int32(220+scenarioIndex*41))
			if err := enc.SetConfig(cbrChange); err != nil {
				t.Fatalf("SetConfig CBR/camera: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
			appendPicture("CBR camera delta", 4, false, false, false)

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-highbd-non420-control-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderHighBitDepthNon420GenericFrameNative(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
		i444     bool
	}{
		{name: "i422-10bit", bitDepth: 10, profile: internalencoder.Profile2},
		{name: "i444-10bit", bitDepth: 10, profile: internalencoder.Profile1, i444: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T1)
			cfg.Profile = tc.profile
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = 38
			cfg.ColorConfigSet = true
			cfg.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     tc.bitDepth,
				SubsamplingX: !tc.i444,
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			frame := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 0), FrameFormat{
				Width:        w,
				Height:       h,
				BitDepth:     tc.bitDepth,
				SubsamplingX: !tc.i444,
				Align:        32,
			})
			out, err := enc.EncodeFrame(frame, false)
			if err != nil {
				t.Fatalf("EncodeFrame native high-bit-depth non-4:2:0: %v", err)
			}
			seq := publicFirstSequenceHeader(t, out.Data)
			if seq.SeqProfile != uint8(tc.profile) ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != tc.bitDepth ||
				seq.ColorConfig.SubsamplingX != !tc.i444 ||
				seq.ColorConfig.SubsamplingY {
				t.Fatalf("sequence profile=%d color=%+v want profile=%d %d-bit i444=%v",
					seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth, tc.i444)
			}

			bad := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 1), FrameFormat{
				Width:        w,
				Height:       h,
				BitDepth:     8,
				SubsamplingX: !tc.i444,
				Align:        32,
			})
			if _, err := enc.EncodeFrame(bad, false); !errors.Is(err, internalframe.ErrInvalidFormat) {
				t.Fatalf("EncodeFrame 8-bit into %d-bit non-4:2:0 err=%v want ErrFrameInvalidFormat", tc.bitDepth, err)
			}
		})
	}
}

func TestPublicRTCEncoderI420HighBitDepthGenericFrameNative(t *testing.T) {
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T1)
	cfg.Profile = EncoderProfile0
	cfg.RateControl = EncoderRateControlCQP
	cfg.Quantizer = 38
	cfg.ColorConfigSet = true
	cfg.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
	}
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	frame := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 0), FrameFormat{
		Width:        w,
		Height:       h,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	out, err := enc.EncodeFrame(frame, false)
	if err != nil {
		t.Fatalf("EncodeFrame native high-bit-depth I420: %v", err)
	}
	seq := publicFirstSequenceHeader(t, out.Data)
	if seq.SeqProfile != uint8(EncoderProfile0) ||
		seq.ColorConfig.MonoChrome ||
		seq.ColorConfig.BitDepth != 10 ||
		!seq.ColorConfig.SubsamplingX ||
		!seq.ColorConfig.SubsamplingY {
		t.Fatalf("sequence profile=%d color=%+v want native 10-bit 4:2:0", seq.SeqProfile, seq.ColorConfig)
	}

	bad := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 1), FrameFormat{
		Width:        w,
		Height:       h,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	if _, err := enc.EncodeFrame(bad, false); !errors.Is(err, internalframe.ErrInvalidFormat) {
		t.Fatalf("EncodeFrame 8-bit into 10-bit I420 err=%v want ErrFrameInvalidFormat", err)
	}
}

func TestPublicRTCEncoderSetConfigNativeHighBitDepthI420TransitionReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit-profile0", bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-profile2", bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			first, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
			if err != nil {
				t.Fatalf("EncodePicture: %v", err)
			}
			colorConfig := enc.Config()
			colorConfig.Profile = tc.profile
			colorConfig.RateControl = EncoderRateControlCQP
			colorConfig.Quantizer = 39
			colorConfig.ColorConfigSet = true
			colorConfig.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:     tc.bitDepth,
				SubsamplingX: true,
				SubsamplingY: true,
			}
			if err := enc.SetConfig(colorConfig); err != nil {
				t.Fatalf("SetConfig high-bit-depth I420: %v", err)
			}
			if enc.Config().Profile != tc.profile ||
				enc.Config().ColorConfig.BitDepth != tc.bitDepth ||
				enc.Config().ColorConfig.MonoChrome ||
				!enc.Config().ColorConfig.SubsamplingX ||
				!enc.Config().ColorConfig.SubsamplingY {
				t.Fatalf("post-SetConfig config=%+v want profile=%d %d-bit 4:2:0", enc.Config(), tc.profile, tc.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := first.Frames[0].FrameID + uint64(first.FrameNum)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte

			key, err := enc.EncodeI420HighBitDepthPicture(publicI420HighBitDepthFrame(w, h, tc.bitDepth, 1), false)
			if err != nil {
				t.Fatalf("EncodeI420HighBitDepthPicture key after SetConfig: %v", err)
			}
			if !key.Keyframe || key.FrameNum != 1 || key.Frames[0].FrameID != nextFrameID {
				t.Fatalf("high-bit-depth I420 transition key=%+v after first=%+v nextFrameID=%d", key, first, nextFrameID)
			}
			seq := publicFirstSequenceHeader(t, key.Frames[0].Data)
			if seq.SeqProfile != uint8(tc.profile) ||
				seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != tc.bitDepth ||
				!seq.ColorConfig.SubsamplingX ||
				!seq.ColorConfig.SubsamplingY {
				t.Fatalf("transition sequence profile=%d color=%+v want profile=%d %d-bit 4:2:0", seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth)
			}
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), key, true, &nextFrameID)
			appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, key)

			delta, err := enc.EncodeI420HighBitDepthPicture(publicI420HighBitDepthFrame(w, h, tc.bitDepth, 2), false)
			if err != nil {
				t.Fatalf("EncodeI420HighBitDepthPicture delta after SetConfig: %v", err)
			}
			if delta.Keyframe || delta.FrameNum != 1 || delta.Frames[0].FrameID != nextFrameID {
				t.Fatalf("high-bit-depth I420 transition delta=%+v want frameID=%d", delta, nextFrameID)
			}
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), delta, false, &nextFrameID)
			appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, delta)

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "public-rtc-i420-"+tc.name+"-transition", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderHighBitDepthMonochromeGenericFrameNative(t *testing.T) {
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T1)
	cfg.RateControl = EncoderRateControlCQP
	cfg.Quantizer = 38
	cfg.ColorConfigSet = true
	cfg.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:   10,
		MonoChrome: true,
	}
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	frame := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 0), FrameFormat{
		Width:      w,
		Height:     h,
		BitDepth:   10,
		MonoChrome: true,
		Align:      32,
	})
	out, err := enc.EncodeFrame(frame, false)
	if err != nil {
		t.Fatalf("EncodeFrame native high-bit-depth mono: %v", err)
	}
	seq := publicFirstSequenceHeader(t, out.Data)
	if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != 10 {
		t.Fatalf("sequence color=%+v want native 10-bit monochrome", seq.ColorConfig)
	}

	bad := publicFrameFromI420(t, publicRTCMatrixFrame(w, h, 1), FrameFormat{
		Width:      w,
		Height:     h,
		BitDepth:   8,
		MonoChrome: true,
		Align:      32,
	})
	if _, err := enc.EncodeFrame(bad, false); !errors.Is(err, internalframe.ErrInvalidFormat) {
		t.Fatalf("EncodeFrame 8-bit into 10-bit mono err=%v want ErrFrameInvalidFormat", err)
	}
}

func TestPublicRTCEncoderSetConfigNativeMonochromeTransition(t *testing.T) {
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	first, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture: %v", err)
	}
	mono := enc.Config()
	mono.ColorConfigSet = true
	mono.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}
	if err := enc.SetConfig(mono); err != nil {
		t.Fatalf("SetConfig mono: %v", err)
	}
	if !enc.Config().ColorConfig.MonoChrome {
		t.Fatalf("post-SetConfig color=%+v", enc.Config().ColorConfig)
	}
	next, err := enc.EncodeI400Picture(publicI400FromI420(publicRTCMatrixFrame(w, h, 1)), false)
	if err != nil {
		t.Fatalf("EncodeI400Picture after SetConfig: %v", err)
	}
	if !next.Keyframe || next.FrameNum != 1 || next.Frames[0].FrameID != first.Frames[0].FrameID+uint64(first.FrameNum) {
		t.Fatalf("mono transition picture=%+v after first=%+v", next, first)
	}
	seq := publicFirstSequenceHeader(t, next.Frames[0].Data)
	if !seq.ColorConfig.MonoChrome {
		t.Fatalf("transition sequence color=%+v want monochrome", seq.ColorConfig)
	}
}

func TestPublicRTCEncoderSetConfigNativeHighBitDepthMonochromeTransitionReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	cases := []struct {
		name     string
		bitDepth uint8
		profile  internalencoder.Profile
	}{
		{name: "10bit-profile0", bitDepth: 10, profile: EncoderProfile0},
		{name: "12bit-profile2", bitDepth: 12, profile: internalencoder.Profile2},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 192, 128
			cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			first, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
			if err != nil {
				t.Fatalf("EncodePicture: %v", err)
			}
			monoConfig := enc.Config()
			monoConfig.Profile = tc.profile
			monoConfig.RateControl = EncoderRateControlCQP
			monoConfig.Quantizer = 39
			monoConfig.ColorConfigSet = true
			monoConfig.ColorConfig = internalencoder.SequenceColorConfig{
				BitDepth:   tc.bitDepth,
				MonoChrome: true,
			}
			if err := enc.SetConfig(monoConfig); err != nil {
				t.Fatalf("SetConfig high-bit-depth mono: %v", err)
			}
			if enc.Config().Profile != tc.profile ||
				enc.Config().ColorConfig.BitDepth != tc.bitDepth ||
				!enc.Config().ColorConfig.MonoChrome {
				t.Fatalf("post-SetConfig config=%+v want profile=%d %d-bit monochrome", enc.Config(), tc.profile, tc.bitDepth)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := first.Frames[0].FrameID + uint64(first.FrameNum)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte

			key, err := enc.EncodeI400HighBitDepthPicture(publicI400HighBitDepthFromI420(publicRTCMatrixFrame(w, h, 1), tc.bitDepth, 1), false)
			if err != nil {
				t.Fatalf("EncodeI400HighBitDepthPicture key after SetConfig: %v", err)
			}
			if !key.Keyframe || key.FrameNum != 1 || key.Frames[0].FrameID != nextFrameID {
				t.Fatalf("high-bit-depth mono transition key=%+v after first=%+v nextFrameID=%d", key, first, nextFrameID)
			}
			seq := publicFirstSequenceHeader(t, key.Frames[0].Data)
			if seq.SeqProfile != uint8(tc.profile) ||
				!seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != tc.bitDepth {
				t.Fatalf("transition sequence profile=%d color=%+v want profile=%d %d-bit monochrome", seq.SeqProfile, seq.ColorConfig, tc.profile, tc.bitDepth)
			}
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), key, true, &nextFrameID)
			appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, key)

			delta, err := enc.EncodeI400HighBitDepthPicture(publicI400HighBitDepthFromI420(publicRTCMatrixFrame(w, h, 2), tc.bitDepth, 2), false)
			if err != nil {
				t.Fatalf("EncodeI400HighBitDepthPicture delta after SetConfig: %v", err)
			}
			if delta.Keyframe || delta.FrameNum != 1 || delta.Frames[0].FrameID != nextFrameID {
				t.Fatalf("high-bit-depth mono transition delta=%+v want frameID=%d", delta, nextFrameID)
			}
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), delta, false, &nextFrameID)
			appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, delta)

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPMonochromeReferenceDecoders(t, decoders, "public-rtc-i400-"+tc.name+"-transition", enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderSetConfigNativeI422TransitionReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	first, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture: %v", err)
	}
	i422Config := enc.Config()
	i422Config.Profile = internalencoder.Profile2
	i422Config.ColorConfigSet = true
	i422Config.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
	}
	if err := enc.SetConfig(i422Config); err != nil {
		t.Fatalf("SetConfig I422: %v", err)
	}
	if enc.Config().Profile != internalencoder.Profile2 ||
		enc.Config().ColorConfig.MonoChrome ||
		enc.Config().ColorConfig.BitDepth != 8 ||
		!enc.Config().ColorConfig.SubsamplingX ||
		enc.Config().ColorConfig.SubsamplingY {
		t.Fatalf("post-SetConfig config=%+v want native 4:2:2", enc.Config())
	}
	key, err := enc.EncodeI422Picture(publicI422NativePattern(w, h, 1), false)
	if err != nil {
		t.Fatalf("EncodeI422Picture key after SetConfig: %v", err)
	}
	if !key.Keyframe || key.FrameNum != 1 || key.Frames[0].FrameID != first.Frames[0].FrameID+uint64(first.FrameNum) {
		t.Fatalf("I422 transition key=%+v after first=%+v", key, first)
	}
	seq := publicFirstSequenceHeader(t, key.Frames[0].Data)
	if seq.SeqProfile != uint8(internalencoder.Profile2) ||
		seq.ColorConfig.MonoChrome ||
		seq.ColorConfig.BitDepth != 8 ||
		!seq.ColorConfig.SubsamplingX ||
		seq.ColorConfig.SubsamplingY {
		t.Fatalf("transition sequence profile=%d color=%+v want native 8-bit 4:2:2", seq.SeqProfile, seq.ColorConfig)
	}
	keyPayload := append([]byte(nil), key.Frames[0].Data...)
	delta, err := enc.EncodeI422Picture(publicI422NativePattern(w, h, 2), false)
	if err != nil {
		t.Fatalf("EncodeI422Picture delta after SetConfig: %v", err)
	}
	if delta.Keyframe || delta.FrameNum != 1 {
		t.Fatalf("I422 transition delta=%+v want single delta picture", delta)
	}
	deltaPayload := append([]byte(nil), delta.Frames[0].Data...)

	frames := []publicIVFFrame{
		{timestamp: 0, payload: keyPayload},
		{timestamp: 1, payload: deltaPayload},
	}
	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, "public-rtc-i422-transition", ivf)
}

func TestPublicRTCEncoderSetConfigNativeI444TransitionReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	first, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture: %v", err)
	}
	i444Config := enc.Config()
	i444Config.Profile = internalencoder.Profile1
	i444Config.ColorConfigSet = true
	i444Config.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth: 8,
	}
	if err := enc.SetConfig(i444Config); err != nil {
		t.Fatalf("SetConfig I444: %v", err)
	}
	if enc.Config().Profile != internalencoder.Profile1 ||
		enc.Config().ColorConfig.MonoChrome ||
		enc.Config().ColorConfig.SubsamplingX ||
		enc.Config().ColorConfig.SubsamplingY {
		t.Fatalf("post-SetConfig config=%+v want native 4:4:4", enc.Config())
	}
	key, err := enc.EncodeI444Picture(publicI444NativePattern(w, h, 1), false)
	if err != nil {
		t.Fatalf("EncodeI444Picture key after SetConfig: %v", err)
	}
	if !key.Keyframe || key.FrameNum != 1 || key.Frames[0].FrameID != first.Frames[0].FrameID+uint64(first.FrameNum) {
		t.Fatalf("I444 transition key=%+v after first=%+v", key, first)
	}
	seq := publicFirstSequenceHeader(t, key.Frames[0].Data)
	if seq.SeqProfile != uint8(internalencoder.Profile1) ||
		seq.ColorConfig.MonoChrome ||
		seq.ColorConfig.BitDepth != 8 ||
		seq.ColorConfig.SubsamplingX ||
		seq.ColorConfig.SubsamplingY {
		t.Fatalf("transition sequence profile=%d color=%+v want native 8-bit 4:4:4", seq.SeqProfile, seq.ColorConfig)
	}
	keyPayload := append([]byte(nil), key.Frames[0].Data...)
	delta, err := enc.EncodeI444Picture(publicI444NativePattern(w, h, 2), false)
	if err != nil {
		t.Fatalf("EncodeI444Picture delta after SetConfig: %v", err)
	}
	if delta.Keyframe || delta.FrameNum != 1 {
		t.Fatalf("I444 transition delta=%+v want single delta picture", delta)
	}
	deltaPayload := append([]byte(nil), delta.Frames[0].Data...)

	frames := []publicIVFFrame{
		{timestamp: 0, payload: keyPayload},
		{timestamp: 1, payload: deltaPayload},
	}
	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, "public-rtc-i444-transition", ivf)
}

func TestPublicEncoderRuntimeOptions(t *testing.T) {
	const w, h = 192, 128
	t.Run("VideoEncoder", func(t *testing.T) {
		enc, err := NewVideoEncoder(VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 400_000, Framerate: 30,
		})
		if err != nil {
			t.Fatalf("NewVideoEncoder: %v", err)
		}
		defer enc.Close()
		key, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("key Encode: %v", err)
		}
		keyData := append([]byte(nil), key.Data...)
		enc.SetTileColumns(4)
		enc.SetGoldenInterval(0)
		delta, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("delta Encode after runtime option change: %v", err)
		}
		if !key.Keyframe || delta.Keyframe {
			t.Fatalf("runtime options key=%v delta=%v", key.Keyframe, delta.Keyframe)
		}
		dec, err := NewDecoder([][]byte{
			keyData,
			append([]byte(nil), delta.Data...),
		})
		if err != nil {
			t.Fatalf("NewDecoder: %v", err)
		}
		defer dec.Close()
		decoded := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !ok {
				break
			}
			decoded += len(batch)
		}
		if decoded != 2 {
			t.Fatalf("decoded %d frames, want 2", decoded)
		}
	})

	t.Run("RTCEncoder", func(t *testing.T) {
		cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeS2T2)
		enc, err := NewRTCEncoderWithConfig(cfg)
		if err != nil {
			t.Fatalf("NewRTCEncoderWithConfig: %v", err)
		}
		defer enc.Close()
		key, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("key EncodePicture: %v", err)
		}
		keyLastFrameID := key.Frames[key.FrameNum-1].FrameID
		var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
		var orderedTUs [][]byte
		appendPublicRTCPictureLayerData(t, &layerTUs, &orderedTUs, key)
		enc.SetTileColumns(4)
		enc.SetGoldenInterval(0)
		delta, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("delta EncodePicture after runtime option change: %v", err)
		}
		if !key.Keyframe || delta.Keyframe {
			t.Fatalf("runtime options key=%v delta=%v", key.Keyframe, delta.Keyframe)
		}
		if delta.Frames[0].FrameID != keyLastFrameID+1 {
			t.Fatalf("delta frame id=%d after key last id=%d", delta.Frames[0].FrameID, keyLastFrameID)
		}
		appendPublicRTCPictureLayerData(t, &layerTUs, &orderedTUs, delta)
		assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
	})

	var zeroVideo VideoEncoder
	zeroVideo.SetTileColumns(4)
	zeroVideo.SetMaxThreads(1)
	zeroVideo.SetGoldenInterval(0)
	if err := zeroVideo.SetEffortLevel(0); err == nil {
		t.Fatal("zero VideoEncoder SetEffortLevel unexpectedly succeeded")
	}
	var zeroRTC RTCEncoder
	zeroRTC.SetTileColumns(4)
	zeroRTC.SetMaxThreads(1)
	zeroRTC.SetGoldenInterval(0)
}

func TestPublicVideoEncoderConfigSpeedValidation(t *testing.T) {
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: 64, Height: 64,
		TargetBitrate: 100_000, Framerate: 30,
		Speed: EncoderWebRTCMinEffortLevel,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoder min effort: %v", err)
	}
	if err := enc.SetEffortLevel(EncoderWebRTCMaxEffortLevel); err != nil {
		t.Fatalf("SetEffortLevel max effort: %v", err)
	}
	_ = enc.Close()

	_, err = NewVideoEncoder(VideoEncoderConfig{
		Width: 64, Height: 64,
		TargetBitrate: 100_000, Framerate: 30,
		Speed: EncoderWebRTCMinEffortLevel - 1,
	})
	if err == nil {
		t.Fatal("NewVideoEncoder accepted invalid effort")
	}
}

func TestPublicEncoderConfigMaxThreadsControlsTileColumns(t *testing.T) {
	const w, h = 640, 360
	t.Run("VideoEncoder", func(t *testing.T) {
		enc, err := NewVideoEncoder(VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 900_000, Framerate: 30,
			MaxThreads: 1,
		})
		if err != nil {
			t.Fatalf("NewVideoEncoder: %v", err)
		}
		defer enc.Close()
		key, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("single-thread key Encode: %v", err)
		}
		if got := publicFirstFrameTileColumns(t, key.Data); got != 1 {
			t.Fatalf("single-thread tile columns=%d want 1", got)
		}

		enc.SetMaxThreads(4)
		key, err = enc.Encode(publicRTCMatrixFrame(w, h, 1), true)
		if err != nil {
			t.Fatalf("four-thread key Encode: %v", err)
		}
		if got := publicFirstFrameTileColumns(t, key.Data); got != 1 {
			t.Fatalf("four-thread tile columns=%d want 1 (one tile, as SVT-AV1)", got)
		}

		enc.SetMaxThreads(0)
		key, err = enc.Encode(publicRTCMatrixFrame(w, h, 2), true)
		if err != nil {
			t.Fatalf("default-thread key Encode: %v", err)
		}
		if got := publicFirstFrameTileColumns(t, key.Data); got != 1 {
			t.Fatalf("default tile columns=%d want 1 (one tile, as SVT-AV1)", got)
		}
	})

	t.Run("RTCEncoder", func(t *testing.T) {
		enc, err := NewRTCEncoder(VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 900_000, Framerate: 30,
			MaxThreads: 1,
		})
		if err != nil {
			t.Fatalf("NewRTCEncoder: %v", err)
		}
		defer enc.Close()
		if got := enc.Config().MaxThreads; got != 1 {
			t.Fatalf("initial Config MaxThreads=%d want 1", got)
		}
		key, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("single-thread key Encode: %v", err)
		}
		if got := publicFirstFrameTileColumns(t, key.Data); got != 1 {
			t.Fatalf("single-thread tile columns=%d want 1", got)
		}

		enc.SetMaxThreads(4)
		if got := enc.Config().MaxThreads; got != 4 {
			t.Fatalf("updated Config MaxThreads=%d want 4", got)
		}
		key, err = enc.Encode(publicRTCMatrixFrame(w, h, 1), true)
		if err != nil {
			t.Fatalf("four-thread key Encode: %v", err)
		}
		if got := publicFirstFrameTileColumns(t, key.Data); got != 1 {
			t.Fatalf("four-thread tile columns=%d want 1 (one tile, as SVT-AV1)", got)
		}

		enc.SetMaxThreads(0)
		if got := enc.Config().MaxThreads; got != 0 {
			t.Fatalf("default Config MaxThreads=%d want 0", got)
		}
		key, err = enc.Encode(publicRTCMatrixFrame(w, h, 2), true)
		if err != nil {
			t.Fatalf("default-thread key Encode: %v", err)
		}
		if got := publicFirstFrameTileColumns(t, key.Data); got != 1 {
			t.Fatalf("default tile columns=%d want 1 (one tile, as SVT-AV1)", got)
		}
	})
}

func TestPublicRTCEncoderDependencyDescriptorBorrowedLifetime(t *testing.T) {
	t.Run("Encode", func(t *testing.T) {
		const w, h = 192, 128
		enc, err := NewRTCEncoder(VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 250_000, Framerate: 30,
			TemporalLayers: 2,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer enc.Close()

		first, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("first Encode: %v", err)
		}
		snapshot := append([]byte(nil), first.DependencyDescriptor...)
		if len(snapshot) == 0 {
			t.Fatal("first Encode returned empty dependency descriptor")
		}

		second, err := enc.Encode(publicRTCMatrixFrame(w, h, 1), false)
		if err != nil {
			t.Fatalf("second Encode: %v", err)
		}
		if bytes.Equal(second.DependencyDescriptor, snapshot) {
			t.Fatal("second descriptor matched first; test did not exercise descriptor replacement")
		}
		if bytes.Equal(first.DependencyDescriptor, snapshot) {
			t.Fatal("retained Encode dependency descriptor stayed stable after a later encode")
		}
	})

	t.Run("EncodePicture", func(t *testing.T) {
		const w, h = 640, 360
		enc, err := NewRTCEncoderWithConfig(publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeL2T2))
		if err != nil {
			t.Fatal(err)
		}
		defer enc.Close()

		first, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("first EncodePicture: %v", err)
		}
		if first.FrameNum != 2 {
			t.Fatalf("first FrameNum=%d want 2", first.FrameNum)
		}
		var snapshots [EncoderWebRTCMaxSpatialLayers][]byte
		for i := 0; i < first.FrameNum; i++ {
			snapshots[i] = append([]byte(nil), first.Frames[i].DependencyDescriptor...)
			if len(snapshots[i]) == 0 {
				t.Fatalf("first frame %d returned empty dependency descriptor", i)
			}
		}

		second, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 1), false)
		if err != nil {
			t.Fatalf("second EncodePicture: %v", err)
		}
		if second.FrameNum != first.FrameNum {
			t.Fatalf("second FrameNum=%d want %d", second.FrameNum, first.FrameNum)
		}
		for i := 0; i < first.FrameNum; i++ {
			if bytes.Equal(second.Frames[i].DependencyDescriptor, snapshots[i]) {
				t.Fatalf("second frame %d descriptor matched first; test did not exercise descriptor replacement", i)
			}
			if bytes.Equal(first.Frames[i].DependencyDescriptor, snapshots[i]) {
				t.Fatalf("retained EncodePicture frame %d dependency descriptor stayed stable after a later encode", i)
			}
		}
	})
}

func TestPublicRTCEncoderEncodePicture1080pHotPathAllocs(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "L1T3", mode: EncoderScalabilityModeL1T3},
		{name: "S3T3", mode: internalencoder.ScalabilityModeS3T3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const width, height = 1920, 1080
			cfg := publicRTCMatrixConfig(width, height, tc.mode)
			cfg.TargetBitrateKbps = 1800
			cfg.MaxBitrateKbps = 2400
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			frames := [...]I420Frame{
				publicRTCMatrixFrame(width, height, 0),
				publicRTCMatrixFrame(width, height, 1),
				publicRTCMatrixFrame(width, height, 2),
				publicRTCMatrixFrame(width, height, 3),
			}
			warmPublicRTCEncoderEncodePictureHotPath(t, enc, frames[:])

			var allocErr error
			frameIndex := 0
			sum := 0
			allocs := testing.AllocsPerRun(10, func() {
				picture, err := enc.EncodePicture(frames[frameIndex&3], false)
				frameIndex++
				if err != nil {
					allocErr = err
					return
				}
				if picture.FrameNum == 0 {
					allocErr = fmt.Errorf("empty picture")
					return
				}
				for i := 0; i < picture.FrameNum; i++ {
					sum += len(picture.Frames[i].Data) + len(picture.Frames[i].DependencyDescriptor)
				}
			})
			publicRTCEncoderHotPathSink = sum
			if allocErr != nil {
				t.Fatalf("EncodePicture allocation run: %v", allocErr)
			}
			if allocs != 0 {
				t.Fatalf("EncodePicture 1080p %s allocations=%f want 0", tc.name, allocs)
			}
		})
	}
}

func warmPublicRTCEncoderEncodePictureHotPath(tb testing.TB, enc *RTCEncoder, frames []I420Frame) {
	tb.Helper()
	for i := 0; i < len(frames)*2; i++ {
		forceKey := i == 0 || i == len(frames)
		picture, err := enc.EncodePicture(frames[i%len(frames)], forceKey)
		if err != nil {
			tb.Fatalf("warm EncodePicture(%d): %v", i, err)
		}
		if picture.FrameNum == 0 {
			tb.Fatalf("warm EncodePicture(%d) emitted no frames", i)
		}
	}
}

func TestPublicRTCEncoderClose(t *testing.T) {
	const w, h = 640, 360
	cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeL2T2)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	if _, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false); err != nil {
		t.Fatalf("EncodePicture before close: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 1), false); err == nil {
		t.Fatal("EncodePicture after Close succeeded")
	}
	if err := enc.SetConfig(cfg); err == nil {
		t.Fatal("SetConfig after Close succeeded")
	}
}

func TestPublicRTCEncoderNormalizeConfig(t *testing.T) {
	cfg := publicRTCMatrixConfig(640, 360, internalencoder.ScalabilityModeL2T2_KEY_SHIFT)
	cfg.BitDepth = 0
	normalized, err := NormalizeRTCEncoderConfig(cfg)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig valid: %v", err)
	}
	if normalized.BitDepth != 8 ||
		normalized.Profile != EncoderProfile0 ||
		normalized.SpatialLayerCount != 2 ||
		normalized.TemporalLayerCount != 2 ||
		normalized.RateControl != EncoderRateControlCBR {
		t.Fatalf("normalized config=%+v", normalized)
	}
	cqp := cfg
	cqp.RateControl = EncoderRateControlCQP
	cqp.Quantizer = 32
	normalizedCQP, err := NormalizeRTCEncoderConfig(cqp)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig CQP: %v", err)
	}
	if normalizedCQP.RateControl != EncoderRateControlCQP || normalizedCQP.Quantizer != 32 {
		t.Fatalf("normalized CQP config=%+v", normalizedCQP)
	}
	mono := publicRTCMatrixConfig(640, 360, internalencoder.ScalabilityModeL2T2)
	mono.ColorConfigSet = true
	mono.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}
	normalizedMono, err := NormalizeRTCEncoderConfig(mono)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig monochrome: %v", err)
	}
	if !normalizedMono.ColorConfig.MonoChrome ||
		normalizedMono.SpatialLayerCount != 2 ||
		normalizedMono.TemporalLayerCount != 2 {
		t.Fatalf("normalized monochrome config=%+v", normalizedMono)
	}
	i444 := cfg
	i444.Profile = internalencoder.Profile1
	i444.ColorConfigSet = true
	i444.ColorConfig = internalencoder.SequenceColorConfig{
		BitDepth: 8,
	}
	normalizedI444, err := NormalizeRTCEncoderConfig(i444)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig I444: %v", err)
	}
	if normalizedI444.Profile != internalencoder.Profile1 ||
		normalizedI444.ColorConfig.BitDepth != 8 ||
		normalizedI444.ColorConfig.SubsamplingX ||
		normalizedI444.ColorConfig.SubsamplingY {
		t.Fatalf("normalized I444 config=%+v", normalizedI444)
	}
	highBitDepthI420 := cfg
	highBitDepthI420.BitDepth = 10
	normalizedHighBitDepthI420, err := NormalizeRTCEncoderConfig(highBitDepthI420)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig high-bit-depth I420: %v", err)
	}
	if normalizedHighBitDepthI420.Profile != EncoderProfile0 ||
		normalizedHighBitDepthI420.BitDepth != 10 ||
		normalizedHighBitDepthI420.ColorConfig.BitDepth != 10 ||
		normalizedHighBitDepthI420.ColorConfig.MonoChrome ||
		!normalizedHighBitDepthI420.ColorConfig.SubsamplingX ||
		!normalizedHighBitDepthI420.ColorConfig.SubsamplingY {
		t.Fatalf("normalized high-bit-depth I420 config=%+v", normalizedHighBitDepthI420)
	}
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig valid: %v", err)
	}
	defer enc.Close()
	wantLive := enc.Config()

	tests := []struct {
		name string
		edit func(*EncoderConfig)
		want error
	}{
		{
			name: "ambiguous-color-config-without-enable",
			edit: func(cfg *EncoderConfig) {
				cfg.ColorConfig = internalencoder.SequenceColorConfig{
					BitDepth:     8,
					SubsamplingX: true,
					SubsamplingY: true,
				}
			},
			want: internalencoder.ErrInvalidConfig,
		},
		{
			name: "cqp-quantizer-zero",
			edit: func(cfg *EncoderConfig) {
				cfg.RateControl = EncoderRateControlCQP
			},
			want: internalencoder.ErrUnsupported,
		},
		{
			name: "cqp-quantizer-too-high",
			edit: func(cfg *EncoderConfig) {
				cfg.RateControl = EncoderRateControlCQP
				cfg.Quantizer = internalencoder.WebRTCMaxQuantizer + 1
			},
			want: internalencoder.ErrInvalidConfig,
		},
		{
			name: "invalid-framerate",
			edit: func(cfg *EncoderConfig) {
				cfg.MaxFramerate = EncoderRational{Num: 1, Den: 2}
			},
			want: internalencoder.ErrInvalidConfig,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bad := cfg
			tc.edit(&bad)
			if _, err := NormalizeRTCEncoderConfig(bad); !errors.Is(err, tc.want) {
				t.Fatalf("NormalizeRTCEncoderConfig err=%v want %v", err, tc.want)
			}
			if _, err := NewRTCEncoderWithConfig(bad); !errors.Is(err, tc.want) {
				t.Fatalf("NewRTCEncoderWithConfig err=%v want %v", err, tc.want)
			}
			if err := enc.SetConfig(bad); !errors.Is(err, tc.want) {
				t.Fatalf("SetConfig err=%v want %v", err, tc.want)
			}
			if got := enc.Config(); got != wantLive {
				t.Fatalf("SetConfig mutated config to %+v want %+v", got, wantLive)
			}
		})
	}
}

func TestPublicRTCEncoderEncodeRejectsMultiSpatialWithoutMutating(t *testing.T) {
	const w, h = 640, 360
	cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeL2T2)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); err != internalencoder.ErrUnsupported {
		t.Fatalf("Encode multi-spatial err=%v want %v", err, internalencoder.ErrUnsupported)
	}
	picture, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 1), false)
	if err != nil {
		t.Fatalf("EncodePicture after rejected Encode: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].FrameID != 0 ||
		picture.Frames[1].FrameID != 1 {
		t.Fatalf("picture after rejected Encode=%+v", picture)
	}
}

func TestPublicRTCFrameAppendRTPPackets(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	src := I420Frame{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	for i := range src.Y {
		src.Y[i] = uint8(40 + i%170)
	}
	for i := range src.U {
		src.U[i] = 120
		src.V[i] = 130
	}
	frame, err := enc.Encode(src, false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 96}
	var receiver RTPDependencyDescriptorState
	assertPublicRTCFrameRTPPackets(t, &receiver, frame, limits, true, true, true)
}

func TestPublicRTCFrameAppendRTPPacketsWithHeaders(t *testing.T) {
	const w, h = 192, 128
	enc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	limits := RTPPayloadSizeLimits{MaxPayloadLen: 96}
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first: %v", err)
	}
	obuScratch := make([]RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full: %v", err)
	}
	packetScratch := make([]RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	packetSpans := make([]EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, packetSpans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets: %v", err)
	}
	if packetCount != size.Packetizer.Packets || packetCount < 2 {
		t.Fatalf("packet count=%d want %d and fragmentation", packetCount, size.Packetizer.Packets)
	}

	config := EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     96,
		SequenceNumber:                  0xfffe,
		Timestamp:                       0x01020304,
		SSRC:                            0xaabbccdd,
		DependencyDescriptorExtensionID: 42,
		MIDExtensionID:                  1,
		MID:                             "video",
		RTPStreamIDExtensionID:          2,
		RTPStreamID:                     "f",
		RepairedRTPStreamIDExtensionID:  3,
		RepairedRTPStreamID:             "r",
		TransportWideCCExtensionID:      4,
		TransportWideCCSequenceNumber:   0x1234,
		TransportWideCC02ExtensionID:    5,
		TransportWideCC02: RTPTransportWideCC02{
			SequenceNumber:        0x2234,
			FeedbackRequest:       true,
			IncludeTimestamps:     true,
			FeedbackSequenceCount: 3,
		},
		VideoOrientationExtensionID: 6,
		VideoOrientation: RTPCoordinationOfVideoOrientation{
			Camera:   true,
			Flip:     true,
			Rotation: 180,
		},
		PlayoutDelayExtensionID:        7,
		PlayoutDelay:                   RTPPlayoutDelay{MinDelayMs: 120, MaxDelayMs: 560},
		AbsoluteSendTimeExtensionID:    8,
		AbsoluteSendTime:               0x00abc1,
		AbsoluteCaptureTimeExtensionID: 9,
		AbsoluteCaptureTime: RTPAbsoluteCaptureTime{
			AbsoluteCaptureTimestamp:           0x0102030405060708,
			EstimatedCaptureClockOffsetPresent: true,
			EstimatedCaptureClockOffset:        -1234,
		},
		VideoContentTypeExtensionID: 10,
		VideoContentType:            RTPVideoContentTypeScreenshare,
		VideoTimingExtensionID:      11,
		VideoTiming: RTPVideoTiming{
			Flags:                        RTPVideoTimingFlagTriggeredByTimer,
			EncodeStartDeltaMs:           1,
			EncodeFinishDeltaMs:          2,
			PacketizationCompleteDeltaMs: 3,
			PacerExitDeltaMs:             4,
			NetworkTimestampDeltaMs:      5,
			NetworkTimestamp2DeltaMs:     6,
		},
		VideoLayersAllocationExtensionID: 13,
		VideoLayersAllocation: RTPVideoLayersAllocation{
			RTPStreamID:    0,
			RTPStreamCount: 1,
			ActiveSpatialLayers: []RTPVideoLayersAllocationLayer{
				{RTPStreamID: 0, SpatialID: 0, TargetBitratesKbps: []uint32{450, 700}, Width: w, Height: h, Framerate: 30},
			},
			HasResolutionAndFramerate: true,
		},
		ColorSpaceExtensionID: 12,
		ColorSpace: RTPColorSpace{
			Primaries:              RTPColorSpacePrimaryBT709,
			Transfer:               RTPColorSpaceTransferBT709,
			Matrix:                 RTPColorSpaceMatrixBT709,
			Range:                  RTPColorSpaceRangeFull,
			ChromaSitingHorizontal: RTPColorSpaceChromaSitingHalf,
			ChromaSitingVertical:   RTPColorSpaceChromaSitingCollocated,
		},
		HeaderExtensionProfile: RTPExtensionProfileTwoByte | 0x0005,
	}
	headerSpans := make([]EncoderWebRTCRTPPacketHeaderSpan, packetCount)
	oneByteConfig := EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     config.PayloadType,
		SequenceNumber:                  config.SequenceNumber,
		Timestamp:                       config.Timestamp,
		SSRC:                            config.SSRC,
		DependencyDescriptorExtensionID: config.DependencyDescriptorExtensionID,
		HeaderExtensionProfile:          RTPExtensionProfileOneByte,
	}
	if _, err := EncoderWebRTCRTPPacketsWithHeadersSize(oneByteConfig, []byte{0xaa}, make([]byte, 17), []EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 17}}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("one-byte dependency descriptor packet size err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	if _, _, err := AppendEncoderWebRTCRTPPacketsWithHeaders(nil, make([]EncoderWebRTCRTPPacketHeaderSpan, 1), oneByteConfig, []byte{0xaa}, make([]byte, 17), []EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 17}}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("one-byte dependency descriptor packet err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	duplicateExtConfig := config
	duplicateExtConfig.MIDExtensionID = config.DependencyDescriptorExtensionID
	if _, err := EncoderWebRTCRTPPacketsWithHeadersSize(duplicateExtConfig, []byte{0xaa}, []byte{0xbb}, []EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 1}}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("duplicate extension id size err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	danglingExtConfig := config
	danglingExtConfig.VideoOrientationExtensionID = 0
	if _, err := EncoderWebRTCRTPPacketsWithHeadersSize(danglingExtConfig, []byte{0xaa}, []byte{0xbb}, []EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 1}}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("dangling extension value size err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	danglingExtConfig = config
	danglingExtConfig.VideoLayersAllocationExtensionID = 0
	if _, err := EncoderWebRTCRTPPacketsWithHeadersSize(danglingExtConfig, []byte{0xaa}, []byte{0xbb}, []EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 1}}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("dangling VLA extension value size err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	twccOnlyConfig := EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                   98,
		SequenceNumber:                0x4100,
		Timestamp:                     0x11121314,
		SSRC:                          0x0badf00d,
		TransportWideCCExtensionID:    4,
		TransportWideCCSequenceNumber: 0x9000,
		HeaderExtensionProfile:        RTPExtensionProfileTwoByte,
	}
	twccOnlySize, err := EncoderWebRTCRTPPacketsWithHeadersSize(
		twccOnlyConfig, rtpPayloads, descriptors, packetSpans[:packetCount])
	if err != nil {
		t.Fatalf("TWCC-only EncoderWebRTCRTPPacketsWithHeadersSize: %v", err)
	}
	twccOnlySpans := make([]EncoderWebRTCRTPPacketHeaderSpan, packetCount)
	twccOnlyPackets, twccOnlyCount, err := AppendEncoderWebRTCRTPPacketsWithHeaders(
		make([]byte, 0, twccOnlySize.Bytes), twccOnlySpans, twccOnlyConfig,
		rtpPayloads, descriptors, packetSpans[:packetCount])
	if err != nil {
		t.Fatalf("TWCC-only AppendEncoderWebRTCRTPPacketsWithHeaders: %v", err)
	}
	if twccOnlyCount != packetCount {
		t.Fatalf("TWCC-only packet count=%d want %d", twccOnlyCount, packetCount)
	}
	twccRaw := twccOnlyPackets[twccOnlySpans[0].Offset : twccOnlySpans[0].Offset+twccOnlySpans[0].Length]
	if _, err := ParseRTPPacketDependencyDescriptor(twccRaw, config.DependencyDescriptorExtensionID, nil); !errors.Is(err, ErrRTPHeaderExtensionNotFound) {
		t.Fatalf("TWCC-only dependency descriptor err=%v want %v", err, ErrRTPHeaderExtensionNotFound)
	}
	twccPacket, err := ParseRTPPacket(twccRaw)
	if err != nil {
		t.Fatalf("TWCC-only ParseRTPPacket: %v", err)
	}
	twccElement, ok, err := FindRTPHeaderExtensionElement(
		twccPacket.Header.ExtensionProfile, twccPacket.Header.ExtensionPayload, twccOnlyConfig.TransportWideCCExtensionID)
	if err != nil || !ok {
		t.Fatalf("TWCC-only extension ok=%v err=%v", ok, err)
	}
	twccOnly, err := ParseRTPTransportWideCCHeaderExtension(twccElement.Payload)
	if err != nil || twccOnly != twccOnlyConfig.TransportWideCCSequenceNumber {
		t.Fatalf("TWCC-only sequence=%d err=%v want %d", twccOnly, err, twccOnlyConfig.TransportWideCCSequenceNumber)
	}
	if twccOnlySpans[0].DependencyDescriptorOffset != -1 || twccOnlySpans[0].DependencyDescriptorLength != 0 {
		t.Fatalf("TWCC-only dependency span=%+v", twccOnlySpans[0])
	}

	sizeInfo, err := EncoderWebRTCRTPPacketsWithHeadersSize(config, rtpPayloads, descriptors, packetSpans[:packetCount])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize: %v", err)
	}
	if sizeInfo.Packets != packetCount || sizeInfo.Bytes <= len(rtpPayloads) ||
		sizeInfo.MaxPacketBytes <= limits.MaxPayloadLen || sizeInfo.MaxHeaderBytes <= RTPHeaderMinSize {
		t.Fatalf("packet header size=%+v packetCount=%d payloadBytes=%d", sizeInfo, packetCount, len(rtpPayloads))
	}
	prefix := []byte{0xde, 0xad, 0xbe}
	fullDst := make([]byte, len(prefix), len(prefix)+sizeInfo.Bytes)
	copy(fullDst, prefix)
	fullPackets, fullCount, err := AppendEncoderWebRTCRTPPacketsWithHeaders(fullDst, headerSpans, config, rtpPayloads, descriptors, packetSpans[:packetCount])
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCRTPPacketsWithHeaders: %v", err)
	}
	if fullCount != packetCount {
		t.Fatalf("full packet count=%d want %d", fullCount, packetCount)
	}
	if len(fullPackets) != len(prefix)+sizeInfo.Bytes || cap(fullPackets) != cap(fullDst) ||
		&fullPackets[0] != &fullDst[0] || !bytes.Equal(fullPackets[:len(prefix)], prefix) {
		t.Fatalf("full packet dst len=%d cap=%d size=%+v prefix=%x", len(fullPackets), cap(fullPackets), sizeInfo, fullPackets[:len(prefix)])
	}
	var allocErr error
	sizeAllocs := testing.AllocsPerRun(50, func() {
		_, allocErr = EncoderWebRTCRTPPacketsWithHeadersSize(config, rtpPayloads, descriptors, packetSpans[:packetCount])
	})
	if allocErr != nil {
		t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize alloc run: %v", allocErr)
	}
	if sizeAllocs != 0 {
		t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize allocations=%f want 0", sizeAllocs)
	}
	allocDst := make([]byte, 0, sizeInfo.Bytes)
	allocHeaderSpans := make([]EncoderWebRTCRTPPacketHeaderSpan, packetCount)
	appendAllocs := testing.AllocsPerRun(50, func() {
		_, _, allocErr = AppendEncoderWebRTCRTPPacketsWithHeaders(allocDst[:0], allocHeaderSpans, config, rtpPayloads, descriptors, packetSpans[:packetCount])
	})
	if allocErr != nil {
		t.Fatalf("AppendEncoderWebRTCRTPPacketsWithHeaders alloc run: %v", allocErr)
	}
	if appendAllocs != 0 {
		t.Fatalf("AppendEncoderWebRTCRTPPacketsWithHeaders allocations=%f want 0", appendAllocs)
	}

	payloadSlices := make([][]byte, packetCount)
	var receiver RTPDependencyDescriptorState
	totalPacketBytes := 0
	for i := 0; i < packetCount; i++ {
		span := headerSpans[i]
		if span.Offset < len(prefix) || span.Length > sizeInfo.MaxPacketBytes || span.HeaderSize > sizeInfo.MaxHeaderBytes {
			t.Fatalf("packet %d span=%+v size=%+v", i, span, sizeInfo)
		}
		totalPacketBytes += span.Length
		raw := fullPackets[span.Offset : span.Offset+span.Length]
		if i == 0 {
			if _, err := ParseRTPPacketDependencyDescriptor(raw, config.DependencyDescriptorExtensionID+1, &RTPDependencyDescriptorState{}); !errors.Is(err, ErrRTPHeaderExtensionNotFound) {
				t.Fatalf("packet %d missing dependency descriptor err=%v want %v", i, err, ErrRTPHeaderExtensionNotFound)
			}
			parsedWithoutState, err := ParseRTPPacketDependencyDescriptor(raw, config.DependencyDescriptorExtensionID, nil)
			if err != nil {
				t.Fatalf("packet %d ParseRTPPacketDependencyDescriptor without state: %v", i, err)
			}
			if parsedWithoutState.Descriptor.Mandatory.FrameNumber != uint16(frame.FrameID) {
				t.Fatalf("packet %d descriptor without state frame=%d want %d", i, parsedWithoutState.Descriptor.Mandatory.FrameNumber, frame.FrameID)
			}
		}
		descriptorPacket, err := ParseRTPPacketDependencyDescriptor(raw, config.DependencyDescriptorExtensionID, &receiver)
		if err != nil {
			t.Fatalf("packet %d ParseRTPPacketDependencyDescriptor: %v", i, err)
		}
		packet := descriptorPacket.Packet
		if packet.Header.PayloadType != config.PayloadType ||
			packet.Header.SequenceNumber != config.SequenceNumber+uint16(i) ||
			packet.Header.Timestamp != config.Timestamp ||
			packet.Header.SSRC != config.SSRC ||
			packet.Header.Marker != packetSpans[i].Marker ||
			packet.Header.ExtensionProfile != config.HeaderExtensionProfile {
			t.Fatalf("packet %d header=%+v span=%+v", i, packet.Header, packetSpans[i])
		}
		if span.HeaderSize != len(raw)-len(packet.Payload) ||
			span.PayloadOffset != span.Offset+span.HeaderSize ||
			span.PayloadLength != len(packet.Payload) ||
			span.SequenceNumber != packet.Header.SequenceNumber ||
			span.Marker != packet.Header.Marker {
			t.Fatalf("packet %d header span=%+v payload=%d", i, span, len(packet.Payload))
		}
		midElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.MIDExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d MID extension ok=%v err=%v", i, ok, err)
		}
		mid, err := ParseRTPMIDHeaderExtension(midElement.Payload)
		if err != nil || mid != config.MID {
			t.Fatalf("packet %d MID=%q err=%v want %q", i, mid, err, config.MID)
		}
		ridElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.RTPStreamIDExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d RID extension ok=%v err=%v", i, ok, err)
		}
		rid, err := ParseRTPStreamIDHeaderExtension(ridElement.Payload)
		if err != nil || rid != config.RTPStreamID {
			t.Fatalf("packet %d RID=%q err=%v want %q", i, rid, err, config.RTPStreamID)
		}
		rridElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.RepairedRTPStreamIDExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d RRID extension ok=%v err=%v", i, ok, err)
		}
		rrid, err := ParseRTPRepairedStreamIDHeaderExtension(rridElement.Payload)
		if err != nil || rrid != config.RepairedRTPStreamID {
			t.Fatalf("packet %d RRID=%q err=%v want %q", i, rrid, err, config.RepairedRTPStreamID)
		}
		twccElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.TransportWideCCExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d TWCC extension ok=%v err=%v", i, ok, err)
		}
		twcc, err := ParseRTPTransportWideCCHeaderExtension(twccElement.Payload)
		if want := config.TransportWideCCSequenceNumber + uint16(i); err != nil || twcc != want {
			t.Fatalf("packet %d TWCC=%d err=%v want %d", i, twcc, err, want)
		}
		twcc02Element, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.TransportWideCC02ExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d TWCC02 extension ok=%v err=%v", i, ok, err)
		}
		twcc02, err := ParseRTPTransportWideCC02HeaderExtension(twcc02Element.Payload)
		wantTWCC02 := config.TransportWideCC02
		wantTWCC02.SequenceNumber += uint16(i)
		if err != nil || twcc02 != wantTWCC02 {
			t.Fatalf("packet %d TWCC02=%+v err=%v want %+v", i, twcc02, err, wantTWCC02)
		}
		orientationElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.VideoOrientationExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d CVO extension ok=%v err=%v", i, ok, err)
		}
		orientation, err := ParseRTPCoordinationOfVideoOrientationHeaderExtension(orientationElement.Payload)
		if err != nil || orientation != config.VideoOrientation {
			t.Fatalf("packet %d CVO=%+v err=%v want %+v", i, orientation, err, config.VideoOrientation)
		}
		playoutElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.PlayoutDelayExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d playout-delay extension ok=%v err=%v", i, ok, err)
		}
		playout, err := ParseRTPPlayoutDelayHeaderExtension(playoutElement.Payload)
		if err != nil || playout != config.PlayoutDelay {
			t.Fatalf("packet %d playout-delay=%+v err=%v want %+v", i, playout, err, config.PlayoutDelay)
		}
		absSendElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.AbsoluteSendTimeExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d abs-send-time extension ok=%v err=%v", i, ok, err)
		}
		absSend, err := ParseRTPAbsoluteSendTimeHeaderExtension(absSendElement.Payload)
		if err != nil || absSend != config.AbsoluteSendTime {
			t.Fatalf("packet %d abs-send-time=%#x err=%v want %#x", i, absSend, err, config.AbsoluteSendTime)
		}
		absCaptureElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.AbsoluteCaptureTimeExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d abs-capture-time extension ok=%v err=%v", i, ok, err)
		}
		absCapture, err := ParseRTPAbsoluteCaptureTimeHeaderExtension(absCaptureElement.Payload)
		if err != nil || absCapture != config.AbsoluteCaptureTime {
			t.Fatalf("packet %d abs-capture-time=%+v err=%v want %+v", i, absCapture, err, config.AbsoluteCaptureTime)
		}
		contentElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.VideoContentTypeExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d content-type extension ok=%v err=%v", i, ok, err)
		}
		contentType, err := ParseRTPVideoContentTypeHeaderExtension(contentElement.Payload)
		if err != nil || contentType != config.VideoContentType {
			t.Fatalf("packet %d content-type=%d err=%v want %d", i, contentType, err, config.VideoContentType)
		}
		timingElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.VideoTimingExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d video-timing extension ok=%v err=%v", i, ok, err)
		}
		timing, err := ParseRTPVideoTimingHeaderExtension(timingElement.Payload)
		if err != nil || timing != config.VideoTiming {
			t.Fatalf("packet %d video-timing=%+v err=%v want %+v", i, timing, err, config.VideoTiming)
		}
		layersElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.VideoLayersAllocationExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d video-layers-allocation extension ok=%v err=%v", i, ok, err)
		}
		layers, err := ParseRTPVideoLayersAllocationHeaderExtension(layersElement.Payload)
		if err != nil || !sameRTPVideoLayersAllocation(layers, config.VideoLayersAllocation) {
			t.Fatalf("packet %d video-layers-allocation=%+v err=%v want %+v", i, layers, err, config.VideoLayersAllocation)
		}
		colorSpaceElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, config.ColorSpaceExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d color-space extension ok=%v err=%v", i, ok, err)
		}
		colorSpace, err := ParseRTPColorSpaceHeaderExtension(colorSpaceElement.Payload)
		if err != nil || colorSpace != config.ColorSpace {
			t.Fatalf("packet %d color-space=%+v err=%v want %+v", i, colorSpace, err, config.ColorSpace)
		}
		if !bytes.Equal(descriptorPacket.DescriptorPayload, fullPackets[span.DependencyDescriptorOffset:span.DependencyDescriptorOffset+span.DependencyDescriptorLength]) {
			t.Fatalf("packet %d dependency descriptor span mismatch", i)
		}
		parsed := descriptorPacket.Descriptor
		if parsed.Mandatory.FrameNumber != uint16(frame.FrameID) ||
			parsed.Mandatory.FirstPacketInFrame != (i == 0) ||
			parsed.Mandatory.LastPacketInFrame != (i == packetCount-1) {
			t.Fatalf("packet %d mandatory=%+v len=%d", i, parsed.Mandatory, len(descriptorPacket.DescriptorPayload))
		}
		payloadSlices[i] = packet.Payload
	}
	if totalPacketBytes != sizeInfo.Bytes {
		t.Fatalf("total packet bytes=%d want %d", totalPacketBytes, sizeInfo.Bytes)
	}
	assertPublicRTPPayloadsAssembleToFrame(t, frame.Data, payloadSlices)
}

func TestPublicRTCFrameAppendRTPPacketsActiveDecodeTargets(t *testing.T) {
	const w, h = 192, 128
	enc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 96}
	allMask, err := frame.AllDecodeTargetsMask()
	if err != nil {
		t.Fatalf("AllDecodeTargetsMask: %v", err)
	}
	if allMask != 0x03 {
		t.Fatalf("all decode targets mask=%#x want 0x03", allMask)
	}
	activeMask, err := frame.ActiveDecodeTargetsMask(0, 0)
	if err != nil {
		t.Fatalf("ActiveDecodeTargetsMask: %v", err)
	}
	if activeMask != 0x01 {
		t.Fatalf("active decode targets mask=%#x want 0x01", activeMask)
	}
	options := EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		ActiveDecodeTargetsPresentOnFirstPacket: true,
		ActiveDecodeTargetsMask:                 activeMask,
	}
	firstSize, err := frame.RTPPacketScratchLenWithOptions(limits, nil, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions first: %v", err)
	}
	obuScratch := make([]RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLenWithOptions(limits, obuScratch, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions full: %v", err)
	}
	if size.Packetizer.Packets <= 1 || size.MaxDescriptorBytes <= RTPDependencyDescriptorMandatorySize {
		t.Fatalf("scratch=%+v want fragmented frame and extended first descriptor", size)
	}
	packetScratch := make([]RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPacketsWithOptions(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch, options)
	if err != nil {
		t.Fatalf("AppendRTPPacketsWithOptions: %v", err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	payloadSlices := make([][]byte, packetCount)
	var receiver RTPDependencyDescriptorState
	for i := 0; i < packetCount; i++ {
		span := spans[i]
		payloadSlices[i] = rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		desc := descriptors[span.DescriptorOffset : span.DescriptorOffset+span.DescriptorLength]
		parsed, consumed, err := receiver.Parse(desc)
		if err != nil {
			t.Fatalf("packet %d descriptor: %v", i, err)
		}
		if consumed != len(desc) ||
			parsed.Mandatory.FirstPacketInFrame != (i == 0) ||
			parsed.Mandatory.LastPacketInFrame != (i == packetCount-1) {
			t.Fatalf("packet %d mandatory=%+v consumed=%d len=%d", i, parsed.Mandatory, consumed, len(desc))
		}
		if i == 0 {
			if !parsed.HasAttachedStructure ||
				!parsed.HasActiveDecodeTargets ||
				parsed.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask {
				t.Fatalf("first descriptor=%+v", parsed)
			}
			continue
		}
		if parsed.HasAttachedStructure || parsed.HasActiveDecodeTargets {
			t.Fatalf("packet %d repeated optional fields: %+v", i, parsed)
		}
	}
	assertPublicRTPPayloadsAssembleToFrame(t, frame.Data, payloadSlices)
}

func TestPublicRTCPictureActiveDecodeTargetsControlMatrix(t *testing.T) {
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 48}
	fps := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
	}

	if _, err := (RTCPicture{}).AllDecodeTargetsMask(); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("empty picture AllDecodeTargetsMask err=%v want %v", err, ErrEncoderInvalidFrame)
	}
	if _, err := (RTCPicture{}).ActiveDecodeTargetsRTPOptions(0, 0); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("empty picture ActiveDecodeTargetsRTPOptions err=%v want %v", err, ErrEncoderInvalidFrame)
	}

	for step, mode := range EncoderWebRTCScalabilityModes() {
		mode := mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = fps[step%len(fps)]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*19))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var receiver RTPDependencyDescriptorState
			key, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
			if err != nil {
				t.Fatalf("key EncodePicture(%s): %v", mode, err)
			}
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &receiver, enc.Config(), key, limits)

			controlChange := enc.Config()
			controlChange.MaxFramerate = fps[(step+1)%len(fps)]
			publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*23)+77)
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig(%s control): %v", mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
			if err != nil {
				t.Fatalf("delta EncodePicture(%s): %v", mode, err)
			}
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &receiver, enc.Config(), delta, limits)
		})
	}
}

func TestPublicLayeredDecoderRTPPacketDecodeTargetSelection(t *testing.T) {
	const dependencyDescriptorExtensionID = 42
	if _, err := RTPDependencyDescriptorDecodeTargetActive(RTPDependencyDescriptorState{}, 0); !errors.Is(err, ErrRTPInvalidDependencyDescriptor) {
		t.Fatalf("empty RTPDependencyDescriptorDecodeTargetActive err=%v want %v", err, ErrRTPInvalidDependencyDescriptor)
	}

	limits := RTPPayloadSizeLimits{MaxPayloadLen: 32}
	fps := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
	}
	for step, mode := range EncoderWebRTCScalabilityModes() {
		mode := mode
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid WebRTC scalability mode %s", mode)
			}
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = fps[step%len(fps)]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*11))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var receiver RTPDependencyDescriptorState
			var selectedPackets [][]byte
			var selectedLowOverheads [][]byte
			var wantMetadata []publicLayeredFrameMetadata
			selectedPacketCount := 0
			discardedPacketCount := 0
			activeSignals := 0

			for frameIndex := 0; frameIndex < 4; frameIndex++ {
				if frameIndex == 2 {
					controlChange := enc.Config()
					controlChange.MaxFramerate = fps[(step+1)%len(fps)]
					publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*17)+47)
					if err := enc.SetConfig(controlChange); err != nil {
						t.Fatalf("SetConfig(%s control): %v", mode, err)
					}
					assertPublicRTCConfigControls(t, enc.Config(), controlChange)
				}

				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), false)
				if err != nil {
					t.Fatalf("EncodePicture(%s, %d): %v", mode, frameIndex, err)
				}
				options, err := picture.ActiveDecodeTargetsRTPOptions(0, 0)
				if err != nil {
					t.Fatalf("%s picture %d ActiveDecodeTargetsRTPOptions(S0,T0): %v", mode, frameIndex, err)
				}
				wantActiveMask := publicExpectedActiveDecodeTargetsMask(spatialLayers, temporalLayers, 0, 0)
				if !options.ActiveDecodeTargetsPresentOnFirstPacket || options.ActiveDecodeTargetsMask != wantActiveMask {
					t.Fatalf("%s picture %d options=%+v want mask %#x", mode, frameIndex, options, wantActiveMask)
				}

				for outputIndex := 0; outputIndex < picture.FrameNum; outputIndex++ {
					frame := picture.Frames[outputIndex]
					wantForward := frame.SpatialID == 0 && frame.TemporalID == 0
					if wantForward {
						selectedLowOverheads = append(selectedLowOverheads, append([]byte(nil), frame.Data...))
						wantMetadata = append(wantMetadata, publicLayeredFrameMetadataFromRTCFrame(frame))
					}
					framePackets := publicDecoderRTPPacketsForFrameWithLimitsAndOptions(t, frame, limits, options)
					for packetIndex, packet := range framePackets {
						parsed, err := ParseRTPPacketDependencyDescriptor(packet, dependencyDescriptorExtensionID, &receiver)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d ParseRTPPacketDependencyDescriptor: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						descriptor := parsed.Descriptor
						if descriptor.FrameDependencies.SpatialID != frame.SpatialID || descriptor.FrameDependencies.TemporalID != frame.TemporalID {
							t.Fatalf("%s picture %d output %d packet %d dependencies=%+v frame=%+v", mode, frameIndex, outputIndex, packetIndex, descriptor.FrameDependencies, frame)
						}
						if packetIndex == 0 {
							activeSignals++
							if !descriptor.HasActiveDecodeTargets || descriptor.ActiveDecodeTargetsMask != wantActiveMask {
								t.Fatalf("%s picture %d output %d first descriptor=%+v want active mask %#x", mode, frameIndex, outputIndex, descriptor, wantActiveMask)
							}
						}
						active, err := RTPDependencyDescriptorDecodeTargetActive(receiver, 0)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d DecodeTargetActive: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						if !active {
							t.Fatalf("%s picture %d output %d packet %d target 0 unexpectedly inactive", mode, frameIndex, outputIndex, packetIndex)
						}
						if spatialLayers*temporalLayers > 1 {
							inactive, err := RTPDependencyDescriptorDecodeTargetActive(receiver, 1)
							if err != nil {
								t.Fatalf("%s picture %d output %d packet %d DecodeTargetActive(target 1): %v", mode, frameIndex, outputIndex, packetIndex, err)
							}
							if inactive {
								t.Fatalf("%s picture %d output %d packet %d target 1 unexpectedly active under mask %#x", mode, frameIndex, outputIndex, packetIndex, wantActiveMask)
							}
							inactiveForward, err := RTPDependencyDescriptorFrameForwardedForDecodeTarget(descriptor, receiver, 1)
							if err != nil {
								t.Fatalf("%s picture %d output %d packet %d FrameForwardedForDecodeTarget(target 1): %v", mode, frameIndex, outputIndex, packetIndex, err)
							}
							if inactiveForward {
								t.Fatalf("%s picture %d output %d packet %d target 1 unexpectedly forwarded under mask %#x", mode, frameIndex, outputIndex, packetIndex, wantActiveMask)
							}
						}
						matches, err := RTPDependencyDescriptorFrameMatchesDecodeTarget(descriptor, 0)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d FrameMatchesDecodeTarget: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						forward, err := RTPDependencyDescriptorFrameForwardedForDecodeTarget(descriptor, receiver, 0)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d FrameForwardedForDecodeTarget: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						if matches != wantForward || forward != wantForward {
							dti, _ := RTPDependencyDescriptorFrameDecodeTargetIndication(descriptor, 0)
							t.Fatalf("%s picture %d output %d packet %d target0 matches/forward=%v/%v want %v dti=%d deps=%+v",
								mode, frameIndex, outputIndex, packetIndex, matches, forward, wantForward, dti, descriptor.FrameDependencies)
						}
						if forward {
							selectedPackets = append(selectedPackets, append([]byte(nil), packet...))
							selectedPacketCount++
						} else {
							discardedPacketCount++
						}
					}
				}
			}

			if activeSignals == 0 || selectedPacketCount == 0 || len(selectedLowOverheads) == 0 {
				t.Fatalf("%s activeSignals=%d selectedPackets=%d selectedFrames=%d", mode, activeSignals, selectedPacketCount, len(selectedLowOverheads))
			}
			if spatialLayers > 1 || temporalLayers > 1 {
				if discardedPacketCount == 0 {
					t.Fatalf("%s discarded no packets for S0/T0 selection", mode)
				}
			}

			want := decodePublicLayeredDecoderLowOverheadDigests(t, selectedLowOverheads...)
			got := decodePublicLayeredDecoderRTPPacketDigests(t, selectedPackets...)
			gotMetadata := decodePublicLayeredDecoderRTPPacketMetadata(t, selectedPackets...)
			if len(got) != len(want) {
				t.Fatalf("%s selected packet decode frames=%d want %d", mode, len(got), len(want))
			}
			assertPublicLayeredFrameMetadata(t, gotMetadata, wantMetadata)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s selected frame %d digest differs: got=%x want=%x", mode, i, got[i], want[i])
				}
			}
		})
	}
}

func TestPublicRTCFrameAppendRTPPacketsSpatialPicture(t *testing.T) {
	const w, h = 640, 360
	enc, err := NewRTCEncoderWithConfig(EncoderConfig{
		Resolution:        EncoderResolution{Width: w, Height: h},
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    200,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 600,
		Scalability:       internalencoder.ScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatal(err)
	}
	picture, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 {
		t.Fatalf("picture key=%v frames=%d, want true 2", picture.Keyframe, picture.FrameNum)
	}
	if !picture.Frames[0].CodedKeyframe || picture.Frames[0].LastFrameInPicture ||
		picture.Frames[1].CodedKeyframe || !picture.Frames[1].LastFrameInPicture {
		t.Fatalf("picture frame coding/position metadata = %+v %+v", picture.Frames[0], picture.Frames[1])
	}

	limits := RTPPayloadSizeLimits{MaxPayloadLen: 96}
	var receiver RTPDependencyDescriptorState
	for i := 0; i < picture.FrameNum; i++ {
		wantAttached := i == 0
		wantNewCodedVideoSequence := i == 0
		wantMarker := i+1 == picture.FrameNum
		assertPublicRTCFrameRTPPackets(t, &receiver, picture.Frames[i], limits, wantAttached, wantNewCodedVideoSequence, wantMarker)
	}
}

func TestPublicRTCEncoderEncodePictureKeyShiftSpatial(t *testing.T) {
	const w, h = 640, 360
	cw, ch := w/2, h/2
	enc, err := NewRTCEncoderWithConfig(EncoderConfig{
		Resolution:        EncoderResolution{Width: w, Height: h},
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    200,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 600,
		Scalability:       internalencoder.ScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatal(err)
	}
	makeFrame := func(n int) I420Frame {
		f := I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(55 + (i/7+n*9)%160)
		}
		for i := range f.U {
			f.U[i] = 118
			f.V[i] = 134
		}
		return f
	}

	key, err := enc.EncodePicture(makeFrame(0), false)
	if err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}
	if key.FrameNum != 2 || !key.Keyframe {
		t.Fatalf("key FrameNum=%d Keyframe=%v, want 2 true", key.FrameNum, key.Keyframe)
	}
	cloneRTCPictureData(&key)
	for i := 0; i < key.FrameNum; i++ {
		frame := key.Frames[i]
		if frame.SpatialID != uint8(i) || frame.TemporalID != 0 || frame.FrameID != uint64(i) || !frame.Keyframe {
			t.Fatalf("key frame %d metadata S%d T%d id=%d key=%v", i, frame.SpatialID, frame.TemporalID, frame.FrameID, frame.Keyframe)
		}
		if len(frame.DependencyDescriptor) == 0 || !rtpPayloadHasLayerOBU(t, frame.Data, 0, uint8(i)) {
			t.Fatalf("key frame %d missing descriptor or layer OBUs", i)
		}
	}

	delta0, err := enc.EncodePicture(makeFrame(1), false)
	if err != nil {
		t.Fatalf("delta0 EncodePicture: %v", err)
	}
	cloneRTCPictureData(&delta0)
	delta1, err := enc.EncodePicture(makeFrame(2), false)
	if err != nil {
		t.Fatalf("delta1 EncodePicture: %v", err)
	}
	cloneRTCPictureData(&delta1)
	want0 := [2]uint8{0, 1}
	want1 := [2]uint8{1, 0}
	for i := 0; i < 2; i++ {
		if delta0.Frames[i].TemporalID != want0[i] || delta0.Frames[i].SpatialID != uint8(i) {
			t.Fatalf("delta0 frame %d S%d T%d, want S%d T%d", i, delta0.Frames[i].SpatialID, delta0.Frames[i].TemporalID, i, want0[i])
		}
		if delta1.Frames[i].TemporalID != want1[i] || delta1.Frames[i].SpatialID != uint8(i) {
			t.Fatalf("delta1 frame %d S%d T%d, want S%d T%d", i, delta1.Frames[i].SpatialID, delta1.Frames[i].TemporalID, i, want1[i])
		}
		if !rtpPayloadHasLayerOBU(t, delta0.Frames[i].Data, want0[i], uint8(i)) ||
			!rtpPayloadHasLayerOBU(t, delta1.Frames[i].Data, want1[i], uint8(i)) {
			t.Fatalf("delta frame %d missing expected OBU layer IDs", i)
		}
	}

	assertPublicRTCSharedReferenceStreamDecodes(t, publicRTCPictureFramesInOrder(key, delta0, delta1))
}

func TestPublicRTCEncoderSupportsWebRTCPixelModeMatrix(t *testing.T) {
	groups := []struct {
		name   string
		width  int
		height int
		modes  []EncoderScalabilityMode
	}{
		{
			name:   "single-spatial",
			width:  192,
			height: 128,
			modes: []EncoderScalabilityMode{
				EncoderScalabilityModeL1T1,
				EncoderScalabilityModeL1T2,
				EncoderScalabilityModeL1T3,
			},
		},
		{
			name:   "two-spatial",
			width:  640,
			height: 360,
			modes: []EncoderScalabilityMode{
				internalencoder.ScalabilityModeL2T1,
				internalencoder.ScalabilityModeL2T1h,
				internalencoder.ScalabilityModeL2T1_KEY,
				internalencoder.ScalabilityModeL2T2,
				internalencoder.ScalabilityModeL2T2h,
				internalencoder.ScalabilityModeL2T2_KEY,
				internalencoder.ScalabilityModeL2T2_KEY_SHIFT,
				internalencoder.ScalabilityModeL2T3,
				internalencoder.ScalabilityModeL2T3h,
				internalencoder.ScalabilityModeL2T3_KEY,
				EncoderScalabilityModeL2T3_KEY_SHIFT,
				internalencoder.ScalabilityModeS2T1,
				internalencoder.ScalabilityModeS2T1h,
				internalencoder.ScalabilityModeS2T2,
				internalencoder.ScalabilityModeS2T2h,
				internalencoder.ScalabilityModeS2T3,
				internalencoder.ScalabilityModeS2T3h,
			},
		},
		{
			name:   "three-spatial",
			width:  1008,
			height: 576,
			modes: []EncoderScalabilityMode{
				internalencoder.ScalabilityModeL3T1,
				internalencoder.ScalabilityModeL3T1h,
				internalencoder.ScalabilityModeL3T1_KEY,
				internalencoder.ScalabilityModeL3T2,
				internalencoder.ScalabilityModeL3T2h,
				internalencoder.ScalabilityModeL3T2_KEY,
				EncoderScalabilityModeL3T2_KEY_SHIFT,
				internalencoder.ScalabilityModeL3T3,
				internalencoder.ScalabilityModeL3T3h,
				internalencoder.ScalabilityModeL3T3_KEY,
				EncoderScalabilityModeL3T3_KEY_SHIFT,
				internalencoder.ScalabilityModeS3T1,
				internalencoder.ScalabilityModeS3T1h,
				internalencoder.ScalabilityModeS3T2,
				internalencoder.ScalabilityModeS3T2h,
				internalencoder.ScalabilityModeS3T3,
				internalencoder.ScalabilityModeS3T3h,
			},
		},
	}
	for _, group := range groups {
		t.Run(group.name, func(t *testing.T) {
			for _, mode := range group.modes {
				cfg := publicRTCMatrixConfig(group.width, group.height, mode)
				enc, err := NewRTCEncoderWithConfig(cfg)
				if err != nil {
					t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
				}
				var receiver RTPDependencyDescriptorState
				var rtpReceiver RTPDependencyDescriptorState
				nextFrameID := uint64(0)
				var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
				var orderedTUs [][]byte
				for frame := 0; frame < 3; frame++ {
					picture, err := enc.EncodePicture(publicRTCMatrixFrame(group.width, group.height, frame), false)
					if err != nil {
						t.Fatalf("EncodePicture(%s, %d): %v", mode, frame, err)
					}
					appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
					assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), picture, frame == 0, &nextFrameID)
				}
				assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			}
		})
	}
}

func TestPublicRTCEncoderSupportsWebRTC1080pThreeLayerSmallStepModes(t *testing.T) {
	modes := []EncoderScalabilityMode{
		internalencoder.ScalabilityModeL3T1h,
		internalencoder.ScalabilityModeL3T2h,
		internalencoder.ScalabilityModeL3T3h,
		internalencoder.ScalabilityModeS3T1h,
		internalencoder.ScalabilityModeS3T2h,
		internalencoder.ScalabilityModeS3T3h,
	}
	for _, mode := range modes {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := publicRTCMatrixConfig(1920, 1080, mode)
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()
			got := enc.Config()
			if got.SpatialLayerCount != 3 || got.SpatialLayers[0].Resolution != (EncoderResolution{Width: 853, Height: 480}) {
				t.Fatalf("%s normalized layers=%d base=%+v", mode, got.SpatialLayerCount, got.SpatialLayers[0].Resolution)
			}
			for frame := 0; frame < 2; frame++ {
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(1920, 1080, frame), false)
				if err != nil {
					t.Fatalf("EncodePicture(%s, %d): %v", mode, frame, err)
				}
				if picture.FrameNum != 3 {
					t.Fatalf("%s picture frames=%d want 3", mode, picture.FrameNum)
				}
			}
		})
	}
}

func TestPublicRTCEncoderCQPScalabilityModeMatrixDecodes(t *testing.T) {
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	for step, mode := range modes {
		t.Run(string(mode), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = uint8(24 + step%32)

			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()
			if got := enc.Config(); got.RateControl != EncoderRateControlCQP || got.Quantizer != cfg.Quantizer {
				t.Fatalf("config rate control=%d q=%d want CQP q=%d", got.RateControl, got.Quantizer, cfg.Quantizer)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			for frame := 0; frame < 2; frame++ {
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frame), false)
				if err != nil {
					t.Fatalf("EncodePicture(%s, %d): %v", mode, frame, err)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, frame == 0, &nextFrameID)
			}
			controlChange := enc.Config()
			controlChange.MaxFramerate = []EncoderRational{
				{Num: 60, Den: 1},
				{Num: 30000, Den: 1001},
				{Num: 24, Den: 1},
			}[step%3]
			controlChange.Quantizer = uint8(31 + step%29)
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig(%s forced-key controls): %v", mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)

			forcedKey, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), true)
			if err != nil {
				t.Fatalf("forced key EncodePicture(%s): %v", mode, err)
			}
			appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, forcedKey)
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), forcedKey, true, &nextFrameID)

			postForceDelta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 3), false)
			if err != nil {
				t.Fatalf("post-forced-key EncodePicture(%s): %v", mode, err)
			}
			appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, postForceDelta)
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), postForceDelta, false, &nextFrameID)
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderSetConfigReconfigure(t *testing.T) {
	const w, h = 640, 360
	cw, ch := w/2, h/2
	makeFrame := func(n int) I420Frame {
		f := I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(45 + (i+n*11)%160)
		}
		for i := range f.U {
			f.U[i] = 119
			f.V[i] = 132
		}
		return f
	}
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: w, Height: h},
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       EncoderScalabilityModeL1T2,
	}
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 3000, Den: 1}) {
		t.Fatalf("initial RTPFrameDuration=%+v err=%v", got, err)
	}
	key, err := enc.Encode(makeFrame(0), false)
	if err != nil {
		t.Fatalf("key Encode: %v", err)
	}
	delta, err := enc.Encode(makeFrame(1), false)
	if err != nil {
		t.Fatalf("delta Encode: %v", err)
	}
	if !key.Keyframe || delta.Keyframe || key.FrameID != 0 || delta.FrameID != 1 {
		t.Fatalf("warm key=%+v delta=%+v", key, delta)
	}

	controlChange := cfg
	controlChange.MaxFramerate = EncoderRational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1200
	controlChange.TargetBitrateKbps = 900
	if err := enc.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 1500, Den: 1}) {
		t.Fatalf("control RTPFrameDuration=%+v err=%v", got, err)
	}
	delta, err = enc.Encode(makeFrame(2), false)
	if err != nil {
		t.Fatalf("delta after control change: %v", err)
	}
	if delta.Keyframe || delta.FrameID != 2 || enc.Config().TargetBitrateKbps != 900 {
		t.Fatalf("control change delta=%+v config=%+v", delta, enc.Config())
	}

	structureChange := controlChange
	structureChange.Scalability = internalencoder.ScalabilityModeS2T2
	if err := enc.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	picture, err := enc.EncodePicture(makeFrame(3), false)
	if err != nil {
		t.Fatalf("key after structure change: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].FrameID != 3 || picture.Frames[1].FrameID != 4 {
		t.Fatalf("structure change picture=%+v", picture)
	}

	fullSVC := structureChange
	fullSVC.Scalability = internalencoder.ScalabilityModeL2T2
	if err := enc.SetConfig(fullSVC); err != nil {
		t.Fatalf("SetConfig full SVC: %v", err)
	}
	picture, err = enc.EncodePicture(makeFrame(4), false)
	if err != nil {
		t.Fatalf("key after full SVC config: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].FrameID != 5 || picture.Frames[1].FrameID != 6 {
		t.Fatalf("full SVC key picture=%+v", picture)
	}
}

func TestPublicRTCEncoderSetConfigCQPTransitionsDecode(t *testing.T) {
	const w, h = 640, 360
	cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeL2T2)
	cfg.RateControl = EncoderRateControlCQP
	cfg.Quantizer = 37

	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var descriptorReceiver RTPDependencyDescriptorState
	var rtpReceiver RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
	var orderedTUs [][]byte
	encode := func(label string, frame int, wantKey bool) RTCPicture {
		t.Helper()
		picture, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, frame), false)
		if err != nil {
			t.Fatalf("%s EncodePicture: %v", label, err)
		}
		appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, wantKey, &nextFrameID)
		return picture
	}

	encode("initial CQP key", 0, true)

	cqpChange := enc.Config()
	cqpChange.Quantizer = 29
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP q change: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	encode("CQP q change delta", 1, false)

	cbrChange := enc.Config()
	cbrChange.RateControl = EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = EncoderRational{Num: 60, Den: 1}
	publicRTCApplyControlBitrates(&cbrChange, 1100)
	if err := enc.SetConfig(cbrChange); err != nil {
		t.Fatalf("SetConfig CBR transition: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
	encode("CBR transition delta", 2, false)

	backToCQP := enc.Config()
	backToCQP.RateControl = EncoderRateControlCQP
	backToCQP.Quantizer = 41
	if err := enc.SetConfig(backToCQP); err != nil {
		t.Fatalf("SetConfig CQP transition: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), backToCQP)
	encode("CQP transition delta", 3, false)

	assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
}

func TestPublicRTCEncoderSingleSpatialSettingsReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 192, 128
	cfg := publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T2)
	publicRTCApplyControlBitrates(&cfg, 420)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var frames []publicIVFFrame
	var sequence SequenceHeader
	appendFrame := func(label string, frameIndex int, wantKey bool, wantScreen bool) {
		t.Helper()
		frame, err := enc.Encode(publicRTCMatrixFrame(w, h, frameIndex), false)
		if err != nil {
			t.Fatalf("%s Encode: %v", label, err)
		}
		if frame.Keyframe != wantKey {
			t.Fatalf("%s key=%v want %v frame=%+v", label, frame.Keyframe, wantKey, frame)
		}
		assertPublicRTCFrameScreenContentHeader(t, label, frame.Data, &sequence, wantScreen)
		frames = append(frames, publicIVFFrame{
			timestamp: uint64(len(frames)),
			payload:   append([]byte(nil), frame.Data...),
		})
	}

	appendFrame("initial CBR key", 0, true, false)
	appendFrame("warm CBR delta", 1, false, false)

	fpsBitrateChange := enc.Config()
	fpsBitrateChange.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
	publicRTCApplyControlBitrates(&fpsBitrateChange, 700)
	if err := enc.SetConfig(fpsBitrateChange); err != nil {
		t.Fatalf("SetConfig fps/bitrate: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 3003, Den: 2}) {
		t.Fatalf("fps/bitrate RTPFrameDuration=%+v err=%v", got, err)
	}
	appendFrame("fps bitrate delta", 2, false, false)

	cqpChange := enc.Config()
	cqpChange.RateControl = EncoderRateControlCQP
	cqpChange.Quantizer = 35
	cqpChange.Content = EncoderContentScreen
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP/screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	appendFrame("CQP screen delta", 3, false, true)

	scalabilityChange := enc.Config()
	scalabilityChange.Scalability = EncoderScalabilityModeL1T3
	scalabilityChange.MaxFramerate = EncoderRational{Num: 24, Den: 1}
	scalabilityChange.Quantizer = 31
	if err := enc.SetConfig(scalabilityChange); err != nil {
		t.Fatalf("SetConfig scalability screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 3750, Den: 1}) {
		t.Fatalf("scalability RTPFrameDuration=%+v err=%v", got, err)
	}
	appendFrame("L1T3 screen key", 4, true, true)
	appendFrame("L1T3 screen delta", 5, false, true)

	cbrChange := enc.Config()
	cbrChange.RateControl = EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
	cbrChange.Content = EncoderContentCamera
	publicRTCApplyControlBitrates(&cbrChange, 520)
	if err := enc.SetConfig(cbrChange); err != nil {
		t.Fatalf("SetConfig CBR/camera: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
	appendFrame("CBR camera delta", 6, false, false)

	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, "single-spatial-settings", ivf)
}

func TestPublicRTCEncoderSimulcastSettingsReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 1008, 576
	cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeS3T3h)
	cfg.MaxFramerate = EncoderRational{Num: 30, Den: 1}
	publicRTCApplyControlBitrates(&cfg, 1500)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	normalized := enc.Config()
	spatialLayers, _, _, ok := normalized.Scalability.Layers()
	if !ok || spatialLayers != 3 {
		t.Fatalf("normalized scalability=%s spatial=%d ok=%v", normalized.Scalability, spatialLayers, ok)
	}
	var layerRes [EncoderWebRTCMaxSpatialLayers]EncoderResolution
	for i := uint8(0); i < spatialLayers; i++ {
		layerRes[i] = normalized.SpatialLayers[i].Resolution
	}

	var receiver RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	var layerFrames [EncoderWebRTCMaxSpatialLayers][]publicIVFFrame
	var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
	appendPicture := func(label string, frameIndex int, wantKey bool, wantScreen bool) {
		t.Helper()
		picture, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, frameIndex), false)
		if err != nil {
			t.Fatalf("%s EncodePicture: %v", label, err)
		}
		if picture.Keyframe != wantKey {
			t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
		}
		current := enc.Config()
		assertPublicRTCPictureDescriptors(t, &receiver, current, picture, wantKey, &nextFrameID)
		for i := 0; i < picture.FrameNum; i++ {
			frame := picture.Frames[i]
			if frame.SpatialID >= spatialLayers {
				t.Fatalf("%s frame %d spatial=%d want <%d", label, i, frame.SpatialID, spatialLayers)
			}
			if current.SpatialLayers[frame.SpatialID].Resolution != layerRes[frame.SpatialID] {
				t.Fatalf("%s spatial %d resolution=%+v want stable %+v",
					label, frame.SpatialID, current.SpatialLayers[frame.SpatialID].Resolution, layerRes[frame.SpatialID])
			}
			assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], wantScreen)
			layerFrames[frame.SpatialID] = append(layerFrames[frame.SpatialID], publicIVFFrame{
				timestamp: uint64(len(layerFrames[frame.SpatialID])),
				payload:   append([]byte(nil), frame.Data...),
			})
		}
	}

	appendPicture("initial S3T3h CBR key", 0, true, false)
	appendPicture("warm S3T3h CBR delta", 1, false, false)

	fpsBitrateChange := enc.Config()
	fpsBitrateChange.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
	publicRTCApplyControlBitrates(&fpsBitrateChange, 1800)
	if err := enc.SetConfig(fpsBitrateChange); err != nil {
		t.Fatalf("SetConfig fps/bitrate: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
	appendPicture("S3T3h fps bitrate delta", 2, false, false)

	cqpChange := enc.Config()
	cqpChange.RateControl = EncoderRateControlCQP
	cqpChange.Quantizer = 33
	cqpChange.Content = EncoderContentScreen
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP/screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	appendPicture("S3T3h CQP screen delta", 3, false, true)

	scalabilityChange := enc.Config()
	scalabilityChange.Scalability = internalencoder.ScalabilityModeS3T2h
	scalabilityChange.MaxFramerate = EncoderRational{Num: 24, Den: 1}
	scalabilityChange.RateControl = EncoderRateControlCQP
	scalabilityChange.Quantizer = 29
	if err := enc.SetConfig(scalabilityChange); err != nil {
		t.Fatalf("SetConfig scalability screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
	appendPicture("S3T2h screen structure key", 4, true, true)
	appendPicture("S3T2h CQP screen delta", 5, false, true)

	cbrChange := enc.Config()
	cbrChange.RateControl = EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
	cbrChange.Content = EncoderContentCamera
	publicRTCApplyControlBitrates(&cbrChange, 1350)
	if err := enc.SetConfig(cbrChange); err != nil {
		t.Fatalf("SetConfig CBR/camera: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
	appendPicture("S3T2h CBR camera delta", 6, false, false)

	for spatialID := uint8(0); spatialID < spatialLayers; spatialID++ {
		res := layerRes[spatialID]
		if len(layerFrames[spatialID]) == 0 {
			t.Fatalf("spatial %d has no frames", spatialID)
		}
		ivf := appendPublicIVF(nil, uint16(res.Width), uint16(res.Height), 30, 1, layerFrames[spatialID])
		assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, fmt.Sprintf("simulcast-spatial-%d", spatialID), ivf)
	}
}

func TestPublicRTCEncoderSharedSVCSettingsReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	const w, h = 640, 360
	cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeL2T2_KEY_SHIFT)
	cfg.MaxFramerate = EncoderRational{Num: 30, Den: 1}
	publicRTCApplyControlBitrates(&cfg, 900)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var descriptorReceiver RTPDependencyDescriptorState
	var activeReceiver RTPDependencyDescriptorState
	activeLimits := RTPPayloadSizeLimits{MaxPayloadLen: 48}
	nextFrameID := uint64(0)
	var sequence SequenceHeader
	var frames []publicIVFFrame
	var lowOverheads [][]byte
	appendPicture := func(label string, frameIndex int, forceKey bool, wantKey bool, wantScreen bool) {
		t.Helper()
		picture, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, frameIndex), forceKey)
		if err != nil {
			t.Fatalf("%s EncodePicture: %v", label, err)
		}
		if picture.Keyframe != wantKey {
			t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
		}
		current := enc.Config()
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
		assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, current, picture, activeLimits)
		for i := 0; i < picture.FrameNum; i++ {
			frame := picture.Frames[i]
			assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &sequence, wantScreen)
			payload := append([]byte(nil), frame.Data...)
			frames = append(frames, publicIVFFrame{
				timestamp: uint64(len(frames)),
				payload:   payload,
			})
			lowOverheads = append(lowOverheads, payload)
		}
	}

	appendPicture("initial L2T2_KEY_SHIFT CBR key", 0, false, true, false)
	appendPicture("warm L2T2_KEY_SHIFT CBR delta", 1, false, false, false)

	fpsBitrateChange := enc.Config()
	fpsBitrateChange.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
	publicRTCApplyControlBitrates(&fpsBitrateChange, 1100)
	if err := enc.SetConfig(fpsBitrateChange); err != nil {
		t.Fatalf("SetConfig fps/bitrate: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 3003, Den: 2}) {
		t.Fatalf("fps/bitrate RTPFrameDuration=%+v err=%v", got, err)
	}
	appendPicture("L2T2_KEY_SHIFT fps bitrate delta", 2, false, false, false)

	cqpChange := enc.Config()
	cqpChange.RateControl = EncoderRateControlCQP
	cqpChange.Quantizer = 32
	cqpChange.Content = EncoderContentScreen
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP/screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	appendPicture("L2T2_KEY_SHIFT CQP screen delta", 3, false, false, true)
	appendPicture("L2T2_KEY_SHIFT forced screen key", 4, true, true, true)

	scalabilityChange := enc.Config()
	scalabilityChange.Scalability = EncoderScalabilityModeL2T3_KEY_SHIFT
	scalabilityChange.MaxFramerate = EncoderRational{Num: 24, Den: 1}
	scalabilityChange.Quantizer = 29
	if err := enc.SetConfig(scalabilityChange); err != nil {
		t.Fatalf("SetConfig scalability screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 3750, Den: 1}) {
		t.Fatalf("scalability RTPFrameDuration=%+v err=%v", got, err)
	}
	appendPicture("L2T3_KEY_SHIFT screen structure key", 5, false, true, true)
	appendPicture("L2T3_KEY_SHIFT CQP screen delta", 6, false, false, true)

	cbrChange := enc.Config()
	cbrChange.RateControl = EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
	cbrChange.Content = EncoderContentCamera
	publicRTCApplyControlBitrates(&cbrChange, 780)
	if err := enc.SetConfig(cbrChange); err != nil {
		t.Fatalf("SetConfig CBR/camera: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
	appendPicture("L2T3_KEY_SHIFT CBR camera delta", 7, false, false, false)

	wantYUV, decodedCount := decodePublicLayeredLowOverheadRawYUV(t, lowOverheads...)
	if decodedCount != len(lowOverheads) {
		t.Fatalf("shared-SVC decoded frames=%d want %d", decodedCount, len(lowOverheads))
	}
	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, "shared-svc-settings", ivf, wantYUV, decodedCount)
}

func TestPublicRTCEncoderScalabilityModeCatalogueReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
	}
	for step, mode := range modes {
		step, mode := step, mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = fpsCycle[step%len(fpsCycle)]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*9))
			if step%2 == 0 {
				cfg.RateControl = EncoderRateControlCQP
				cfg.Quantizer = uint8(24 + step%31)
			}
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerFrames [EncoderWebRTCMaxSpatialLayers][]publicIVFFrame
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var orderedFrames []publicIVFFrame
			var lowOverheads [][]byte
			var sharedSequence SequenceHeader
			appendPicture := func(label string, frameIndex int, forceKey bool, wantKey bool, wantScreen bool) {
				t.Helper()
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), forceKey)
				if err != nil {
					t.Fatalf("%s EncodePicture(%s): %v", label, mode, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					payload := append([]byte(nil), frame.Data...)
					if publicRTCSharedReferenceSlotMode(mode) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), payload, &sharedSequence, wantScreen)
						orderedFrames = append(orderedFrames, publicIVFFrame{
							timestamp: uint64(len(orderedFrames)),
							payload:   payload,
						})
						lowOverheads = append(lowOverheads, payload)
						continue
					}
					if frame.SpatialID >= EncoderWebRTCMaxSpatialLayers {
						t.Fatalf("%s spatial id=%d", label, frame.SpatialID)
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), payload, &layerSequences[frame.SpatialID], wantScreen)
					layerFrames[frame.SpatialID] = append(layerFrames[frame.SpatialID], publicIVFFrame{
						timestamp: uint64(len(layerFrames[frame.SpatialID])),
						payload:   payload,
					})
				}
			}

			appendPicture("initial key", 0, false, true, cfg.Content == EncoderContentScreen)

			controlChange := enc.Config()
			controlChange.MaxFramerate = fpsCycle[(step+1)%len(fpsCycle)]
			if controlChange.RateControl == EncoderRateControlCQP {
				controlChange.RateControl = EncoderRateControlCBR
				controlChange.Quantizer = 0
				publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*11)+57)
			} else {
				controlChange.RateControl = EncoderRateControlCQP
				controlChange.Quantizer = uint8(29 + step%27)
			}
			if step%3 == 0 {
				controlChange.Content = EncoderContentScreen
			} else {
				controlChange.Content = EncoderContentCamera
			}
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig(%s control): %v", mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			appendPicture("control delta", 1, false, false, controlChange.Content == EncoderContentScreen)
			appendPicture("forced key", 2, true, true, controlChange.Content == EncoderContentScreen)

			secondControlChange := enc.Config()
			secondControlChange.MaxFramerate = fpsCycle[(step+2)%len(fpsCycle)]
			if secondControlChange.RateControl == EncoderRateControlCQP {
				secondControlChange.RateControl = EncoderRateControlCBR
				secondControlChange.Quantizer = 0
				publicRTCApplyControlBitrates(&secondControlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*13)+91)
			} else {
				secondControlChange.RateControl = EncoderRateControlCQP
				secondControlChange.Quantizer = uint8(31 + step%23)
			}
			if secondControlChange.Content == EncoderContentScreen {
				secondControlChange.Content = EncoderContentCamera
			} else {
				secondControlChange.Content = EncoderContentScreen
			}
			if err := enc.SetConfig(secondControlChange); err != nil {
				t.Fatalf("SetConfig(%s second control): %v", mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), secondControlChange)
			appendPicture("second control delta", 3, false, false, secondControlChange.Content == EncoderContentScreen)

			normalized := enc.Config()
			if publicRTCSharedReferenceSlotMode(mode) {
				wantYUV, decodedCount := decodePublicLayeredLowOverheadRawYUV(t, lowOverheads...)
				if decodedCount != len(lowOverheads) {
					t.Fatalf("%s shared-SVC decoded frames=%d want %d", mode, decodedCount, len(lowOverheads))
				}
				ivf := appendPublicIVF(nil, uint16(width), uint16(height), 30, 1, orderedFrames)
				assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, "catalogue-"+mode.String(), ivf, wantYUV, decodedCount)
				return
			}

			for spatialID := uint8(0); spatialID < normalized.SpatialLayerCount; spatialID++ {
				if len(layerFrames[spatialID]) == 0 {
					t.Fatalf("%s spatial %d has no frames", mode, spatialID)
				}
				res := normalized.SpatialLayers[spatialID].Resolution
				ivf := appendPublicIVF(nil, uint16(res.Width), uint16(res.Height), 30, 1, layerFrames[spatialID])
				assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, fmt.Sprintf("catalogue-%s-spatial-%d", mode, spatialID), ivf)
			}
		})
	}
}

func TestPublicRTCEncoderSetConfigScalabilityTransitionReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
		{Num: 15, Den: 1},
	}
	type transitionPhase struct {
		layerFrames    [EncoderWebRTCMaxSpatialLayers][]publicIVFFrame
		layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
		orderedFrames  []publicIVFFrame
		lowOverheads   [][]byte
		sharedSequence SequenceHeader
	}

	for step, initial := range modes {
		step, initial := step, initial
		reconfig := publicRTCMatrixReconfigMode(t, initial)
		t.Run(initial.String()+"-to-"+reconfig.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, initial)
			cfg := publicRTCMatrixConfig(width, height, initial)
			cfg.MaxFramerate = fpsCycle[step%len(fpsCycle)]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, initial)+int32(step*7))
			if step%2 == 1 {
				cfg.RateControl = EncoderRateControlCQP
				cfg.Quantizer = uint8(22 + step%37)
				cfg.Content = EncoderContentScreen
			}

			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", initial, err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			appendPicture := func(phase *transitionPhase, label string, frameIndex int, forceKey bool, wantKey bool, wantScreen bool) {
				t.Helper()
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), forceKey)
				if err != nil {
					t.Fatalf("%s EncodePicture: %v", label, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					payload := append([]byte(nil), frame.Data...)
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), payload, &phase.sharedSequence, wantScreen)
						phase.orderedFrames = append(phase.orderedFrames, publicIVFFrame{
							timestamp: uint64(len(phase.orderedFrames)),
							payload:   payload,
						})
						phase.lowOverheads = append(phase.lowOverheads, payload)
						continue
					}
					if frame.SpatialID >= EncoderWebRTCMaxSpatialLayers {
						t.Fatalf("%s spatial id=%d", label, frame.SpatialID)
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), payload, &phase.layerSequences[frame.SpatialID], wantScreen)
					phase.layerFrames[frame.SpatialID] = append(phase.layerFrames[frame.SpatialID], publicIVFFrame{
						timestamp: uint64(len(phase.layerFrames[frame.SpatialID])),
						payload:   payload,
					})
				}
			}
			assertPhaseReferenceDecoders := func(label string, cfg EncoderConfig, phase transitionPhase) {
				t.Helper()
				if publicRTCSharedReferenceSlotMode(cfg.Scalability) {
					if len(phase.lowOverheads) == 0 || len(phase.orderedFrames) == 0 {
						t.Fatalf("%s has no shared-reference frames", label)
					}
					wantYUV, decodedCount := decodePublicLayeredLowOverheadRawYUV(t, phase.lowOverheads...)
					if decodedCount != len(phase.lowOverheads) {
						t.Fatalf("%s shared-SVC decoded frames=%d want %d", label, decodedCount, len(phase.lowOverheads))
					}
					ivf := appendPublicIVF(nil, uint16(cfg.Resolution.Width), uint16(cfg.Resolution.Height), 30, 1, phase.orderedFrames)
					assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, label, ivf, wantYUV, decodedCount)
					return
				}
				for spatialID := uint8(0); spatialID < cfg.SpatialLayerCount; spatialID++ {
					if len(phase.layerFrames[spatialID]) == 0 {
						t.Fatalf("%s spatial %d has no frames", label, spatialID)
					}
					res := cfg.SpatialLayers[spatialID].Resolution
					ivf := appendPublicIVF(nil, uint16(res.Width), uint16(res.Height), 30, 1, phase.layerFrames[spatialID])
					assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, fmt.Sprintf("%s-spatial-%d", label, spatialID), ivf)
				}
			}

			var initialPhase transitionPhase
			appendPicture(&initialPhase, "initial key", 0, false, true, cfg.Content == EncoderContentScreen)

			controlChange := enc.Config()
			controlChange.MaxFramerate = fpsCycle[(step+1)%len(fpsCycle)]
			if controlChange.RateControl == EncoderRateControlCQP {
				controlChange.RateControl = EncoderRateControlCBR
				controlChange.Quantizer = 0
				publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, initial)+int32(step*11)+61)
			} else {
				controlChange.RateControl = EncoderRateControlCQP
				controlChange.Quantizer = uint8(28 + step%29)
			}
			if step%3 == 0 {
				controlChange.Content = EncoderContentScreen
			} else {
				controlChange.Content = EncoderContentCamera
			}
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig(%s controls): %v", initial, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			appendPicture(&initialPhase, "control delta", 1, false, false, controlChange.Content == EncoderContentScreen)
			assertPhaseReferenceDecoders("transition-"+initial.String()+"-controls", enc.Config(), initialPhase)

			structureChange := enc.Config()
			structureChange.Scalability = reconfig
			structureChange.MaxFramerate = fpsCycle[(step+2)%len(fpsCycle)]
			if structureChange.RateControl == EncoderRateControlCQP {
				structureChange.RateControl = EncoderRateControlCBR
				structureChange.Quantizer = 0
				publicRTCApplyControlBitrates(&structureChange, publicRTCMatrixControlBitrateKbps(t, reconfig)+int32(step*13)+83)
			} else {
				structureChange.RateControl = EncoderRateControlCQP
				structureChange.Quantizer = uint8(30 + step%23)
			}
			structureChange.Content = EncoderContentScreen
			wantStructureKey := publicRTCSetConfigRequiresKey(t, enc.Config(), structureChange)
			if err := enc.SetConfig(structureChange); err != nil {
				t.Fatalf("SetConfig(%s to %s): %v", initial, reconfig, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), structureChange)
			if !wantStructureKey {
				t.Fatalf("SetConfig(%s to %s) did not require a key picture", initial, reconfig)
			}

			var reconfigPhase transitionPhase
			appendPicture(&reconfigPhase, "structure key", 2, false, true, true)
			appendPicture(&reconfigPhase, "post-structure delta", 3, false, false, true)
			assertPhaseReferenceDecoders("transition-"+initial.String()+"-to-"+reconfig.String(), enc.Config(), reconfigPhase)
		})
	}
}

func TestPublicRTCEncoderPairwiseControlRTPReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	type step struct {
		label       string
		fps         EncoderRational
		rateControl EncoderRateControlMode
		quantizer   uint8
		targetKbps  int32
		content     EncoderContentHint
		forceKey    bool
	}
	scenarios := []struct {
		name  string
		mode  EncoderScalabilityMode
		steps []step
	}{
		{
			name: "single-spatial-l1t3",
			mode: EncoderScalabilityModeL1T3,
			steps: []step{
				{label: "camera-cbr-30", fps: EncoderRational{Num: 30, Den: 1}, rateControl: EncoderRateControlCBR, targetKbps: 420, content: EncoderContentCamera},
				{label: "screen-cqp-ntsc", fps: EncoderRational{Num: 60000, Den: 1001}, rateControl: EncoderRateControlCQP, quantizer: 35, targetKbps: 520, content: EncoderContentScreen},
				{label: "forced-screen-cqp-24", fps: EncoderRational{Num: 24, Den: 1}, rateControl: EncoderRateControlCQP, quantizer: 29, targetKbps: 540, content: EncoderContentScreen, forceKey: true},
				{label: "camera-cbr-120", fps: EncoderRational{Num: 120, Den: 1}, rateControl: EncoderRateControlCBR, targetKbps: 700, content: EncoderContentCamera},
			},
		},
		{
			name: "shared-svc-l2t3-key-shift",
			mode: EncoderScalabilityModeL2T3_KEY_SHIFT,
			steps: []step{
				{label: "screen-cqp-24", fps: EncoderRational{Num: 24, Den: 1}, rateControl: EncoderRateControlCQP, quantizer: 31, targetKbps: 900, content: EncoderContentScreen},
				{label: "camera-cbr-30", fps: EncoderRational{Num: 30, Den: 1}, rateControl: EncoderRateControlCBR, targetKbps: 1100, content: EncoderContentCamera},
				{label: "screen-cqp-ntsc", fps: EncoderRational{Num: 60000, Den: 1001}, rateControl: EncoderRateControlCQP, quantizer: 33, targetKbps: 1000, content: EncoderContentScreen},
				{label: "forced-camera-cbr-15", fps: EncoderRational{Num: 15, Den: 1}, rateControl: EncoderRateControlCBR, targetKbps: 780, content: EncoderContentCamera, forceKey: true},
			},
		},
		{
			name: "simulcast-s3t2h",
			mode: internalencoder.ScalabilityModeS3T2h,
			steps: []step{
				{label: "camera-cbr-30", fps: EncoderRational{Num: 30, Den: 1}, rateControl: EncoderRateControlCBR, targetKbps: 1500, content: EncoderContentCamera},
				{label: "screen-cqp-ntsc", fps: EncoderRational{Num: 60000, Den: 1001}, rateControl: EncoderRateControlCQP, quantizer: 32, targetKbps: 1600, content: EncoderContentScreen},
				{label: "forced-camera-cbr-24", fps: EncoderRational{Num: 24, Den: 1}, rateControl: EncoderRateControlCBR, targetKbps: 1700, content: EncoderContentCamera, forceKey: true},
				{label: "screen-cqp-120", fps: EncoderRational{Num: 120, Den: 1}, rateControl: EncoderRateControlCQP, quantizer: 27, targetKbps: 1450, content: EncoderContentScreen},
			},
		},
	}

	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.name, func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, scenario.mode)
			configForStep := func(s step) EncoderConfig {
				cfg := publicRTCMatrixConfig(width, height, scenario.mode)
				cfg.MaxFramerate = s.fps
				cfg.RateControl = s.rateControl
				cfg.Quantizer = s.quantizer
				cfg.Content = s.content
				if cfg.RateControl == EncoderRateControlCBR {
					cfg.Quantizer = 0
				}
				publicRTCApplyControlBitrates(&cfg, s.targetKbps)
				return cfg
			}

			var enc *RTCEncoder
			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			var activeReceiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader
			nextFrameID := uint64(0)
			activeLimits := RTPPayloadSizeLimits{MaxPayloadLen: 48}

			for i, s := range scenario.steps {
				cfg := configForStep(s)
				wantKey := i == 0 || s.forceKey
				if enc == nil {
					var err error
					enc, err = NewRTCEncoderWithConfig(cfg)
					if err != nil {
						t.Fatalf("NewRTCEncoderWithConfig(%s): %v", scenario.mode, err)
					}
					defer enc.Close()
				} else {
					wantSetConfigKey := publicRTCSetConfigRequiresKey(t, enc.Config(), cfg)
					wantKey = wantKey || wantSetConfigKey
					if err := enc.SetConfig(cfg); err != nil {
						t.Fatalf("%s SetConfig: %v", s.label, err)
					}
				}
				assertPublicRTCConfigControls(t, enc.Config(), cfg)
				wantDuration, err := EncoderWebRTCRTPFrameDuration(enc.Config())
				if err != nil {
					t.Fatalf("%s EncoderWebRTCRTPFrameDuration: %v", s.label, err)
				}
				if got, err := enc.RTPFrameDuration(); err != nil || got != wantDuration {
					t.Fatalf("%s RTPFrameDuration=%+v err=%v want %+v", s.label, got, err, wantDuration)
				}

				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, i), s.forceKey)
				if err != nil {
					t.Fatalf("%s EncodePicture: %v", s.label, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", s.label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, current, picture, activeLimits)
				for frameIndex := 0; frameIndex < picture.FrameNum; frameIndex++ {
					frame := picture.Frames[frameIndex]
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", s.label, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", s.label, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			normalized := enc.Config()
			assertPublicRTCLayerStreamsDecode(t, normalized, layerTUs, orderedTUs)
			if publicRTCSharedReferenceSlotMode(normalized.Scalability) {
				wantYUV, decodedCount := decodePublicLayeredLowOverheadRawYUV(t, orderedTUs...)
				if decodedCount != len(orderedTUs) {
					t.Fatalf("%s shared-reference decoded frames=%d want %d", scenario.name, decodedCount, len(orderedTUs))
				}
				frames := make([]publicIVFFrame, 0, len(orderedTUs))
				for i, payload := range orderedTUs {
					frames = append(frames, publicIVFFrame{timestamp: uint64(i), payload: append([]byte(nil), payload...)})
				}
				ivf := appendPublicIVF(nil, uint16(normalized.Resolution.Width), uint16(normalized.Resolution.Height), 30, 1, frames)
				assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, scenario.name, ivf, wantYUV, decodedCount)
				return
			}
			for spatialID := uint8(0); spatialID < normalized.SpatialLayerCount; spatialID++ {
				if len(layerTUs[spatialID]) == 0 {
					t.Fatalf("%s spatial %d has no RTP-assembled frames", scenario.name, spatialID)
				}
				frames := make([]publicIVFFrame, 0, len(layerTUs[spatialID]))
				for i, payload := range layerTUs[spatialID] {
					frames = append(frames, publicIVFFrame{timestamp: uint64(i), payload: append([]byte(nil), payload...)})
				}
				res := normalized.SpatialLayers[spatialID].Resolution
				ivf := appendPublicIVF(nil, uint16(res.Width), uint16(res.Height), 30, 1, frames)
				assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, fmt.Sprintf("%s-spatial-%d", scenario.name, spatialID), ivf)
			}
		})
	}
}

func TestPublicRTCEncoderDeterministicControlChurnSoakReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	fpsCycle := []EncoderRational{
		{Num: 15, Den: 1},
		{Num: 24, Den: 1},
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 60, Den: 1},
		{Num: 120, Den: 1},
	}
	speedCycle := []int8{
		EncoderWebRTCMinEffortLevel,
		0,
		1,
		EncoderWebRTCMaxEffortLevel,
	}
	scenarios := []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "single-spatial-l1t3", mode: EncoderScalabilityModeL1T3},
		{name: "shared-svc-l2t3-key-shift", mode: EncoderScalabilityModeL2T3_KEY_SHIFT},
		{name: "shared-svc-l3t3-key-shift", mode: EncoderScalabilityModeL3T3_KEY_SHIFT},
		{name: "simulcast-s3t3h", mode: internalencoder.ScalabilityModeS3T3h},
	}

	for scenarioIndex, scenario := range scenarios {
		scenarioIndex, scenario := scenarioIndex, scenario
		t.Run(scenario.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(0x5eed0000 + scenarioIndex*0x101)))
			width, height := publicRTCMatrixGeometry(t, scenario.mode)
			baseKbps := publicRTCMatrixControlBitrateKbps(t, scenario.mode)
			configForStep := func(step int) EncoderConfig {
				cfg := publicRTCMatrixConfig(width, height, scenario.mode)
				cfg.MaxFramerate = fpsCycle[(step+rng.Intn(len(fpsCycle)))%len(fpsCycle)]
				cfg.Speed = speedCycle[(step+rng.Intn(len(speedCycle)))%len(speedCycle)]
				cfg.MaxThreads = int32(1 + rng.Intn(3))
				cfg.Content = EncoderContentCamera
				if rng.Intn(2) == 0 {
					cfg.Content = EncoderContentScreen
				}
				targetKbps := baseKbps + int32(step*53+rng.Intn(int(baseKbps/2)+1))
				publicRTCApplyControlBitrates(&cfg, targetKbps)
				if rng.Intn(2) == 0 {
					cfg.RateControl = EncoderRateControlCQP
					cfg.Quantizer = uint8(22 + rng.Intn(23))
				} else {
					cfg.RateControl = EncoderRateControlCBR
					cfg.Quantizer = 0
				}
				return cfg
			}

			var enc *RTCEncoder
			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			var activeReceiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader
			nextFrameID := uint64(0)
			activeLimits := RTPPayloadSizeLimits{MaxPayloadLen: 48}

			for step := 0; step < 8; step++ {
				cfg := configForStep(step)
				forceKey := step == 3 || step == 6 || (step > 0 && rng.Intn(5) == 0)
				wantKey := step == 0 || forceKey
				if enc == nil {
					var err error
					enc, err = NewRTCEncoderWithConfig(cfg)
					if err != nil {
						t.Fatalf("NewRTCEncoderWithConfig(%s): %v", scenario.mode, err)
					}
					defer enc.Close()
				} else {
					wantKey = wantKey || publicRTCSetConfigRequiresKey(t, enc.Config(), cfg)
					if err := enc.SetConfig(cfg); err != nil {
						t.Fatalf("step %d SetConfig(%s): %v", step, scenario.mode, err)
					}
				}
				assertPublicRTCConfigControls(t, enc.Config(), cfg)
				wantDuration, err := EncoderWebRTCRTPFrameDuration(enc.Config())
				if err != nil {
					t.Fatalf("step %d EncoderWebRTCRTPFrameDuration: %v", step, err)
				}
				if got, err := enc.RTPFrameDuration(); err != nil || got != wantDuration {
					t.Fatalf("step %d RTPFrameDuration=%+v err=%v want %+v", step, got, err, wantDuration)
				}

				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, step*7+rng.Intn(7)), forceKey)
				if err != nil {
					t.Fatalf("step %d EncodePicture(%s): %v", step, scenario.mode, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("step %d key=%v want %v picture=%+v", step, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, current, picture, activeLimits)
				for frameIndex := 0; frameIndex < picture.FrameNum; frameIndex++ {
					frame := picture.Frames[frameIndex]
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("step-%d S%d", step, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("step-%d S%d", step, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			normalized := enc.Config()
			assertPublicRTCLayerStreamsDecode(t, normalized, layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "control-churn-soak-"+scenario.name, normalized, layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderFeedbackControlRTPReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
		{Num: 15, Den: 1},
	}

	for step, mode := range modes {
		step, mode := step, mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = fpsCycle[step%len(fpsCycle)]
			cfg.Content = EncoderContentCamera
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*17))
			if step%2 == 1 {
				cfg.RateControl = EncoderRateControlCQP
				cfg.Quantizer = uint8(23 + step%31)
				cfg.Content = EncoderContentScreen
			}

			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			var activeReceiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader
			activeLimits := RTPPayloadSizeLimits{MaxPayloadLen: 48}
			nextFrameID := uint64(0)
			frameIndex := 0

			appendPicture := func(label string, forceKey bool, wantKey bool) RTCPicture {
				t.Helper()
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), forceKey)
				if err != nil {
					t.Fatalf("%s EncodePicture(%s): %v", label, mode, err)
				}
				frameIndex++
				if picture.Keyframe != wantKey {
					t.Fatalf("%s key=%v want %v picture=%+v", label, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, current, picture, activeLimits)
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
				return picture
			}

			appendPicture("initial key", false, true)

			rembKbps := publicRTCMatrixControlBitrateKbps(t, mode) + int32(step*29) + 73
			rembCompound := publicRTCREMBFeedbackCompound(t, uint64(rembKbps)*1000)
			beforeREMB := enc.Config()
			wantREMB, err := EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
				beforeREMB,
				RTCPReceiverEstimatedMaximumBitrate{BitrateBps: uint64(rembKbps) * 1000},
			)
			if err != nil {
				t.Fatalf("expected REMB config: %v", err)
			}
			ok, packets, err := enc.ApplyRTCPReceiverEstimatedMaximumBitrate(
				rembCompound,
				make([]RTCPPacket, 0, 2),
				make([]uint32, 0, 1),
			)
			if err != nil {
				t.Fatalf("ApplyRTCPReceiverEstimatedMaximumBitrate: %v", err)
			}
			if !ok || len(packets) != 1 {
				t.Fatalf("REMB apply ok=%v packet len=%d want true,1", ok, len(packets))
			}
			afterREMB := enc.Config()
			assertPublicRTCConfigControls(t, afterREMB, wantREMB)
			if afterREMB.TargetBitrateKbps != rembKbps {
				t.Fatalf("REMB target=%d want %d", afterREMB.TargetBitrateKbps, rembKbps)
			}
			if afterREMB.Scalability != beforeREMB.Scalability ||
				afterREMB.MaxFramerate != beforeREMB.MaxFramerate ||
				afterREMB.RateControl != beforeREMB.RateControl ||
				afterREMB.Quantizer != beforeREMB.Quantizer ||
				afterREMB.Speed != beforeREMB.Speed ||
				afterREMB.MaxThreads != beforeREMB.MaxThreads ||
				afterREMB.KeyFrameInterval != beforeREMB.KeyFrameInterval ||
				afterREMB.Content != beforeREMB.Content {
				t.Fatalf("REMB mutated non-bitrate controls before=%+v after=%+v", beforeREMB, afterREMB)
			}
			if publicRTCSetConfigRequiresKey(t, beforeREMB, afterREMB) {
				t.Fatalf("REMB-only config change for %s requires a key picture", mode)
			}
			appendPicture("REMB delta", false, false)

			pliCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBPictureLossIndicationFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  0x05060708,
			})
			forcePLI, pliPackets, err := enc.RTCPRequiresKeyFrame(
				pliCompound,
				make([]RTCPPacket, 0, 1),
				nil,
				nil,
			)
			if err != nil {
				t.Fatalf("PLI compound decision: %v", err)
			}
			if !forcePLI || len(pliPackets) != 1 {
				t.Fatalf("PLI force=%v packet len=%d want true,1", forcePLI, len(pliPackets))
			}
			appendPicture("PLI recovery key", forcePLI, true)

			entry := testAV1RTCPValidLayerRefreshEntry(t, mode)
			lrrCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBLayerRefreshRequestFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  entry.SSRC,
				FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{entry}),
			})
			forceLRR, lrrPackets, err := enc.RTCPRequiresKeyFrame(
				lrrCompound,
				make([]RTCPPacket, 0, 1),
				nil,
				make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("LRR compound decision: %v", err)
			}
			if !forceLRR || len(lrrPackets) != 1 {
				t.Fatalf("LRR force=%v packet len=%d want true,1", forceLRR, len(lrrPackets))
			}
			target, ok, targetPackets, err := enc.RTCPLayerRefreshTarget(
				lrrCompound,
				make([]RTCPPacket, 0, 1),
				make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("LRR target compound: %v", err)
			}
			if !ok || len(targetPackets) != 1 || target != entry.Target {
				t.Fatalf("LRR target=%+v ok=%v packet len=%d want %+v,true,1", target, ok, len(targetPackets), entry.Target)
			}
			lrrRecovery := appendPicture("LRR recovery key", forceLRR, true)
			options, err := lrrRecovery.ActiveDecodeTargetsRTPOptions(target.SpatialID, target.TemporalID)
			if err != nil {
				t.Fatalf("LRR recovery ActiveDecodeTargetsRTPOptions: %v", err)
			}
			spatialLayers, temporalLayers, _, ok := enc.Config().Scalability.Layers()
			if !ok {
				t.Fatalf("invalid mode after LRR recovery %s", enc.Config().Scalability)
			}
			wantMask := publicExpectedActiveDecodeTargetsMask(spatialLayers, temporalLayers, target.SpatialID, target.TemporalID)
			if options.ActiveDecodeTargetsMask != wantMask {
				t.Fatalf("LRR active mask=%#x want %#x", options.ActiveDecodeTargetsMask, wantMask)
			}
			var lrrActiveReceiver RTPDependencyDescriptorState
			for i := 0; i < lrrRecovery.FrameNum; i++ {
				assertPublicRTCFrameRTPPacketsWithActiveDecodeTargets(t, &lrrActiveReceiver, lrrRecovery.Frames[i], activeLimits, options)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "feedback-controls-"+mode.String(), enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderControllerSettingsReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
		{Num: 15, Den: 1},
	}
	speedCycle := []int8{
		EncoderWebRTCMinEffortLevel,
		0,
		EncoderWebRTCMaxEffortLevel,
		1,
	}

	for modeIndex, mode := range modes {
		modeIndex, mode := modeIndex, mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			configForStep := func(step int) EncoderConfig {
				cfg := publicRTCMatrixConfig(width, height, mode)
				cfg.MaxFramerate = fpsCycle[(modeIndex+step)%len(fpsCycle)]
				cfg.Speed = speedCycle[(modeIndex+step)%len(speedCycle)]
				cfg.MaxThreads = int32(1 + (modeIndex+step)%3)
				cfg.KeyFrameInterval = int32(2 + (modeIndex+step)%3)
				cfg.Content = EncoderContentCamera
				publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(modeIndex*17+step*53))
				switch step % 3 {
				case 1:
					cfg.RateControl = EncoderRateControlCQP
					cfg.Quantizer = uint8(24 + (modeIndex+step)%35)
					cfg.Content = EncoderContentScreen
				case 2:
					cfg.Content = EncoderContentScreen
				}
				if cfg.RateControl == EncoderRateControlCBR {
					cfg.Quantizer = 0
				}
				return cfg
			}

			initial := configForStep(0)
			rtc, err := NewRTCEncoderWithConfig(initial)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer rtc.Close()
			controller, err := newPublicRTCController(initial)
			if err != nil {
				t.Fatalf("controller(%s): %v", mode, err)
			}

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader
			nextFrameID := uint64(0)

			for step := 0; step < 5; step++ {
				cfg := configForStep(step)
				forceKey := step == 3
				if step > 0 {
					if err := rtc.SetConfig(cfg); err != nil {
						t.Fatalf("%s step %d RTC SetConfig: %v", mode, step, err)
					}
					if err := controller.SetConfig(cfg); err != nil {
						t.Fatalf("%s step %d controller SetConfig: %v", mode, step, err)
					}
				}
				assertPublicRTCConfigControls(t, rtc.Config(), cfg)
				assertPublicRTCConfigControls(t, controller.Config(), cfg)

				unit, err := controller.NextTemporalUnit(forceKey)
				if err != nil {
					t.Fatalf("%s step %d controller NextTemporalUnit: %v", mode, step, err)
				}
				controllerState := controller.State()
				picture, err := rtc.EncodePicture(publicRTCMatrixFrame(width, height, step), forceKey)
				if err != nil {
					t.Fatalf("%s step %d EncodePicture: %v", mode, step, err)
				}
				frameNum := publicRTCUnitFrameNum(unit)
				if picture.FrameNum != int(frameNum) || picture.Keyframe != unit.Key {
					t.Fatalf("%s step %d picture frames=%d key=%v want frames=%d key=%v",
						mode, step, picture.FrameNum, picture.Keyframe, frameNum, unit.Key)
				}
				current := rtc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, unit.Key, &nextFrameID)
				for frameIndex := 0; frameIndex < picture.FrameNum; frameIndex++ {
					frame := picture.Frames[frameIndex]
					control, _, err := publicRTCUnitFrameControl(unit, controllerState, uint8(frameIndex))
					if err != nil {
						t.Fatalf("%s step %d frame %d controller frame control: %v", mode, step, frameIndex, err)
					}
					settings := control.Settings
					info := control.GenericFrameInfo
					wantCodedKey := unit.Key && (current.Scalability.IsSimulcast() || settings.Type == internalencoder.FrameTypeKey)
					if frame.FrameID != info.FrameID ||
						frame.TemporalID != info.TemporalID ||
						frame.SpatialID != info.SpatialID ||
						frame.CodedKeyframe != wantCodedKey ||
						frame.Keyframe != unit.Key ||
						frame.LastFrameInPicture != (frameIndex+1 == picture.FrameNum) {
						t.Fatalf("%s step %d frame %d actual id=%d S%d/T%d key=%v codedKey=%v last=%v control=%+v settings=%+v want codedKey=%v",
							mode, step, frameIndex, frame.FrameID, frame.SpatialID, frame.TemporalID, frame.Keyframe, frame.CodedKeyframe, frame.LastFrameInPicture, info, settings, wantCodedKey)
					}
					if settings.RateControl != current.RateControl ||
						settings.Quantizer != current.Quantizer ||
						settings.EffortLevel != current.Speed {
						t.Fatalf("%s step %d frame %d settings=%+v config rate=%d q=%d speed=%d",
							mode, step, frameIndex, settings, current.RateControl, current.Quantizer, current.Speed)
					}
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s step %d S%d", mode, step, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
						continue
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s step %d S%d", mode, step, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			assertPublicRTCLayerStreamsDecode(t, rtc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "controller-settings-"+mode.String(), rtc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCEncoderRTPHeaderTimestampCadenceControlMatrix(t *testing.T) {
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
		{Num: 15, Den: 1},
	}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 72}

	for modeIndex, mode := range modes {
		modeIndex, mode := modeIndex, mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			configForStep := func(step int) EncoderConfig {
				cfg := publicRTCMatrixConfig(width, height, mode)
				cfg.MaxFramerate = fpsCycle[(modeIndex+step)%len(fpsCycle)]
				cfg.Content = EncoderContentCamera
				publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(modeIndex*19+step*41))
				if step%2 == 1 {
					cfg.RateControl = EncoderRateControlCQP
					cfg.Quantizer = uint8(25 + (modeIndex+step*3)%31)
					cfg.Content = EncoderContentScreen
				}
				if cfg.RateControl == EncoderRateControlCBR {
					cfg.Quantizer = 0
				}
				return cfg
			}

			enc, err := NewRTCEncoderWithConfig(configForStep(0))
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			sequence := uint16(0x3000 + modeIndex*257)
			timestampUnits := uint64(0x01000000+modeIndex*0x10000) * 2
			var receiver RTPDependencyDescriptorState
			for step := 0; step < 5; step++ {
				cfg := configForStep(step)
				if step > 0 {
					if publicRTCSetConfigRequiresKey(t, enc.Config(), cfg) {
						t.Fatalf("%s same-mode timestamp step %d unexpectedly requires key", mode, step)
					}
					if err := enc.SetConfig(cfg); err != nil {
						t.Fatalf("%s step %d SetConfig: %v", mode, step, err)
					}
				}
				assertPublicRTCConfigControls(t, enc.Config(), cfg)
				duration, err := enc.RTPFrameDuration()
				if err != nil {
					t.Fatalf("%s step %d RTPFrameDuration: %v", mode, step, err)
				}
				timestamp := uint32(timestampUnits / 2)

				forceKey := step == 3
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, step), forceKey)
				if err != nil {
					t.Fatalf("%s step %d EncodePicture: %v", mode, step, err)
				}
				if picture.Keyframe != (step == 0 || forceKey) {
					t.Fatalf("%s step %d key=%v forceKey=%v", mode, step, picture.Keyframe, forceKey)
				}
				publicAssertRTCPictureRTPHeaderTimestamp(t, &receiver, picture, limits, &sequence, timestamp)
				timestampUnits += publicRTPTimestampHalfTicks(t, mode, step, duration)
			}
		})
	}
}

type publicReferenceAV1Decoder struct {
	name string
	path string
	args func(outPath string, ivfPath string) []string
}

func publicReferenceAV1Decoders(t *testing.T) []publicReferenceAV1Decoder {
	t.Helper()
	requireAll := os.Getenv("GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS") == "1"
	candidates := []publicReferenceAV1Decoder{
		{
			name: "aomdec",
			args: func(outPath string, ivfPath string) []string {
				return []string{"--rawvideo", "--all-layers", "-o", outPath, ivfPath}
			},
		},
		{
			name: "dav1d",
			args: func(outPath string, ivfPath string) []string {
				return []string{"--alllayers", "1", "--muxer", "yuv", "-o", outPath, "-i", ivfPath}
			},
		},
	}
	decoders := make([]publicReferenceAV1Decoder, 0, len(candidates))
	missing := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		path, err := exec.LookPath(candidate.name)
		if err != nil {
			missing = append(missing, candidate.name)
			t.Logf("%s not on PATH", candidate.name)
			continue
		}
		candidate.path = path
		decoders = append(decoders, candidate)
	}
	if requireAll && len(missing) > 0 {
		t.Fatalf("required reference AV1 decoder(s) not on PATH: %s", strings.Join(missing, ", "))
	}
	if len(decoders) == 0 {
		t.Skip("no reference AV1 decoder on PATH")
	}
	return decoders
}

func publicFirstSequenceHeader(t *testing.T, frameData []byte) SequenceHeader {
	t.Helper()
	it := obu.NewLowOverheadIterator(frameData)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("OBU iteration: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type != obu.TypeSequenceHeader {
			continue
		}
		seq, err := internalparser.ParseSequenceHeader(unit.Payload)
		if err != nil {
			t.Fatalf("ParseSequenceHeader: %v", err)
		}
		return seq
	}
	t.Fatal("missing sequence header")
	return SequenceHeader{}
}

func publicFirstFrameTileColumns(t *testing.T, frameData []byte) int {
	t.Helper()
	var sequence SequenceHeader
	haveSequence := false
	it := obu.NewLowOverheadIterator(frameData)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("OBU iteration: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeSequenceHeader:
			sequence, err = internalparser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			haveSequence = true
		case obu.TypeFrameHeader, obu.TypeFrame:
			if !haveSequence {
				t.Fatal("frame header appeared before sequence header")
			}
			prefix, err := internalparser.ParseFrameHeaderPrefix(unit.Payload, sequence)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			var size FrameSize
			if prefix.UsesIntraFrameSizePath() {
				size, err = internalparser.ParseIntraFrameSize(unit.Payload, sequence, prefix, unit.Header.TemporalID, unit.Header.SpatialID)
			} else {
				size, err = internalparser.ParseFrameSize(unit.Payload, sequence, prefix, nil, unit.Header.TemporalID, unit.Header.SpatialID)
			}
			if err != nil {
				t.Fatalf("ParseFrameSize: %v", err)
			}
			tiles, err := internalparser.ParseTileInfo(unit.Payload, sequence, prefix, size)
			if err != nil {
				t.Fatalf("ParseTileInfo: %v", err)
			}
			return int(tiles.Cols)
		}
	}
	t.Fatal("missing frame header")
	return 0
}

func assertPublicRTCFrameScreenContentHeader(
	t *testing.T, label string, frameData []byte, sequence *SequenceHeader, wantScreen bool,
) {
	t.Helper()
	it := obu.NewLowOverheadIterator(frameData)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("%s OBU iteration: %v", label, err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeSequenceHeader:
			seq, err := internalparser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("%s ParseSequenceHeader: %v", label, err)
			}
			*sequence = seq
		case obu.TypeFrameHeader, obu.TypeFrame:
			if sequence.ColorConfig.BitDepth == 0 {
				t.Fatalf("%s frame header appeared before sequence header", label)
			}
			prefix, err := internalparser.ParseFrameHeaderPrefix(unit.Payload, *sequence)
			if err != nil {
				t.Fatalf("%s ParseFrameHeaderPrefix: %v", label, err)
			}
			// SVT-AV1 rtc: screen-content tools on intra frames of screen
			// content only (palette on I-slices, no IntraBC); force_integer_mv
			// is implied on intra frames and 0 otherwise.
			intra := prefix.FrameType == internalparser.FrameTypeKey || prefix.FrameType == internalparser.FrameTypeIntraOnly
			if prefix.AllowScreenContentTools != (wantScreen && intra) || prefix.ForceIntegerMV != intra {
				t.Fatalf("%s screen flags allow=%v integerMV=%v want screen=%v intra=%v prefix=%+v",
					label, prefix.AllowScreenContentTools, prefix.ForceIntegerMV, wantScreen, intra, prefix)
			}
			return
		}
	}
	t.Fatalf("%s missing frame header OBU", label)
}

func assertPublicIVFMatchesReferenceDecodersRawYUV(
	t *testing.T, decoders []publicReferenceAV1Decoder, name string, ivf []byte,
) {
	t.Helper()
	decoded, err := DecodeIVF(ivf)
	if err != nil {
		t.Fatalf("%s DecodeIVF: %v", name, err)
	}
	want := publicDecodedFramesRawYUV(decoded)

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, name+".ivf")
	if err := os.WriteFile(ivfPath, ivf, 0o644); err != nil {
		t.Fatalf("%s write IVF: %v", name, err)
	}
	for _, decoder := range decoders {
		outPath := filepath.Join(dir, name+"-"+decoder.name+".yuv")
		out, err := exec.Command(decoder.path, decoder.args(outPath, ivfPath)...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, decoder.name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("%s read %s output: %v", name, decoder.name, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("%s %s output len=%d matches=%v want len=%d", name, decoder.name, len(got), bytes.Equal(got, want), len(want))
		}
		t.Logf("%s %s: %d frames bit-exact", name, decoder.name, len(decoded))
	}
}

func assertPublicIVFMatchesReferenceDecodersRawYUVBytes(
	t *testing.T, decoders []publicReferenceAV1Decoder, name string, ivf []byte, want []byte, frameCount int,
) {
	t.Helper()
	dir := t.TempDir()
	ivfPath := filepath.Join(dir, name+".ivf")
	if err := os.WriteFile(ivfPath, ivf, 0o644); err != nil {
		t.Fatalf("%s write IVF: %v", name, err)
	}
	for _, decoder := range decoders {
		outPath := filepath.Join(dir, name+"-"+decoder.name+".yuv")
		out, err := exec.Command(decoder.path, decoder.args(outPath, ivfPath)...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, decoder.name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("%s read %s output: %v", name, decoder.name, err)
		}
		if !bytes.Equal(got, want) {
			offset := firstPublicByteDiff(got, want)
			var gotByte, wantByte byte
			if offset >= 0 && offset < len(got) {
				gotByte = got[offset]
			}
			if offset >= 0 && offset < len(want) {
				wantByte = want[offset]
			}
			t.Fatalf("%s %s output len=%d want len=%d first_diff=%d got=%#02x want=%#02x",
				name, decoder.name, len(got), len(want), offset, gotByte, wantByte)
		}
		t.Logf("%s %s: %d frames bit-exact", name, decoder.name, frameCount)
	}
}

func assertPublicIVFMonochromeMatchesReferenceDecoders(
	t *testing.T, decoders []publicReferenceAV1Decoder, name string, ivf []byte, width int, height int,
) {
	t.Helper()
	decoded, err := DecodeIVF(ivf)
	if err != nil {
		t.Fatalf("%s DecodeIVF: %v", name, err)
	}
	wantY := make([]byte, 0, len(decoded)*width*height)
	for i, frame := range decoded {
		if frame.ChromaWidth != 0 || frame.ChromaHeight != 0 || len(frame.U) != 0 || len(frame.V) != 0 {
			t.Fatalf("%s decoded frame %d chroma=%dx%d U=%d V=%d want monochrome", name, i, frame.ChromaWidth, frame.ChromaHeight, len(frame.U), len(frame.V))
		}
		wantY = append(wantY, frame.Y...)
	}

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, name+".ivf")
	if err := os.WriteFile(ivfPath, ivf, 0o644); err != nil {
		t.Fatalf("%s write IVF: %v", name, err)
	}
	yLen := width * height
	i420Len := yLen + 2*(width/2)*(height/2)
	for _, decoder := range decoders {
		outPath := filepath.Join(dir, name+"-"+decoder.name+".yuv")
		out, err := exec.Command(decoder.path, decoder.args(outPath, ivfPath)...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, decoder.name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("%s read %s output: %v", name, decoder.name, err)
		}
		switch len(got) {
		case len(wantY):
			if !bytes.Equal(got, wantY) {
				offset := firstPublicByteDiff(got, wantY)
				t.Fatalf("%s %s Y-only output differs first at byte %d", name, decoder.name, offset)
			}
		case len(decoded) * i420Len:
			for frame := range decoded {
				gotY := got[frame*i420Len : frame*i420Len+yLen]
				wantFrameY := wantY[frame*yLen : frame*yLen+yLen]
				if !bytes.Equal(gotY, wantFrameY) {
					offset := firstPublicByteDiff(gotY, wantFrameY)
					t.Fatalf("%s %s expanded frame %d luma differs first at byte %d", name, decoder.name, frame, offset)
				}
			}
		default:
			t.Fatalf("%s %s output len=%d want %d Y-only or %d yuv420 bytes", name, decoder.name, len(got), len(wantY), len(decoded)*i420Len)
		}
		t.Logf("%s %s: %d monochrome frames luma bit-exact", name, decoder.name, len(decoded))
	}
}

func assertPublicIVFHighBitDepthMonochromeMatchesReferenceDecoders(
	t *testing.T, decoders []publicReferenceAV1Decoder, name string, ivf []byte, wantY []byte, width int, height int, bytesPerSample int,
) {
	t.Helper()
	decoded, err := DecodeIVF(ivf)
	if err != nil {
		t.Fatalf("%s DecodeIVF: %v", name, err)
	}
	yLen := width * height * bytesPerSample
	if yLen <= 0 || len(wantY)%yLen != 0 {
		t.Fatalf("%s invalid expected luma length %d for %dx%d %d-byte samples", name, len(wantY), width, height, bytesPerSample)
	}
	wantFrames := len(wantY) / yLen
	if len(decoded) != wantFrames {
		t.Fatalf("%s decoded %d frames, want %d", name, len(decoded), wantFrames)
	}
	gotLocalY := make([]byte, 0, len(wantY))
	for i, frame := range decoded {
		if frame.Width != width || frame.Height != height ||
			frame.BytesPerSample != bytesPerSample ||
			frame.ChromaWidth != 0 || frame.ChromaHeight != 0 ||
			len(frame.U) != 0 || len(frame.V) != 0 {
			t.Fatalf("%s decoded frame %d=%+v want %dx%d %d-byte monochrome", name, i, frame, width, height, bytesPerSample)
		}
		gotLocalY = append(gotLocalY, frame.Y...)
	}
	if !bytes.Equal(gotLocalY, wantY) {
		offset := firstPublicByteDiff(gotLocalY, wantY)
		t.Fatalf("%s local decoded luma differs first at byte %d", name, offset)
	}

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, name+".ivf")
	if err := os.WriteFile(ivfPath, ivf, 0o644); err != nil {
		t.Fatalf("%s write IVF: %v", name, err)
	}
	i420Len := yLen + 2*(width/2)*(height/2)*bytesPerSample
	for _, decoder := range decoders {
		outPath := filepath.Join(dir, name+"-"+decoder.name+".yuv")
		out, err := exec.Command(decoder.path, decoder.args(outPath, ivfPath)...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, decoder.name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("%s read %s output: %v", name, decoder.name, err)
		}
		switch len(got) {
		case len(wantY):
			if !bytes.Equal(got, wantY) {
				offset := firstPublicByteDiff(got, wantY)
				t.Fatalf("%s %s Y-only output differs first at byte %d", name, decoder.name, offset)
			}
		case wantFrames * i420Len:
			gotY := make([]byte, 0, len(wantY))
			for frame := range wantFrames {
				gotY = append(gotY, got[frame*i420Len:frame*i420Len+yLen]...)
			}
			if !bytes.Equal(gotY, wantY) {
				offset := firstPublicByteDiff(gotY, wantY)
				t.Fatalf("%s %s expanded luma differs first at byte %d", name, decoder.name, offset)
			}
		default:
			t.Fatalf("%s %s output len=%d want %d Y-only or %d yuv420 bytes", name, decoder.name, len(got), len(wantY), wantFrames*i420Len)
		}
		t.Logf("%s %s: %d high-bit-depth monochrome frames luma bit-exact", name, decoder.name, wantFrames)
	}
}

func publicRTCREMBFeedbackCompound(t *testing.T, bitrateBps uint64) []byte {
	t.Helper()
	fci, err := AppendRTCPReceiverEstimatedMaximumBitrateFCI(
		make([]byte, 0, RTCPReceiverEstimatedMaximumBitrateFCIMinSize+RTCPReceiverEstimatedMaximumBitrateSSRCSize),
		bitrateBps,
		[]uint32{0x05060708},
	)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	return testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        fci,
	})
}

func assertPublicRTCRTPReferenceDecoders(
	t *testing.T,
	decoders []publicReferenceAV1Decoder,
	name string,
	cfg EncoderConfig,
	layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte,
	orderedTUs [][]byte,
) {
	t.Helper()
	if publicRTCSharedReferenceSlotMode(cfg.Scalability) {
		if len(orderedTUs) == 0 {
			t.Fatalf("%s has no shared-reference RTP-assembled frames", name)
		}
		wantYUV, decodedCount := decodePublicLayeredLowOverheadRawYUV(t, orderedTUs...)
		if decodedCount != len(orderedTUs) {
			t.Fatalf("%s shared-reference decoded frames=%d want %d", name, decodedCount, len(orderedTUs))
		}
		frames := make([]publicIVFFrame, 0, len(orderedTUs))
		for i, payload := range orderedTUs {
			frames = append(frames, publicIVFFrame{timestamp: uint64(i), payload: append([]byte(nil), payload...)})
		}
		ivf := appendPublicIVF(nil, uint16(cfg.Resolution.Width), uint16(cfg.Resolution.Height), 30, 1, frames)
		assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, name, ivf, wantYUV, decodedCount)
		return
	}

	for spatialID := uint8(0); spatialID < cfg.SpatialLayerCount; spatialID++ {
		if len(layerTUs[spatialID]) == 0 {
			t.Fatalf("%s spatial %d has no RTP-assembled frames", name, spatialID)
		}
		frames := make([]publicIVFFrame, 0, len(layerTUs[spatialID]))
		for i, payload := range layerTUs[spatialID] {
			frames = append(frames, publicIVFFrame{timestamp: uint64(i), payload: append([]byte(nil), payload...)})
		}
		res := cfg.SpatialLayers[spatialID].Resolution
		ivf := appendPublicIVF(nil, uint16(res.Width), uint16(res.Height), 30, 1, frames)
		assertPublicIVFMatchesReferenceDecodersRawYUV(t, decoders, fmt.Sprintf("%s-spatial-%d", name, spatialID), ivf)
	}
}

func assertPublicRTCRTPMonochromeReferenceDecoders(
	t *testing.T,
	decoders []publicReferenceAV1Decoder,
	name string,
	cfg EncoderConfig,
	layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte,
	orderedTUs [][]byte,
) {
	t.Helper()
	if publicRTCSharedReferenceSlotMode(cfg.Scalability) {
		if len(orderedTUs) == 0 {
			t.Fatalf("%s has no shared-reference RTP-assembled frames", name)
		}
		wantY, spans := decodePublicLayeredLowOverheadMonochromeRawY(t, orderedTUs...)
		frames := make([]publicIVFFrame, 0, len(orderedTUs))
		for i, payload := range orderedTUs {
			frames = append(frames, publicIVFFrame{timestamp: uint64(i), payload: append([]byte(nil), payload...)})
		}
		ivf := appendPublicIVF(nil, uint16(cfg.Resolution.Width), uint16(cfg.Resolution.Height), 30, 1, frames)
		assertPublicIVFMonochromeVariableReferenceDecoders(t, decoders, name, ivf, wantY, spans)
		return
	}

	for spatialID := uint8(0); spatialID < cfg.SpatialLayerCount; spatialID++ {
		if len(layerTUs[spatialID]) == 0 {
			t.Fatalf("%s spatial %d has no RTP-assembled frames", name, spatialID)
		}
		frames := make([]publicIVFFrame, 0, len(layerTUs[spatialID]))
		for i, payload := range layerTUs[spatialID] {
			frames = append(frames, publicIVFFrame{timestamp: uint64(i), payload: append([]byte(nil), payload...)})
		}
		res := cfg.SpatialLayers[spatialID].Resolution
		ivf := appendPublicIVF(nil, uint16(res.Width), uint16(res.Height), 30, 1, frames)
		assertPublicIVFMonochromeMatchesReferenceDecoders(t, decoders, fmt.Sprintf("%s-spatial-%d", name, spatialID), ivf, int(res.Width), int(res.Height))
	}
}

type publicMonochromeFrameSpan struct {
	yLen        int
	expandedLen int
}

func assertPublicIVFMonochromeVariableReferenceDecoders(
	t *testing.T, decoders []publicReferenceAV1Decoder, name string, ivf []byte, wantY []byte, spans []publicMonochromeFrameSpan,
) {
	t.Helper()
	expandedLen := 0
	for _, span := range spans {
		expandedLen += span.expandedLen
	}

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, name+".ivf")
	if err := os.WriteFile(ivfPath, ivf, 0o644); err != nil {
		t.Fatalf("%s write IVF: %v", name, err)
	}
	for _, decoder := range decoders {
		outPath := filepath.Join(dir, name+"-"+decoder.name+".yuv")
		out, err := exec.Command(decoder.path, decoder.args(outPath, ivfPath)...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, decoder.name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("%s read %s output: %v", name, decoder.name, err)
		}
		switch len(got) {
		case len(wantY):
			if !bytes.Equal(got, wantY) {
				offset := firstPublicByteDiff(got, wantY)
				t.Fatalf("%s %s Y-only output differs first at byte %d", name, decoder.name, offset)
			}
		case expandedLen:
			gotOffset, wantOffset := 0, 0
			for frame, span := range spans {
				gotY := got[gotOffset : gotOffset+span.yLen]
				wantFrameY := wantY[wantOffset : wantOffset+span.yLen]
				if !bytes.Equal(gotY, wantFrameY) {
					offset := firstPublicByteDiff(gotY, wantFrameY)
					t.Fatalf("%s %s expanded frame %d luma differs first at byte %d", name, decoder.name, frame, offset)
				}
				gotOffset += span.expandedLen
				wantOffset += span.yLen
			}
		default:
			t.Fatalf("%s %s output len=%d want %d Y-only or %d expanded bytes", name, decoder.name, len(got), len(wantY), expandedLen)
		}
		t.Logf("%s %s: %d monochrome frames luma bit-exact", name, decoder.name, len(spans))
	}
}

func firstPublicByteDiff(a []byte, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func TestPublicRTCEncoderSetConfigSpatialCycleDecodes(t *testing.T) {
	const w, h = 1008, 576
	modes := []EncoderScalabilityMode{
		EncoderScalabilityModeL1T2,
		internalencoder.ScalabilityModeS2T2,
		EncoderScalabilityModeL3T2_KEY_SHIFT,
		EncoderScalabilityModeL1T3,
	}
	fps := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
	}
	targets := []int32{520, 940, 1500, 640}

	cfg := publicRTCMatrixConfig(w, h, modes[0])
	cfg.MaxFramerate = fps[0]
	publicRTCApplyControlBitrates(&cfg, targets[0])
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig(%s): %v", modes[0], err)
	}
	defer enc.Close()

	var receiver RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	frameIndex := 0
	for step, mode := range modes {
		if step > 0 {
			cfg = publicRTCMatrixConfig(w, h, mode)
			cfg.MaxFramerate = fps[step]
			publicRTCApplyControlBitrates(&cfg, targets[step])
			if err := enc.SetConfig(cfg); err != nil {
				t.Fatalf("SetConfig(%s): %v", mode, err)
			}
			assertPublicRTCConfigBitrates(t, enc.Config(), cfg)
		}

		var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
		var orderedTUs [][]byte
		key, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, frameIndex), false)
		if err != nil {
			t.Fatalf("key EncodePicture(%s): %v", mode, err)
		}
		frameIndex++
		appendPublicRTCPictureLayerData(t, &layerTUs, &orderedTUs, key)
		assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), key, true, &nextFrameID)

		delta, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, frameIndex), false)
		if err != nil {
			t.Fatalf("delta EncodePicture(%s): %v", mode, err)
		}
		frameIndex++
		appendPublicRTCPictureLayerData(t, &layerTUs, &orderedTUs, delta)
		assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), delta, false, &nextFrameID)
		assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
	}
}

func TestPublicRTCEncoderKeyFrameScalabilityMetadata(t *testing.T) {
	const w, h = 640, 360
	cfg := publicRTCMatrixConfig(w, h, internalencoder.ScalabilityModeL2T2)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig L2T2: %v", err)
	}
	picture, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture L2T2: %v", err)
	}
	if picture.FrameNum < 1 || !picture.Keyframe {
		t.Fatalf("picture=%+v", picture)
	}
	assertPublicRTCFrameScalabilityMetadata(t, picture.Frames[0].Data, obu.ScalabilityModeL2T2, true)

	ssCfg := publicRTCMatrixConfig(960, 540, internalencoder.ScalabilityModeL3T1h)
	ssEnc, err := NewRTCEncoderWithConfig(ssCfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig L3T1h: %v", err)
	}
	ssPicture, err := ssEnc.EncodePicture(publicRTCMatrixFrame(960, 540, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture L3T1h: %v", err)
	}
	if ssPicture.FrameNum < 1 || !ssPicture.Keyframe {
		t.Fatalf("ss picture=%+v", ssPicture)
	}
	assertPublicRTCFrameScalabilitySSMetadata(t, ssPicture.Frames[0].Data,
		[]uint16{426, 640, 960},
		[]uint16{240, 360, 540},
		[]uint8{0xff, 0, 1},
		[]uint8{0},
		[]uint8{1},
	)

	plain, err := NewRTCEncoderWithConfig(publicRTCMatrixConfig(w, h, EncoderScalabilityModeL1T1))
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig L1T1: %v", err)
	}
	plainPicture, err := plain.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
	if err != nil {
		t.Fatalf("EncodePicture L1T1: %v", err)
	}
	assertPublicRTCFrameScalabilityMetadata(t, plainPicture.Frames[0].Data, 0, false)
}

func TestPublicRTCEncoderSetConfigFullModeCatalogueDecodes(t *testing.T) {
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fps := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 15, Den: 1},
		{Num: 120, Den: 1},
	}

	var enc *RTCEncoder
	var descriptorReceiver RTPDependencyDescriptorState
	var rtpReceiver RTPDependencyDescriptorState
	var activeLayerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
	var activeOrderedTUs [][]byte
	nextFrameID := uint64(0)
	frameIndex := 0

	for step, mode := range modes {
		width, height := publicRTCMatrixGeometry(t, mode)
		cfg := publicRTCMatrixConfig(width, height, mode)
		cfg.MaxFramerate = fps[step%len(fps)]
		targetKbps := publicRTCMatrixControlBitrateKbps(t, mode) + int32((step%5)*37)
		publicRTCApplyControlBitrates(&cfg, targetKbps)
		if step%2 == 1 {
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = uint8(24 + step%31)
		}

		wantKey := true
		if enc == nil {
			var err error
			enc, err = NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()
		} else {
			wantKey = publicRTCSetConfigRequiresKey(t, enc.Config(), cfg)
			if err := enc.SetConfig(cfg); err != nil {
				t.Fatalf("SetConfig(%s): %v", mode, err)
			}
		}
		assertPublicRTCConfigControls(t, enc.Config(), cfg)
		if wantKey {
			activeLayerTUs = [EncoderWebRTCMaxSpatialLayers][][]byte{}
			activeOrderedTUs = nil
		}

		picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), false)
		if err != nil {
			t.Fatalf("catalogue key/delta EncodePicture(%s): %v", mode, err)
		}
		frameIndex++
		appendPublicRTCPictureRTPData(t, &rtpReceiver, &activeLayerTUs, &activeOrderedTUs, picture)
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), picture, wantKey, &nextFrameID)

		delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), false)
		if err != nil {
			t.Fatalf("catalogue delta EncodePicture(%s): %v", mode, err)
		}
		frameIndex++
		appendPublicRTCPictureRTPData(t, &rtpReceiver, &activeLayerTUs, &activeOrderedTUs, delta)
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), delta, false, &nextFrameID)

		controlChange := enc.Config()
		controlChange.MaxFramerate = fps[(step+1)%len(fps)]
		if controlChange.RateControl == EncoderRateControlCQP {
			controlChange.Quantizer = uint8(20 + (step*7)%37)
		} else {
			publicRTCApplyControlBitrates(&controlChange, targetKbps+53)
		}
		if err := enc.SetConfig(controlChange); err != nil {
			t.Fatalf("SetConfig(%s control): %v", mode, err)
		}
		assertPublicRTCConfigControls(t, enc.Config(), controlChange)
		controlDelta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), false)
		if err != nil {
			t.Fatalf("catalogue control delta EncodePicture(%s): %v", mode, err)
		}
		frameIndex++
		appendPublicRTCPictureRTPData(t, &rtpReceiver, &activeLayerTUs, &activeOrderedTUs, controlDelta)
		assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), controlDelta, false, &nextFrameID)

		assertPublicRTCLayerStreamsDecode(t, enc.Config(), activeLayerTUs, activeOrderedTUs)
	}
}

func TestPublicRTCEncoderControlChurnSoakMatrix(t *testing.T) {
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60000, Den: 1001},
		{Num: 60, Den: 1},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
		{Num: 15, Den: 1},
	}

	for modeIndex, mode := range modes {
		modeIndex, mode := modeIndex, mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			configForStep := func(step int) EncoderConfig {
				cfg := publicRTCMatrixConfig(width, height, mode)
				cfg.MaxFramerate = fpsCycle[(modeIndex+step)%len(fpsCycle)]
				cfg.Content = EncoderContentCamera
				target := publicRTCMatrixControlBitrateKbps(t, mode) + int32(modeIndex*13+step*37)
				publicRTCApplyControlBitrates(&cfg, target)
				if step%2 == 1 {
					cfg.RateControl = EncoderRateControlCQP
					cfg.Quantizer = uint8(23 + (modeIndex+step*5)%37)
					cfg.Content = EncoderContentScreen
				}
				if step%4 == 3 {
					cfg.Content = EncoderContentScreen
				}
				return cfg
			}

			enc, err := NewRTCEncoderWithConfig(configForStep(0))
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			var rtpReceiver RTPDependencyDescriptorState
			var activeReceiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			var layerSequences [EncoderWebRTCMaxSpatialLayers]SequenceHeader
			var sharedSequence SequenceHeader
			activeLimits := RTPPayloadSizeLimits{MaxPayloadLen: 56}
			nextFrameID := uint64(0)

			for step := 0; step < 8; step++ {
				cfg := configForStep(step)
				if step > 0 {
					if publicRTCSetConfigRequiresKey(t, enc.Config(), cfg) {
						t.Fatalf("%s same-mode control step %d unexpectedly requires key", mode, step)
					}
					if err := enc.SetConfig(cfg); err != nil {
						t.Fatalf("%s step %d SetConfig: %v", mode, step, err)
					}
				}
				assertPublicRTCConfigControls(t, enc.Config(), cfg)
				wantDuration, err := EncoderWebRTCRTPFrameDuration(cfg)
				if err != nil {
					t.Fatalf("%s step %d EncoderWebRTCRTPFrameDuration: %v", mode, step, err)
				}
				if got, err := enc.RTPFrameDuration(); err != nil || got != wantDuration {
					t.Fatalf("%s step %d RTPFrameDuration=%+v err=%v want %+v", mode, step, got, err, wantDuration)
				}

				forceKey := step == 4
				wantKey := step == 0 || forceKey
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, step), forceKey)
				if err != nil {
					t.Fatalf("%s step %d EncodePicture: %v", mode, step, err)
				}
				if picture.Keyframe != wantKey {
					t.Fatalf("%s step %d key=%v want %v picture=%+v", mode, step, picture.Keyframe, wantKey, picture)
				}
				current := enc.Config()
				assertPublicRTCPictureDescriptors(t, &descriptorReceiver, current, picture, wantKey, &nextFrameID)
				assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, current, picture, activeLimits)
				for frameIndex := 0; frameIndex < picture.FrameNum; frameIndex++ {
					frame := picture.Frames[frameIndex]
					if publicRTCSharedReferenceSlotMode(current.Scalability) {
						assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("step-%d S%d", step, frame.SpatialID), frame.Data, &sharedSequence, current.Content == EncoderContentScreen)
						continue
					}
					if frame.SpatialID >= EncoderWebRTCMaxSpatialLayers {
						t.Fatalf("%s step %d spatial id=%d", mode, step, frame.SpatialID)
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("step-%d S%d", step, frame.SpatialID), frame.Data, &layerSequences[frame.SpatialID], current.Content == EncoderContentScreen)
				}
				appendPublicRTCPictureRTPData(t, &rtpReceiver, &layerTUs, &orderedTUs, picture)
			}

			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicRTCSharedReferenceSVCDecodeRTPPayloads(t *testing.T) {
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 24}
	covered := 0
	for step, mode := range EncoderWebRTCScalabilityModes() {
		if !publicRTCSharedReferenceSlotMode(mode) {
			continue
		}
		covered++
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = []EncoderRational{
				{Num: 30, Den: 1},
				{Num: 60, Den: 1},
				{Num: 30000, Den: 1001},
				{Num: 24, Den: 1},
			}[step%4]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*13))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var lowOverheads [][]byte
			var rtpPayloads [][]byte
			var rtpPackets [][]byte
			var rtpPayloadLabels []string
			var wantMetadata []publicLayeredFrameMetadata
			for frameIndex := 0; frameIndex < 3; frameIndex++ {
				if frameIndex == 2 {
					controlChange := enc.Config()
					controlChange.MaxFramerate = EncoderRational{Num: 120, Den: 1}
					publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*17)+53)
					if err := enc.SetConfig(controlChange); err != nil {
						t.Fatalf("SetConfig(%s control): %v", mode, err)
					}
					assertPublicRTCConfigControls(t, enc.Config(), controlChange)
				}
				picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, frameIndex), false)
				if err != nil {
					t.Fatalf("EncodePicture(%s, %d): %v", mode, frameIndex, err)
				}
				for outputIndex := 0; outputIndex < picture.FrameNum; outputIndex++ {
					frame := picture.Frames[outputIndex]
					lowOverheads = append(lowOverheads, append([]byte(nil), frame.Data...))
					wantMetadata = append(wantMetadata, publicLayeredFrameMetadataFromRTCFrame(frame))
					framePayloads := publicDecoderRTPPayloadsForFrameWithLimits(t, frame, limits)
					framePackets := publicDecoderRTPPacketsForFrameWithLimits(t, frame, limits)
					if len(framePayloads) != len(framePackets) {
						t.Fatalf("frame payload/packet count=%d/%d", len(framePayloads), len(framePackets))
					}
					for packetIndex, payload := range framePayloads {
						rtpPayloads = append(rtpPayloads, payload)
						rtpPackets = append(rtpPackets, framePackets[packetIndex])
						rtpPayloadLabels = append(rtpPayloadLabels, fmt.Sprintf("picture=%d output=%d packet=%d/%d S%d T%d key=%v frameID=%d",
							frameIndex, outputIndex, packetIndex, len(framePayloads), frame.SpatialID, frame.TemporalID, frame.Keyframe, frame.FrameID))
					}
				}
			}

			want := decodePublicLayeredDecoderLowOverheadDigests(t, lowOverheads...)
			got := decodePublicLayeredRTPPayloadDigestsWithLabels(t, rtpPayloadLabels, rtpPayloads...)
			gotLayeredQueued := decodePublicLayeredDecoderRTPPayloadDigests(t, rtpPayloads...)
			gotLayeredLive := decodePublicLayeredLiveDecoderRTPPayloadDigests(t, rtpPayloads, rtpPayloads...)
			gotLayeredQueuedPackets := decodePublicLayeredDecoderRTPPacketDigests(t, rtpPackets...)
			gotLayeredLivePackets := decodePublicLayeredLiveDecoderRTPPacketDigests(t, rtpPackets, rtpPackets...)
			gotLayeredLowMetadata := decodePublicLayeredDecoderLowOverheadMetadata(t, lowOverheads...)
			gotLayeredQueuedMetadata := decodePublicLayeredDecoderRTPPayloadMetadata(t, rtpPayloads...)
			gotLayeredLiveMetadata := decodePublicLayeredLiveDecoderRTPPayloadMetadata(t, rtpPayloads, rtpPayloads...)
			gotLayeredQueuedPacketMetadata := decodePublicLayeredDecoderRTPPacketMetadata(t, rtpPackets...)
			gotLayeredLivePacketMetadata := decodePublicLayeredLiveDecoderRTPPacketMetadata(t, rtpPackets, rtpPackets...)
			if len(got) != len(want) || len(got) != len(lowOverheads) {
				t.Fatalf("decoded frames got=%d want=%d low-overheads=%d", len(got), len(want), len(lowOverheads))
			}
			if len(gotLayeredQueued) != len(want) || len(gotLayeredLive) != len(want) ||
				len(gotLayeredQueuedPackets) != len(want) || len(gotLayeredLivePackets) != len(want) {
				t.Fatalf("layered decoded frames payload queued/live=%d/%d packet queued/live=%d/%d want=%d",
					len(gotLayeredQueued), len(gotLayeredLive), len(gotLayeredQueuedPackets), len(gotLayeredLivePackets), len(want))
			}
			assertPublicLayeredFrameMetadata(t, gotLayeredLowMetadata, wantMetadata)
			assertPublicLayeredFrameMetadata(t, gotLayeredQueuedMetadata, wantMetadata)
			assertPublicLayeredFrameMetadata(t, gotLayeredLiveMetadata, wantMetadata)
			assertPublicLayeredFrameMetadata(t, gotLayeredQueuedPacketMetadata, wantMetadata)
			assertPublicLayeredFrameMetadata(t, gotLayeredLivePacketMetadata, wantMetadata)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("frame %d digest differs: rtp=%x low=%x", i, got[i], want[i])
				}
				if gotLayeredQueued[i] != want[i] {
					t.Fatalf("frame %d layered queued digest differs: rtp=%x low=%x", i, gotLayeredQueued[i], want[i])
				}
				if gotLayeredLive[i] != want[i] {
					t.Fatalf("frame %d layered live digest differs: rtp=%x low=%x", i, gotLayeredLive[i], want[i])
				}
				if gotLayeredQueuedPackets[i] != want[i] {
					t.Fatalf("frame %d layered queued packet digest differs: got=%x want=%x", i, gotLayeredQueuedPackets[i], want[i])
				}
				if gotLayeredLivePackets[i] != want[i] {
					t.Fatalf("frame %d layered live packet digest differs: got=%x want=%x", i, gotLayeredLivePackets[i], want[i])
				}
			}
		})
	}
	if covered == 0 {
		t.Fatal("no shared-reference WebRTC SVC modes covered")
	}
}

func TestPublicRTCEncoderRTPSequencerImpairmentRecoveryMatrix(t *testing.T) {
	const dependencyDescriptorExtensionID = 42
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 192}
	fps := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
	}

	for step, mode := range EncoderWebRTCScalabilityModes() {
		mode := mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = fps[step%len(fps)]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*23))
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			key0, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
			if err != nil {
				t.Fatalf("key0 EncodePicture(%s): %v", mode, err)
			}
			cloneRTCPictureData(&key0)
			lost, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
			if err != nil {
				t.Fatalf("lost delta EncodePicture(%s): %v", mode, err)
			}
			cloneRTCPictureData(&lost)
			recovery, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), true)
			if err != nil {
				t.Fatalf("recovery key EncodePicture(%s): %v", mode, err)
			}
			cloneRTCPictureData(&recovery)

			key0PacketsRaw := publicRTCPictureRTPPackets(t, key0, limits)
			lostPacketsRaw := publicRTCPictureRTPPackets(t, lost, limits)
			recoveryPacketsRaw := publicRTCPictureRTPPackets(t, recovery, limits)
			if len(key0PacketsRaw) < 3 || len(lostPacketsRaw) == 0 || len(recoveryPacketsRaw) == 0 {
				t.Fatalf("%s packet counts key0=%d lost=%d recovery=%d", mode, len(key0PacketsRaw), len(lostPacketsRaw), len(recoveryPacketsRaw))
			}

			next := uint16(0x5000)
			key0Packets := publicRewriteRTPPacketSequences(t, key0PacketsRaw, &next)
			lostPackets := publicRewriteRTPPacketSequences(t, lostPacketsRaw, &next)
			recoveryPackets := publicRewriteRTPPacketSequences(t, recoveryPacketsRaw, &next)
			if len(lostPackets)+1 >= 1<<15 {
				t.Fatalf("%s lost packet gap=%d exceeds RTP sequencer window", mode, len(lostPackets))
			}
			maxPacketLen := 0
			for _, packetSet := range [][][]byte{key0Packets, lostPackets, recoveryPackets} {
				for _, packet := range packetSet {
					if len(packet) > maxPacketLen {
						maxPacketLen = len(packet)
					}
				}
			}
			cacheSlots := make([]RTPRetransmissionCacheSlot, len(key0Packets)+len(lostPackets)+len(recoveryPackets)+8)
			for i := range cacheSlots {
				cacheSlots[i].Packet = make([]byte, 0, maxPacketLen+RTPHeaderMinSize+2)
			}
			retransmissionCache, err := BindRTPRetransmissionCache(cacheSlots)
			if err != nil {
				t.Fatalf("BindRTPRetransmissionCache(%s): %v", mode, err)
			}
			for _, packetSet := range [][][]byte{key0Packets, lostPackets, recoveryPackets} {
				for _, packet := range packetSet {
					if err := retransmissionCache.Store(packet); err != nil {
						t.Fatalf("%s RTP retransmission cache Store seq=%#x: %v", mode, publicRTPPacketSequence(t, packet), err)
					}
				}
			}

			selectedProbePackets := publicRTCPictureDecodeTargetRTPPackets(t, key0, limits, 0, 0)
			selectedProbePackets = append(selectedProbePackets, publicRTCPictureDecodeTargetRTPPackets(t, recovery, limits, 0, 0)...)
			if len(selectedProbePackets) == 0 {
				t.Fatalf("%s produced no S0/T0 probe packets", mode)
			}
			dec, err := NewLayeredDecoderFromRTPPackets(selectedProbePackets)
			if err != nil {
				t.Fatalf("NewLayeredDecoderFromRTPPackets(%s): %v", mode, err)
			}
			defer dec.Close()
			if err := dec.Reset(); err != nil {
				t.Fatalf("LayeredDecoder Reset(%s): %v", mode, err)
			}

			slotCount := len(key0Packets) + len(lostPackets) + len(recoveryPackets) + 8
			if slotCount < 16 {
				slotCount = 16
			}
			if slotCount > 1<<15 {
				slotCount = 1 << 15
			}
			sequencer, err := BindRTPPacketSequencer(make([]RTPPacketSequencerSlot, slotCount))
			if err != nil {
				t.Fatalf("BindRTPPacketSequencer(%s): %v", mode, err)
			}
			events := make([]RTPSequencedPacket, 0, 8)
			var descriptorReceiver RTPDependencyDescriptorState
			var got [][16]byte
			var gotMetadata []publicLayeredFrameMetadata
			pendingLoss := false

			decodeEvents := func(label string, in []RTPSequencedPacket) {
				t.Helper()
				for i, event := range in {
					parsed, err := ParseRTPPacketDependencyDescriptor(event.RawPacket, dependencyDescriptorExtensionID, &descriptorReceiver)
					if err != nil {
						t.Fatalf("%s %s event %d ParseRTPPacketDependencyDescriptor: %v", mode, label, i, err)
					}
					forward, err := RTPDependencyDescriptorFrameForwardedForDecodeTarget(parsed.Descriptor, descriptorReceiver, 0)
					if err != nil {
						t.Fatalf("%s %s event %d FrameForwardedForDecodeTarget: %v", mode, label, i, err)
					}
					if event.AfterLoss {
						pendingLoss = true
					}
					if !forward {
						continue
					}
					if pendingLoss {
						event.AfterLoss = true
						pendingLoss = false
					}
					frames, err := dec.DecodeRTPSequencedPacketWithMetadata(event)
					if err != nil {
						t.Fatalf("%s %s event %d DecodeRTPSequencedPacketWithMetadata: %v", mode, label, i, err)
					}
					for _, frame := range frames {
						got = append(got, frameMD5Visible(frame.Frame))
						gotMetadata = append(gotMetadata, publicLayeredFrameMetadataFromDecoded(t, frame))
					}
				}
			}
			push := func(label string, packet []byte) []RTPSequencedPacket {
				t.Helper()
				out, err := sequencer.Push(packet, events[:0])
				if err != nil {
					t.Fatalf("%s %s Push: %v", mode, label, err)
				}
				return out
			}

			decodeEvents("key0-packet-0", push("key0-packet-0", key0Packets[0]))
			gapped := push("key0-packet-2", key0Packets[2])
			if len(gapped) != 0 || sequencer.Buffered() != 1 {
				t.Fatalf("%s key0 gap released=%d buffered=%d want 0/1", mode, len(gapped), sequencer.Buffered())
			}
			missing, err := sequencer.AppendMissingSequenceNumbers(make([]uint16, 0, 1))
			if err != nil {
				t.Fatalf("%s AppendMissingSequenceNumbers key0 gap: %v", mode, err)
			}
			wantGapSeq := publicRTPPacketSequence(t, key0Packets[1])
			if len(missing) != 1 || missing[0] != wantGapSeq {
				t.Fatalf("%s key0 missing=%#v want [%#x]", mode, missing, wantGapSeq)
			}
			pairs, err := sequencer.AppendMissingRTCPGenericNACKPairs(make([]RTCPGenericNACKPair, 0, 1))
			if err != nil {
				t.Fatalf("%s AppendMissingRTCPGenericNACKPairs key0 gap: %v", mode, err)
			}
			if len(pairs) != 1 || pairs[0] != (RTCPGenericNACKPair{PacketID: wantGapSeq}) {
				t.Fatalf("%s key0 NACK pairs=%+v want PID %#x", mode, pairs, wantGapSeq)
			}
			duplicate := push("key0-packet-2-duplicate", key0Packets[2])
			if len(duplicate) != 0 || sequencer.Buffered() != 1 {
				t.Fatalf("%s duplicate released=%d buffered=%d want 0/1", mode, len(duplicate), sequencer.Buffered())
			}
			originalGapPacket, err := ParseRTPPacket(key0Packets[1])
			if err != nil {
				t.Fatalf("%s ParseRTPPacket original gap: %v", mode, err)
			}
			rtxConfig := RTPRTXPacketConfig{
				PayloadType:    97,
				SequenceNumber: uint16(0x7000 + step),
				SSRC:           0x55667700 + uint32(step),
			}
			rtxSpans := make([]RTPRTXPacketSpan, 1)
			rtxPackets, rtxCount, nextRTXConfig, err := retransmissionCache.AppendRTXPacketsForRTCPGenericNACKPairs(
				make([]byte, 0, maxPacketLen+RTPHeaderMinSize+2),
				rtxSpans,
				pairs,
				rtxConfig,
			)
			if err != nil {
				t.Fatalf("%s AppendRTXPacketsForRTCPGenericNACKPairs key0 gap: %v", mode, err)
			}
			if rtxCount != 1 || nextRTXConfig.SequenceNumber != rtxConfig.SequenceNumber+1 {
				t.Fatalf("%s RTX count=%d next=%#x want 1/%#x", mode, rtxCount, nextRTXConfig.SequenceNumber, rtxConfig.SequenceNumber+1)
			}
			rtxPacket := rtxPackets[rtxSpans[0].Offset : rtxSpans[0].Offset+rtxSpans[0].Length]
			parsedRTX, err := ParseRTPRTXPacket(rtxPacket)
			if err != nil {
				t.Fatalf("%s ParseRTPRTXPacket key0 gap: %v", mode, err)
			}
			if rtxSpans[0].OriginalSequenceNumber != wantGapSeq ||
				rtxSpans[0].RTXSequenceNumber != rtxConfig.SequenceNumber ||
				parsedRTX.OriginalSequenceNumber != wantGapSeq ||
				parsedRTX.Header.PayloadType != rtxConfig.PayloadType ||
				parsedRTX.Header.SequenceNumber != rtxConfig.SequenceNumber ||
				parsedRTX.Header.SSRC != rtxConfig.SSRC {
				t.Fatalf("%s RTX repair span=%+v parsed=%+v osn=%#x", mode, rtxSpans[0], parsedRTX.Header, parsedRTX.OriginalSequenceNumber)
			}
			restoredGap, err := AppendRTPPacketFromRTX(
				make([]byte, 0, len(key0Packets[1])),
				rtxPacket,
				originalGapPacket.Header.PayloadType,
				originalGapPacket.Header.SSRC,
			)
			if err != nil {
				t.Fatalf("%s AppendRTPPacketFromRTX key0 gap: %v", mode, err)
			}
			if !bytes.Equal(restoredGap, key0Packets[1]) {
				t.Fatalf("%s restored RTX packet differs from original gap packet", mode)
			}
			decodeEvents("key0-gap-rtx-repair", push("key0-packet-1-rtx-restored", restoredGap))
			for i := 3; i < len(key0Packets); i++ {
				decodeEvents(fmt.Sprintf("key0-packet-%d", i), push("key0-tail", key0Packets[i]))
			}

			wantKey0 := publicRTCPictureDecodeTargetLowOverheads(key0, 0, 0)
			if len(got) != len(wantKey0) {
				t.Fatalf("%s decoded key0 target frames=%d want %d", mode, len(got), len(wantKey0))
			}

			recoveryFirst := push("recovery-first-before-skip", recoveryPackets[0])
			if len(recoveryFirst) != 0 || sequencer.Buffered() != 1 {
				t.Fatalf("%s recovery pre-skip released=%d buffered=%d want 0/1", mode, len(recoveryFirst), sequencer.Buffered())
			}
			missing, err = sequencer.AppendMissingSequenceNumbers(make([]uint16, 0, len(lostPackets)))
			if err != nil {
				t.Fatalf("%s AppendMissingSequenceNumbers lost picture: %v", mode, err)
			}
			firstLostSeq := publicRTPPacketSequence(t, lostPackets[0])
			firstMissing := uint16(0)
			if len(missing) > 0 {
				firstMissing = missing[0]
			}
			if len(missing) != len(lostPackets) || firstMissing != firstLostSeq {
				t.Fatalf("%s lost missing len=%d first=%#x want len=%d first=%#x", mode, len(missing), firstMissing, len(lostPackets), firstLostSeq)
			}
			nackCap := (len(lostPackets) + 15) / 16
			pairs, err = sequencer.AppendMissingRTCPGenericNACKPairs(make([]RTCPGenericNACKPair, 0, nackCap))
			if err != nil {
				t.Fatalf("%s AppendMissingRTCPGenericNACKPairs lost picture: %v", mode, err)
			}
			if len(pairs) == 0 || pairs[0].PacketID != firstLostSeq {
				t.Fatalf("%s lost NACK pairs=%+v want first PID %#x", mode, pairs, firstLostSeq)
			}

			skipped := sequencer.SkipMissing(events[:0])
			if len(skipped) != 1 || !skipped[0].AfterLoss {
				t.Fatalf("%s SkipMissing released=%d afterLoss=%v want 1/true", mode, len(skipped), len(skipped) == 1 && skipped[0].AfterLoss)
			}
			decodeEvents("recovery-after-loss-first", skipped)
			late := push("late-lost-packet", lostPackets[0])
			if len(late) != 0 {
				t.Fatalf("%s late lost packet released %d events, want 0", mode, len(late))
			}
			lateRecoveryDuplicate := push("late-recovery-duplicate", recoveryPackets[0])
			if len(lateRecoveryDuplicate) != 0 {
				t.Fatalf("%s late recovery duplicate released %d events, want 0", mode, len(lateRecoveryDuplicate))
			}
			for i := 1; i < len(recoveryPackets); i++ {
				decodeEvents(fmt.Sprintf("recovery-packet-%d", i), push("recovery-tail", recoveryPackets[i]))
			}

			wantLowOverheads := append(wantKey0, publicRTCPictureDecodeTargetLowOverheads(recovery, 0, 0)...)
			want := decodePublicLayeredDecoderLowOverheadDigests(t, wantLowOverheads...)
			var wantMetadata []publicLayeredFrameMetadata
			for _, picture := range []RTCPicture{key0, recovery} {
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					if frame.SpatialID == 0 && frame.TemporalID == 0 {
						wantMetadata = append(wantMetadata, publicLayeredFrameMetadataFromRTCFrame(frame))
					}
				}
			}
			if len(got) != len(want) {
				t.Fatalf("%s recovered frames=%d want %d", mode, len(got), len(want))
			}
			assertPublicLayeredFrameMetadata(t, gotMetadata, wantMetadata)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s recovered frame %d digest differs after RTP impairment: got=%x want=%x", mode, i, got[i], want[i])
				}
			}
		})
	}
}

func TestPublicRTCEncoderControlImpairmentRTPReferenceDecoders(t *testing.T) {
	const dependencyDescriptorExtensionID = 42
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 96}
	decoders := publicReferenceAV1Decoders(t)
	scenarios := []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "single-spatial-l1t3", mode: EncoderScalabilityModeL1T3},
		{name: "shared-svc-l2t3-key-shift", mode: EncoderScalabilityModeL2T3_KEY_SHIFT},
		{name: "simulcast-s3t2h", mode: internalencoder.ScalabilityModeS3T2h},
	}

	for scenarioIndex, scenario := range scenarios {
		scenarioIndex, scenario := scenarioIndex, scenario
		t.Run(scenario.name, func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, scenario.mode)
			cfg := publicRTCMatrixConfig(width, height, scenario.mode)
			cfg.MaxFramerate = EncoderRational{Num: 30, Den: 1}
			cfg.Content = EncoderContentCamera
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(scenarioIndex*97))

			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", scenario.mode, err)
			}
			defer enc.Close()

			var descriptorReceiver RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			key0, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
			if err != nil {
				t.Fatalf("key0 EncodePicture(%s): %v", scenario.mode, err)
			}
			cloneRTCPictureData(&key0)
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), key0, true, &nextFrameID)

			lostConfig := enc.Config()
			lostConfig.MaxFramerate = EncoderRational{Num: 60000, Den: 1001}
			lostConfig.RateControl = EncoderRateControlCQP
			lostConfig.Quantizer = uint8(27 + scenarioIndex*5)
			lostConfig.Content = EncoderContentScreen
			if publicRTCSetConfigRequiresKey(t, enc.Config(), lostConfig) {
				t.Fatalf("%s same-mode lost-control change unexpectedly requires key", scenario.name)
			}
			if err := enc.SetConfig(lostConfig); err != nil {
				t.Fatalf("SetConfig(%s lost controls): %v", scenario.mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), lostConfig)
			lost, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
			if err != nil {
				t.Fatalf("lost EncodePicture(%s): %v", scenario.mode, err)
			}
			if lost.Keyframe {
				t.Fatalf("%s lost control delta unexpectedly became key picture", scenario.name)
			}
			cloneRTCPictureData(&lost)
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), lost, false, &nextFrameID)

			recoveryConfig := enc.Config()
			recoveryConfig.MaxFramerate = EncoderRational{Num: 120, Den: 1}
			recoveryConfig.RateControl = EncoderRateControlCBR
			recoveryConfig.Quantizer = 0
			recoveryConfig.Content = EncoderContentScreen
			publicRTCApplyControlBitrates(&recoveryConfig, publicRTCMatrixControlBitrateKbps(t, scenario.mode)+int32(scenarioIndex*113)+211)
			if publicRTCSetConfigRequiresKey(t, enc.Config(), recoveryConfig) {
				t.Fatalf("%s same-mode recovery-control change unexpectedly requires key", scenario.name)
			}
			if err := enc.SetConfig(recoveryConfig); err != nil {
				t.Fatalf("SetConfig(%s recovery controls): %v", scenario.mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), recoveryConfig)

			pliCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBPictureLossIndicationFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  0x05060708,
			})
			forceKey, packets, err := EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
				enc.Config(),
				pliCompound,
				make([]RTCPPacket, 0, 1),
				nil,
				nil,
			)
			if err != nil {
				t.Fatalf("%s PLI decision: %v", scenario.name, err)
			}
			if !forceKey || len(packets) != 1 {
				t.Fatalf("%s PLI force=%v packets=%d want true,1", scenario.name, forceKey, len(packets))
			}
			recovery, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), forceKey)
			if err != nil {
				t.Fatalf("recovery EncodePicture(%s): %v", scenario.mode, err)
			}
			if !recovery.Keyframe {
				t.Fatalf("%s recovery picture key=%v want true", scenario.name, recovery.Keyframe)
			}
			cloneRTCPictureData(&recovery)
			assertPublicRTCPictureDescriptors(t, &descriptorReceiver, enc.Config(), recovery, true, &nextFrameID)

			key0PacketsRaw := publicRTCPictureRTPPackets(t, key0, limits)
			lostPacketsRaw := publicRTCPictureRTPPackets(t, lost, limits)
			recoveryPacketsRaw := publicRTCPictureRTPPackets(t, recovery, limits)
			if len(key0PacketsRaw) == 0 || len(lostPacketsRaw) == 0 || len(recoveryPacketsRaw) == 0 {
				t.Fatalf("%s packet counts key0=%d lost=%d recovery=%d", scenario.name, len(key0PacketsRaw), len(lostPacketsRaw), len(recoveryPacketsRaw))
			}
			if len(lostPacketsRaw)+1 >= 1<<15 {
				t.Fatalf("%s lost packet gap=%d exceeds RTP sequencer window", scenario.name, len(lostPacketsRaw))
			}

			nextSequence := uint16(0x6200 + scenarioIndex*0x1000)
			key0Packets := publicRewriteRTPPacketSequences(t, key0PacketsRaw, &nextSequence)
			lostPackets := publicRewriteRTPPacketSequences(t, lostPacketsRaw, &nextSequence)
			recoveryPackets := publicRewriteRTPPacketSequences(t, recoveryPacketsRaw, &nextSequence)

			probePackets := publicRTCPictureDecodeTargetRTPPackets(t, key0, limits, 0, 0)
			probePackets = append(probePackets, publicRTCPictureDecodeTargetRTPPackets(t, recovery, limits, 0, 0)...)
			if len(probePackets) == 0 {
				t.Fatalf("%s produced no S0/T0 probe packets", scenario.name)
			}
			dec, err := NewLayeredDecoderFromRTPPackets(probePackets)
			if err != nil {
				t.Fatalf("NewLayeredDecoderFromRTPPackets(%s): %v", scenario.name, err)
			}
			defer dec.Close()
			if err := dec.Reset(); err != nil {
				t.Fatalf("LayeredDecoder Reset(%s): %v", scenario.name, err)
			}

			slotCount := len(key0Packets) + len(lostPackets) + len(recoveryPackets) + 8
			if slotCount < 16 {
				slotCount = 16
			}
			sequencer, err := BindRTPPacketSequencer(make([]RTPPacketSequencerSlot, slotCount))
			if err != nil {
				t.Fatalf("BindRTPPacketSequencer(%s): %v", scenario.name, err)
			}
			events := make([]RTPSequencedPacket, 0, 8)
			var rtpDescriptorReceiver RTPDependencyDescriptorState
			var got [][16]byte
			var gotMetadata []publicLayeredFrameMetadata
			pendingLoss := false
			decodeEvents := func(label string, in []RTPSequencedPacket) {
				t.Helper()
				for i, event := range in {
					parsed, err := ParseRTPPacketDependencyDescriptor(event.RawPacket, dependencyDescriptorExtensionID, &rtpDescriptorReceiver)
					if err != nil {
						t.Fatalf("%s %s event %d ParseRTPPacketDependencyDescriptor: %v", scenario.name, label, i, err)
					}
					forward, err := RTPDependencyDescriptorFrameForwardedForDecodeTarget(parsed.Descriptor, rtpDescriptorReceiver, 0)
					if err != nil {
						t.Fatalf("%s %s event %d FrameForwardedForDecodeTarget: %v", scenario.name, label, i, err)
					}
					if event.AfterLoss {
						pendingLoss = true
					}
					if !forward {
						continue
					}
					if pendingLoss {
						event.AfterLoss = true
						pendingLoss = false
					}
					frames, err := dec.DecodeRTPSequencedPacketWithMetadata(event)
					if err != nil {
						t.Fatalf("%s %s event %d DecodeRTPSequencedPacketWithMetadata: %v", scenario.name, label, i, err)
					}
					for _, frame := range frames {
						got = append(got, frameMD5Visible(frame.Frame))
						gotMetadata = append(gotMetadata, publicLayeredFrameMetadataFromDecoded(t, frame))
					}
				}
			}
			push := func(label string, packet []byte) []RTPSequencedPacket {
				t.Helper()
				out, err := sequencer.Push(packet, events[:0])
				if err != nil {
					t.Fatalf("%s %s Push: %v", scenario.name, label, err)
				}
				return out
			}

			for i, packet := range key0Packets {
				decodeEvents(fmt.Sprintf("key0-packet-%d", i), push("key0", packet))
			}
			recoveryFirst := push("recovery-first-before-loss-skip", recoveryPackets[0])
			if len(recoveryFirst) != 0 || sequencer.Buffered() != 1 {
				t.Fatalf("%s recovery pre-skip released=%d buffered=%d want 0/1", scenario.name, len(recoveryFirst), sequencer.Buffered())
			}
			missing, err := sequencer.AppendMissingSequenceNumbers(make([]uint16, 0, len(lostPackets)))
			if err != nil {
				t.Fatalf("%s AppendMissingSequenceNumbers lost picture: %v", scenario.name, err)
			}
			firstLostSeq := publicRTPPacketSequence(t, lostPackets[0])
			if len(missing) != len(lostPackets) || missing[0] != firstLostSeq {
				t.Fatalf("%s missing len=%d first=%#x want len=%d first=%#x", scenario.name, len(missing), missing[0], len(lostPackets), firstLostSeq)
			}
			pairs, err := sequencer.AppendMissingRTCPGenericNACKPairs(make([]RTCPGenericNACKPair, 0, (len(lostPackets)+15)/16))
			if err != nil {
				t.Fatalf("%s AppendMissingRTCPGenericNACKPairs lost picture: %v", scenario.name, err)
			}
			if len(pairs) == 0 || pairs[0].PacketID != firstLostSeq {
				t.Fatalf("%s NACK pairs=%+v want first PID %#x", scenario.name, pairs, firstLostSeq)
			}
			skipped := sequencer.SkipMissing(events[:0])
			if len(skipped) != 1 || !skipped[0].AfterLoss {
				t.Fatalf("%s SkipMissing released=%d afterLoss=%v want 1/true", scenario.name, len(skipped), len(skipped) == 1 && skipped[0].AfterLoss)
			}
			decodeEvents("recovery-after-loss-first", skipped)
			lateLost := push("late-lost-packet", lostPackets[0])
			if len(lateLost) != 0 {
				t.Fatalf("%s late lost packet released %d events, want 0", scenario.name, len(lateLost))
			}
			for i := 1; i < len(recoveryPackets); i++ {
				decodeEvents(fmt.Sprintf("recovery-packet-%d", i), push("recovery-tail", recoveryPackets[i]))
			}

			wantLowOverheads := append(publicRTCPictureDecodeTargetLowOverheads(key0, 0, 0), publicRTCPictureDecodeTargetLowOverheads(recovery, 0, 0)...)
			want := decodePublicLayeredDecoderLowOverheadDigests(t, wantLowOverheads...)
			var wantMetadata []publicLayeredFrameMetadata
			for _, picture := range []RTCPicture{key0, recovery} {
				for i := 0; i < picture.FrameNum; i++ {
					frame := picture.Frames[i]
					if frame.SpatialID == 0 && frame.TemporalID == 0 {
						wantMetadata = append(wantMetadata, publicLayeredFrameMetadataFromRTCFrame(frame))
					}
				}
			}
			if len(got) != len(want) {
				t.Fatalf("%s recovered frames=%d want %d", scenario.name, len(got), len(want))
			}
			assertPublicLayeredFrameMetadata(t, gotMetadata, wantMetadata)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s recovered frame %d digest differs after controls+RTP impairment: got=%x want=%x", scenario.name, i, got[i], want[i])
				}
			}

			var deliveredReceiver RTPDependencyDescriptorState
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte
			appendPublicRTCPictureRTPData(t, &deliveredReceiver, &layerTUs, &orderedTUs, key0)
			appendPublicRTCPictureRTPData(t, &deliveredReceiver, &layerTUs, &orderedTUs, recovery)
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
			assertPublicRTCRTPReferenceDecoders(t, decoders, "control-impairment-"+scenario.name, enc.Config(), layerTUs, orderedTUs)
		})
	}
}

func TestPublicLayeredDecoderRTPPayloadAfterLossSharedSVC(t *testing.T) {
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 24}
	mode := internalencoder.ScalabilityModeL2T1
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	key0, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
	if err != nil {
		t.Fatalf("key0 EncodePicture: %v", err)
	}
	key0LowOverheads := publicRTCPictureFramesInOrder(key0)
	key0Payloads := publicRTCPictureRTPPayloads(t, key0, limits)
	key0Packets := publicRTCPictureRTPPackets(t, key0, limits)
	delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	deltaPayloads := publicRTCPictureRTPPayloads(t, delta, limits)
	deltaPackets := publicRTCPictureRTPPackets(t, delta, limits)
	key2, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), true)
	if err != nil {
		t.Fatalf("key2 EncodePicture: %v", err)
	}
	key2LowOverheads := publicRTCPictureFramesInOrder(key2)
	key2Payloads := publicRTCPictureRTPPayloads(t, key2, limits)
	key2Packets := publicRTCPictureRTPPackets(t, key2, limits)
	if len(deltaPayloads) == 0 || len(key2Payloads) == 0 {
		t.Fatalf("payloads delta=%d key2=%d", len(deltaPayloads), len(key2Payloads))
	}
	probePayloads := append(append([][]byte(nil), key0Payloads...), key2Payloads...)
	probePackets := append(append([][]byte(nil), key0Packets...), key2Packets...)

	dec, err := NewLayeredDecoderFromRTPPayloads(probePayloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPayloads: %v", err)
	}
	defer dec.Close()
	if err := dec.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}

	var got [][16]byte
	for i, payload := range key0Payloads {
		frames, err := dec.DecodeRTPPayload(payload)
		if err != nil {
			t.Fatalf("DecodeRTPPayload key0 packet %d: %v", i, err)
		}
		for _, frame := range frames {
			got = append(got, frameMD5Visible(frame))
		}
	}
	frames, err := dec.DecodeRTPPayload(deltaPayloads[0])
	if err != nil {
		t.Fatalf("DecodeRTPPayload dropped delta prefix: %v", err)
	}
	if len(frames) != 0 {
		t.Fatalf("dropped delta prefix produced %d frames", len(frames))
	}
	frames, err = dec.DecodeRTPPayloadAfterLoss(key2Payloads[0])
	if err != nil {
		t.Fatalf("DecodeRTPPayloadAfterLoss key2 first packet: %v", err)
	}
	for _, frame := range frames {
		got = append(got, frameMD5Visible(frame))
	}
	for i, payload := range key2Payloads[1:] {
		frames, err := dec.DecodeRTPPayload(payload)
		if err != nil {
			t.Fatalf("DecodeRTPPayload key2 tail packet %d: %v", i, err)
		}
		for _, frame := range frames {
			got = append(got, frameMD5Visible(frame))
		}
	}

	wantPayloads := append(append([][]byte(nil), key0LowOverheads...), key2LowOverheads...)
	want := decodePublicLayeredDecoderLowOverheadDigests(t, wantPayloads...)
	if len(got) != len(want) {
		t.Fatalf("decoded frames got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frame %d digest differs after loss: got=%x want=%x", i, got[i], want[i])
		}
	}

	packetDec, err := NewLayeredDecoderFromRTPPackets(probePackets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets: %v", err)
	}
	defer packetDec.Close()
	if err := packetDec.Reset(); err != nil {
		t.Fatalf("packet Reset: %v", err)
	}
	got = got[:0]
	for i, packet := range key0Packets {
		frames, err := packetDec.DecodeRTPPacket(packet)
		if err != nil {
			t.Fatalf("DecodeRTPPacket key0 packet %d: %v", i, err)
		}
		for _, frame := range frames {
			got = append(got, frameMD5Visible(frame))
		}
	}
	frames, err = packetDec.DecodeRTPPacket(deltaPackets[0])
	if err != nil {
		t.Fatalf("DecodeRTPPacket dropped delta prefix: %v", err)
	}
	if len(frames) != 0 {
		t.Fatalf("dropped delta packet prefix produced %d frames", len(frames))
	}
	frames, err = packetDec.DecodeRTPPacketAfterLoss(key2Packets[0])
	if err != nil {
		t.Fatalf("DecodeRTPPacketAfterLoss key2 first packet: %v", err)
	}
	for _, frame := range frames {
		got = append(got, frameMD5Visible(frame))
	}
	for i, packet := range key2Packets[1:] {
		frames, err := packetDec.DecodeRTPPacket(packet)
		if err != nil {
			t.Fatalf("DecodeRTPPacket key2 tail packet %d: %v", i, err)
		}
		for _, frame := range frames {
			got = append(got, frameMD5Visible(frame))
		}
	}
	if len(got) != len(want) {
		t.Fatalf("decoded packet frames got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("packet frame %d digest differs after loss: got=%x want=%x", i, got[i], want[i])
		}
	}
}

func TestPublicLayeredDecoderRTPPacketRejectsMalformedWithoutMutating(t *testing.T) {
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 24}
	mode := internalencoder.ScalabilityModeL2T1
	width, height := publicRTCMatrixGeometry(t, mode)
	enc, err := NewRTCEncoderWithConfig(publicRTCMatrixConfig(width, height, mode))
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	key0, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
	if err != nil {
		t.Fatalf("key0 EncodePicture: %v", err)
	}
	key0Packets := publicRTCPictureRTPPackets(t, key0, limits)
	key0LowOverheads := publicRTCPictureFramesInOrder(key0)
	delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	deltaPackets := publicRTCPictureRTPPackets(t, delta, limits)
	deltaLowOverheads := publicRTCPictureFramesInOrder(delta)
	if len(deltaPackets) < 2 {
		t.Fatalf("delta picture packetized into %d RTP packets, want fragmentation", len(deltaPackets))
	}
	key2, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), true)
	if err != nil {
		t.Fatalf("key2 EncodePicture: %v", err)
	}
	key2Packets := publicRTCPictureRTPPackets(t, key2, limits)
	key2LowOverheads := publicRTCPictureFramesInOrder(key2)

	probePackets := append(append([][]byte(nil), key0Packets...), deltaPackets...)
	probePackets = append(probePackets, key2Packets...)
	for _, tc := range publicMalformedRTPPacketCases() {
		_, err := NewLayeredDecoderFromRTPPackets([][]byte{key0Packets[0], tc.packet})
		publicAssertMalformedRTPPacketError(t, "NewLayeredDecoderFromRTPPackets "+tc.name, err, tc.want)
	}

	dec, err := NewLayeredDecoderFromRTPPackets(probePackets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets: %v", err)
	}
	defer dec.Close()
	if err := dec.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}

	var got [][16]byte
	var gotMetadata []publicLayeredFrameMetadata
	appendLayered := func(frames []LayeredFrame) {
		t.Helper()
		for _, frame := range frames {
			got = append(got, frameMD5Visible(frame.Frame))
			gotMetadata = append(gotMetadata, publicLayeredFrameMetadataFromDecoded(t, frame))
		}
	}
	for i, packet := range key0Packets {
		frames, err := dec.DecodeRTPPacketWithMetadata(packet)
		if err != nil {
			t.Fatalf("DecodeRTPPacketWithMetadata key0 packet %d: %v", i, err)
		}
		appendLayered(frames)
	}
	if len(got) != key0.FrameNum {
		t.Fatalf("initial key decoded %d frames, want %d", len(got), key0.FrameNum)
	}
	for _, tc := range publicMalformedRTPPacketCases() {
		frames, err := dec.DecodeRTPPacketWithMetadata(tc.packet)
		publicAssertMalformedRTPPacketError(t, "DecodeRTPPacketWithMetadata "+tc.name, err, tc.want)
		if len(frames) != 0 {
			t.Fatalf("DecodeRTPPacketWithMetadata %s returned %d frames on malformed input", tc.name, len(frames))
		}
	}

	frames, err := dec.DecodeRTPPacketWithMetadata(deltaPackets[0])
	if err != nil {
		t.Fatalf("DecodeRTPPacketWithMetadata delta prefix: %v", err)
	}
	if len(frames) != 0 {
		t.Fatalf("delta prefix decoded %d frames, want 0", len(frames))
	}
	for _, tc := range publicMalformedRTPPacketCases() {
		frames, err := dec.DecodeRTPPacketAfterLossWithMetadata(tc.packet)
		publicAssertMalformedRTPPacketError(t, "DecodeRTPPacketAfterLossWithMetadata "+tc.name, err, tc.want)
		if len(frames) != 0 {
			t.Fatalf("DecodeRTPPacketAfterLossWithMetadata %s returned %d frames on malformed input", tc.name, len(frames))
		}
	}
	for i, packet := range deltaPackets[1:] {
		frames, err := dec.DecodeRTPPacketWithMetadata(packet)
		if err != nil {
			t.Fatalf("DecodeRTPPacketWithMetadata delta tail %d after malformed after-loss packets: %v", i, err)
		}
		appendLayered(frames)
	}
	for i, packet := range key2Packets {
		frames, err := dec.DecodeRTPPacketWithMetadata(packet)
		if err != nil {
			t.Fatalf("DecodeRTPPacketWithMetadata key2 packet %d after malformed packets: %v", i, err)
		}
		appendLayered(frames)
	}

	wantPayloads := append(append([][]byte(nil), key0LowOverheads...), deltaLowOverheads...)
	wantPayloads = append(wantPayloads, key2LowOverheads...)
	want := decodePublicLayeredDecoderLowOverheadDigests(t, wantPayloads...)
	if len(got) != len(want) {
		t.Fatalf("layered RTP packet decoder produced %d frames after malformed packets, low-overhead decoded %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("layered frame %d digest differs after malformed RTP packet errors: got=%x want=%x", i, got[i], want[i])
		}
	}
	var wantMetadata []publicLayeredFrameMetadata
	for _, picture := range []RTCPicture{key0, delta, key2} {
		for i := 0; i < picture.FrameNum; i++ {
			wantMetadata = append(wantMetadata, publicLayeredFrameMetadataFromRTCFrame(picture.Frames[i]))
		}
	}
	assertPublicLayeredFrameMetadata(t, gotMetadata, wantMetadata)
}

func TestPublicLayeredDecoderRTPPacketSequencerAfterLossSharedSVC(t *testing.T) {
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 24}
	mode := internalencoder.ScalabilityModeL2T1
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	key0, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
	if err != nil {
		t.Fatalf("key0 EncodePicture: %v", err)
	}
	key0PacketsRaw := publicRTCPictureRTPPackets(t, key0, limits)
	key0LowOverheads := publicRTCPictureFramesInOrder(key0)
	delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	deltaPacketsRaw := publicRTCPictureRTPPackets(t, delta, limits)
	key2, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), true)
	if err != nil {
		t.Fatalf("key2 EncodePicture: %v", err)
	}
	key2PacketsRaw := publicRTCPictureRTPPackets(t, key2, limits)
	key2LowOverheads := publicRTCPictureFramesInOrder(key2)

	next := uint16(0x4000)
	key0Packets := publicRewriteRTPPacketSequences(t, key0PacketsRaw, &next)
	deltaPackets := publicRewriteRTPPacketSequences(t, deltaPacketsRaw, &next)
	key2Packets := publicRewriteRTPPacketSequences(t, key2PacketsRaw, &next)
	probePackets := append(append([][]byte(nil), key0Packets...), deltaPackets...)
	probePackets = append(probePackets, key2Packets...)

	sequencer, err := BindRTPPacketSequencer(make([]RTPPacketSequencerSlot, 2048))
	if err != nil {
		t.Fatalf("BindRTPPacketSequencer: %v", err)
	}
	dec, err := NewLayeredDecoderFromRTPPackets(probePackets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets: %v", err)
	}
	defer dec.Close()

	var got [][16]byte
	var gotMetadata []publicLayeredFrameMetadata
	decodeEvents := func(label string, events []RTPSequencedPacket) {
		t.Helper()
		for i, event := range events {
			frames, err := dec.DecodeRTPSequencedPacketWithMetadata(event)
			if err != nil {
				t.Fatalf("%s event %d DecodeRTPSequencedPacketWithMetadata: %v", label, i, err)
			}
			for _, frame := range frames {
				got = append(got, frameMD5Visible(frame.Frame))
				gotMetadata = append(gotMetadata, publicLayeredFrameMetadataFromDecoded(t, frame))
			}
		}
	}

	events := make([]RTPSequencedPacket, 0, 16)
	for i, packet := range key0Packets {
		events, err = sequencer.Push(packet, events[:0])
		if err != nil {
			t.Fatalf("Push key0 packet %d: %v", i, err)
		}
		decodeEvents("key0", events)
	}
	if len(got) != key0.FrameNum {
		t.Fatalf("initial key decoded %d frames, want %d", len(got), key0.FrameNum)
	}

	events, err = sequencer.Push(key2Packets[0], events[:0])
	if err != nil {
		t.Fatalf("Push recovery key first packet: %v", err)
	}
	if len(events) != 0 || sequencer.Buffered() != 1 {
		t.Fatalf("future recovery packet released=%d buffered=%d want 0/1", len(events), sequencer.Buffered())
	}
	events = sequencer.SkipMissing(events[:0])
	if len(events) != 1 || !events[0].AfterLoss {
		t.Fatalf("SkipMissing events=%d afterLoss=%v", len(events), len(events) == 1 && events[0].AfterLoss)
	}
	decodeEvents("recovery first", events)

	events, err = sequencer.Push(deltaPackets[0], events[:0])
	if err != nil {
		t.Fatalf("Push late delta packet: %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("late delta released %d packets, want 0", len(events))
	}
	for i, packet := range key2Packets[1:] {
		events, err = sequencer.Push(packet, events[:0])
		if err != nil {
			t.Fatalf("Push recovery key tail %d: %v", i, err)
		}
		decodeEvents("recovery tail", events)
	}

	lowOverheads := append(append([][]byte(nil), key0LowOverheads...), key2LowOverheads...)
	want := decodePublicLayeredDecoderLowOverheadDigests(t, lowOverheads...)
	var wantMetadata []publicLayeredFrameMetadata
	for _, picture := range []RTCPicture{key0, key2} {
		for i := 0; i < picture.FrameNum; i++ {
			wantMetadata = append(wantMetadata, publicLayeredFrameMetadataFromRTCFrame(picture.Frames[i]))
		}
	}
	if len(got) != len(want) {
		t.Fatalf("sequenced layered RTP decoder recovered %d frames, low-overhead decoded %d", len(got), len(want))
	}
	assertPublicLayeredFrameMetadata(t, gotMetadata, wantMetadata)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sequenced layered frame %d digest differs after loss: got=%x want=%x", i, got[i], want[i])
		}
	}
}

func TestPublicRTCEncoderRTCPCompoundFeedbackForcesKeyPicture(t *testing.T) {
	mode := internalencoder.ScalabilityModeL3T2h
	width, height := publicRTCMatrixGeometry(t, mode)
	enc, err := NewRTCEncoderWithConfig(publicRTCMatrixConfig(width, height, mode))
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var receiver RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
	var orderedTUs [][]byte

	key, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
	if err != nil {
		t.Fatalf("initial key EncodePicture: %v", err)
	}
	appendPublicRTCPictureRTPData(t, &receiver, &layerTUs, &orderedTUs, key)
	assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), key, true, &nextFrameID)

	delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	appendPublicRTCPictureRTPData(t, &receiver, &layerTUs, &orderedTUs, delta)
	assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), delta, false, &nextFrameID)

	compound, err := AppendRTCPSenderReportPacket(make([]byte, 0, 128), RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	entry := testAV1RTCPValidLayerRefreshEntry(t, mode)
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  entry.SSRC,
		FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{entry}),
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket LRR: %v", err)
	}

	forceKey, packets, err := enc.RTCPRequiresKeyFrame(
		compound,
		make([]RTCPPacket, 0, 2),
		nil,
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame: %v", err)
	}
	if !forceKey || len(packets) != 2 {
		t.Fatalf("forceKey=%v packet len=%d want true,2", forceKey, len(packets))
	}
	target, ok, targetPackets, err := enc.RTCPLayerRefreshTarget(
		compound,
		make([]RTCPPacket, 0, 2),
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget: %v", err)
	}
	if !ok || len(targetPackets) != 2 || target != entry.Target {
		t.Fatalf("target=%+v ok=%v packet len=%d want %+v,true,2", target, ok, len(targetPackets), entry.Target)
	}

	recovery, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), forceKey)
	if err != nil {
		t.Fatalf("recovery EncodePicture: %v", err)
	}
	appendPublicRTCPictureRTPData(t, &receiver, &layerTUs, &orderedTUs, recovery)
	assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), recovery, true, &nextFrameID)
	spatialLayers, temporalLayers, _, ok := enc.Config().Scalability.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", enc.Config().Scalability)
	}
	options, err := recovery.ActiveDecodeTargetsRTPOptions(target.SpatialID, target.TemporalID)
	if err != nil {
		t.Fatalf("recovery ActiveDecodeTargetsRTPOptions: %v", err)
	}
	wantMask := publicExpectedActiveDecodeTargetsMask(spatialLayers, temporalLayers, target.SpatialID, target.TemporalID)
	if options.ActiveDecodeTargetsMask != wantMask {
		t.Fatalf("LRR active mask=%#x want %#x", options.ActiveDecodeTargetsMask, wantMask)
	}
	var activeReceiver RTPDependencyDescriptorState
	for i := 0; i < recovery.FrameNum; i++ {
		assertPublicRTCFrameRTPPacketsWithActiveDecodeTargets(t, &activeReceiver, recovery.Frames[i], RTPPayloadSizeLimits{MaxPayloadLen: 48}, options)
	}
	assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
}

func TestPublicRTCEncoderRTCPFeedbackMethodsAllocs(t *testing.T) {
	mode := internalencoder.ScalabilityModeL3T2h
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	publicRTCApplyControlBitrates(&cfg, 1500)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	rembCompound := publicRTCREMBFeedbackCompound(t, 700_000)
	pliCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	entry := testAV1RTCPValidLayerRefreshEntry(t, mode)
	lrrCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  entry.SSRC,
		FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{entry}),
	})
	packetScratch := make([]RTCPPacket, 0, 1)
	ssrcScratch := make([]uint32, 0, 1)
	lrrScratch := make([]AV1RTCPLayerRefreshRequestEntry, 0, 1)

	allocs := testing.AllocsPerRun(1000, func() {
		if ok, packets, err := enc.ApplyRTCPReceiverEstimatedMaximumBitrate(rembCompound, packetScratch[:0], ssrcScratch[:0]); err != nil || !ok || len(packets) != 1 {
			t.Fatalf("alloc REMB ok=%v packets=%d err=%v", ok, len(packets), err)
		}
		if force, packets, err := enc.RTCPRequiresKeyFrame(pliCompound, packetScratch[:0], nil, nil); err != nil || !force || len(packets) != 1 {
			t.Fatalf("alloc PLI force=%v packets=%d err=%v", force, len(packets), err)
		}
		if target, ok, packets, err := enc.RTCPLayerRefreshTarget(lrrCompound, packetScratch[:0], lrrScratch[:0]); err != nil || !ok || len(packets) != 1 || target != entry.Target {
			t.Fatalf("alloc LRR target=%+v ok=%v packets=%d err=%v", target, ok, len(packets), err)
		}
	})
	if allocs != 0 {
		t.Fatalf("RTCEncoder RTCP feedback method allocs/run=%f want 0", allocs)
	}
}

func TestPublicRTCEncoderSettingsMatrixDependencyDescriptors(t *testing.T) {
	for _, initial := range EncoderWebRTCScalabilityModes() {
		initial := initial
		reconfig := publicRTCMatrixReconfigMode(t, initial)
		width, height := publicRTCMatrixGeometry(t, initial)
		t.Run(initial.String()+"-to-"+reconfig.String(), func(t *testing.T) {
			cfg := publicRTCMatrixConfig(width, height, initial)
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", initial, err)
			}
			var receiver RTPDependencyDescriptorState
			var packetReceiver RTPDependencyDescriptorState
			var activeReceiver RTPDependencyDescriptorState
			activeLimits := RTPPayloadSizeLimits{MaxPayloadLen: 48}
			nextFrameID := uint64(0)
			var layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte

			key, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
			if err != nil {
				t.Fatalf("initial key EncodePicture: %v", err)
			}
			appendPublicRTCPictureRTPData(t, &packetReceiver, &layerTUs, &orderedTUs, key)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), key, true, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), key, activeLimits)

			controlChange := enc.Config()
			controlChange.MaxFramerate = EncoderRational{Num: 60, Den: 1}
			publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, initial))
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig control change: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			controlDelta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
			if err != nil {
				t.Fatalf("control delta EncodePicture: %v", err)
			}
			appendPublicRTCPictureRTPData(t, &packetReceiver, &layerTUs, &orderedTUs, controlDelta)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), controlDelta, false, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), controlDelta, activeLimits)
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)

			structureChange := enc.Config()
			structureChange.Scalability = reconfig
			publicRTCApplyControlBitrates(&structureChange, publicRTCMatrixControlBitrateKbps(t, reconfig)+90)
			if err := enc.SetConfig(structureChange); err != nil {
				t.Fatalf("SetConfig structure change: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), structureChange)
			var reconfigLayerTUs [EncoderWebRTCMaxSpatialLayers][][]byte
			var reconfigOrderedTUs [][]byte
			structureKey, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), false)
			if err != nil {
				t.Fatalf("structure key EncodePicture: %v", err)
			}
			appendPublicRTCPictureRTPData(t, &packetReceiver, &reconfigLayerTUs, &reconfigOrderedTUs, structureKey)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), structureKey, true, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), structureKey, activeLimits)

			postReconfigDelta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 3), false)
			if err != nil {
				t.Fatalf("post-reconfigure delta EncodePicture: %v", err)
			}
			appendPublicRTCPictureRTPData(t, &packetReceiver, &reconfigLayerTUs, &reconfigOrderedTUs, postReconfigDelta)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), postReconfigDelta, false, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), postReconfigDelta, activeLimits)
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), reconfigLayerTUs, reconfigOrderedTUs)
		})
	}
}

func publicRTCMatrixGeometry(t *testing.T, mode EncoderScalabilityMode) (int, int) {
	t.Helper()
	spatial, _, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", mode)
	}
	switch spatial {
	case 1:
		return 192, 128
	case 2:
		return 640, 360
	case 3:
		return 1008, 576
	default:
		t.Fatalf("unsupported spatial layer count %d for %s", spatial, mode)
		return 0, 0
	}
}

func publicRTCMatrixReconfigMode(t *testing.T, mode EncoderScalabilityMode) EncoderScalabilityMode {
	t.Helper()
	spatial, temporal, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", mode)
	}
	wantTemporal := temporal + 1
	if temporal >= 3 {
		wantTemporal = temporal - 1
	}
	for _, candidate := range EncoderWebRTCScalabilityModes() {
		cSpatial, cTemporal, _, ok := candidate.Layers()
		if ok && candidate != mode && cSpatial == spatial && cTemporal == wantTemporal {
			return candidate
		}
	}
	for _, candidate := range EncoderWebRTCScalabilityModes() {
		cSpatial, _, _, ok := candidate.Layers()
		if ok && candidate != mode && cSpatial == spatial {
			return candidate
		}
	}
	t.Fatalf("no reconfigure mode for %s", mode)
	return EncoderScalabilityModeL1T1
}

func publicRTCMatrixControlBitrateKbps(t *testing.T, mode EncoderScalabilityMode) int32 {
	t.Helper()
	spatial, _, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", mode)
	}
	switch spatial {
	case 1:
		return 420
	case 2:
		return 900
	case 3:
		return 1500
	default:
		t.Fatalf("unsupported spatial layer count %d for %s", spatial, mode)
		return 0
	}
}

func publicRTCApplyControlBitrates(cfg *EncoderConfig, targetKbps int32) {
	cfg.MinBitrateKbps = targetKbps / 4
	cfg.MaxBitrateKbps = targetKbps * 2
	cfg.TargetBitrateKbps = targetKbps
	spatialLayers, _, _, ok := cfg.Scalability.Layers()
	if !ok {
		return
	}
	for i := uint8(0); i < spatialLayers; i++ {
		layer := &cfg.SpatialLayers[i]
		if spatialLayers == 1 {
			layer.MinBitrateKbps = cfg.MinBitrateKbps
			layer.MaxBitrateKbps = cfg.MaxBitrateKbps
			layer.TargetBitrateKbps = cfg.TargetBitrateKbps
			continue
		}
		layerTarget := targetKbps * int32(i+1) / int32(spatialLayers)
		layer.MinBitrateKbps = layerTarget / 2
		layer.MaxBitrateKbps = layerTarget * 2
		layer.TargetBitrateKbps = layerTarget
	}
}

func assertPublicRTCConfigBitrates(t *testing.T, got EncoderConfig, want EncoderConfig) {
	t.Helper()
	spatialLayers, _, _, ok := got.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid normalized mode %s", got.Scalability)
	}
	want, err := SetWebRTCEncoderSVCConfig(want, want.TemporalLayerCount, want.SpatialLayerCount)
	if err != nil {
		t.Fatalf("normalize expected config %s: %v", want.Scalability, err)
	}
	if got.SpatialLayerCount != spatialLayers {
		t.Fatalf("config spatial layers=%d want %d", got.SpatialLayerCount, spatialLayers)
	}
	if want.SpatialLayerCount != spatialLayers {
		t.Fatalf("expected config spatial layers=%d want %d", want.SpatialLayerCount, spatialLayers)
	}
	for i := uint8(0); i < spatialLayers; i++ {
		gotLayer := got.SpatialLayers[i]
		wantLayer := want.SpatialLayers[i]
		if gotLayer.MinBitrateKbps != wantLayer.MinBitrateKbps ||
			gotLayer.MaxBitrateKbps != wantLayer.MaxBitrateKbps ||
			gotLayer.TargetBitrateKbps != wantLayer.TargetBitrateKbps {
			t.Fatalf("layer %d bitrate got %+v want min=%d max=%d target=%d", i, gotLayer, wantLayer.MinBitrateKbps, wantLayer.MaxBitrateKbps, wantLayer.TargetBitrateKbps)
		}
	}
}

func assertPublicRTCConfigControls(t *testing.T, got EncoderConfig, want EncoderConfig) {
	t.Helper()
	want, err := SetWebRTCEncoderSVCConfig(want, want.TemporalLayerCount, want.SpatialLayerCount)
	if err != nil {
		t.Fatalf("normalize expected config %s: %v", want.Scalability, err)
	}
	if got.Scalability != want.Scalability ||
		got.SpatialLayerCount != want.SpatialLayerCount ||
		got.TemporalLayerCount != want.TemporalLayerCount {
		t.Fatalf("config layers got mode=%s S%d/T%d want mode=%s S%d/T%d",
			got.Scalability, got.SpatialLayerCount, got.TemporalLayerCount,
			want.Scalability, want.SpatialLayerCount, want.TemporalLayerCount)
	}
	if got.MaxFramerate != want.MaxFramerate || got.RTPTimebase != want.RTPTimebase {
		t.Fatalf("config timing fps=%+v timebase=%+v want fps=%+v timebase=%+v",
			got.MaxFramerate, got.RTPTimebase, want.MaxFramerate, want.RTPTimebase)
	}
	gotDuration, err := EncoderWebRTCRTPFrameDuration(got)
	if err != nil {
		t.Fatalf("got RTP frame duration: %v", err)
	}
	wantDuration, err := EncoderWebRTCRTPFrameDuration(want)
	if err != nil {
		t.Fatalf("want RTP frame duration: %v", err)
	}
	if gotDuration != wantDuration {
		t.Fatalf("RTP frame duration=%+v want %+v", gotDuration, wantDuration)
	}
	if got.RateControl != want.RateControl || got.Quantizer != want.Quantizer || got.Content != want.Content {
		t.Fatalf("config rate control=%d q=%d content=%d want %d q=%d content=%d",
			got.RateControl, got.Quantizer, got.Content, want.RateControl, want.Quantizer, want.Content)
	}
	if got.Speed != want.Speed || got.MaxThreads != want.MaxThreads || got.KeyFrameInterval != want.KeyFrameInterval {
		t.Fatalf("config speed=%d threads=%d keyInterval=%d want speed=%d threads=%d keyInterval=%d",
			got.Speed, got.MaxThreads, got.KeyFrameInterval, want.Speed, want.MaxThreads, want.KeyFrameInterval)
	}
	for i := uint8(0); i < want.SpatialLayerCount; i++ {
		gotLayer := got.SpatialLayers[i]
		wantLayer := want.SpatialLayers[i]
		if gotLayer.Active != wantLayer.Active ||
			gotLayer.Resolution != wantLayer.Resolution ||
			gotLayer.ScalingFactor != wantLayer.ScalingFactor ||
			gotLayer.MaxFramerate != wantLayer.MaxFramerate ||
			gotLayer.TemporalLayers != wantLayer.TemporalLayers {
			t.Fatalf("layer %d controls got %+v want %+v", i, gotLayer, wantLayer)
		}
	}
	assertPublicRTCConfigBitrates(t, got, want)
}

func publicRTCSetConfigRequiresKey(t *testing.T, before EncoderConfig, after EncoderConfig) bool {
	t.Helper()
	normalizedBefore, err := SetWebRTCEncoderSVCConfig(before, before.TemporalLayerCount, before.SpatialLayerCount)
	if err != nil {
		t.Fatalf("normalize previous config %s: %v", before.Scalability, err)
	}
	normalizedAfter, err := SetWebRTCEncoderSVCConfig(after, after.TemporalLayerCount, after.SpatialLayerCount)
	if err != nil {
		t.Fatalf("normalize next config %s: %v", after.Scalability, err)
	}
	if !publicRTCConfigLayerGeometryEqual(normalizedBefore, normalizedAfter) {
		return true
	}
	beforeStructure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalizedBefore)
	if err != nil {
		t.Fatalf("previous dependency structure %s: %v", normalizedBefore.Scalability, err)
	}
	afterStructure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalizedAfter)
	if err != nil {
		t.Fatalf("next dependency structure %s: %v", normalizedAfter.Scalability, err)
	}
	return beforeStructure != afterStructure
}

func publicRTCConfigLayerGeometryEqual(a EncoderConfig, b EncoderConfig) bool {
	if a.SpatialLayerCount != b.SpatialLayerCount {
		return false
	}
	for i := uint8(0); i < a.SpatialLayerCount; i++ {
		if a.SpatialLayers[i].Resolution != b.SpatialLayers[i].Resolution {
			return false
		}
	}
	return true
}

func cloneRTCPictureData(p *RTCPicture) {
	for i := 0; i < p.FrameNum; i++ {
		p.Frames[i].Data = append([]byte(nil), p.Frames[i].Data...)
		p.Frames[i].DependencyDescriptor = append([]byte(nil), p.Frames[i].DependencyDescriptor...)
	}
}

func publicRTCMatrixConfig(width, height int, mode EncoderScalabilityMode) EncoderConfig {
	return EncoderConfig{
		Resolution:        EncoderResolution{Width: int32(width), Height: int32(height)},
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    120,
		MaxBitrateKbps:    1600,
		TargetBitrateKbps: 900,
		Scalability:       mode,
	}
}

func publicRTCMatrixFrame(width, height int, n int) I420Frame {
	cw, ch := width/2, height/2
	f := I420Frame{
		Y: make([]byte, width*height), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: width, ChromaStride: cw, Width: width, Height: height,
	}
	for y := 0; y < height; y++ {
		row := f.Y[y*width : y*width+width]
		for x := range row {
			row[x] = uint8(45 + (x*3+y*5+n*17)%175)
		}
	}
	for i := range f.U {
		f.U[i] = uint8(116 + (i+n)%9)
		f.V[i] = uint8(132 - (i+n)%7)
	}
	return f
}

func publicI420NeutralChroma(src I420Frame) I420Frame {
	out := src
	out.Y = append([]byte(nil), src.Y...)
	cw, ch := src.Width/2, src.Height/2
	out.U = make([]byte, cw*ch)
	out.V = make([]byte, cw*ch)
	for i := range out.U {
		out.U[i] = 128
		out.V[i] = 128
	}
	out.YStride = src.YStride
	out.ChromaStride = cw
	return out
}

func publicI422FromI420(src I420Frame) I422Frame {
	cw := src.Width / 2
	u := make([]byte, cw*src.Height)
	v := make([]byte, cw*src.Height)
	for y := 0; y < src.Height; y++ {
		srcY := (y / 2) * src.ChromaStride
		copy(u[y*cw:y*cw+cw], src.U[srcY:srcY+cw])
		copy(v[y*cw:y*cw+cw], src.V[srcY:srcY+cw])
	}
	return I422Frame{
		Y:            src.Y,
		U:            u,
		V:            v,
		YStride:      src.YStride,
		ChromaStride: cw,
		Width:        src.Width,
		Height:       src.Height,
	}
}

func publicI444FromI420(src I420Frame) I444Frame {
	u := make([]byte, src.Width*src.Height)
	v := make([]byte, src.Width*src.Height)
	cw := src.Width / 2
	for y := 0; y < src.Height; y++ {
		srcY := (y / 2) * src.ChromaStride
		dstU := u[y*src.Width : y*src.Width+src.Width]
		dstV := v[y*src.Width : y*src.Width+src.Width]
		for x := 0; x < cw; x++ {
			dstU[x*2] = src.U[srcY+x]
			dstU[x*2+1] = src.U[srcY+x]
			dstV[x*2] = src.V[srcY+x]
			dstV[x*2+1] = src.V[srcY+x]
		}
	}
	return I444Frame{
		Y:       src.Y,
		U:       u,
		V:       v,
		YStride: src.YStride,
		UStride: src.Width,
		VStride: src.Width,
		Width:   src.Width,
		Height:  src.Height,
	}
}

func publicI422NativePattern(width int, height int, frame int) I422Frame {
	base := publicRTCMatrixFrame(width, height, frame)
	chromaWidth := width / 2
	u := make([]byte, chromaWidth*height)
	v := make([]byte, chromaWidth*height)
	for y := 0; y < height; y++ {
		ur := u[y*chromaWidth : y*chromaWidth+chromaWidth]
		vr := v[y*chromaWidth : y*chromaWidth+chromaWidth]
		for x := 0; x < chromaWidth; x++ {
			ur[x] = byte((x*9 + y*5 + frame*17) & 0xff)
			vr[x] = byte(255 - ((x*7 + y*13 + frame*11) & 0xff))
		}
	}
	return I422Frame{
		Y:            base.Y,
		U:            u,
		V:            v,
		YStride:      base.YStride,
		ChromaStride: chromaWidth,
		Width:        width,
		Height:       height,
	}
}

func publicI444NativePattern(width int, height int, frame int) I444Frame {
	base := publicRTCMatrixFrame(width, height, frame)
	u := make([]byte, width*height)
	v := make([]byte, width*height)
	for y := 0; y < height; y++ {
		ur := u[y*width : y*width+width]
		vr := v[y*width : y*width+width]
		for x := 0; x < width; x++ {
			ur[x] = byte((x*7 + y*3 + frame*17) & 0xff)
			vr[x] = byte(255 - ((x*5 + y*11 + frame*13) & 0xff))
		}
	}
	return I444Frame{
		Y:       base.Y,
		U:       u,
		V:       v,
		YStride: base.YStride,
		UStride: width,
		VStride: width,
		Width:   width,
		Height:  height,
	}
}

func publicI400FromI420(src I420Frame) I400Frame {
	return I400Frame{
		Y:       src.Y,
		YStride: src.YStride,
		Width:   src.Width,
		Height:  src.Height,
	}
}

func publicI400HighBitDepthFromI420(src I420Frame, bitDepth uint8, n int) I400HighBitDepthFrame {
	y := make([]uint16, src.Width*src.Height)
	lowMask := uint16(0)
	if bitDepth > 8 {
		lowMask = uint16((1 << (bitDepth - 8)) - 1)
	}
	for yy := 0; yy < src.Height; yy++ {
		srcRow := src.Y[yy*src.YStride : yy*src.YStride+src.Width]
		dstRow := y[yy*src.Width : yy*src.Width+src.Width]
		for x, sample := range srcRow {
			dstRow[x] = publicFrameHighBitSample(sample, bitDepth) | (uint16(x+yy+n) & lowMask)
		}
	}
	return I400HighBitDepthFrame{
		Y:        y,
		YStride:  src.Width,
		Width:    src.Width,
		Height:   src.Height,
		BitDepth: bitDepth,
	}
}

func publicFrameFromI420(tb testing.TB, src I420Frame, format FrameFormat) Frame {
	tb.Helper()
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		tb.Fatalf("FrameRequiredSize: %v", err)
	}
	frame, err := internalframe.Bind(make([]byte, layout.Size), format)
	if err != nil {
		tb.Fatalf("BindFrame: %v", err)
	}
	bytesPerSample := layout.BytesPerSample
	for y := 0; y < src.Height; y++ {
		srcRow := src.Y[y*src.YStride : y*src.YStride+src.Width]
		for x, sample := range srcRow {
			setPublicFrameSample(frame.Y, bytesPerSample, x, y, publicFrameHighBitSample(sample, frame.Format.BitDepth))
		}
	}
	if frame.Format.MonoChrome {
		return frame
	}
	switch {
	case frame.Format.SubsamplingX && frame.Format.SubsamplingY:
		for y := 0; y < src.Height/2; y++ {
			for x := 0; x < src.Width/2; x++ {
				u := src.U[y*src.ChromaStride+x]
				v := src.V[y*src.ChromaStride+x]
				setPublicFrameSample(frame.U, bytesPerSample, x, y, publicFrameHighBitSample(u, frame.Format.BitDepth))
				setPublicFrameSample(frame.V, bytesPerSample, x, y, publicFrameHighBitSample(v, frame.Format.BitDepth))
			}
		}
	case frame.Format.SubsamplingX:
		for y := 0; y < src.Height; y++ {
			srcY := (y / 2) * src.ChromaStride
			for x := 0; x < src.Width/2; x++ {
				u := src.U[srcY+x]
				v := src.V[srcY+x]
				setPublicFrameSample(frame.U, bytesPerSample, x, y, publicFrameHighBitSample(u, frame.Format.BitDepth))
				setPublicFrameSample(frame.V, bytesPerSample, x, y, publicFrameHighBitSample(v, frame.Format.BitDepth))
			}
		}
	case !frame.Format.SubsamplingY:
		for y := 0; y < src.Height; y++ {
			srcY := (y / 2) * src.ChromaStride
			for x := 0; x < src.Width; x++ {
				u := src.U[srcY+x/2]
				v := src.V[srcY+x/2]
				setPublicFrameSample(frame.U, bytesPerSample, x, y, publicFrameHighBitSample(u, frame.Format.BitDepth))
				setPublicFrameSample(frame.V, bytesPerSample, x, y, publicFrameHighBitSample(v, frame.Format.BitDepth))
			}
		}
	default:
		tb.Fatalf("unsupported test frame format: %+v", frame.Format)
	}
	return frame
}

func publicFrameHighBitSample(sample byte, bitDepth uint8) uint16 {
	if bitDepth <= 8 {
		return uint16(sample)
	}
	return uint16(sample) << (bitDepth - 8)
}

func publicNV12FromI420(src I420Frame) NV12Frame {
	uv := make([]byte, src.Width*src.Height/2)
	cw, ch := src.Width/2, src.Height/2
	for y := 0; y < ch; y++ {
		dstRow := uv[y*src.Width : y*src.Width+src.Width]
		uRow := src.U[y*src.ChromaStride : y*src.ChromaStride+cw]
		vRow := src.V[y*src.ChromaStride : y*src.ChromaStride+cw]
		for x := 0; x < cw; x++ {
			dstRow[x*2] = uRow[x]
			dstRow[x*2+1] = vRow[x]
		}
	}
	return NV12Frame{
		Y:        src.Y,
		UV:       uv,
		YStride:  src.YStride,
		UVStride: src.Width,
		Width:    src.Width,
		Height:   src.Height,
	}
}

func publicNV21FromI420(src I420Frame) NV21Frame {
	vu := make([]byte, src.Width*src.Height/2)
	cw, ch := src.Width/2, src.Height/2
	for y := 0; y < ch; y++ {
		dstRow := vu[y*src.Width : y*src.Width+src.Width]
		uRow := src.U[y*src.ChromaStride : y*src.ChromaStride+cw]
		vRow := src.V[y*src.ChromaStride : y*src.ChromaStride+cw]
		for x := 0; x < cw; x++ {
			dstRow[x*2] = vRow[x]
			dstRow[x*2+1] = uRow[x]
		}
	}
	return NV21Frame{
		Y:        src.Y,
		VU:       vu,
		YStride:  src.YStride,
		VUStride: src.Width,
		Width:    src.Width,
		Height:   src.Height,
	}
}

func assertPublicRTCPictureDescriptors(t *testing.T, receiver *RTPDependencyDescriptorState, cfg EncoderConfig, picture RTCPicture, wantKey bool, nextFrameID *uint64) {
	t.Helper()
	normalized, err := SetWebRTCEncoderSVCConfig(cfg, cfg.TemporalLayerCount, cfg.SpatialLayerCount)
	if err != nil {
		t.Fatalf("SetWebRTCEncoderSVCConfig(%s): %v", cfg.Scalability, err)
	}
	spatialLayers, temporalLayers, _, ok := normalized.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", normalized.Scalability)
	}
	if picture.Keyframe != wantKey || picture.FrameNum != int(spatialLayers) {
		t.Fatalf("picture key=%v frames=%d want key=%v frames=%d", picture.Keyframe, picture.FrameNum, wantKey, spatialLayers)
	}
	for i := 0; i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		wantFrameID := *nextFrameID + uint64(i)
		if frame.FrameID != wantFrameID || frame.SpatialID != uint8(i) || frame.TemporalID >= temporalLayers || frame.Keyframe != wantKey {
			t.Fatalf("frame %d metadata=%+v want id=%d spatial=%d key=%v temporal<%d", i, frame, wantFrameID, i, wantKey, temporalLayers)
		}
		wantCodedKeyframe := wantKey && (normalized.Scalability.IsSimulcast() || frame.SpatialID == 0)
		wantLastFrame := i+1 == picture.FrameNum
		if frame.CodedKeyframe != wantCodedKeyframe || frame.LastFrameInPicture != wantLastFrame {
			t.Fatalf("frame %d coding metadata=%+v want codedKey=%v last=%v mode=%s", i, frame, wantCodedKeyframe, wantLastFrame, normalized.Scalability)
		}
		if len(frame.DependencyDescriptor) == 0 {
			t.Fatalf("frame %d missing dependency descriptor", i)
		}
		parsed, consumed, err := receiver.Parse(frame.DependencyDescriptor)
		if err != nil {
			t.Fatalf("frame %d dependency descriptor parse: %v", i, err)
		}
		if consumed != len(frame.DependencyDescriptor) ||
			parsed.Mandatory.FrameNumber != uint16(frame.FrameID) ||
			!parsed.Mandatory.FirstPacketInFrame ||
			!parsed.Mandatory.LastPacketInFrame {
			t.Fatalf("frame %d parsed mandatory=%+v consumed=%d len=%d frame=%+v", i, parsed.Mandatory, consumed, len(frame.DependencyDescriptor), frame)
		}
		wantAttached := wantKey && i == 0
		if parsed.HasAttachedStructure != wantAttached {
			t.Fatalf("frame %d attached=%v want %v descriptor=%+v", i, parsed.HasAttachedStructure, wantAttached, parsed)
		}
		if wantAttached {
			assertPublicRTCAttachedStructure(t, parsed.AttachedStructure, normalized)
		}
		deps := parsed.FrameDependencies
		if deps.SpatialID != frame.SpatialID || deps.TemporalID != frame.TemporalID {
			t.Fatalf("frame %d dependencies=%+v frame=%+v", i, deps, frame)
		}
		if !rtpPayloadHasLayerOBU(t, frame.Data, frame.TemporalID, frame.SpatialID) {
			t.Fatalf("frame %d missing layer-tagged OBU", i)
		}
	}
	*nextFrameID += uint64(picture.FrameNum)
}

func assertPublicRTCFrameRTPPackets(t *testing.T, receiver *RTPDependencyDescriptorState, frame RTCFrame, limits RTPPayloadSizeLimits, wantAttachedStructure bool, wantNewCodedVideoSequence bool, wantMarker bool) {
	t.Helper()
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	obuScratch := make([]RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if size.Packetizer.Packets <= 1 {
		t.Fatalf("packetizer packets=%d want fragmented frame S%d", size.Packetizer.Packets, frame.SpatialID)
	}
	packetScratch := make([]RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	payloadSlices := make([][]byte, packetCount)
	for i := range packetCount {
		span := spans[i]
		payloadSlices[i] = rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		header, _, err := ParseRTPAggregationHeader(payloadSlices[i])
		if err != nil {
			t.Fatalf("packet %d aggregation header S%d: %v", i, frame.SpatialID, err)
		}
		if header.StartsNewCodedVideoSequence != (wantNewCodedVideoSequence && i == 0) {
			t.Fatalf("packet %d N=%v want %v frame=%+v", i, header.StartsNewCodedVideoSequence, wantNewCodedVideoSequence && i == 0, frame)
		}
		desc := descriptors[span.DescriptorOffset : span.DescriptorOffset+span.DescriptorLength]
		parsed, consumed, err := receiver.Parse(desc)
		if err != nil {
			t.Fatalf("packet %d descriptor S%d: %v", i, frame.SpatialID, err)
		}
		if consumed != len(desc) ||
			parsed.Mandatory.FrameNumber != uint16(frame.FrameID) ||
			parsed.Mandatory.FirstPacketInFrame != (i == 0) ||
			parsed.Mandatory.LastPacketInFrame != (i == packetCount-1) {
			t.Fatalf("packet %d mandatory=%+v consumed=%d len=%d frame=%+v", i, parsed.Mandatory, consumed, len(desc), frame)
		}
		if parsed.HasAttachedStructure != (wantAttachedStructure && i == 0) {
			t.Fatalf("packet %d attached=%v want %v", i, parsed.HasAttachedStructure, wantAttachedStructure && i == 0)
		}
		if parsed.FrameDependencies.SpatialID != frame.SpatialID || parsed.FrameDependencies.TemporalID != frame.TemporalID {
			t.Fatalf("packet %d deps=%+v frame=%+v", i, parsed.FrameDependencies, frame)
		}
		if span.Marker != (wantMarker && i == packetCount-1) {
			t.Fatalf("packet %d marker=%v", i, span.Marker)
		}
	}
	assertPublicRTPPayloadsAssembleToFrame(t, frame.Data, payloadSlices)
}

func publicRTPTimestampHalfTicks(t *testing.T, mode EncoderScalabilityMode, step int, duration EncoderRational) uint64 {
	t.Helper()
	if duration.Num <= 0 || duration.Den <= 0 || 2%duration.Den != 0 {
		t.Fatalf("%s step %d RTPFrameDuration=%+v want positive duration representable in half-ticks", mode, step, duration)
	}
	return uint64(duration.Num) * uint64(2/duration.Den)
}

func publicAssertRTCPictureRTPHeaderTimestamp(t *testing.T, receiver *RTPDependencyDescriptorState, picture RTCPicture, limits RTPPayloadSizeLimits, sequence *uint16, timestamp uint32) {
	t.Helper()
	if picture.FrameNum <= 0 || picture.FrameNum > EncoderWebRTCMaxSpatialLayers {
		t.Fatalf("invalid picture frame count %d", picture.FrameNum)
	}
	totalPackets := 0
	for frameIndex := 0; frameIndex < picture.FrameNum; frameIndex++ {
		frame := picture.Frames[frameIndex]
		firstSize, err := frame.RTPPacketScratchLen(limits, nil)
		if err != nil {
			t.Fatalf("RTPPacketScratchLen first frame %d S%d/T%d: %v", frameIndex, frame.SpatialID, frame.TemporalID, err)
		}
		obuScratch := make([]RTPPacketizerOBU, firstSize.Packetizer.OBUs)
		size, err := frame.RTPPacketScratchLen(limits, obuScratch)
		if err != nil {
			t.Fatalf("RTPPacketScratchLen full frame %d S%d/T%d: %v", frameIndex, frame.SpatialID, frame.TemporalID, err)
		}
		packetScratch := make([]RTPPacketPlan, size.Packetizer.Packets)
		workScratch := make([]RTPPacketPlan, size.Packetizer.Work)
		payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
		descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
		packetSpans := make([]EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
		rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, packetSpans, limits, obuScratch, packetScratch, workScratch)
		if err != nil {
			t.Fatalf("AppendRTPPackets frame %d S%d/T%d: %v", frameIndex, frame.SpatialID, frame.TemporalID, err)
		}
		if packetCount == 0 || packetCount != size.Packetizer.Packets {
			t.Fatalf("frame %d packet count=%d want %d", frameIndex, packetCount, size.Packetizer.Packets)
		}

		startSequence := *sequence
		headerConfig := EncoderWebRTCRTPPacketHeaderConfig{
			PayloadType:                     96,
			SequenceNumber:                  startSequence,
			Timestamp:                       timestamp,
			SSRC:                            0x77880000 + uint32(frame.SpatialID),
			DependencyDescriptorExtensionID: 42,
		}
		packetSize, err := EncoderWebRTCRTPPacketsWithHeadersSize(headerConfig, rtpPayloads, descriptors, packetSpans[:packetCount])
		if err != nil {
			t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize frame %d S%d/T%d: %v", frameIndex, frame.SpatialID, frame.TemporalID, err)
		}
		headerSpans := make([]EncoderWebRTCRTPPacketHeaderSpan, packetCount)
		fullPackets, fullCount, err := AppendEncoderWebRTCRTPPacketsWithHeaders(make([]byte, 0, packetSize.Bytes), headerSpans, headerConfig, rtpPayloads, descriptors, packetSpans[:packetCount])
		if err != nil {
			t.Fatalf("AppendEncoderWebRTCRTPPacketsWithHeaders frame %d S%d/T%d: %v", frameIndex, frame.SpatialID, frame.TemporalID, err)
		}
		if fullCount != packetCount {
			t.Fatalf("full packet count=%d want %d", fullCount, packetCount)
		}

		payloadSlices := make([][]byte, packetCount)
		for packetIndex := 0; packetIndex < packetCount; packetIndex++ {
			packetSpan := packetSpans[packetIndex]
			headerSpan := headerSpans[packetIndex]
			raw := fullPackets[headerSpan.Offset : headerSpan.Offset+headerSpan.Length]
			descriptorPacket, err := ParseRTPPacketDependencyDescriptor(raw, headerConfig.DependencyDescriptorExtensionID, receiver)
			if err != nil {
				t.Fatalf("packet %d frame %d ParseRTPPacketDependencyDescriptor: %v", packetIndex, frameIndex, err)
			}
			packet := descriptorPacket.Packet
			wantSeq := startSequence + uint16(packetIndex)
			wantMarker := frameIndex+1 == picture.FrameNum && packetIndex+1 == packetCount
			if packet.Header.PayloadType != headerConfig.PayloadType ||
				packet.Header.SequenceNumber != wantSeq ||
				packet.Header.Timestamp != timestamp ||
				packet.Header.SSRC != headerConfig.SSRC ||
				packet.Header.Marker != wantMarker {
				t.Fatalf("packet %d frame %d header=%+v want seq=%d ts=%d ssrc=%#x marker=%v",
					packetIndex, frameIndex, packet.Header, wantSeq, timestamp, headerConfig.SSRC, wantMarker)
			}
			if headerSpan.SequenceNumber != wantSeq ||
				headerSpan.Marker != wantMarker ||
				headerSpan.PayloadLength != len(packet.Payload) ||
				headerSpan.PayloadOffset != headerSpan.Offset+headerSpan.HeaderSize {
				t.Fatalf("packet %d frame %d header span=%+v want seq=%d marker=%v payload=%d",
					packetIndex, frameIndex, headerSpan, wantSeq, wantMarker, len(packet.Payload))
			}
			if packetSpan.Marker != wantMarker {
				t.Fatalf("packet %d frame %d packet marker=%v want %v", packetIndex, frameIndex, packetSpan.Marker, wantMarker)
			}
			if descriptorPacket.Descriptor.FrameDependencies.SpatialID != frame.SpatialID ||
				descriptorPacket.Descriptor.FrameDependencies.TemporalID != frame.TemporalID {
				t.Fatalf("packet %d frame %d dependencies=%+v frame S%d/T%d",
					packetIndex, frameIndex, descriptorPacket.Descriptor.FrameDependencies, frame.SpatialID, frame.TemporalID)
			}
			payloadSlices[packetIndex] = packet.Payload
			totalPackets++
		}
		assertPublicRTPPayloadsAssembleToFrame(t, frame.Data, payloadSlices)
		*sequence += uint16(packetCount)
	}
	if totalPackets == 0 {
		t.Fatal("picture produced no RTP packets")
	}
}

func assertPublicRTCPictureActiveDecodeTargetOptions(t *testing.T, receiver *RTPDependencyDescriptorState, cfg EncoderConfig, picture RTCPicture, limits RTPPayloadSizeLimits) {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := cfg.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid mode %s", cfg.Scalability)
	}
	wantAll := publicExpectedActiveDecodeTargetsMask(spatialLayers, temporalLayers, spatialLayers-1, temporalLayers-1)
	all, err := picture.AllDecodeTargetsMask()
	if err != nil {
		t.Fatalf("%s picture AllDecodeTargetsMask: %v", cfg.Scalability, err)
	}
	if all != wantAll {
		t.Fatalf("%s all decode-target mask=%#x want %#x", cfg.Scalability, all, wantAll)
	}
	for maxSpatialID := uint8(0); maxSpatialID < spatialLayers; maxSpatialID++ {
		for maxTemporalID := uint8(0); maxTemporalID < temporalLayers; maxTemporalID++ {
			wantMask := publicExpectedActiveDecodeTargetsMask(spatialLayers, temporalLayers, maxSpatialID, maxTemporalID)
			mask, err := picture.ActiveDecodeTargetsMask(maxSpatialID, maxTemporalID)
			if err != nil {
				t.Fatalf("%s ActiveDecodeTargetsMask(S%d,T%d): %v", cfg.Scalability, maxSpatialID, maxTemporalID, err)
			}
			if mask != wantMask {
				t.Fatalf("%s active mask S%d/T%d=%#x want %#x", cfg.Scalability, maxSpatialID, maxTemporalID, mask, wantMask)
			}
			options, err := picture.ActiveDecodeTargetsRTPOptions(maxSpatialID, maxTemporalID)
			if err != nil {
				t.Fatalf("%s ActiveDecodeTargetsRTPOptions(S%d,T%d): %v", cfg.Scalability, maxSpatialID, maxTemporalID, err)
			}
			if !options.ActiveDecodeTargetsPresentOnFirstPacket || options.ActiveDecodeTargetsMask != wantMask {
				t.Fatalf("%s options S%d/T%d=%+v want mask %#x", cfg.Scalability, maxSpatialID, maxTemporalID, options, wantMask)
			}
			for i := 0; i < picture.FrameNum; i++ {
				assertPublicRTCFrameRTPPacketsWithActiveDecodeTargets(t, receiver, picture.Frames[i], limits, options)
			}
		}
	}
	if _, err := picture.ActiveDecodeTargetsMask(spatialLayers, 0); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("%s invalid spatial active mask err=%v want %v", cfg.Scalability, err, ErrEncoderInvalidFrame)
	}
	if _, err := picture.ActiveDecodeTargetsMask(0, temporalLayers); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("%s invalid temporal active mask err=%v want %v", cfg.Scalability, err, ErrEncoderInvalidFrame)
	}
}

func assertPublicRTCFrameRTPPacketsWithActiveDecodeTargets(t *testing.T, receiver *RTPDependencyDescriptorState, frame RTCFrame, limits RTPPayloadSizeLimits, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) {
	t.Helper()
	firstSize, err := frame.RTPPacketScratchLenWithOptions(limits, nil, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions first S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	obuScratch := make([]RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLenWithOptions(limits, obuScratch, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions full S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	packetScratch := make([]RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPacketsWithOptions(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch, options)
	if err != nil {
		t.Fatalf("AppendRTPPacketsWithOptions S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	payloadSlices := make([][]byte, packetCount)
	for i := range packetCount {
		span := spans[i]
		payloadSlices[i] = rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		desc := descriptors[span.DescriptorOffset : span.DescriptorOffset+span.DescriptorLength]
		parsed, consumed, err := receiver.Parse(desc)
		if err != nil {
			t.Fatalf("packet %d active descriptor S%d T%d: %v", i, frame.SpatialID, frame.TemporalID, err)
		}
		if consumed != len(desc) ||
			parsed.Mandatory.FrameNumber != uint16(frame.FrameID) ||
			parsed.Mandatory.FirstPacketInFrame != (i == 0) ||
			parsed.Mandatory.LastPacketInFrame != (i == packetCount-1) {
			t.Fatalf("packet %d mandatory=%+v consumed=%d len=%d frame=%+v", i, parsed.Mandatory, consumed, len(desc), frame)
		}
		if parsed.FrameDependencies.SpatialID != frame.SpatialID || parsed.FrameDependencies.TemporalID != frame.TemporalID {
			t.Fatalf("packet %d dependencies=%+v frame=%+v", i, parsed.FrameDependencies, frame)
		}
		if i == 0 {
			if !parsed.HasActiveDecodeTargets || parsed.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask {
				t.Fatalf("packet %d active descriptor=%+v options=%+v", i, parsed, options)
			}
			continue
		}
		if parsed.HasActiveDecodeTargets {
			t.Fatalf("packet %d repeated active decode targets: %+v", i, parsed)
		}
	}
	assertPublicRTPPayloadsAssembleToFrame(t, frame.Data, payloadSlices)
}

func publicExpectedActiveDecodeTargetsMask(spatialLayers uint8, temporalLayers uint8, maxSpatialID uint8, maxTemporalID uint8) uint32 {
	var mask uint32
	for spatialID := uint8(0); spatialID < spatialLayers; spatialID++ {
		for temporalID := uint8(0); temporalID < temporalLayers; temporalID++ {
			if spatialID > maxSpatialID || temporalID > maxTemporalID {
				continue
			}
			target := spatialID*temporalLayers + temporalID
			mask |= uint32(1) << target
		}
	}
	return mask
}

func assertPublicRTPPayloadsAssembleToFrame(t *testing.T, frameData []byte, payloadSlices [][]byte) {
	t.Helper()
	assembledLen, obuCount, err := AssembleRTPFrameSize(payloadSlices)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	assembled := make([]byte, assembledLen)
	assembledOBUs := make([]RTPFrameOBU, obuCount)
	wrote, gotOBUs, err := AssembleRTPFrame(assembled, payloadSlices, assembledOBUs)
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	var expected []byte
	it := obu.NewLowOverheadIterator(frameData)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("source OBU iteration: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeTemporalDelimiter, obu.TypeTileList, obu.TypePadding:
			continue
		default:
			expected = append(expected, unit.Raw...)
		}
	}
	if wrote != len(expected) || gotOBUs != obuCount || string(assembled[:wrote]) != string(expected) {
		t.Fatalf("assembled len=%d obus=%d match=%v want len=%d obus=%d", wrote, gotOBUs, string(assembled[:wrote]) == string(expected), len(expected), obuCount)
	}
}

func assertPublicRTCFrameScalabilityMetadata(t *testing.T, frameData []byte, wantIDC uint8, wantPresent bool) {
	t.Helper()
	it := obu.NewLowOverheadIterator(frameData)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := it.Next(); err != nil || !ok || seq.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	next, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("next OBU ok=%v err=%v header=%+v", ok, err, next.Header)
	}
	if !wantPresent {
		if next.Header.Type == obu.TypeMetadata {
			t.Fatalf("unexpected metadata payload=% x", next.Payload)
		}
		if next.Header.Type != obu.TypeFrameHeader && next.Header.Type != obu.TypeFrame {
			t.Fatalf("next OBU after sequence=%+v", next.Header)
		}
		return
	}
	if next.Header.Type != obu.TypeMetadata {
		t.Fatalf("metadata header=%+v", next.Header)
	}
	meta, err := obu.ParseMetadata(next.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != obu.MetadataTypeScalability || meta.Scalability.ModeIDC != wantIDC || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v want idc=%d", meta, wantIDC)
	}
	frame, ok, err := it.Next()
	if err != nil || !ok || (frame.Header.Type != obu.TypeFrameHeader && frame.Header.Type != obu.TypeFrame) {
		t.Fatalf("frame after metadata ok=%v err=%v header=%+v", ok, err, frame.Header)
	}
}

func assertPublicRTCFrameScalabilitySSMetadata(t *testing.T, frameData []byte, widths []uint16, heights []uint16, refIDs []uint8, groupTIDs []uint8, groupDiffs []uint8) {
	t.Helper()
	it := obu.NewLowOverheadIterator(frameData)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := it.Next(); err != nil || !ok || seq.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	next, ok, err := it.Next()
	if err != nil || !ok || next.Header.Type != obu.TypeMetadata {
		t.Fatalf("metadata ok=%v err=%v header=%+v", ok, err, next.Header)
	}
	meta, err := obu.ParseMetadata(next.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != obu.MetadataTypeScalability ||
		meta.Scalability.ModeIDC != obu.ScalabilityModeSS ||
		!meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v", meta)
	}
	structure := meta.Scalability.Structure
	if !structure.SpatialLayerDimensionsPresent ||
		!structure.SpatialLayerDescriptionPresent ||
		!structure.TemporalGroupDescriptionPresent ||
		structure.SpatialLayersCountMinus1 != uint8(len(widths)-1) {
		t.Fatalf("structure flags=%+v", structure)
	}
	for i := range widths {
		if structure.SpatialLayerMaxWidth[i] != widths[i] || structure.SpatialLayerMaxHeight[i] != heights[i] {
			t.Fatalf("dimensions[%d]=%dx%d want %dx%d structure=%+v", i,
				structure.SpatialLayerMaxWidth[i], structure.SpatialLayerMaxHeight[i],
				widths[i], heights[i], structure)
		}
		if structure.SpatialLayerRefID[i] != refIDs[i] {
			t.Fatalf("refID[%d]=%d want %d structure=%+v", i, structure.SpatialLayerRefID[i], refIDs[i], structure)
		}
	}
	if structure.TemporalGroupSize != uint8(len(groupTIDs)) || len(groupTIDs) != len(groupDiffs) {
		t.Fatalf("temporal group size=%d tids=%d diffs=%d", structure.TemporalGroupSize, len(groupTIDs), len(groupDiffs))
	}
	for i, tid := range groupTIDs {
		entry := structure.TemporalGroup[i]
		if entry.TemporalID != tid || entry.RefCount != 1 || entry.RefPicDiff[0] != groupDiffs[i] {
			t.Fatalf("group[%d]=%+v want tid=%d diff=%d", i, entry, tid, groupDiffs[i])
		}
		wantTemporalSwitchingUp := true
		if len(groupTIDs) == 4 && i == 2 {
			wantTemporalSwitchingUp = false
		}
		if entry.TemporalSwitchingUp != wantTemporalSwitchingUp || entry.SpatialSwitchingUp {
			t.Fatalf("group[%d] switching=%+v want temporal=%v spatial=false", i, entry, wantTemporalSwitchingUp)
		}
	}
	frame, ok, err := it.Next()
	if err != nil || !ok || (frame.Header.Type != obu.TypeFrameHeader && frame.Header.Type != obu.TypeFrame) {
		t.Fatalf("frame after metadata ok=%v err=%v header=%+v", ok, err, frame.Header)
	}
}

func publicDecodedFramesRawYUV(frames []DecodedFrame) []byte {
	var out []byte
	for _, frame := range frames {
		out = append(out, frame.Y...)
		out = append(out, frame.U...)
		out = append(out, frame.V...)
	}
	return out
}

func appendPublicI400HighBitDepthRaw(dst []byte, frame I400HighBitDepthFrame) []byte {
	for y := range frame.Height {
		row := frame.Y[y*frame.YStride : y*frame.YStride+frame.Width]
		for _, sample := range row {
			dst = append(dst, byte(sample), byte(sample>>8))
		}
	}
	return dst
}

func appendPublicI420HighBitDepthRaw(dst []byte, frame I420HighBitDepthFrame) []byte {
	dst = appendPublicI420HighBitDepthPlaneRaw(dst, frame.Y, frame.YStride, frame.Width, frame.Height)
	chromaWidth, chromaHeight := frame.Width/2, frame.Height/2
	dst = appendPublicI420HighBitDepthPlaneRaw(dst, frame.U, frame.ChromaStride, chromaWidth, chromaHeight)
	return appendPublicI420HighBitDepthPlaneRaw(dst, frame.V, frame.ChromaStride, chromaWidth, chromaHeight)
}

func appendPublicI420HighBitDepthPlaneRaw(dst []byte, samples []uint16, stride, width, height int) []byte {
	for y := range height {
		row := samples[y*stride : y*stride+width]
		for _, sample := range row {
			dst = append(dst, byte(sample), byte(sample>>8))
		}
	}
	return dst
}

func publicI420HighBitDepthFrame(width, height int, bitDepth uint8, seed int) I420HighBitDepthFrame {
	chromaWidth, chromaHeight := width/2, height/2
	frame := I420HighBitDepthFrame{
		Y:            make([]uint16, width*height),
		U:            make([]uint16, chromaWidth*chromaHeight),
		V:            make([]uint16, chromaWidth*chromaHeight),
		YStride:      width,
		ChromaStride: chromaWidth,
		Width:        width,
		Height:       height,
		BitDepth:     bitDepth,
	}
	maxSample := (1 << bitDepth) - 1
	for y := range height {
		for x := range width {
			frame.Y[y*frame.YStride+x] = uint16((seed*17 + x*29 + y*23 + (x*y)%257) & maxSample)
		}
	}
	for y := range chromaHeight {
		for x := range chromaWidth {
			off := y*frame.ChromaStride + x
			frame.U[off] = uint16((seed*31 + x*19 + y*43 + (x*y)%113) & maxSample)
			frame.V[off] = uint16((seed*47 + x*41 + y*13 + (x*y)%167) & maxSample)
		}
	}
	return frame
}

func publicI422HighBitDepthFrame(width, height int, bitDepth uint8, seed int) I422HighBitDepthFrame {
	chromaWidth := width / 2
	frame := I422HighBitDepthFrame{
		Y:            make([]uint16, width*height),
		U:            make([]uint16, chromaWidth*height),
		V:            make([]uint16, chromaWidth*height),
		YStride:      width,
		ChromaStride: chromaWidth,
		Width:        width,
		Height:       height,
		BitDepth:     bitDepth,
	}
	maxSample := (1 << bitDepth) - 1
	for y := range height {
		for x := range width {
			frame.Y[y*frame.YStride+x] = uint16((seed*17 + x*29 + y*23 + (x*y)%257) & maxSample)
		}
	}
	for y := range height {
		for x := range chromaWidth {
			off := y*frame.ChromaStride + x
			frame.U[off] = uint16((seed*31 + x*19 + y*43 + (x*y)%113) & maxSample)
			frame.V[off] = uint16((seed*47 + x*41 + y*13 + (x*y)%167) & maxSample)
		}
	}
	return frame
}

func publicI444HighBitDepthFrame(width, height int, bitDepth uint8, seed int) I444HighBitDepthFrame {
	frame := I444HighBitDepthFrame{
		Y:        make([]uint16, width*height),
		U:        make([]uint16, width*height),
		V:        make([]uint16, width*height),
		YStride:  width,
		UStride:  width,
		VStride:  width,
		Width:    width,
		Height:   height,
		BitDepth: bitDepth,
	}
	maxSample := (1 << bitDepth) - 1
	for y := range height {
		for x := range width {
			off := y*frame.YStride + x
			frame.Y[off] = uint16((seed*17 + x*29 + y*23 + (x*y)%257) & maxSample)
			frame.U[y*frame.UStride+x] = uint16((seed*31 + x*19 + y*43 + (x*y)%113) & maxSample)
			frame.V[y*frame.VStride+x] = uint16((seed*47 + x*41 + y*13 + (x*y)%167) & maxSample)
		}
	}
	return frame
}

func publicClampUint16(v int, max uint16) uint16 {
	if v < 0 {
		return 0
	}
	if v > int(max) {
		return max
	}
	return uint16(v)
}

func publicUint16Equal(a []uint16, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func publicI420HighBitDepthFrameEqual(a I420HighBitDepthFrame, b I420HighBitDepthFrame) bool {
	if a.Width != b.Width || a.Height != b.Height || a.BitDepth != b.BitDepth {
		return false
	}
	for y := range a.Height {
		ar := a.Y[y*a.YStride : y*a.YStride+a.Width]
		br := b.Y[y*b.YStride : y*b.YStride+b.Width]
		for x := range ar {
			if ar[x] != br[x] {
				return false
			}
		}
	}
	cw, ch := a.Width/2, a.Height/2
	for y := range ch {
		au := a.U[y*a.ChromaStride : y*a.ChromaStride+cw]
		bu := b.U[y*b.ChromaStride : y*b.ChromaStride+cw]
		av := a.V[y*a.ChromaStride : y*a.ChromaStride+cw]
		bv := b.V[y*b.ChromaStride : y*b.ChromaStride+cw]
		for x := range cw {
			if au[x] != bu[x] || av[x] != bv[x] {
				return false
			}
		}
	}
	return true
}

func publicI400HighBitDepthFrameEqual(a I400HighBitDepthFrame, b I400HighBitDepthFrame) bool {
	if a.Width != b.Width || a.Height != b.Height || a.BitDepth != b.BitDepth {
		return false
	}
	for y := range a.Height {
		ar := a.Y[y*a.YStride : y*a.YStride+a.Width]
		br := b.Y[y*b.YStride : y*b.YStride+b.Width]
		for x := range ar {
			if ar[x] != br[x] {
				return false
			}
		}
	}
	return true
}

func appendPublicFrameRawYUV(dst []byte, frame *Frame) []byte {
	if frame == nil {
		return dst
	}
	bytesPerSample := frame.Layout.BytesPerSample
	dst = appendPublicFramePlaneRawYUV(dst, frame.Y, bytesPerSample)
	dst = appendPublicFramePlaneRawYUV(dst, frame.U, bytesPerSample)
	dst = appendPublicFramePlaneRawYUV(dst, frame.V, bytesPerSample)
	return dst
}

func appendPublicFramePlaneRawYUV(dst []byte, plane FramePlane, bytesPerSample int) []byte {
	if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
		return dst
	}
	rowBytes := plane.Width * bytesPerSample
	for row := 0; row < plane.Height; row++ {
		off := row * plane.Stride
		dst = append(dst, plane.Pix[off:off+rowBytes]...)
	}
	return dst
}

func assertPublicRTCAttachedStructure(t *testing.T, structure RTPDependencyDescriptorStructure, cfg EncoderConfig) {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := cfg.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid attached-structure mode %s", cfg.Scalability)
	}
	if structure.NumDecodeTargets != spatialLayers*temporalLayers ||
		structure.NumChains != spatialLayers ||
		structure.ResolutionNum != spatialLayers {
		t.Fatalf("attached structure shape=%+v want spatial=%d temporal=%d", structure, spatialLayers, temporalLayers)
	}
	for i := uint8(0); i < spatialLayers; i++ {
		want := cfg.SpatialLayers[i].Resolution
		got := structure.Resolutions[i]
		if int32(got.Width) != want.Width || int32(got.Height) != want.Height {
			t.Fatalf("attached resolution[%d]=%+v want %+v", i, got, want)
		}
	}
}

func appendPublicRTCPictureLayerData(t *testing.T, layerTUs *[EncoderWebRTCMaxSpatialLayers][][]byte, orderedTUs *[][]byte, picture RTCPicture) {
	t.Helper()
	for i := 0; i < picture.FrameNum; i++ {
		spatialID := picture.Frames[i].SpatialID
		if spatialID >= EncoderWebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", i, spatialID)
		}
		tu := append([]byte(nil), picture.Frames[i].Data...)
		layerTUs[spatialID] = append(layerTUs[spatialID], tu)
		*orderedTUs = append(*orderedTUs, tu)
	}
}

func appendPublicRTCPictureRTPData(t *testing.T, receiver *RTPDependencyDescriptorState, layerTUs *[EncoderWebRTCMaxSpatialLayers][][]byte, orderedTUs *[][]byte, picture RTCPicture) {
	t.Helper()
	for i := 0; i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		spatialID := frame.SpatialID
		if spatialID >= EncoderWebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", i, spatialID)
		}
		tu := publicRTCPacketizeAndAssembleFrame(t, receiver, frame)
		layerTUs[spatialID] = append(layerTUs[spatialID], tu)
		*orderedTUs = append(*orderedTUs, tu)
	}
}

func publicRTCPacketizeAndAssembleFrame(t *testing.T, receiver *RTPDependencyDescriptorState, frame RTCFrame) []byte {
	t.Helper()
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 96}
	const dependencyDescriptorExtensionID = 42
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	obuScratch := make([]RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	packetScratch := make([]RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	headerConfig := EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     96,
		SequenceNumber:                  uint16(frame.FrameID * 29),
		Timestamp:                       uint32(frame.FrameID * 3000),
		SSRC:                            0x01020304 + uint32(frame.SpatialID),
		DependencyDescriptorExtensionID: dependencyDescriptorExtensionID,
		MIDExtensionID:                  1,
		MID:                             "video",
		RTPStreamIDExtensionID:          2,
		RTPStreamID:                     fmt.Sprintf("s%d", frame.SpatialID),
		TransportWideCCExtensionID:      3,
		TransportWideCCSequenceNumber:   uint16(frame.FrameID * 31),
		AbsoluteSendTimeExtensionID:     4,
		AbsoluteSendTime:                uint32(frame.FrameID*37) & RTPAbsoluteSendTimeMaxValue,
		VideoContentTypeExtensionID:     5,
		VideoContentType:                RTPVideoContentTypeScreenshare,
	}
	packetSize, err := EncoderWebRTCRTPPacketsWithHeadersSize(headerConfig, rtpPayloads, descriptors, spans[:packetCount])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if packetSize.Packets != packetCount || packetSize.Bytes <= len(rtpPayloads) ||
		packetSize.MaxHeaderBytes <= RTPHeaderMinSize {
		t.Fatalf("full RTP size=%+v packetCount=%d payloadBytes=%d", packetSize, packetCount, len(rtpPayloads))
	}
	headerSpans := make([]EncoderWebRTCRTPPacketHeaderSpan, packetCount)
	fullPackets, fullCount, err := AppendEncoderWebRTCRTPPacketsWithHeaders(make([]byte, 0, packetSize.Bytes), headerSpans, headerConfig, rtpPayloads, descriptors, spans[:packetCount])
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCRTPPacketsWithHeaders S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if fullCount != packetCount || len(fullPackets) != packetSize.Bytes {
		t.Fatalf("full packet count=%d/%d bytes=%d/%d", fullCount, packetCount, len(fullPackets), packetSize.Bytes)
	}
	payloadSlices := make([][]byte, packetCount)
	for i := range packetCount {
		span := spans[i]
		headerSpan := headerSpans[i]
		raw := fullPackets[headerSpan.Offset : headerSpan.Offset+headerSpan.Length]
		descriptorPacket, err := ParseRTPPacketDependencyDescriptor(raw, dependencyDescriptorExtensionID, receiver)
		if err != nil {
			t.Fatalf("packet %d ParseRTPPacketDependencyDescriptor S%d: %v", i, frame.SpatialID, err)
		}
		packet := descriptorPacket.Packet
		if packet.Header.PayloadType != headerConfig.PayloadType ||
			packet.Header.SequenceNumber != headerConfig.SequenceNumber+uint16(i) ||
			packet.Header.Timestamp != headerConfig.Timestamp ||
			packet.Header.SSRC != headerConfig.SSRC ||
			packet.Header.Marker != span.Marker ||
			packet.Header.ExtensionProfile != RTPExtensionProfileTwoByte {
			t.Fatalf("packet %d RTP header=%+v span=%+v config=%+v", i, packet.Header, span, headerConfig)
		}
		if headerSpan.HeaderSize != len(raw)-len(packet.Payload) ||
			headerSpan.PayloadOffset != headerSpan.Offset+headerSpan.HeaderSize ||
			headerSpan.PayloadLength != len(packet.Payload) ||
			headerSpan.SequenceNumber != packet.Header.SequenceNumber ||
			headerSpan.Marker != packet.Header.Marker {
			t.Fatalf("packet %d header span=%+v payload=%d", i, headerSpan, len(packet.Payload))
		}
		payloadSlices[i] = packet.Payload
		wantPayload := rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		if !bytes.Equal(payloadSlices[i], wantPayload) {
			t.Fatalf("packet %d payload mismatch len=%d want=%d", i, len(payloadSlices[i]), len(wantPayload))
		}
		midElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, headerConfig.MIDExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d MID extension ok=%v err=%v", i, ok, err)
		}
		mid, err := ParseRTPMIDHeaderExtension(midElement.Payload)
		if err != nil || mid != headerConfig.MID {
			t.Fatalf("packet %d MID=%q err=%v want %q", i, mid, err, headerConfig.MID)
		}
		ridElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, headerConfig.RTPStreamIDExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d RID extension ok=%v err=%v", i, ok, err)
		}
		rid, err := ParseRTPStreamIDHeaderExtension(ridElement.Payload)
		if err != nil || rid != headerConfig.RTPStreamID {
			t.Fatalf("packet %d RID=%q err=%v want %q", i, rid, err, headerConfig.RTPStreamID)
		}
		twccElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, headerConfig.TransportWideCCExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d TWCC extension ok=%v err=%v", i, ok, err)
		}
		twcc, err := ParseRTPTransportWideCCHeaderExtension(twccElement.Payload)
		if want := headerConfig.TransportWideCCSequenceNumber + uint16(i); err != nil || twcc != want {
			t.Fatalf("packet %d TWCC=%d err=%v want %d", i, twcc, err, want)
		}
		absSendElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, headerConfig.AbsoluteSendTimeExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d abs-send-time extension ok=%v err=%v", i, ok, err)
		}
		absSend, err := ParseRTPAbsoluteSendTimeHeaderExtension(absSendElement.Payload)
		if err != nil || absSend != headerConfig.AbsoluteSendTime {
			t.Fatalf("packet %d abs-send-time=%#x err=%v want %#x", i, absSend, err, headerConfig.AbsoluteSendTime)
		}
		contentElement, ok, err := FindRTPHeaderExtensionElement(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, headerConfig.VideoContentTypeExtensionID)
		if err != nil || !ok {
			t.Fatalf("packet %d content-type extension ok=%v err=%v", i, ok, err)
		}
		content, err := ParseRTPVideoContentTypeHeaderExtension(contentElement.Payload)
		if err != nil || content != headerConfig.VideoContentType {
			t.Fatalf("packet %d content-type=%d err=%v want %d", i, content, err, headerConfig.VideoContentType)
		}
		header, _, err := ParseRTPAggregationHeader(payloadSlices[i])
		if err != nil {
			t.Fatalf("packet %d aggregation header S%d: %v", i, frame.SpatialID, err)
		}
		if header.StartsNewCodedVideoSequence != (frame.CodedKeyframe && i == 0) {
			t.Fatalf("packet %d N=%v want %v frame=%+v", i, header.StartsNewCodedVideoSequence, frame.CodedKeyframe && i == 0, frame)
		}
		desc := descriptorPacket.DescriptorPayload
		wantDesc := descriptors[span.DescriptorOffset : span.DescriptorOffset+span.DescriptorLength]
		if !bytes.Equal(desc, wantDesc) ||
			!bytes.Equal(desc, fullPackets[headerSpan.DependencyDescriptorOffset:headerSpan.DependencyDescriptorOffset+headerSpan.DependencyDescriptorLength]) {
			t.Fatalf("packet %d dependency descriptor span mismatch", i)
		}
		parsed := descriptorPacket.Descriptor
		if parsed.Mandatory.FrameNumber != uint16(frame.FrameID) ||
			parsed.Mandatory.FirstPacketInFrame != (i == 0) ||
			parsed.Mandatory.LastPacketInFrame != (i == packetCount-1) {
			t.Fatalf("packet %d mandatory=%+v len=%d frame=%+v", i, parsed.Mandatory, len(desc), frame)
		}
		if span.Marker != (frame.LastFrameInPicture && i == packetCount-1) {
			t.Fatalf("packet %d marker=%v frame=%+v", i, span.Marker, frame)
		}
	}
	assembledLen, obuCount, err := AssembleRTPFrameSize(payloadSlices)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	assembled := make([]byte, assembledLen)
	assembledOBUs := make([]RTPFrameOBU, obuCount)
	wrote, gotOBUs, err := AssembleRTPFrame(assembled, payloadSlices, assembledOBUs)
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	if wrote != assembledLen || gotOBUs != obuCount {
		t.Fatalf("assembled wrote=%d/%d obus=%d/%d", wrote, assembledLen, gotOBUs, obuCount)
	}
	return assembled[:wrote]
}

func assertPublicRTCLayerStreamsDecode(t *testing.T, cfg EncoderConfig, layerTUs [EncoderWebRTCMaxSpatialLayers][][]byte, orderedTUs [][]byte) {
	t.Helper()
	if publicRTCSharedReferenceSlotMode(cfg.Scalability) {
		assertPublicRTCSharedReferenceStreamDecodes(t, orderedTUs)
		return
	}
	spatialLayers := int(cfg.SpatialLayerCount)
	for spatialID := 0; spatialID < spatialLayers; spatialID++ {
		assertPublicRTCSpatialLayerDecodes(t, spatialID, layerTUs[spatialID])
	}
}

func assertPublicRTCSpatialLayerDecodes(t *testing.T, spatialID int, tus [][]byte) {
	t.Helper()
	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatalf("spatial layer %d decoder: %v", spatialID, err)
	}
	defer dec.Close()
	n := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("spatial layer %d decode: %v", spatialID, err)
		}
		if !ok {
			break
		}
		n += len(batch)
	}
	if n != len(tus) {
		t.Fatalf("spatial layer %d decoded %d frames, want %d", spatialID, n, len(tus))
	}
}

func publicRTCSharedReferenceSlotMode(mode EncoderScalabilityMode) bool {
	return mode.UsesSharedReferenceSlots()
}

type publicLayeredFrameMetadata struct {
	TemporalID    uint8
	SpatialID     uint8
	FrameType     FrameType
	CodedKeyframe bool
	ShowExisting  bool
	ShowFrame     bool
	CodedWidth    int
	Height        int
}

func publicLayeredFrameMetadataFromRTCFrame(frame RTCFrame) publicLayeredFrameMetadata {
	frameType := internalparser.FrameTypeInter
	if frame.CodedKeyframe {
		frameType = internalparser.FrameTypeKey
	}
	return publicLayeredFrameMetadata{
		TemporalID:    frame.TemporalID,
		SpatialID:     frame.SpatialID,
		FrameType:     frameType,
		CodedKeyframe: frame.CodedKeyframe,
		ShowFrame:     true,
	}
}

func publicLayeredFrameMetadataFromDecoded(t *testing.T, frame LayeredFrame) publicLayeredFrameMetadata {
	t.Helper()
	if frame.Frame == nil {
		t.Fatal("layered metadata has nil frame")
	}
	if frame.FrameSize.CodedWidth == 0 || frame.FrameSize.Height == 0 {
		t.Fatalf("layered metadata has empty size: %+v", frame.FrameSize)
	}
	if int(frame.FrameSize.CodedWidth) != frame.Frame.Format.Width ||
		int(frame.FrameSize.Height) != frame.Frame.Format.Height {
		t.Fatalf("layered metadata size=%dx%d frame format=%dx%d",
			frame.FrameSize.CodedWidth, frame.FrameSize.Height, frame.Frame.Format.Width, frame.Frame.Format.Height)
	}
	return publicLayeredFrameMetadata{
		TemporalID:    frame.TemporalID,
		SpatialID:     frame.SpatialID,
		FrameType:     frame.FrameType,
		CodedKeyframe: frame.CodedKeyframe,
		ShowExisting:  frame.ShowExistingFrame,
		ShowFrame:     frame.ShowFrame,
		CodedWidth:    int(frame.FrameSize.CodedWidth),
		Height:        int(frame.FrameSize.Height),
	}
}

func assertPublicLayeredFrameMetadata(t *testing.T, got []publicLayeredFrameMetadata, want []publicLayeredFrameMetadata) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("layered metadata len got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i].TemporalID != want[i].TemporalID ||
			got[i].SpatialID != want[i].SpatialID ||
			got[i].FrameType != want[i].FrameType ||
			got[i].CodedKeyframe != want[i].CodedKeyframe ||
			got[i].ShowExisting != want[i].ShowExisting ||
			got[i].ShowFrame != want[i].ShowFrame {
			t.Fatalf("layered metadata[%d]=T%d/S%d type=%v codedKey=%v showExisting=%v showFrame=%v; want T%d/S%d type=%v codedKey=%v showExisting=%v showFrame=%v",
				i,
				got[i].TemporalID, got[i].SpatialID, got[i].FrameType, got[i].CodedKeyframe, got[i].ShowExisting, got[i].ShowFrame,
				want[i].TemporalID, want[i].SpatialID, want[i].FrameType, want[i].CodedKeyframe, want[i].ShowExisting, want[i].ShowFrame)
		}
		if got[i].CodedWidth == 0 || got[i].Height == 0 {
			t.Fatalf("layered metadata[%d] empty geometry: %+v", i, got[i])
		}
	}
}

func publicRTCPictureFramesInOrder(pictures ...RTCPicture) [][]byte {
	var out [][]byte
	for _, picture := range pictures {
		for i := 0; i < picture.FrameNum; i++ {
			out = append(out, append([]byte(nil), picture.Frames[i].Data...))
		}
	}
	return out
}

func publicRTCPictureRTPPayloads(t *testing.T, picture RTCPicture, limits RTPPayloadSizeLimits) [][]byte {
	t.Helper()
	var out [][]byte
	for i := 0; i < picture.FrameNum; i++ {
		out = append(out, publicDecoderRTPPayloadsForFrameWithLimits(t, picture.Frames[i], limits)...)
	}
	return out
}

func publicRTCPictureRTPPackets(t *testing.T, picture RTCPicture, limits RTPPayloadSizeLimits) [][]byte {
	t.Helper()
	var out [][]byte
	for i := 0; i < picture.FrameNum; i++ {
		out = append(out, publicDecoderRTPPacketsForFrameWithLimits(t, picture.Frames[i], limits)...)
	}
	return out
}

func publicRTCPictureDecodeTargetRTPPackets(t *testing.T, picture RTCPicture, limits RTPPayloadSizeLimits, spatialID uint8, temporalID uint8) [][]byte {
	t.Helper()
	var out [][]byte
	for i := 0; i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		if frame.SpatialID != spatialID || frame.TemporalID != temporalID {
			continue
		}
		out = append(out, publicDecoderRTPPacketsForFrameWithLimits(t, frame, limits)...)
	}
	return out
}

func publicRTCPictureDecodeTargetLowOverheads(picture RTCPicture, spatialID uint8, temporalID uint8) [][]byte {
	var out [][]byte
	for i := 0; i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		if frame.SpatialID != spatialID || frame.TemporalID != temporalID {
			continue
		}
		out = append(out, append([]byte(nil), frame.Data...))
	}
	return out
}

func publicRTPPacketSequence(t testing.TB, packet []byte) uint16 {
	t.Helper()
	parsed, err := ParseRTPPacket(packet)
	if err != nil {
		t.Fatalf("ParseRTPPacket sequence: %v", err)
	}
	return parsed.Header.SequenceNumber
}

func assertPublicRTCSharedReferenceStreamDecodes(t *testing.T, tus [][]byte) {
	t.Helper()
	frames := 0
	decodePublicLayeredLowOverheadFrames(t, tus, func(int, *Frame) { frames++ })
	if frames != len(tus) {
		t.Fatalf("shared-reference stream decoded %d frames, want %d", frames, len(tus))
	}
}

// decodePublicLayeredLowOverheadFrames decodes ordered low-overhead temporal
// units through LayeredDecoder and hands every visible frame to visit before
// the next decode call can recycle its surface.
func decodePublicLayeredLowOverheadFrames(t *testing.T, payloads [][]byte, visit func(payloadIndex int, frame *Frame)) {
	t.Helper()
	dec, err := NewLayeredDecoder(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoder: %v", err)
	}
	defer dec.Close()
	for payloadIndex := 0; ; payloadIndex++ {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("payload %d LayeredDecoder DecodeNext: %v", payloadIndex, err)
		}
		if !ok {
			return
		}
		for _, frame := range frames {
			if frame == nil {
				t.Fatalf("payload %d decoded nil frame", payloadIndex)
			}
			visit(payloadIndex, frame)
		}
	}
}

func decodePublicLayeredLowOverheadRawYUV(t *testing.T, payloads ...[]byte) ([]byte, int) {
	t.Helper()
	var out []byte
	frames := 0
	decodePublicLayeredLowOverheadFrames(t, payloads, func(_ int, frame *Frame) {
		out = appendPublicFrameRawYUV(out, frame)
		frames++
	})
	return out, frames
}

func decodePublicLayeredLowOverheadMonochromeRawY(t *testing.T, payloads ...[]byte) ([]byte, []publicMonochromeFrameSpan) {
	t.Helper()
	var out []byte
	var spans []publicMonochromeFrameSpan
	decodePublicLayeredLowOverheadFrames(t, payloads, func(payloadIndex int, frame *Frame) {
		if !frame.Format.MonoChrome || len(frame.U.Pix) != 0 || len(frame.V.Pix) != 0 {
			t.Fatalf("payload %d decoded frame format=%+v U=%d V=%d want monochrome", payloadIndex, frame.Format, len(frame.U.Pix), len(frame.V.Pix))
		}
		before := len(out)
		out = appendPublicFramePlaneRawYUV(out, frame.Y, frame.Layout.BytesPerSample)
		yLen := len(out) - before
		cw, ch := frame.Y.Width/2, frame.Y.Height/2
		spans = append(spans, publicMonochromeFrameSpan{
			yLen:        yLen,
			expandedLen: yLen + 2*cw*ch*frame.Layout.BytesPerSample,
		})
	})
	return out, spans
}

func decodePublicLayeredRTPPayloadDigestsWithLabels(t *testing.T, labels []string, payloads ...[]byte) [][16]byte {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPayloads(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPayloads: %v", err)
	}
	defer dec.Close()
	var out [][16]byte
	for payloadIndex := 0; ; payloadIndex++ {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("payload %d %s LayeredDecoder RTP DecodeNext %s: %v",
				payloadIndex, publicRTCPayloadLabel(labels, payloadIndex), publicRTCRTPPayloadSummary(payloads[payloadIndex]), err)
		}
		if !ok {
			return out
		}
		for _, frame := range frames {
			out = append(out, frameMD5Visible(frame))
		}
	}
}

func decodePublicLayeredDecoderLowOverheadDigests(t *testing.T, payloads ...[]byte) [][16]byte {
	t.Helper()
	dec, err := NewLayeredDecoder(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoder: %v", err)
	}
	defer dec.Close()

	var out [][16]byte
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("LayeredDecoder low-overhead DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range frames {
			out = append(out, frameMD5Visible(frame))
		}
	}
	return out
}

func decodePublicLayeredDecoderLowOverheadMetadata(t *testing.T, payloads ...[]byte) []publicLayeredFrameMetadata {
	t.Helper()
	dec, err := NewLayeredDecoder(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoder metadata: %v", err)
	}
	defer dec.Close()

	var out []publicLayeredFrameMetadata
	for {
		frames, ok, err := dec.DecodeNextWithMetadata()
		if err != nil {
			t.Fatalf("LayeredDecoder low-overhead DecodeNextWithMetadata: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range frames {
			out = append(out, publicLayeredFrameMetadataFromDecoded(t, frame))
		}
	}
	return out
}

func decodePublicLayeredDecoderRTPPayloadDigests(t *testing.T, payloads ...[]byte) [][16]byte {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPayloads(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPayloads: %v", err)
	}
	defer dec.Close()

	var out [][16]byte
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("LayeredDecoder DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range frames {
			out = append(out, frameMD5Visible(frame))
		}
	}
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder Reset: %v", err)
	}
	if _, ok, err := dec.DecodeNext(); err != nil || !ok {
		t.Fatalf("LayeredDecoder DecodeNext after Reset ok=%v err=%v", ok, err)
	}
	return out
}

func decodePublicLayeredDecoderRTPPayloadMetadata(t *testing.T, payloads ...[]byte) []publicLayeredFrameMetadata {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPayloads(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPayloads metadata: %v", err)
	}
	defer dec.Close()

	var out []publicLayeredFrameMetadata
	for {
		frames, ok, err := dec.DecodeNextWithMetadata()
		if err != nil {
			t.Fatalf("LayeredDecoder DecodeNextWithMetadata: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range frames {
			out = append(out, publicLayeredFrameMetadataFromDecoded(t, frame))
		}
	}
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder metadata Reset: %v", err)
	}
	if frames, ok, err := dec.DecodeNextWithMetadata(); err != nil || !ok {
		t.Fatalf("LayeredDecoder DecodeNextWithMetadata after Reset ok=%v err=%v", ok, err)
	} else {
		for _, frame := range frames {
			_ = publicLayeredFrameMetadataFromDecoded(t, frame)
		}
	}
	return out
}

func decodePublicLayeredDecoderRTPPacketDigests(t *testing.T, packets ...[]byte) [][16]byte {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPackets(packets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets: %v", err)
	}
	defer dec.Close()

	var out [][16]byte
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("LayeredDecoder packet DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range frames {
			out = append(out, frameMD5Visible(frame))
		}
	}
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder packet Reset: %v", err)
	}
	if _, ok, err := dec.DecodeNext(); err != nil || !ok {
		t.Fatalf("LayeredDecoder packet DecodeNext after Reset ok=%v err=%v", ok, err)
	}
	return out
}

func decodePublicLayeredDecoderRTPPacketMetadata(t *testing.T, packets ...[]byte) []publicLayeredFrameMetadata {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPackets(packets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets metadata: %v", err)
	}
	defer dec.Close()

	var out []publicLayeredFrameMetadata
	for {
		frames, ok, err := dec.DecodeNextWithMetadata()
		if err != nil {
			t.Fatalf("LayeredDecoder packet DecodeNextWithMetadata: %v", err)
		}
		if !ok {
			break
		}
		for _, frame := range frames {
			out = append(out, publicLayeredFrameMetadataFromDecoded(t, frame))
		}
	}
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder packet metadata Reset: %v", err)
	}
	if frames, ok, err := dec.DecodeNextWithMetadata(); err != nil || !ok {
		t.Fatalf("LayeredDecoder packet DecodeNextWithMetadata after Reset ok=%v err=%v", ok, err)
	} else {
		for _, frame := range frames {
			_ = publicLayeredFrameMetadataFromDecoded(t, frame)
		}
	}
	return out
}

func decodePublicLayeredLiveDecoderRTPPayloadDigests(t *testing.T, probePayloads [][]byte, payloads ...[]byte) [][16]byte {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPayloads(probePayloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPayloads live: %v", err)
	}
	defer dec.Close()
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder live Reset: %v", err)
	}

	var out [][16]byte
	for i, payload := range payloads {
		frames, err := dec.DecodeRTPPayload(payload)
		if err != nil {
			t.Fatalf("LayeredDecoder DecodeRTPPayload packet %d: %v", i, err)
		}
		for _, frame := range frames {
			out = append(out, frameMD5Visible(frame))
		}
	}
	return out
}

func decodePublicLayeredLiveDecoderRTPPacketDigests(t *testing.T, probePackets [][]byte, packets ...[]byte) [][16]byte {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPackets(probePackets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets live: %v", err)
	}
	defer dec.Close()
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder live packet Reset: %v", err)
	}

	var out [][16]byte
	for i, packet := range packets {
		frames, err := dec.DecodeRTPPacket(packet)
		if err != nil {
			t.Fatalf("LayeredDecoder DecodeRTPPacket packet %d: %v", i, err)
		}
		for _, frame := range frames {
			out = append(out, frameMD5Visible(frame))
		}
	}
	return out
}

func decodePublicLayeredLiveDecoderRTPPayloadMetadata(t *testing.T, probePayloads [][]byte, payloads ...[]byte) []publicLayeredFrameMetadata {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPayloads(probePayloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPayloads live metadata: %v", err)
	}
	defer dec.Close()
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder live metadata Reset: %v", err)
	}

	var out []publicLayeredFrameMetadata
	for i, payload := range payloads {
		frames, err := dec.DecodeRTPPayloadWithMetadata(payload)
		if err != nil {
			t.Fatalf("LayeredDecoder DecodeRTPPayloadWithMetadata packet %d: %v", i, err)
		}
		for _, frame := range frames {
			out = append(out, publicLayeredFrameMetadataFromDecoded(t, frame))
		}
	}
	return out
}

func decodePublicLayeredLiveDecoderRTPPacketMetadata(t *testing.T, probePackets [][]byte, packets ...[]byte) []publicLayeredFrameMetadata {
	t.Helper()
	dec, err := NewLayeredDecoderFromRTPPackets(probePackets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets live metadata: %v", err)
	}
	defer dec.Close()
	if err := dec.Reset(); err != nil {
		t.Fatalf("LayeredDecoder live packet metadata Reset: %v", err)
	}

	var out []publicLayeredFrameMetadata
	for i, packet := range packets {
		frames, err := dec.DecodeRTPPacketWithMetadata(packet)
		if err != nil {
			t.Fatalf("LayeredDecoder DecodeRTPPacketWithMetadata packet %d: %v", i, err)
		}
		for _, frame := range frames {
			out = append(out, publicLayeredFrameMetadataFromDecoded(t, frame))
		}
	}
	return out
}

func publicRTCPayloadLabel(labels []string, index int) string {
	if index < 0 || index >= len(labels) || labels[index] == "" {
		return ""
	}
	return labels[index]
}

func publicRTCRTPPayloadSummary(payload []byte) string {
	it, err := NewRTPPayloadIterator(payload)
	if err != nil {
		return fmt.Sprintf("rtp=%dB iteratorErr=%v", len(payload), err)
	}
	header := it.Header()
	summary := fmt.Sprintf("rtp=%dB Z=%v Y=%v W=%d N=%v",
		len(payload), header.ContinuesPrevious, header.ContinuesNext, header.ElementCount, header.StartsNewCodedVideoSequence)
	for {
		elem, ok, err := it.Next()
		if err != nil {
			return summary + fmt.Sprintf(" elemErr=%v", err)
		}
		if !ok {
			return summary
		}
		elemSummary := fmt.Sprintf(" elem%d=%dB prev=%v next=%v inferred=%v", elem.Index, len(elem.Data), elem.ContinuesPrevious, elem.ContinuesNext, elem.InferredLastLength)
		if !elem.ContinuesPrevious && !elem.ContinuesNext {
			if unit, err := obu.ParseElement(elem.Data); err == nil {
				elemSummary += fmt.Sprintf(" obu=%v T%d S%d", unit.Header.Type, unit.Header.TemporalID, unit.Header.SpatialID)
			}
		}
		summary += elemSummary
	}
}

// publicRTCController replays the WebRTC temporal-unit schedule from the
// internal encoder state machine so tests can check every RTCEncoder picture
// against an independently stepped schedule. A config change that alters the
// sequence header or dependency structure schedules the next key picture.
type publicRTCController struct {
	config EncoderConfig
	state  internalencoder.WebRTCEncoderState
}

func newPublicRTCController(config EncoderConfig) (*publicRTCController, error) {
	c := &publicRTCController{}
	if err := c.SetConfig(config); err != nil {
		return nil, err
	}
	c.state = internalencoder.WebRTCEncoderState{}
	return c, nil
}

func (c *publicRTCController) Config() EncoderConfig { return c.config }

func (c *publicRTCController) State() internalencoder.WebRTCEncoderState { return c.state }

func (c *publicRTCController) SetConfig(config EncoderConfig) error {
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return err
	}
	seq, err := internalencoder.SequenceHeaderForConfig(normalized)
	if err != nil {
		return err
	}
	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalized)
	if err != nil {
		return err
	}
	prevSeq, prevSeqErr := internalencoder.SequenceHeaderForConfig(c.config)
	prevStructure, prevStructureErr := internalencoder.WebRTCFrameDependencyStructureForConfig(c.config)
	if prevSeqErr != nil || prevSeq != seq || prevStructureErr != nil || prevStructure != structure {
		c.state = internalencoder.WebRTCEncoderState{NextOrderHint: c.state.NextOrderHint, NextFrameID: c.state.NextFrameID}
	}
	c.config = normalized
	return nil
}

func (c *publicRTCController) NextTemporalUnit(forceKeyFrame bool) (internalencoder.WebRTCPictureTemporalUnit, error) {
	unit, next, err := internalencoder.WebRTCNextTemporalUnitForState(c.config, c.state, forceKeyFrame)
	if err != nil {
		return internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	c.state = next
	return unit, nil
}

func publicRTCUnitFrameNum(unit internalencoder.WebRTCPictureTemporalUnit) uint8 {
	if unit.Key {
		return unit.KeyUnit.FrameNum
	}
	if unit.Delta {
		return unit.DeltaUnit.FrameNum
	}
	return 0
}

func publicRTCUnitFrameControl(unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) (internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	if unit.Key == unit.Delta {
		return internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.Control.FrameNum {
			return internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
		}
		return unit.KeyUnit.Control.Frames[frameIndex], unit.KeyUnit.Control.DependencyStructure, nil
	}
	if frameIndex >= unit.DeltaUnit.Control.FrameNum || !state.DependencyStructureState.Valid {
		return internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return unit.DeltaUnit.Control.Frames[frameIndex], state.DependencyStructureState.Structure, nil
}

func rtpPayloadHasLayerOBU(t *testing.T, data []byte, temporalID uint8, spatialID uint8) bool {
	t.Helper()
	it := obu.NewLowOverheadIterator(data)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			return false
		}
		switch unit.Header.Type {
		case obu.TypeFrameHeader, obu.TypeFrame, obu.TypeTileGroup, obu.TypeRedundantFrameHeader:
			if unit.Header.TemporalID != temporalID || unit.Header.SpatialID != spatialID {
				t.Fatalf("OBU %v layer T%d S%d, want T%d S%d", unit.Header.Type, unit.Header.TemporalID, unit.Header.SpatialID, temporalID, spatialID)
			}
			return true
		}
	}
}

func TestPublicRTCEncoderL1T3(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 300_000, Framerate: 30,
		TemporalLayers: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantTIDs := []uint8{0, 2, 1, 2, 0, 2, 1, 2, 0}
	var tus [][]byte
	for i, wantTID := range wantTIDs {
		f := I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for j := range f.Y {
			f.Y[j] = uint8(50 + (j/7+i*11)%140)
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if out.TemporalID != wantTID {
			t.Fatalf("frame %d temporal id %d want %d", i, out.TemporalID, wantTID)
		}
		if len(out.DependencyDescriptor) == 0 {
			t.Fatalf("frame %d missing dependency descriptor", i)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}
	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	n := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		n += len(batch)
	}
	if n != len(wantTIDs) {
		t.Fatalf("decoded %d frames, want %d", n, len(wantTIDs))
	}
}

func TestPublicRTCEncoderGoldenInterval(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	base := make([]byte, w*h)
	for i := range base {
		base[i] = uint8(50 + rng.Intn(170))
	}
	mk := func(boxX int) I420Frame {
		f := I420Frame{
			Y: append([]byte(nil), base...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		if boxX >= 0 {
			for y := 32; y < 96; y++ {
				for x := boxX; x < boxX+64 && x < w; x++ {
					f.Y[y*w+x] = 235
				}
			}
		}
		return f
	}
	frames := []I420Frame{
		mk(-1),
		mk(64),
		mk(-1),
	}
	encode := func(goldenInterval int) [][]byte {
		enc, err := NewRTCEncoder(VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 2_000_000, Framerate: 30,
			MinQIndex:      40,
			MaxQIndex:      160,
			TemporalLayers: 1,
			GoldenInterval: goldenInterval,
		})
		if err != nil {
			t.Fatal(err)
		}
		var tus [][]byte
		for i, f := range frames {
			out, err := enc.Encode(f, false)
			if err != nil {
				t.Fatalf("encode %d: %v", i, err)
			}
			tus = append(tus, append([]byte(nil), out.Data...))
		}
		return tus
	}
	lastOnly := encode(-1)
	withGolden := encode(16)
	t.Logf("rtc reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
	if len(withGolden[2])*5 >= len(lastOnly[2])*4 {
		t.Fatalf("golden interval did not materially reduce reveal frame: last-only %dB, golden %dB", len(lastOnly[2]), len(withGolden[2]))
	}
}

func TestPublicEncoderZeroValueGuards(t *testing.T) {
	var video VideoEncoder
	if _, err := video.Encode(I420Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder Encode returned nil error")
	}
	if _, err := video.EncodeI422(I422Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeI422 returned nil error")
	}
	if _, err := video.EncodeI444(I444Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeI444 returned nil error")
	}
	if _, err := video.EncodeI400(I400Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeI400 returned nil error")
	}
	if _, err := video.EncodeNV12(NV12Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeNV12 returned nil error")
	}
	if _, err := video.EncodeNV21(NV21Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeNV21 returned nil error")
	}
	video.SetDecisionStatsEnabled(true)
	video.ResetDecisionStats()
	if got := video.DecisionStats(); got != (EncoderDecisionStats{}) {
		t.Fatalf("zero VideoEncoder stats=%+v", got)
	}
	if got := video.QIndex(); got != 0 {
		t.Fatalf("zero VideoEncoder QIndex=%d", got)
	}
	if got := video.Reconstruction(); got.Width != 0 || got.Height != 0 {
		t.Fatalf("zero VideoEncoder reconstruction=%+v", got)
	}

	var rtc RTCEncoder
	if _, err := rtc.Encode(I420Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder Encode returned nil error")
	}
	if _, err := rtc.EncodeI422(I422Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI422 returned nil error")
	}
	if _, err := rtc.EncodeI444(I444Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI444 returned nil error")
	}
	if _, err := rtc.EncodeI400(I400Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI400 returned nil error")
	}
	if _, err := rtc.EncodeNV12(NV12Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV12 returned nil error")
	}
	if _, err := rtc.EncodeNV21(NV21Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV21 returned nil error")
	}
	if _, err := rtc.EncodeNV12Picture(NV12Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV12Picture returned nil error")
	}
	if _, err := rtc.EncodeNV21Picture(NV21Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV21Picture returned nil error")
	}
	if _, err := rtc.EncodeI422Picture(I422Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI422Picture returned nil error")
	}
	if _, err := rtc.EncodeI444Picture(I444Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI444Picture returned nil error")
	}
	if _, err := rtc.EncodeI400Picture(I400Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI400Picture returned nil error")
	}
}

func TestPublicEncoderRejectsInvalidI420Frames(t *testing.T) {
	const w, h = 64, 64
	invalid := I420Frame{
		Y:            make([]byte, w*h-1),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	video, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := video.Encode(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short Y plane")
	}
	invalid.Y = make([]byte, w*h)
	invalid.ChromaStride = w/2 - 1
	if _, err := video.Encode(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid chroma stride")
	}

	rtc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	invalid.ChromaStride = w / 2
	invalid.U = invalid.U[:len(invalid.U)-1]
	if _, err := rtc.Encode(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short U plane")
	}
}

func TestPublicEncoderRejectsInvalidI422Frames(t *testing.T) {
	const w, h = 64, 64
	valid := publicI422FromI420(publicRTCMatrixFrame(w, h, 0))
	invalid := valid
	invalid.U = invalid.U[:len(invalid.U)-1]
	video, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer video.Close()
	if _, err := video.EncodeI422(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short I422 U plane")
	}
	invalid = valid
	invalid.ChromaStride = w/2 - 1
	if _, err := video.EncodeI422(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid I422 chroma stride")
	}

	rtc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rtc.Close()
	invalid = valid
	invalid.V = invalid.V[:len(invalid.V)-1]
	if _, err := rtc.EncodeI422(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short I422 V plane")
	}
	if _, err := rtc.EncodeI422Picture(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short I422 V plane for picture encode")
	}
}

func TestPublicEncoderRejectsInvalidI444Frames(t *testing.T) {
	const w, h = 64, 64
	valid := publicI444FromI420(publicRTCMatrixFrame(w, h, 0))
	invalid := valid
	invalid.U = invalid.U[:len(invalid.U)-1]
	video, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer video.Close()
	if _, err := video.EncodeI444(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short I444 U plane")
	}
	invalid = valid
	invalid.UStride = w - 1
	if _, err := video.EncodeI444(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid I444 U stride")
	}

	rtc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rtc.Close()
	invalid = valid
	invalid.VStride = w - 1
	if _, err := rtc.EncodeI444(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted an invalid I444 V stride")
	}
	if _, err := rtc.EncodeI444Picture(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted an invalid I444 V stride for picture encode")
	}
}

func TestPublicEncoderRejectsInvalidI400Frames(t *testing.T) {
	const w, h = 64, 64
	valid := publicI400FromI420(publicRTCMatrixFrame(w, h, 0))
	invalid := valid
	invalid.Y = invalid.Y[:len(invalid.Y)-1]
	video, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer video.Close()
	if _, err := video.EncodeI400(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short I400 Y plane")
	}
	invalid = valid
	invalid.YStride = w - 1
	if _, err := video.EncodeI400(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid I400 Y stride")
	}

	rtc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rtc.Close()
	invalid = valid
	invalid.Y = invalid.Y[:len(invalid.Y)-1]
	if _, err := rtc.EncodeI400(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short I400 Y plane")
	}
	if _, err := rtc.EncodeI400Picture(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short I400 Y plane for picture encode")
	}
}

func TestPublicEncoderRejectsInvalidNV12Frames(t *testing.T) {
	const w, h = 64, 64
	valid := publicNV12FromI420(publicRTCMatrixFrame(w, h, 0))
	invalid := valid
	invalid.Y = invalid.Y[:len(invalid.Y)-1]
	video, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer video.Close()
	if _, err := video.EncodeNV12(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short NV12 Y plane")
	}
	invalid = valid
	invalid.UVStride = w - 1
	if _, err := video.EncodeNV12(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid NV12 UV stride")
	}

	rtc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rtc.Close()
	invalid = valid
	invalid.UV = invalid.UV[:len(invalid.UV)-1]
	if _, err := rtc.EncodeNV12(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short NV12 UV plane")
	}
	if _, err := rtc.EncodeNV12Picture(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short NV12 UV plane for picture encode")
	}
}

func TestPublicEncoderRejectsInvalidNV21Frames(t *testing.T) {
	const w, h = 64, 64
	valid := publicNV21FromI420(publicRTCMatrixFrame(w, h, 0))
	invalid := valid
	invalid.Y = invalid.Y[:len(invalid.Y)-1]
	video, err := NewVideoEncoder(VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer video.Close()
	if _, err := video.EncodeNV21(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short NV21 Y plane")
	}
	invalid = valid
	invalid.VUStride = w - 1
	if _, err := video.EncodeNV21(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid NV21 VU stride")
	}

	rtc, err := NewRTCEncoder(VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rtc.Close()
	invalid = valid
	invalid.VU = invalid.VU[:len(invalid.VU)-1]
	if _, err := rtc.EncodeNV21(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short NV21 VU plane")
	}
	if _, err := rtc.EncodeNV21Picture(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short NV21 VU plane for picture encode")
	}
}
