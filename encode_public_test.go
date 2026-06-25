package goav1_test

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

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

func TestPublicEncoderWebRTCScalabilityModeCatalogue(t *testing.T) {
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
	modes := goav1.EncoderWebRTCScalabilityModes()
	if len(modes) != len(want) {
		t.Fatalf("EncoderWebRTCScalabilityModes len=%d want %d", len(modes), len(want))
	}
	prefixed := goav1.AppendEncoderWebRTCScalabilityModes([]goav1.EncoderScalabilityMode{goav1.EncoderScalabilityModeL3T3})
	if len(prefixed) != len(want)+1 || prefixed[0] != goav1.EncoderScalabilityModeL3T3 {
		t.Fatalf("AppendEncoderWebRTCScalabilityModes prefix len=%d first=%s", len(prefixed), prefixed[0])
	}
	for i, name := range want {
		mode := modes[i]
		if prefixed[i+1] != mode {
			t.Fatalf("prefixed mode %d=%s want %s", i+1, prefixed[i+1], mode)
		}
		parsed, ok := goav1.ParseEncoderScalabilityMode(name)
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
			mode.UsesKeyFrameInterLayerDependency() != strings.Contains(name, "_KEY") ||
			mode.UsesKeyFrameInterLayerDependencyShift() != strings.Contains(name, "_KEY_SHIFT") {
			t.Fatalf("%q flags simulcast=%v small=%v key=%v shift=%v",
				name,
				mode.IsSimulcast(),
				mode.UsesSmallResolutionStep(),
				mode.UsesKeyFrameInterLayerDependency(),
				mode.UsesKeyFrameInterLayerDependencyShift())
		}
	}
	if _, ok := goav1.ParseEncoderScalabilityMode("L4T4"); ok {
		t.Fatal("ParseEncoderScalabilityMode accepted invalid L4T4")
	}
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
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
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
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	dec, err := goav1.NewDecoder(tus)
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

func TestPublicVideoEncoderClose(t *testing.T) {
	const w, h = 192, 128
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	for i := range 6 {
		f := goav1.I420Frame{
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
	dec, err := goav1.NewDecoder(tus)
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
	cfg := goav1.VideoEncoderConfig{
		Width: w, Height: h,
		QIndex:         80,
		TemporalLayers: 2,
	}
	formats := []struct {
		name   string
		source func(int) goav1.I420Frame
		encode func(*goav1.VideoEncoder, goav1.I420Frame, bool) (goav1.EncodedFrame, error)
	}{
		{
			name: "I422",
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				return enc.EncodeI422(publicI422FromI420(src), forceKey)
			},
		},
		{
			name: "I444",
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				return enc.EncodeI444(publicI444FromI420(src), forceKey)
			},
		},
		{
			name: "I400",
			source: func(frame int) goav1.I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(w, h, frame))
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				return enc.EncodeI400(publicI400FromI420(src), forceKey)
			},
		},
		{
			name: "Frame420-8bit",
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(frame int) goav1.I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(w, h, frame))
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				return enc.EncodeNV12(publicNV12FromI420(src), forceKey)
			},
		},
		{
			name: "NV21",
			source: func(frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(w, h, frame)
			},
			encode: func(enc *goav1.VideoEncoder, src goav1.I420Frame, forceKey bool) (goav1.EncodedFrame, error) {
				return enc.EncodeNV21(publicNV21FromI420(src), forceKey)
			},
		},
	}
	for _, format := range formats {
		t.Run(format.name, func(t *testing.T) {
			i420Enc, err := goav1.NewVideoEncoder(cfg)
			if err != nil {
				t.Fatalf("NewVideoEncoder I420: %v", err)
			}
			defer i420Enc.Close()
			testEnc, err := goav1.NewVideoEncoder(cfg)
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
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: 16, Height: 16,
		QIndex: 80,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer enc.Close()

	format := goav1.FrameFormat{
		Width:        16,
		Height:       16,
		BitDepth:     8,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := goav1.FrameRequiredSize(format)
	if err != nil {
		t.Fatalf("FrameRequiredSize: %v", err)
	}
	frame, err := goav1.BindFrame(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatalf("BindFrame: %v", err)
	}
	if _, err := enc.EncodeFrame(frame, false); !errors.Is(err, goav1.ErrFrameInvalidFormat) {
		t.Fatalf("EncodeFrame err=%v want %v", err, goav1.ErrFrameInvalidFormat)
	}
}

func TestPublicRTCEncoderInputFormatsEncodeAndPictureDecode(t *testing.T) {
	formats := []struct {
		name          string
		source        func(int, int, int) goav1.I420Frame
		encode        func(*goav1.RTCEncoder, goav1.I420Frame, bool) (goav1.RTCFrame, error)
		encodePicture func(*goav1.RTCEncoder, goav1.I420Frame, bool) (goav1.RTCPicture, error)
	}{
		{
			name: "I422",
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				return enc.EncodeI422(publicI422FromI420(src), forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				return enc.EncodeI422Picture(publicI422FromI420(src), forceKey)
			},
		},
		{
			name: "I444",
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				return enc.EncodeI444(publicI444FromI420(src), forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				return enc.EncodeI444Picture(publicI444FromI420(src), forceKey)
			},
		},
		{
			name: "I400",
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(width, height, frame))
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				return enc.EncodeI400(publicI400FromI420(src), forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				return enc.EncodeI400Picture(publicI400FromI420(src), forceKey)
			},
		},
		{
			name: "Frame420-8bit",
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     8,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
					Width:        src.Width,
					Height:       src.Height,
					BitDepth:     10,
					SubsamplingX: true,
					Align:        32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
					Width:    src.Width,
					Height:   src.Height,
					BitDepth: 12,
					Align:    32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicI420NeutralChroma(publicRTCMatrixFrame(width, height, frame))
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
					Width:      src.Width,
					Height:     src.Height,
					BitDepth:   10,
					MonoChrome: true,
					Align:      32,
				})
				return enc.EncodeFrame(frame, forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				frame := publicFrameFromI420(t, src, goav1.FrameFormat{
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
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				return enc.EncodeNV12(publicNV12FromI420(src), forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				return enc.EncodeNV12Picture(publicNV12FromI420(src), forceKey)
			},
		},
		{
			name: "NV21",
			source: func(width int, height int, frame int) goav1.I420Frame {
				return publicRTCMatrixFrame(width, height, frame)
			},
			encode: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCFrame, error) {
				return enc.EncodeNV21(publicNV21FromI420(src), forceKey)
			},
			encodePicture: func(enc *goav1.RTCEncoder, src goav1.I420Frame, forceKey bool) (goav1.RTCPicture, error) {
				return enc.EncodeNV21Picture(publicNV21FromI420(src), forceKey)
			},
		},
	}
	for _, format := range formats {
		t.Run(format.name+"/single-spatial", func(t *testing.T) {
			const w, h = 192, 128
			enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
			dec, err := goav1.NewDecoder(tus)
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
			cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeS2T2)
			cfg.RateControl = goav1.EncoderRateControlCQP
			cfg.Quantizer = 34
			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			defer enc.Close()

			var descriptorReceiver goav1.RTPDependencyDescriptorState
			var rtpReceiver goav1.RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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
		})
	}
}

func TestPublicEncoderRuntimeOptions(t *testing.T) {
	const w, h = 192, 128
	t.Run("VideoEncoder", func(t *testing.T) {
		enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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
		enc.SetTileColumns(4)
		enc.SetGoldenInterval(0)
		delta, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("delta Encode after runtime option change: %v", err)
		}
		if !key.Keyframe || delta.Keyframe {
			t.Fatalf("runtime options key=%v delta=%v", key.Keyframe, delta.Keyframe)
		}
		dec, err := goav1.NewDecoder([][]byte{
			append([]byte(nil), key.Data...),
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
		cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeS2T2)
		enc, err := goav1.NewRTCEncoderWithConfig(cfg)
		if err != nil {
			t.Fatalf("NewRTCEncoderWithConfig: %v", err)
		}
		defer enc.Close()
		key, err := enc.EncodePicture(publicRTCMatrixFrame(w, h, 0), false)
		if err != nil {
			t.Fatalf("key EncodePicture: %v", err)
		}
		keyLastFrameID := key.Frames[key.FrameNum-1].FrameID
		var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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

	var zeroVideo goav1.VideoEncoder
	zeroVideo.SetTileColumns(4)
	zeroVideo.SetGoldenInterval(0)
	var zeroRTC goav1.RTCEncoder
	zeroRTC.SetTileColumns(4)
	zeroRTC.SetGoldenInterval(0)
}

func TestPublicRTCEncoderDependencyDescriptorOwnership(t *testing.T) {
	t.Run("Encode", func(t *testing.T) {
		const w, h = 192, 128
		enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
		if !bytes.Equal(first.DependencyDescriptor, snapshot) {
			t.Fatal("retained Encode dependency descriptor changed after a later encode")
		}
	})

	t.Run("EncodePicture", func(t *testing.T) {
		const w, h = 640, 360
		enc, err := goav1.NewRTCEncoderWithConfig(publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL2T2))
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
		var snapshots [goav1.EncoderWebRTCMaxSpatialLayers][]byte
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
			if !bytes.Equal(first.Frames[i].DependencyDescriptor, snapshots[i]) {
				t.Fatalf("retained EncodePicture frame %d dependency descriptor changed after a later encode", i)
			}
		}
	})
}

func TestPublicRTCEncoderClose(t *testing.T) {
	const w, h = 640, 360
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL2T2)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
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
	cfg := publicRTCMatrixConfig(640, 360, goav1.EncoderScalabilityModeL2T2_KEY_SHIFT)
	cfg.BitDepth = 0
	normalized, err := goav1.NormalizeRTCEncoderConfig(cfg)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig valid: %v", err)
	}
	if normalized.BitDepth != 8 ||
		normalized.Profile != goav1.EncoderProfile0 ||
		normalized.SpatialLayerCount != 2 ||
		normalized.TemporalLayerCount != 2 ||
		normalized.RateControl != goav1.EncoderRateControlCBR {
		t.Fatalf("normalized config=%+v", normalized)
	}
	cqp := cfg
	cqp.RateControl = goav1.EncoderRateControlCQP
	cqp.Quantizer = 32
	normalizedCQP, err := goav1.NormalizeRTCEncoderConfig(cqp)
	if err != nil {
		t.Fatalf("NormalizeRTCEncoderConfig CQP: %v", err)
	}
	if normalizedCQP.RateControl != goav1.EncoderRateControlCQP || normalizedCQP.Quantizer != 32 {
		t.Fatalf("normalized CQP config=%+v", normalizedCQP)
	}
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig valid: %v", err)
	}
	defer enc.Close()
	wantLive := enc.Config()

	tests := []struct {
		name string
		edit func(*goav1.EncoderConfig)
		want error
	}{
		{
			name: "profile1-444",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.Profile = goav1.EncoderProfile1
			},
			want: goav1.ErrEncoderUnsupported,
		},
		{
			name: "high-bit-depth",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.BitDepth = 10
			},
			want: goav1.ErrEncoderUnsupported,
		},
		{
			name: "explicit-monochrome-color-config",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = goav1.EncoderSequenceColorConfig{
					BitDepth:   8,
					MonoChrome: true,
				}
			},
			want: goav1.ErrEncoderUnsupported,
		},
		{
			name: "ambiguous-color-config-without-enable",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.ColorConfig = goav1.EncoderSequenceColorConfig{
					BitDepth:     8,
					SubsamplingX: true,
					SubsamplingY: true,
				}
			},
			want: goav1.ErrEncoderInvalidConfig,
		},
		{
			name: "cqp-quantizer-zero",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.RateControl = goav1.EncoderRateControlCQP
			},
			want: goav1.ErrEncoderUnsupported,
		},
		{
			name: "cqp-quantizer-too-high",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.RateControl = goav1.EncoderRateControlCQP
				cfg.Quantizer = goav1.EncoderWebRTCMaxQuantizer + 1
			},
			want: goav1.ErrEncoderInvalidConfig,
		},
		{
			name: "invalid-framerate",
			edit: func(cfg *goav1.EncoderConfig) {
				cfg.MaxFramerate = goav1.EncoderRational{Num: 1, Den: 2}
			},
			want: goav1.ErrEncoderInvalidConfig,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bad := cfg
			tc.edit(&bad)
			if _, err := goav1.NormalizeRTCEncoderConfig(bad); !errors.Is(err, tc.want) {
				t.Fatalf("NormalizeRTCEncoderConfig err=%v want %v", err, tc.want)
			}
			if _, err := goav1.NewRTCEncoderWithConfig(bad); !errors.Is(err, tc.want) {
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
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL2T2)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	if _, err := enc.Encode(publicRTCMatrixFrame(w, h, 0), false); err != goav1.ErrEncoderUnsupported {
		t.Fatalf("Encode multi-spatial err=%v want %v", err, goav1.ErrEncoderUnsupported)
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
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	src := goav1.I420Frame{
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
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
	var receiver goav1.RTPDependencyDescriptorState
	assertPublicRTCFrameRTPPackets(t, &receiver, frame, limits, true, true, true)
}

func TestPublicRTCFrameAppendRTPPacketsWithHeaders(t *testing.T) {
	const w, h = 192, 128
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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

	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first: %v", err)
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full: %v", err)
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	packetSpans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, packetSpans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets: %v", err)
	}
	if packetCount != size.Packetizer.Packets || packetCount < 2 {
		t.Fatalf("packet count=%d want %d and fragmentation", packetCount, size.Packetizer.Packets)
	}

	config := goav1.EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     96,
		SequenceNumber:                  0xfffe,
		Timestamp:                       0x01020304,
		SSRC:                            0xaabbccdd,
		DependencyDescriptorExtensionID: 42,
		HeaderExtensionProfile:          goav1.RTPExtensionProfileTwoByte | 0x0005,
	}
	headerSpans := make([]goav1.EncoderWebRTCRTPPacketHeaderSpan, packetCount)
	oneByteConfig := goav1.EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     config.PayloadType,
		SequenceNumber:                  config.SequenceNumber,
		Timestamp:                       config.Timestamp,
		SSRC:                            config.SSRC,
		DependencyDescriptorExtensionID: config.DependencyDescriptorExtensionID,
		HeaderExtensionProfile:          goav1.RTPExtensionProfileOneByte,
	}
	if _, err := goav1.EncoderWebRTCRTPPacketsWithHeadersSize(oneByteConfig, []byte{0xaa}, make([]byte, 17), []goav1.EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 17}}); !errors.Is(err, goav1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("one-byte dependency descriptor packet size err=%v want %v", err, goav1.ErrRTPInvalidHeaderExtension)
	}
	if _, _, err := goav1.AppendEncoderWebRTCRTPPacketsWithHeaders(nil, make([]goav1.EncoderWebRTCRTPPacketHeaderSpan, 1), oneByteConfig, []byte{0xaa}, make([]byte, 17), []goav1.EncoderWebRTCRTPPacketSpan{{PayloadLength: 1, DescriptorLength: 17}}); !errors.Is(err, goav1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("one-byte dependency descriptor packet err=%v want %v", err, goav1.ErrRTPInvalidHeaderExtension)
	}

	sizeInfo, err := goav1.EncoderWebRTCRTPPacketsWithHeadersSize(config, rtpPayloads, descriptors, packetSpans[:packetCount])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize: %v", err)
	}
	if sizeInfo.Packets != packetCount || sizeInfo.Bytes <= len(rtpPayloads) ||
		sizeInfo.MaxPacketBytes <= limits.MaxPayloadLen || sizeInfo.MaxHeaderBytes <= goav1.RTPHeaderMinSize {
		t.Fatalf("packet header size=%+v packetCount=%d payloadBytes=%d", sizeInfo, packetCount, len(rtpPayloads))
	}
	prefix := []byte{0xde, 0xad, 0xbe}
	fullDst := make([]byte, len(prefix), len(prefix)+sizeInfo.Bytes)
	copy(fullDst, prefix)
	fullPackets, fullCount, err := goav1.AppendEncoderWebRTCRTPPacketsWithHeaders(fullDst, headerSpans, config, rtpPayloads, descriptors, packetSpans[:packetCount])
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

	payloadSlices := make([][]byte, packetCount)
	var receiver goav1.RTPDependencyDescriptorState
	totalPacketBytes := 0
	for i := 0; i < packetCount; i++ {
		span := headerSpans[i]
		if span.Offset < len(prefix) || span.Length > sizeInfo.MaxPacketBytes || span.HeaderSize > sizeInfo.MaxHeaderBytes {
			t.Fatalf("packet %d span=%+v size=%+v", i, span, sizeInfo)
		}
		totalPacketBytes += span.Length
		raw := fullPackets[span.Offset : span.Offset+span.Length]
		if i == 0 {
			if _, err := goav1.ParseRTPPacketDependencyDescriptor(raw, config.DependencyDescriptorExtensionID+1, &goav1.RTPDependencyDescriptorState{}); !errors.Is(err, goav1.ErrRTPHeaderExtensionNotFound) {
				t.Fatalf("packet %d missing dependency descriptor err=%v want %v", i, err, goav1.ErrRTPHeaderExtensionNotFound)
			}
			parsedWithoutState, err := goav1.ParseRTPPacketDependencyDescriptor(raw, config.DependencyDescriptorExtensionID, nil)
			if err != nil {
				t.Fatalf("packet %d ParseRTPPacketDependencyDescriptor without state: %v", i, err)
			}
			if parsedWithoutState.Descriptor.Mandatory.FrameNumber != uint16(frame.FrameID) {
				t.Fatalf("packet %d descriptor without state frame=%d want %d", i, parsedWithoutState.Descriptor.Mandatory.FrameNumber, frame.FrameID)
			}
		}
		descriptorPacket, err := goav1.ParseRTPPacketDependencyDescriptor(raw, config.DependencyDescriptorExtensionID, &receiver)
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
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
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
	options := goav1.EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		ActiveDecodeTargetsPresentOnFirstPacket: true,
		ActiveDecodeTargetsMask:                 activeMask,
	}
	firstSize, err := frame.RTPPacketScratchLenWithOptions(limits, nil, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions first: %v", err)
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLenWithOptions(limits, obuScratch, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions full: %v", err)
	}
	if size.Packetizer.Packets <= 1 || size.MaxDescriptorBytes <= goav1.RTPDependencyDescriptorMandatorySize {
		t.Fatalf("scratch=%+v want fragmented frame and extended first descriptor", size)
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPacketsWithOptions(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch, options)
	if err != nil {
		t.Fatalf("AppendRTPPacketsWithOptions: %v", err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	payloadSlices := make([][]byte, packetCount)
	var receiver goav1.RTPDependencyDescriptorState
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
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 48}
	fps := []goav1.EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 120, Den: 1},
	}

	if _, err := (goav1.RTCPicture{}).AllDecodeTargetsMask(); !errors.Is(err, goav1.ErrEncoderInvalidFrame) {
		t.Fatalf("empty picture AllDecodeTargetsMask err=%v want %v", err, goav1.ErrEncoderInvalidFrame)
	}
	if _, err := (goav1.RTCPicture{}).ActiveDecodeTargetsRTPOptions(0, 0); !errors.Is(err, goav1.ErrEncoderInvalidFrame) {
		t.Fatalf("empty picture ActiveDecodeTargetsRTPOptions err=%v want %v", err, goav1.ErrEncoderInvalidFrame)
	}

	for step, mode := range goav1.EncoderWebRTCScalabilityModes() {
		mode := mode
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = fps[step%len(fps)]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*19))
			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var receiver goav1.RTPDependencyDescriptorState
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
	if _, err := goav1.RTPDependencyDescriptorDecodeTargetActive(goav1.RTPDependencyDescriptorState{}, 0); !errors.Is(err, goav1.ErrRTPInvalidDependencyDescriptor) {
		t.Fatalf("empty RTPDependencyDescriptorDecodeTargetActive err=%v want %v", err, goav1.ErrRTPInvalidDependencyDescriptor)
	}

	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 32}
	fps := []goav1.EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
	}
	for step, mode := range goav1.EncoderWebRTCScalabilityModes() {
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
			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var receiver goav1.RTPDependencyDescriptorState
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
						parsed, err := goav1.ParseRTPPacketDependencyDescriptor(packet, dependencyDescriptorExtensionID, &receiver)
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
						active, err := goav1.RTPDependencyDescriptorDecodeTargetActive(receiver, 0)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d DecodeTargetActive: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						if !active {
							t.Fatalf("%s picture %d output %d packet %d target 0 unexpectedly inactive", mode, frameIndex, outputIndex, packetIndex)
						}
						if spatialLayers*temporalLayers > 1 {
							inactive, err := goav1.RTPDependencyDescriptorDecodeTargetActive(receiver, 1)
							if err != nil {
								t.Fatalf("%s picture %d output %d packet %d DecodeTargetActive(target 1): %v", mode, frameIndex, outputIndex, packetIndex, err)
							}
							if inactive {
								t.Fatalf("%s picture %d output %d packet %d target 1 unexpectedly active under mask %#x", mode, frameIndex, outputIndex, packetIndex, wantActiveMask)
							}
							inactiveForward, err := goav1.RTPDependencyDescriptorFrameForwardedForDecodeTarget(descriptor, receiver, 1)
							if err != nil {
								t.Fatalf("%s picture %d output %d packet %d FrameForwardedForDecodeTarget(target 1): %v", mode, frameIndex, outputIndex, packetIndex, err)
							}
							if inactiveForward {
								t.Fatalf("%s picture %d output %d packet %d target 1 unexpectedly forwarded under mask %#x", mode, frameIndex, outputIndex, packetIndex, wantActiveMask)
							}
						}
						matches, err := goav1.RTPDependencyDescriptorFrameMatchesDecodeTarget(descriptor, 0)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d FrameMatchesDecodeTarget: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						forward, err := goav1.RTPDependencyDescriptorFrameForwardedForDecodeTarget(descriptor, receiver, 0)
						if err != nil {
							t.Fatalf("%s picture %d output %d packet %d FrameForwardedForDecodeTarget: %v", mode, frameIndex, outputIndex, packetIndex, err)
						}
						if matches != wantForward || forward != wantForward {
							dti, _ := goav1.RTPDependencyDescriptorFrameDecodeTargetIndication(descriptor, 0)
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
	enc, err := goav1.NewRTCEncoderWithConfig(goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: w, Height: h},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    200,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 600,
		Scalability:       goav1.EncoderScalabilityModeL2T2_KEY_SHIFT,
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

	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
	var receiver goav1.RTPDependencyDescriptorState
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
	enc, err := goav1.NewRTCEncoderWithConfig(goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: w, Height: h},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    200,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 600,
		Scalability:       goav1.EncoderScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatal(err)
	}
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
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
		modes  []goav1.EncoderScalabilityMode
	}{
		{
			name:   "single-spatial",
			width:  192,
			height: 128,
			modes: []goav1.EncoderScalabilityMode{
				goav1.EncoderScalabilityModeL1T1,
				goav1.EncoderScalabilityModeL1T2,
				goav1.EncoderScalabilityModeL1T3,
			},
		},
		{
			name:   "two-spatial",
			width:  640,
			height: 360,
			modes: []goav1.EncoderScalabilityMode{
				goav1.EncoderScalabilityModeL2T1,
				goav1.EncoderScalabilityModeL2T1h,
				goav1.EncoderScalabilityModeL2T1_KEY,
				goav1.EncoderScalabilityModeL2T2,
				goav1.EncoderScalabilityModeL2T2h,
				goav1.EncoderScalabilityModeL2T2_KEY,
				goav1.EncoderScalabilityModeL2T2_KEY_SHIFT,
				goav1.EncoderScalabilityModeL2T3,
				goav1.EncoderScalabilityModeL2T3h,
				goav1.EncoderScalabilityModeL2T3_KEY,
				goav1.EncoderScalabilityModeL2T3_KEY_SHIFT,
				goav1.EncoderScalabilityModeS2T1,
				goav1.EncoderScalabilityModeS2T1h,
				goav1.EncoderScalabilityModeS2T2,
				goav1.EncoderScalabilityModeS2T2h,
				goav1.EncoderScalabilityModeS2T3,
				goav1.EncoderScalabilityModeS2T3h,
			},
		},
		{
			name:   "three-spatial",
			width:  1008,
			height: 576,
			modes: []goav1.EncoderScalabilityMode{
				goav1.EncoderScalabilityModeL3T1,
				goav1.EncoderScalabilityModeL3T1h,
				goav1.EncoderScalabilityModeL3T1_KEY,
				goav1.EncoderScalabilityModeL3T2,
				goav1.EncoderScalabilityModeL3T2h,
				goav1.EncoderScalabilityModeL3T2_KEY,
				goav1.EncoderScalabilityModeL3T2_KEY_SHIFT,
				goav1.EncoderScalabilityModeL3T3,
				goav1.EncoderScalabilityModeL3T3h,
				goav1.EncoderScalabilityModeL3T3_KEY,
				goav1.EncoderScalabilityModeL3T3_KEY_SHIFT,
				goav1.EncoderScalabilityModeS3T1,
				goav1.EncoderScalabilityModeS3T1h,
				goav1.EncoderScalabilityModeS3T2,
				goav1.EncoderScalabilityModeS3T2h,
				goav1.EncoderScalabilityModeS3T3,
				goav1.EncoderScalabilityModeS3T3h,
			},
		},
	}
	for _, group := range groups {
		t.Run(group.name, func(t *testing.T) {
			for _, mode := range group.modes {
				cfg := publicRTCMatrixConfig(group.width, group.height, mode)
				enc, err := goav1.NewRTCEncoderWithConfig(cfg)
				if err != nil {
					t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
				}
				var receiver goav1.RTPDependencyDescriptorState
				var rtpReceiver goav1.RTPDependencyDescriptorState
				nextFrameID := uint64(0)
				var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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

func TestPublicRTCEncoderCQPScalabilityModeMatrixDecodes(t *testing.T) {
	modes := goav1.EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	for step, mode := range modes {
		t.Run(string(mode), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.RateControl = goav1.EncoderRateControlCQP
			cfg.Quantizer = uint8(24 + step%32)

			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()
			if got := enc.Config(); got.RateControl != goav1.EncoderRateControlCQP || got.Quantizer != cfg.Quantizer {
				t.Fatalf("config rate control=%d q=%d want CQP q=%d", got.RateControl, got.Quantizer, cfg.Quantizer)
			}

			var descriptorReceiver goav1.RTPDependencyDescriptorState
			var rtpReceiver goav1.RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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
			controlChange.MaxFramerate = []goav1.EncoderRational{
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
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
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
	cfg := goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: w, Height: h},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       goav1.EncoderScalabilityModeL1T2,
	}
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	if got, err := enc.RTPFrameDuration(); err != nil || got != (goav1.EncoderRational{Num: 3000, Den: 1}) {
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
	controlChange.MaxFramerate = goav1.EncoderRational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1200
	controlChange.TargetBitrateKbps = 900
	if err := enc.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	if got, err := enc.RTPFrameDuration(); err != nil || got != (goav1.EncoderRational{Num: 1500, Den: 1}) {
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
	structureChange.Scalability = goav1.EncoderScalabilityModeS2T2
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
	fullSVC.Scalability = goav1.EncoderScalabilityModeL2T2
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
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL2T2)
	cfg.RateControl = goav1.EncoderRateControlCQP
	cfg.Quantizer = 37

	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var descriptorReceiver goav1.RTPDependencyDescriptorState
	var rtpReceiver goav1.RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
	var orderedTUs [][]byte
	encode := func(label string, frame int, wantKey bool) goav1.RTCPicture {
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
	cbrChange.RateControl = goav1.EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = goav1.EncoderRational{Num: 60, Den: 1}
	publicRTCApplyControlBitrates(&cbrChange, 1100)
	if err := enc.SetConfig(cbrChange); err != nil {
		t.Fatalf("SetConfig CBR transition: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
	encode("CBR transition delta", 2, false)

	backToCQP := enc.Config()
	backToCQP.RateControl = goav1.EncoderRateControlCQP
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
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL1T2)
	publicRTCApplyControlBitrates(&cfg, 420)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var frames []publicIVFFrame
	var sequence goav1.SequenceHeader
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
	fpsBitrateChange.MaxFramerate = goav1.EncoderRational{Num: 60000, Den: 1001}
	publicRTCApplyControlBitrates(&fpsBitrateChange, 700)
	if err := enc.SetConfig(fpsBitrateChange); err != nil {
		t.Fatalf("SetConfig fps/bitrate: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (goav1.EncoderRational{Num: 3003, Den: 2}) {
		t.Fatalf("fps/bitrate RTPFrameDuration=%+v err=%v", got, err)
	}
	appendFrame("fps bitrate delta", 2, false, false)

	cqpChange := enc.Config()
	cqpChange.RateControl = goav1.EncoderRateControlCQP
	cqpChange.Quantizer = 35
	cqpChange.Content = goav1.EncoderContentScreen
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP/screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	appendFrame("CQP screen delta", 3, false, true)

	scalabilityChange := enc.Config()
	scalabilityChange.Scalability = goav1.EncoderScalabilityModeL1T3
	scalabilityChange.MaxFramerate = goav1.EncoderRational{Num: 24, Den: 1}
	scalabilityChange.Quantizer = 31
	if err := enc.SetConfig(scalabilityChange); err != nil {
		t.Fatalf("SetConfig scalability screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (goav1.EncoderRational{Num: 3750, Den: 1}) {
		t.Fatalf("scalability RTPFrameDuration=%+v err=%v", got, err)
	}
	appendFrame("L1T3 screen key", 4, true, true)
	appendFrame("L1T3 screen delta", 5, false, true)

	cbrChange := enc.Config()
	cbrChange.RateControl = goav1.EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = goav1.EncoderRational{Num: 120, Den: 1}
	cbrChange.Content = goav1.EncoderContentCamera
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
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeS3T3h)
	cfg.MaxFramerate = goav1.EncoderRational{Num: 30, Den: 1}
	publicRTCApplyControlBitrates(&cfg, 1500)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	normalized := enc.Config()
	spatialLayers, _, _, ok := normalized.Scalability.Layers()
	if !ok || spatialLayers != 3 {
		t.Fatalf("normalized scalability=%s spatial=%d ok=%v", normalized.Scalability, spatialLayers, ok)
	}
	var layerRes [goav1.EncoderWebRTCMaxSpatialLayers]goav1.EncoderResolution
	for i := uint8(0); i < spatialLayers; i++ {
		layerRes[i] = normalized.SpatialLayers[i].Resolution
	}

	var receiver goav1.RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	var layerFrames [goav1.EncoderWebRTCMaxSpatialLayers][]publicIVFFrame
	var layerSequences [goav1.EncoderWebRTCMaxSpatialLayers]goav1.SequenceHeader
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
	fpsBitrateChange.MaxFramerate = goav1.EncoderRational{Num: 60000, Den: 1001}
	publicRTCApplyControlBitrates(&fpsBitrateChange, 1800)
	if err := enc.SetConfig(fpsBitrateChange); err != nil {
		t.Fatalf("SetConfig fps/bitrate: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
	appendPicture("S3T3h fps bitrate delta", 2, false, false)

	cqpChange := enc.Config()
	cqpChange.RateControl = goav1.EncoderRateControlCQP
	cqpChange.Quantizer = 33
	cqpChange.Content = goav1.EncoderContentScreen
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP/screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	appendPicture("S3T3h CQP screen delta", 3, false, true)

	scalabilityChange := enc.Config()
	scalabilityChange.Scalability = goav1.EncoderScalabilityModeS3T2h
	scalabilityChange.MaxFramerate = goav1.EncoderRational{Num: 24, Den: 1}
	scalabilityChange.RateControl = goav1.EncoderRateControlCQP
	scalabilityChange.Quantizer = 29
	if err := enc.SetConfig(scalabilityChange); err != nil {
		t.Fatalf("SetConfig scalability screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
	appendPicture("S3T2h screen structure key", 4, true, true)
	appendPicture("S3T2h CQP screen delta", 5, false, true)

	cbrChange := enc.Config()
	cbrChange.RateControl = goav1.EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = goav1.EncoderRational{Num: 120, Den: 1}
	cbrChange.Content = goav1.EncoderContentCamera
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
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL2T2_KEY_SHIFT)
	cfg.MaxFramerate = goav1.EncoderRational{Num: 30, Den: 1}
	publicRTCApplyControlBitrates(&cfg, 900)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var descriptorReceiver goav1.RTPDependencyDescriptorState
	var activeReceiver goav1.RTPDependencyDescriptorState
	activeLimits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 48}
	nextFrameID := uint64(0)
	var sequence goav1.SequenceHeader
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
	fpsBitrateChange.MaxFramerate = goav1.EncoderRational{Num: 60000, Den: 1001}
	publicRTCApplyControlBitrates(&fpsBitrateChange, 1100)
	if err := enc.SetConfig(fpsBitrateChange); err != nil {
		t.Fatalf("SetConfig fps/bitrate: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), fpsBitrateChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (goav1.EncoderRational{Num: 3003, Den: 2}) {
		t.Fatalf("fps/bitrate RTPFrameDuration=%+v err=%v", got, err)
	}
	appendPicture("L2T2_KEY_SHIFT fps bitrate delta", 2, false, false, false)

	cqpChange := enc.Config()
	cqpChange.RateControl = goav1.EncoderRateControlCQP
	cqpChange.Quantizer = 32
	cqpChange.Content = goav1.EncoderContentScreen
	if err := enc.SetConfig(cqpChange); err != nil {
		t.Fatalf("SetConfig CQP/screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cqpChange)
	appendPicture("L2T2_KEY_SHIFT CQP screen delta", 3, false, false, true)
	appendPicture("L2T2_KEY_SHIFT forced screen key", 4, true, true, true)

	scalabilityChange := enc.Config()
	scalabilityChange.Scalability = goav1.EncoderScalabilityModeL2T3_KEY_SHIFT
	scalabilityChange.MaxFramerate = goav1.EncoderRational{Num: 24, Den: 1}
	scalabilityChange.Quantizer = 29
	if err := enc.SetConfig(scalabilityChange); err != nil {
		t.Fatalf("SetConfig scalability screen: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), scalabilityChange)
	if got, err := enc.RTPFrameDuration(); err != nil || got != (goav1.EncoderRational{Num: 3750, Den: 1}) {
		t.Fatalf("scalability RTPFrameDuration=%+v err=%v", got, err)
	}
	appendPicture("L2T3_KEY_SHIFT screen structure key", 5, false, true, true)
	appendPicture("L2T3_KEY_SHIFT CQP screen delta", 6, false, false, true)

	cbrChange := enc.Config()
	cbrChange.RateControl = goav1.EncoderRateControlCBR
	cbrChange.Quantizer = 0
	cbrChange.MaxFramerate = goav1.EncoderRational{Num: 120, Den: 1}
	cbrChange.Content = goav1.EncoderContentCamera
	publicRTCApplyControlBitrates(&cbrChange, 780)
	if err := enc.SetConfig(cbrChange); err != nil {
		t.Fatalf("SetConfig CBR/camera: %v", err)
	}
	assertPublicRTCConfigControls(t, enc.Config(), cbrChange)
	appendPicture("L2T3_KEY_SHIFT CBR camera delta", 7, false, false, false)

	wantYUV, decodedCount := decodePublicRTCLayerPoolLowOverheadRawYUV(t, lowOverheads...)
	if decodedCount != len(lowOverheads) {
		t.Fatalf("shared-SVC decoded frames=%d want %d", decodedCount, len(lowOverheads))
	}
	ivf := appendPublicIVF(nil, w, h, 30, 1, frames)
	assertPublicIVFMatchesReferenceDecodersRawYUVBytes(t, decoders, "shared-svc-settings", ivf, wantYUV, decodedCount)
}

func TestPublicRTCEncoderScalabilityModeCatalogueReferenceDecoders(t *testing.T) {
	decoders := publicReferenceAV1Decoders(t)
	modes := goav1.EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fpsCycle := []goav1.EncoderRational{
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
				cfg.RateControl = goav1.EncoderRateControlCQP
				cfg.Quantizer = uint8(24 + step%31)
			}
			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
			}
			defer enc.Close()

			var descriptorReceiver goav1.RTPDependencyDescriptorState
			nextFrameID := uint64(0)
			var layerFrames [goav1.EncoderWebRTCMaxSpatialLayers][]publicIVFFrame
			var layerSequences [goav1.EncoderWebRTCMaxSpatialLayers]goav1.SequenceHeader
			var orderedFrames []publicIVFFrame
			var lowOverheads [][]byte
			var sharedSequence goav1.SequenceHeader
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
					if frame.SpatialID >= goav1.EncoderWebRTCMaxSpatialLayers {
						t.Fatalf("%s spatial id=%d", label, frame.SpatialID)
					}
					assertPublicRTCFrameScreenContentHeader(t, fmt.Sprintf("%s S%d", label, frame.SpatialID), payload, &layerSequences[frame.SpatialID], wantScreen)
					layerFrames[frame.SpatialID] = append(layerFrames[frame.SpatialID], publicIVFFrame{
						timestamp: uint64(len(layerFrames[frame.SpatialID])),
						payload:   payload,
					})
				}
			}

			appendPicture("initial key", 0, false, true, cfg.Content == goav1.EncoderContentScreen)

			controlChange := enc.Config()
			controlChange.MaxFramerate = fpsCycle[(step+1)%len(fpsCycle)]
			if controlChange.RateControl == goav1.EncoderRateControlCQP {
				controlChange.RateControl = goav1.EncoderRateControlCBR
				controlChange.Quantizer = 0
				publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*11)+57)
			} else {
				controlChange.RateControl = goav1.EncoderRateControlCQP
				controlChange.Quantizer = uint8(29 + step%27)
			}
			if step%3 == 0 {
				controlChange.Content = goav1.EncoderContentScreen
			} else {
				controlChange.Content = goav1.EncoderContentCamera
			}
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig(%s control): %v", mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			appendPicture("control delta", 1, false, false, controlChange.Content == goav1.EncoderContentScreen)
			appendPicture("forced key", 2, true, true, controlChange.Content == goav1.EncoderContentScreen)

			secondControlChange := enc.Config()
			secondControlChange.MaxFramerate = fpsCycle[(step+2)%len(fpsCycle)]
			if secondControlChange.RateControl == goav1.EncoderRateControlCQP {
				secondControlChange.RateControl = goav1.EncoderRateControlCBR
				secondControlChange.Quantizer = 0
				publicRTCApplyControlBitrates(&secondControlChange, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*13)+91)
			} else {
				secondControlChange.RateControl = goav1.EncoderRateControlCQP
				secondControlChange.Quantizer = uint8(31 + step%23)
			}
			if secondControlChange.Content == goav1.EncoderContentScreen {
				secondControlChange.Content = goav1.EncoderContentCamera
			} else {
				secondControlChange.Content = goav1.EncoderContentScreen
			}
			if err := enc.SetConfig(secondControlChange); err != nil {
				t.Fatalf("SetConfig(%s second control): %v", mode, err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), secondControlChange)
			appendPicture("second control delta", 3, false, false, secondControlChange.Content == goav1.EncoderContentScreen)

			normalized := enc.Config()
			if publicRTCSharedReferenceSlotMode(mode) {
				wantYUV, decodedCount := decodePublicRTCLayerPoolLowOverheadRawYUV(t, lowOverheads...)
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

func assertPublicRTCFrameScreenContentHeader(
	t *testing.T, label string, frameData []byte, sequence *goav1.SequenceHeader, wantScreen bool,
) {
	t.Helper()
	it := goav1.NewLowOverheadIterator(frameData)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("%s OBU iteration: %v", label, err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case goav1.OBUSequenceHeader:
			seq, err := goav1.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("%s ParseSequenceHeader: %v", label, err)
			}
			*sequence = seq
		case goav1.OBUFrameHeader, goav1.OBUFrame:
			if sequence.ColorConfig.BitDepth == 0 {
				t.Fatalf("%s frame header appeared before sequence header", label)
			}
			prefix, err := goav1.ParseFrameHeaderPrefix(unit.Payload, *sequence)
			if err != nil {
				t.Fatalf("%s ParseFrameHeaderPrefix: %v", label, err)
			}
			if prefix.AllowScreenContentTools != wantScreen || (wantScreen && !prefix.ForceIntegerMV) {
				t.Fatalf("%s screen flags allow=%v integerMV=%v want screen=%v prefix=%+v",
					label, prefix.AllowScreenContentTools, prefix.ForceIntegerMV, wantScreen, prefix)
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
	decoded, err := goav1.DecodeIVF(ivf)
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
	modes := []goav1.EncoderScalabilityMode{
		goav1.EncoderScalabilityModeL1T2,
		goav1.EncoderScalabilityModeS2T2,
		goav1.EncoderScalabilityModeL3T2_KEY_SHIFT,
		goav1.EncoderScalabilityModeL1T3,
	}
	fps := []goav1.EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
	}
	targets := []int32{520, 940, 1500, 640}

	cfg := publicRTCMatrixConfig(w, h, modes[0])
	cfg.MaxFramerate = fps[0]
	publicRTCApplyControlBitrates(&cfg, targets[0])
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig(%s): %v", modes[0], err)
	}
	defer enc.Close()

	var receiver goav1.RTPDependencyDescriptorState
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

		var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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
	cfg := publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL2T2)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
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
	assertPublicRTCFrameScalabilityMetadata(t, picture.Frames[0].Data, goav1.MetadataScalabilityModeL2T2, true)

	ssCfg := publicRTCMatrixConfig(960, 540, goav1.EncoderScalabilityModeL3T1h)
	ssEnc, err := goav1.NewRTCEncoderWithConfig(ssCfg)
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

	plain, err := goav1.NewRTCEncoderWithConfig(publicRTCMatrixConfig(w, h, goav1.EncoderScalabilityModeL1T1))
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
	modes := goav1.EncoderWebRTCScalabilityModes()
	if len(modes) == 0 {
		t.Fatal("no WebRTC scalability modes")
	}
	fps := []goav1.EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 15, Den: 1},
		{Num: 120, Den: 1},
	}

	var enc *goav1.RTCEncoder
	var descriptorReceiver goav1.RTPDependencyDescriptorState
	var rtpReceiver goav1.RTPDependencyDescriptorState
	var activeLayerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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
			cfg.RateControl = goav1.EncoderRateControlCQP
			cfg.Quantizer = uint8(24 + step%31)
		}

		wantKey := true
		if enc == nil {
			var err error
			enc, err = goav1.NewRTCEncoderWithConfig(cfg)
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
			activeLayerTUs = [goav1.EncoderWebRTCMaxSpatialLayers][][]byte{}
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
		if controlChange.RateControl == goav1.EncoderRateControlCQP {
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

func TestPublicRTCSharedReferenceSVCDecodeRTPPayloads(t *testing.T) {
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 24}
	covered := 0
	for step, mode := range goav1.EncoderWebRTCScalabilityModes() {
		if !publicRTCSharedReferenceSlotMode(mode) {
			continue
		}
		covered++
		t.Run(mode.String(), func(t *testing.T) {
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			cfg.MaxFramerate = []goav1.EncoderRational{
				{Num: 30, Den: 1},
				{Num: 60, Den: 1},
				{Num: 30000, Den: 1001},
				{Num: 24, Den: 1},
			}[step%4]
			publicRTCApplyControlBitrates(&cfg, publicRTCMatrixControlBitrateKbps(t, mode)+int32(step*13))
			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
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
					controlChange.MaxFramerate = goav1.EncoderRational{Num: 120, Den: 1}
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

			want := decodePublicRTCLayerPoolLowOverheadDigests(t, lowOverheads...)
			got := decodePublicRTCLayerPoolRTPPayloadDigestsWithLabels(t, rtpPayloadLabels, rtpPayloads...)
			gotLayeredLow := decodePublicLayeredDecoderLowOverheadDigests(t, lowOverheads...)
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
			if len(gotLayeredLow) != len(want) || len(gotLayeredQueued) != len(want) || len(gotLayeredLive) != len(want) ||
				len(gotLayeredQueuedPackets) != len(want) || len(gotLayeredLivePackets) != len(want) {
				t.Fatalf("layered decoded frames low=%d payload queued/live=%d/%d packet queued/live=%d/%d want=%d",
					len(gotLayeredLow), len(gotLayeredQueued), len(gotLayeredLive), len(gotLayeredQueuedPackets), len(gotLayeredLivePackets), len(want))
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
				if gotLayeredLow[i] != want[i] {
					t.Fatalf("frame %d layered low-overhead digest differs: got=%x want=%x", i, gotLayeredLow[i], want[i])
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

func TestPublicLayeredDecoderRTPPayloadAfterLossSharedSVC(t *testing.T) {
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 24}
	mode := goav1.EncoderScalabilityModeL2T1
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
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

	dec, err := goav1.NewLayeredDecoderFromRTPPayloads(probePayloads)
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
	want := decodePublicRTCLayerPoolLowOverheadDigests(t, wantPayloads...)
	if len(got) != len(want) {
		t.Fatalf("decoded frames got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frame %d digest differs after loss: got=%x want=%x", i, got[i], want[i])
		}
	}

	packetDec, err := goav1.NewLayeredDecoderFromRTPPackets(probePackets)
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

func TestPublicLayeredDecoderRTPPacketSequencerAfterLossSharedSVC(t *testing.T) {
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 24}
	mode := goav1.EncoderScalabilityModeL2T1
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
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

	sequencer, err := goav1.BindRTPPacketSequencer(make([]goav1.RTPPacketSequencerSlot, 2048))
	if err != nil {
		t.Fatalf("BindRTPPacketSequencer: %v", err)
	}
	dec, err := goav1.NewLayeredDecoderFromRTPPackets(probePackets)
	if err != nil {
		t.Fatalf("NewLayeredDecoderFromRTPPackets: %v", err)
	}
	defer dec.Close()

	var got [][16]byte
	var gotMetadata []publicLayeredFrameMetadata
	decodeEvents := func(label string, events []goav1.RTPSequencedPacket) {
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

	events := make([]goav1.RTPSequencedPacket, 0, 16)
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
	for _, picture := range []goav1.RTCPicture{key0, key2} {
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
	mode := goav1.EncoderScalabilityModeL3T2h
	width, height := publicRTCMatrixGeometry(t, mode)
	enc, err := goav1.NewRTCEncoderWithConfig(publicRTCMatrixConfig(width, height, mode))
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	defer enc.Close()

	var receiver goav1.RTPDependencyDescriptorState
	nextFrameID := uint64(0)
	var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
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

	compound, err := goav1.AppendRTCPSenderReportPacket(make([]byte, 0, 128), goav1.RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	entry := testAV1RTCPValidLayerRefreshEntry(t, mode)
	compound, err = goav1.AppendRTCPFeedbackPacket(compound, goav1.RTCPFeedbackPacket{
		PacketType: goav1.RTCPPSFBPacketType,
		FMT:        goav1.RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  entry.SSRC,
		FCI:        testAV1RTCPLayerRefreshFCI(t, []goav1.AV1RTCPLayerRefreshRequestEntry{entry}),
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket LRR: %v", err)
	}

	forceKey, packets, err := goav1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		enc.Config(),
		compound,
		make([]goav1.RTCPPacket, 0, 2),
		nil,
		make([]goav1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame: %v", err)
	}
	if !forceKey || len(packets) != 2 {
		t.Fatalf("forceKey=%v packet len=%d want true,2", forceKey, len(packets))
	}
	target, ok, targetPackets, err := goav1.EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
		enc.Config(),
		compound,
		make([]goav1.RTCPPacket, 0, 2),
		make([]goav1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
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
	var activeReceiver goav1.RTPDependencyDescriptorState
	for i := 0; i < recovery.FrameNum; i++ {
		assertPublicRTCFrameRTPPacketsWithActiveDecodeTargets(t, &activeReceiver, recovery.Frames[i], goav1.RTPPayloadSizeLimits{MaxPayloadLen: 48}, options)
	}
	assertPublicRTCLayerStreamsDecode(t, enc.Config(), layerTUs, orderedTUs)
}

func TestPublicRTCEncoderSettingsMatrixDependencyDescriptors(t *testing.T) {
	for _, initial := range goav1.EncoderWebRTCScalabilityModes() {
		initial := initial
		reconfig := publicRTCMatrixReconfigMode(t, initial)
		width, height := publicRTCMatrixGeometry(t, initial)
		t.Run(initial.String()+"-to-"+reconfig.String(), func(t *testing.T) {
			cfg := publicRTCMatrixConfig(width, height, initial)
			enc, err := goav1.NewRTCEncoderWithConfig(cfg)
			if err != nil {
				t.Fatalf("NewRTCEncoderWithConfig(%s): %v", initial, err)
			}
			var receiver goav1.RTPDependencyDescriptorState
			var activeReceiver goav1.RTPDependencyDescriptorState
			activeLimits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 48}
			nextFrameID := uint64(0)
			var layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
			var orderedTUs [][]byte

			key, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 0), false)
			if err != nil {
				t.Fatalf("initial key EncodePicture: %v", err)
			}
			appendPublicRTCPictureLayerData(t, &layerTUs, &orderedTUs, key)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), key, true, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), key, activeLimits)

			controlChange := enc.Config()
			controlChange.MaxFramerate = goav1.EncoderRational{Num: 60, Den: 1}
			publicRTCApplyControlBitrates(&controlChange, publicRTCMatrixControlBitrateKbps(t, initial))
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig control change: %v", err)
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			controlDelta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
			if err != nil {
				t.Fatalf("control delta EncodePicture: %v", err)
			}
			appendPublicRTCPictureLayerData(t, &layerTUs, &orderedTUs, controlDelta)
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
			var reconfigLayerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte
			var reconfigOrderedTUs [][]byte
			structureKey, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), false)
			if err != nil {
				t.Fatalf("structure key EncodePicture: %v", err)
			}
			appendPublicRTCPictureLayerData(t, &reconfigLayerTUs, &reconfigOrderedTUs, structureKey)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), structureKey, true, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), structureKey, activeLimits)

			postReconfigDelta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 3), false)
			if err != nil {
				t.Fatalf("post-reconfigure delta EncodePicture: %v", err)
			}
			appendPublicRTCPictureLayerData(t, &reconfigLayerTUs, &reconfigOrderedTUs, postReconfigDelta)
			assertPublicRTCPictureDescriptors(t, &receiver, enc.Config(), postReconfigDelta, false, &nextFrameID)
			assertPublicRTCPictureActiveDecodeTargetOptions(t, &activeReceiver, enc.Config(), postReconfigDelta, activeLimits)
			assertPublicRTCLayerStreamsDecode(t, enc.Config(), reconfigLayerTUs, reconfigOrderedTUs)
		})
	}
}

func publicRTCMatrixGeometry(t *testing.T, mode goav1.EncoderScalabilityMode) (int, int) {
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

func publicRTCMatrixReconfigMode(t *testing.T, mode goav1.EncoderScalabilityMode) goav1.EncoderScalabilityMode {
	t.Helper()
	spatial, temporal, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", mode)
	}
	wantTemporal := temporal + 1
	if temporal >= 3 {
		wantTemporal = temporal - 1
	}
	for _, candidate := range goav1.EncoderWebRTCScalabilityModes() {
		cSpatial, cTemporal, _, ok := candidate.Layers()
		if ok && candidate != mode && cSpatial == spatial && cTemporal == wantTemporal {
			return candidate
		}
	}
	for _, candidate := range goav1.EncoderWebRTCScalabilityModes() {
		cSpatial, _, _, ok := candidate.Layers()
		if ok && candidate != mode && cSpatial == spatial {
			return candidate
		}
	}
	t.Fatalf("no reconfigure mode for %s", mode)
	return goav1.EncoderScalabilityModeL1T1
}

func publicRTCMatrixControlBitrateKbps(t *testing.T, mode goav1.EncoderScalabilityMode) int32 {
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

func publicRTCApplyControlBitrates(cfg *goav1.EncoderConfig, targetKbps int32) {
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

func assertPublicRTCConfigBitrates(t *testing.T, got goav1.EncoderConfig, want goav1.EncoderConfig) {
	t.Helper()
	spatialLayers, _, _, ok := got.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid normalized mode %s", got.Scalability)
	}
	want, err := goav1.SetWebRTCEncoderSVCConfig(want, want.TemporalLayerCount, want.SpatialLayerCount)
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

func assertPublicRTCConfigControls(t *testing.T, got goav1.EncoderConfig, want goav1.EncoderConfig) {
	t.Helper()
	want, err := goav1.SetWebRTCEncoderSVCConfig(want, want.TemporalLayerCount, want.SpatialLayerCount)
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
	gotDuration, err := goav1.EncoderWebRTCRTPFrameDuration(got)
	if err != nil {
		t.Fatalf("got RTP frame duration: %v", err)
	}
	wantDuration, err := goav1.EncoderWebRTCRTPFrameDuration(want)
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

func publicRTCSetConfigRequiresKey(t *testing.T, before goav1.EncoderConfig, after goav1.EncoderConfig) bool {
	t.Helper()
	normalizedBefore, err := goav1.SetWebRTCEncoderSVCConfig(before, before.TemporalLayerCount, before.SpatialLayerCount)
	if err != nil {
		t.Fatalf("normalize previous config %s: %v", before.Scalability, err)
	}
	normalizedAfter, err := goav1.SetWebRTCEncoderSVCConfig(after, after.TemporalLayerCount, after.SpatialLayerCount)
	if err != nil {
		t.Fatalf("normalize next config %s: %v", after.Scalability, err)
	}
	if !publicRTCConfigLayerGeometryEqual(normalizedBefore, normalizedAfter) {
		return true
	}
	beforeStructure, err := goav1.EncoderWebRTCFrameDependencyStructureForConfig(normalizedBefore)
	if err != nil {
		t.Fatalf("previous dependency structure %s: %v", normalizedBefore.Scalability, err)
	}
	afterStructure, err := goav1.EncoderWebRTCFrameDependencyStructureForConfig(normalizedAfter)
	if err != nil {
		t.Fatalf("next dependency structure %s: %v", normalizedAfter.Scalability, err)
	}
	return beforeStructure != afterStructure
}

func publicRTCConfigLayerGeometryEqual(a goav1.EncoderConfig, b goav1.EncoderConfig) bool {
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

func cloneRTCPictureData(p *goav1.RTCPicture) {
	for i := 0; i < p.FrameNum; i++ {
		p.Frames[i].Data = append([]byte(nil), p.Frames[i].Data...)
		p.Frames[i].DependencyDescriptor = append([]byte(nil), p.Frames[i].DependencyDescriptor...)
	}
}

func publicRTCMatrixConfig(width, height int, mode goav1.EncoderScalabilityMode) goav1.EncoderConfig {
	return goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: int32(width), Height: int32(height)},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    120,
		MaxBitrateKbps:    1600,
		TargetBitrateKbps: 900,
		Scalability:       mode,
	}
}

func publicRTCMatrixFrame(width, height int, n int) goav1.I420Frame {
	cw, ch := width/2, height/2
	f := goav1.I420Frame{
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

func publicI420NeutralChroma(src goav1.I420Frame) goav1.I420Frame {
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

func publicI422FromI420(src goav1.I420Frame) goav1.I422Frame {
	cw := src.Width / 2
	u := make([]byte, cw*src.Height)
	v := make([]byte, cw*src.Height)
	for y := 0; y < src.Height; y++ {
		srcY := (y / 2) * src.ChromaStride
		copy(u[y*cw:y*cw+cw], src.U[srcY:srcY+cw])
		copy(v[y*cw:y*cw+cw], src.V[srcY:srcY+cw])
	}
	return goav1.I422Frame{
		Y:            src.Y,
		U:            u,
		V:            v,
		YStride:      src.YStride,
		ChromaStride: cw,
		Width:        src.Width,
		Height:       src.Height,
	}
}

func publicI444FromI420(src goav1.I420Frame) goav1.I444Frame {
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
	return goav1.I444Frame{
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

func publicI400FromI420(src goav1.I420Frame) goav1.I400Frame {
	return goav1.I400Frame{
		Y:       src.Y,
		YStride: src.YStride,
		Width:   src.Width,
		Height:  src.Height,
	}
}

func publicFrameFromI420(tb testing.TB, src goav1.I420Frame, format goav1.FrameFormat) goav1.Frame {
	tb.Helper()
	layout, err := goav1.FrameRequiredSize(format)
	if err != nil {
		tb.Fatalf("FrameRequiredSize: %v", err)
	}
	frame, err := goav1.BindFrame(make([]byte, layout.Size), format)
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

func publicNV12FromI420(src goav1.I420Frame) goav1.NV12Frame {
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
	return goav1.NV12Frame{
		Y:        src.Y,
		UV:       uv,
		YStride:  src.YStride,
		UVStride: src.Width,
		Width:    src.Width,
		Height:   src.Height,
	}
}

func publicNV21FromI420(src goav1.I420Frame) goav1.NV21Frame {
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
	return goav1.NV21Frame{
		Y:        src.Y,
		VU:       vu,
		YStride:  src.YStride,
		VUStride: src.Width,
		Width:    src.Width,
		Height:   src.Height,
	}
}

func assertPublicRTCPictureDescriptors(t *testing.T, receiver *goav1.RTPDependencyDescriptorState, cfg goav1.EncoderConfig, picture goav1.RTCPicture, wantKey bool, nextFrameID *uint64) {
	t.Helper()
	normalized, err := goav1.SetWebRTCEncoderSVCConfig(cfg, cfg.TemporalLayerCount, cfg.SpatialLayerCount)
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

func assertPublicRTCFrameRTPPackets(t *testing.T, receiver *goav1.RTPDependencyDescriptorState, frame goav1.RTCFrame, limits goav1.RTPPayloadSizeLimits, wantAttachedStructure bool, wantNewCodedVideoSequence bool, wantMarker bool) {
	t.Helper()
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if size.Packetizer.Packets <= 1 {
		t.Fatalf("packetizer packets=%d want fragmented frame S%d", size.Packetizer.Packets, frame.SpatialID)
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
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
		header, _, err := goav1.ParseRTPAggregationHeader(payloadSlices[i])
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

func assertPublicRTCPictureActiveDecodeTargetOptions(t *testing.T, receiver *goav1.RTPDependencyDescriptorState, cfg goav1.EncoderConfig, picture goav1.RTCPicture, limits goav1.RTPPayloadSizeLimits) {
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
	if _, err := picture.ActiveDecodeTargetsMask(spatialLayers, 0); !errors.Is(err, goav1.ErrEncoderInvalidFrame) {
		t.Fatalf("%s invalid spatial active mask err=%v want %v", cfg.Scalability, err, goav1.ErrEncoderInvalidFrame)
	}
	if _, err := picture.ActiveDecodeTargetsMask(0, temporalLayers); !errors.Is(err, goav1.ErrEncoderInvalidFrame) {
		t.Fatalf("%s invalid temporal active mask err=%v want %v", cfg.Scalability, err, goav1.ErrEncoderInvalidFrame)
	}
}

func assertPublicRTCFrameRTPPacketsWithActiveDecodeTargets(t *testing.T, receiver *goav1.RTPDependencyDescriptorState, frame goav1.RTCFrame, limits goav1.RTPPayloadSizeLimits, options goav1.EncoderWebRTCRTPPacketDependencyDescriptorOptions) {
	t.Helper()
	firstSize, err := frame.RTPPacketScratchLenWithOptions(limits, nil, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions first S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLenWithOptions(limits, obuScratch, options)
	if err != nil {
		t.Fatalf("RTPPacketScratchLenWithOptions full S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
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
	assembledLen, obuCount, err := goav1.AssembleRTPFrameSize(payloadSlices)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	assembled := make([]byte, assembledLen)
	assembledOBUs := make([]goav1.RTPFrameOBU, obuCount)
	wrote, gotOBUs, err := goav1.AssembleRTPFrame(assembled, payloadSlices, assembledOBUs)
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	var expected []byte
	it := goav1.NewLowOverheadIterator(frameData)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("source OBU iteration: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case goav1.OBUTemporalDelimiter, goav1.OBUTileList, goav1.OBUPadding:
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
	it := goav1.NewLowOverheadIterator(frameData)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != goav1.OBUTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := it.Next(); err != nil || !ok || seq.Header.Type != goav1.OBUSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	next, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("next OBU ok=%v err=%v header=%+v", ok, err, next.Header)
	}
	if !wantPresent {
		if next.Header.Type == goav1.OBUMetadata {
			t.Fatalf("unexpected metadata payload=% x", next.Payload)
		}
		if next.Header.Type != goav1.OBUFrameHeader && next.Header.Type != goav1.OBUFrame {
			t.Fatalf("next OBU after sequence=%+v", next.Header)
		}
		return
	}
	if next.Header.Type != goav1.OBUMetadata {
		t.Fatalf("metadata header=%+v", next.Header)
	}
	meta, err := goav1.ParseMetadataOBU(next.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != goav1.MetadataTypeScalability || meta.Scalability.ModeIDC != wantIDC || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v want idc=%d", meta, wantIDC)
	}
	frame, ok, err := it.Next()
	if err != nil || !ok || (frame.Header.Type != goav1.OBUFrameHeader && frame.Header.Type != goav1.OBUFrame) {
		t.Fatalf("frame after metadata ok=%v err=%v header=%+v", ok, err, frame.Header)
	}
}

func assertPublicRTCFrameScalabilitySSMetadata(t *testing.T, frameData []byte, widths []uint16, heights []uint16, refIDs []uint8, groupTIDs []uint8, groupDiffs []uint8) {
	t.Helper()
	it := goav1.NewLowOverheadIterator(frameData)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != goav1.OBUTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := it.Next(); err != nil || !ok || seq.Header.Type != goav1.OBUSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	next, ok, err := it.Next()
	if err != nil || !ok || next.Header.Type != goav1.OBUMetadata {
		t.Fatalf("metadata ok=%v err=%v header=%+v", ok, err, next.Header)
	}
	meta, err := goav1.ParseMetadataOBU(next.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != goav1.MetadataTypeScalability ||
		meta.Scalability.ModeIDC != goav1.MetadataScalabilityModeSS ||
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
	if err != nil || !ok || (frame.Header.Type != goav1.OBUFrameHeader && frame.Header.Type != goav1.OBUFrame) {
		t.Fatalf("frame after metadata ok=%v err=%v header=%+v", ok, err, frame.Header)
	}
}

func publicDecodedFramesRawYUV(frames []goav1.DecodedFrame) []byte {
	var out []byte
	for _, frame := range frames {
		out = append(out, frame.Y...)
		out = append(out, frame.U...)
		out = append(out, frame.V...)
	}
	return out
}

func appendPublicFrameRawYUV(dst []byte, frame *goav1.Frame) []byte {
	if frame == nil {
		return dst
	}
	bytesPerSample := frame.Layout.BytesPerSample
	dst = appendPublicFramePlaneRawYUV(dst, frame.Y, bytesPerSample)
	dst = appendPublicFramePlaneRawYUV(dst, frame.U, bytesPerSample)
	dst = appendPublicFramePlaneRawYUV(dst, frame.V, bytesPerSample)
	return dst
}

func appendPublicFramePlaneRawYUV(dst []byte, plane goav1.FramePlane, bytesPerSample int) []byte {
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

func assertPublicRTCAttachedStructure(t *testing.T, structure goav1.RTPDependencyDescriptorStructure, cfg goav1.EncoderConfig) {
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

func appendPublicRTCPictureLayerData(t *testing.T, layerTUs *[goav1.EncoderWebRTCMaxSpatialLayers][][]byte, orderedTUs *[][]byte, picture goav1.RTCPicture) {
	t.Helper()
	for i := 0; i < picture.FrameNum; i++ {
		spatialID := picture.Frames[i].SpatialID
		if spatialID >= goav1.EncoderWebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", i, spatialID)
		}
		tu := append([]byte(nil), picture.Frames[i].Data...)
		layerTUs[spatialID] = append(layerTUs[spatialID], tu)
		*orderedTUs = append(*orderedTUs, tu)
	}
}

func appendPublicRTCPictureRTPData(t *testing.T, receiver *goav1.RTPDependencyDescriptorState, layerTUs *[goav1.EncoderWebRTCMaxSpatialLayers][][]byte, orderedTUs *[][]byte, picture goav1.RTCPicture) {
	t.Helper()
	for i := 0; i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		spatialID := frame.SpatialID
		if spatialID >= goav1.EncoderWebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", i, spatialID)
		}
		tu := publicRTCPacketizeAndAssembleFrame(t, receiver, frame)
		layerTUs[spatialID] = append(layerTUs[spatialID], tu)
		*orderedTUs = append(*orderedTUs, tu)
	}
}

func publicRTCPacketizeAndAssembleFrame(t *testing.T, receiver *goav1.RTPDependencyDescriptorState, frame goav1.RTCFrame) []byte {
	t.Helper()
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
	const dependencyDescriptorExtensionID = 42
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	headerConfig := goav1.EncoderWebRTCRTPPacketHeaderConfig{
		PayloadType:                     96,
		SequenceNumber:                  uint16(frame.FrameID * 29),
		Timestamp:                       uint32(frame.FrameID * 3000),
		SSRC:                            0x01020304 + uint32(frame.SpatialID),
		DependencyDescriptorExtensionID: dependencyDescriptorExtensionID,
	}
	packetSize, err := goav1.EncoderWebRTCRTPPacketsWithHeadersSize(headerConfig, rtpPayloads, descriptors, spans[:packetCount])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTPPacketsWithHeadersSize S%d T%d: %v", frame.SpatialID, frame.TemporalID, err)
	}
	if packetSize.Packets != packetCount || packetSize.Bytes <= len(rtpPayloads) ||
		packetSize.MaxHeaderBytes <= goav1.RTPHeaderMinSize {
		t.Fatalf("full RTP size=%+v packetCount=%d payloadBytes=%d", packetSize, packetCount, len(rtpPayloads))
	}
	headerSpans := make([]goav1.EncoderWebRTCRTPPacketHeaderSpan, packetCount)
	fullPackets, fullCount, err := goav1.AppendEncoderWebRTCRTPPacketsWithHeaders(make([]byte, 0, packetSize.Bytes), headerSpans, headerConfig, rtpPayloads, descriptors, spans[:packetCount])
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
		descriptorPacket, err := goav1.ParseRTPPacketDependencyDescriptor(raw, dependencyDescriptorExtensionID, receiver)
		if err != nil {
			t.Fatalf("packet %d ParseRTPPacketDependencyDescriptor S%d: %v", i, frame.SpatialID, err)
		}
		packet := descriptorPacket.Packet
		if packet.Header.PayloadType != headerConfig.PayloadType ||
			packet.Header.SequenceNumber != headerConfig.SequenceNumber+uint16(i) ||
			packet.Header.Timestamp != headerConfig.Timestamp ||
			packet.Header.SSRC != headerConfig.SSRC ||
			packet.Header.Marker != span.Marker ||
			packet.Header.ExtensionProfile != goav1.RTPExtensionProfileTwoByte {
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
		header, _, err := goav1.ParseRTPAggregationHeader(payloadSlices[i])
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
	assembledLen, obuCount, err := goav1.AssembleRTPFrameSize(payloadSlices)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	assembled := make([]byte, assembledLen)
	assembledOBUs := make([]goav1.RTPFrameOBU, obuCount)
	wrote, gotOBUs, err := goav1.AssembleRTPFrame(assembled, payloadSlices, assembledOBUs)
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	if wrote != assembledLen || gotOBUs != obuCount {
		t.Fatalf("assembled wrote=%d/%d obus=%d/%d", wrote, assembledLen, gotOBUs, obuCount)
	}
	return assembled[:wrote]
}

func assertPublicRTCLayerStreamsDecode(t *testing.T, cfg goav1.EncoderConfig, layerTUs [goav1.EncoderWebRTCMaxSpatialLayers][][]byte, orderedTUs [][]byte) {
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
	dec, err := goav1.NewDecoder(tus)
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

func publicRTCSharedReferenceSlotMode(mode goav1.EncoderScalabilityMode) bool {
	spatial, _, _, ok := mode.Layers()
	return ok && spatial > 1 && !mode.IsSimulcast()
}

type publicLayeredFrameMetadata struct {
	TemporalID    uint8
	SpatialID     uint8
	FrameType     goav1.FrameType
	CodedKeyframe bool
	ShowExisting  bool
	ShowFrame     bool
	CodedWidth    int
	Height        int
}

func publicLayeredFrameMetadataFromRTCFrame(frame goav1.RTCFrame) publicLayeredFrameMetadata {
	frameType := goav1.FrameTypeInter
	if frame.CodedKeyframe {
		frameType = goav1.FrameTypeKey
	}
	return publicLayeredFrameMetadata{
		TemporalID:    frame.TemporalID,
		SpatialID:     frame.SpatialID,
		FrameType:     frameType,
		CodedKeyframe: frame.CodedKeyframe,
		ShowFrame:     true,
	}
}

func publicLayeredFrameMetadataFromDecoded(t *testing.T, frame goav1.LayeredFrame) publicLayeredFrameMetadata {
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

func publicRTCPictureFramesInOrder(pictures ...goav1.RTCPicture) [][]byte {
	var out [][]byte
	for _, picture := range pictures {
		for i := 0; i < picture.FrameNum; i++ {
			out = append(out, append([]byte(nil), picture.Frames[i].Data...))
		}
	}
	return out
}

func publicRTCPictureRTPPayloads(t *testing.T, picture goav1.RTCPicture, limits goav1.RTPPayloadSizeLimits) [][]byte {
	t.Helper()
	var out [][]byte
	for i := 0; i < picture.FrameNum; i++ {
		out = append(out, publicDecoderRTPPayloadsForFrameWithLimits(t, picture.Frames[i], limits)...)
	}
	return out
}

func publicRTCPictureRTPPackets(t *testing.T, picture goav1.RTCPicture, limits goav1.RTPPayloadSizeLimits) [][]byte {
	t.Helper()
	var out [][]byte
	for i := 0; i < picture.FrameNum; i++ {
		out = append(out, publicDecoderRTPPacketsForFrameWithLimits(t, picture.Frames[i], limits)...)
	}
	return out
}

func frameDigestsVisible(frames []*goav1.Frame) [][16]byte {
	out := make([][16]byte, 0, len(frames))
	for _, frame := range frames {
		out = append(out, frameMD5Visible(frame))
	}
	return out
}

func assertPublicRTCSharedReferenceStreamDecodes(t *testing.T, tus [][]byte) {
	t.Helper()
	frames := decodePublicRTCLayerPoolLowOverheads(t, tus...)
	if len(frames) != len(tus) {
		t.Fatalf("shared-reference stream decoded %d frames, want %d", len(frames), len(tus))
	}
}

func decodePublicRTCLayerPoolLowOverheads(t *testing.T, payloads ...[]byte) []*goav1.Frame {
	t.Helper()

	h := newPublicRTCLayerPoolDecodeHarness(t, len(payloads))
	defer h.close(t)
	events := make([]goav1.DecoderEvent, 16)

	for payloadIndex, payload := range payloads {
		count, err := h.stream.PushLowOverhead(payload, events)
		if err != nil {
			t.Fatalf("payload %d PushLowOverhead: %v", payloadIndex, err)
		}
		h.runEvents(t, payloadIndex, events[:count])
	}
	return h.outputs
}

func decodePublicRTCLayerPoolRTPPayloads(t *testing.T, payloads ...[]byte) []*goav1.Frame {
	t.Helper()
	return decodePublicRTCLayerPoolRTPPayloadsWithLabels(t, nil, payloads...)
}

func decodePublicRTCLayerPoolRTPPayloadsWithLabels(t *testing.T, labels []string, payloads ...[]byte) []*goav1.Frame {
	t.Helper()

	h := newPublicRTCLayerPoolDecodeHarness(t, len(payloads))
	defer h.close(t)
	var (
		rtpUsed   int
		rtpBuffer []byte
		rtpSpans  []goav1.RTPObuSpan
		events    []goav1.DecoderEvent
	)

	for payloadIndex, payload := range payloads {
		plannedUsed, eventCount, err := h.stream.PushRTPPayloadSize(rtpUsed, payload)
		if err != nil {
			t.Fatalf("payload %d PushRTPPayloadSize: %v", payloadIndex, err)
		}
		if cap(rtpBuffer) < plannedUsed {
			next := make([]byte, plannedUsed)
			copy(next, rtpBuffer[:rtpUsed])
			rtpBuffer = next
		}
		rtpBuffer = rtpBuffer[:plannedUsed]
		if cap(rtpSpans) < eventCount {
			rtpSpans = make([]goav1.RTPObuSpan, eventCount)
		}
		rtpSpans = rtpSpans[:eventCount]
		if cap(events) < eventCount {
			events = make([]goav1.DecoderEvent, eventCount)
		}
		events = events[:eventCount]

		used, count, err := h.stream.PushRTPPayload(rtpBuffer, rtpUsed, rtpSpans, events, payload)
		if err != nil {
			t.Fatalf("payload %d %s PushRTPPayload used=%d planned=%d events=%d/%d inFragment=%v %s: %v",
				payloadIndex, publicRTCPayloadLabel(labels, payloadIndex), used, plannedUsed, count, eventCount, h.stream.InRTPFragment(), publicRTCRTPPayloadSummary(payload), err)
		}
		if used != plannedUsed || count > eventCount {
			t.Fatalf("payload %d RTP used/count=%d/%d planned=%d/%d", payloadIndex, used, count, plannedUsed, eventCount)
		}
		rtpUsed = used
		h.runEvents(t, payloadIndex, events[:count])
		if !h.stream.InRTPFragment() {
			rtpUsed = 0
		}
	}
	return h.outputs
}

func decodePublicRTCLayerPoolLowOverheadDigests(t *testing.T, payloads ...[]byte) [][16]byte {
	t.Helper()

	h := newPublicRTCLayerPoolDecodeHarness(t, len(payloads))
	defer h.close(t)
	events := make([]goav1.DecoderEvent, 16)
	var out [][16]byte

	for payloadIndex, payload := range payloads {
		count, err := h.stream.PushLowOverhead(payload, events)
		if err != nil {
			t.Fatalf("payload %d PushLowOverhead: %v", payloadIndex, err)
		}
		start := len(h.outputs)
		h.runEvents(t, payloadIndex, events[:count])
		for _, frame := range h.outputs[start:] {
			out = append(out, frameMD5Visible(frame))
		}
	}
	return out
}

func decodePublicRTCLayerPoolLowOverheadRawYUV(t *testing.T, payloads ...[]byte) ([]byte, int) {
	t.Helper()

	h := newPublicRTCLayerPoolDecodeHarness(t, len(payloads))
	defer h.close(t)
	events := make([]goav1.DecoderEvent, 16)
	var out []byte
	frames := 0

	for payloadIndex, payload := range payloads {
		count, err := h.stream.PushLowOverhead(payload, events)
		if err != nil {
			t.Fatalf("payload %d PushLowOverhead: %v", payloadIndex, err)
		}
		start := len(h.outputs)
		h.runEvents(t, payloadIndex, events[:count])
		for _, frame := range h.outputs[start:] {
			out = appendPublicFrameRawYUV(out, frame)
			frames++
		}
	}
	return out, frames
}

func decodePublicRTCLayerPoolRTPPayloadDigestsWithLabels(t *testing.T, labels []string, payloads ...[]byte) [][16]byte {
	t.Helper()

	h := newPublicRTCLayerPoolDecodeHarness(t, len(payloads))
	defer h.close(t)
	var (
		rtpUsed   int
		rtpBuffer []byte
		rtpSpans  []goav1.RTPObuSpan
		events    []goav1.DecoderEvent
		out       [][16]byte
	)

	for payloadIndex, payload := range payloads {
		plannedUsed, eventCount, err := h.stream.PushRTPPayloadSize(rtpUsed, payload)
		if err != nil {
			t.Fatalf("payload %d PushRTPPayloadSize: %v", payloadIndex, err)
		}
		if cap(rtpBuffer) < plannedUsed {
			next := make([]byte, plannedUsed)
			copy(next, rtpBuffer[:rtpUsed])
			rtpBuffer = next
		}
		rtpBuffer = rtpBuffer[:plannedUsed]
		if cap(rtpSpans) < eventCount {
			rtpSpans = make([]goav1.RTPObuSpan, eventCount)
		}
		rtpSpans = rtpSpans[:eventCount]
		if cap(events) < eventCount {
			events = make([]goav1.DecoderEvent, eventCount)
		}
		events = events[:eventCount]

		used, count, err := h.stream.PushRTPPayload(rtpBuffer, rtpUsed, rtpSpans, events, payload)
		if err != nil {
			t.Fatalf("payload %d %s PushRTPPayload used=%d planned=%d events=%d/%d inFragment=%v %s: %v",
				payloadIndex, publicRTCPayloadLabel(labels, payloadIndex), used, plannedUsed, count, eventCount, h.stream.InRTPFragment(), publicRTCRTPPayloadSummary(payload), err)
		}
		if used != plannedUsed || count > eventCount {
			t.Fatalf("payload %d RTP used/count=%d/%d planned=%d/%d", payloadIndex, used, count, plannedUsed, eventCount)
		}
		rtpUsed = used
		start := len(h.outputs)
		h.runEvents(t, payloadIndex, events[:count])
		for _, frame := range h.outputs[start:] {
			out = append(out, frameMD5Visible(frame))
		}
		if !h.stream.InRTPFragment() {
			rtpUsed = 0
		}
	}
	return out
}

func decodePublicLayeredDecoderLowOverheadDigests(t *testing.T, payloads ...[]byte) [][16]byte {
	t.Helper()
	dec, err := goav1.NewLayeredDecoder(payloads)
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
	dec, err := goav1.NewLayeredDecoder(payloads)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPayloads(payloads)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPayloads(payloads)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPackets(packets)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPackets(packets)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPayloads(probePayloads)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPackets(probePackets)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPayloads(probePayloads)
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
	dec, err := goav1.NewLayeredDecoderFromRTPPackets(probePackets)
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
	it, err := goav1.NewRTPPayloadIterator(payload)
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

type publicRTCLayerPoolDecodeHarness struct {
	workers    int
	workerPool *goav1.TileWorkerPool

	layerPool goav1.FrameLayerPool
	adapter   goav1.DecoderFrameLayerPool

	stream        goav1.DecoderStream
	refs          goav1.DecoderSurfaceReferences
	states        [goav1.EncoderWebRTCMaxSpatialLayers]goav1.DecoderFrameWorkState
	frameContexts goav1.DecoderSharedFrameContextStore
	stats         goav1.DecoderFrameWorkTileResidualStats
	postFilter    goav1.DecoderFrameWorkReusableSupportedPostFilterRunner
	sequence      goav1.SequenceHeader
	haveSequence  bool

	referenceSurfaces []int
	referenceFrames   []*goav1.Frame
	releases          []int
	planSpans         []goav1.TileSpan
	planJobs          []goav1.TileJob
	planBatches       []goav1.TileBatch
	outputs           []*goav1.Frame
}

func newPublicRTCLayerPoolDecodeHarness(t *testing.T, outputCap int) *publicRTCLayerPoolDecodeHarness {
	t.Helper()
	const workers = 1
	workerPool, err := goav1.NewTileWorkerPool(workers)
	if err != nil {
		t.Fatalf("NewTileWorkerPool: %v", err)
	}
	layerPool := newPublicDecoderLayerPool(t, goav1.EncoderWebRTCMaxSpatialLayers+1, goav1.RefFrames+1)
	h := &publicRTCLayerPoolDecodeHarness{
		workers:           workers,
		workerPool:        workerPool,
		layerPool:         layerPool,
		referenceSurfaces: make([]int, goav1.InterRefsPerFrame),
		referenceFrames:   make([]*goav1.Frame, goav1.InterRefsPerFrame),
		releases:          make([]int, goav1.RefFrames),
		planSpans:         make([]goav1.TileSpan, goav1.MaxTiles),
		planJobs:          make([]goav1.TileJob, goav1.MaxTiles),
		planBatches:       make([]goav1.TileBatch, goav1.MaxTiles),
		outputs:           make([]*goav1.Frame, 0, outputCap),
	}
	h.adapter = goav1.NewDecoderFrameLayerPool(&h.layerPool)
	return h
}

func (h *publicRTCLayerPoolDecodeHarness) close(t *testing.T) {
	t.Helper()
	h.workerPool.Close()
}

func (h *publicRTCLayerPoolDecodeHarness) stateForEvent(t *testing.T, payloadIndex, eventIndex int, event goav1.DecoderEvent) *goav1.DecoderFrameWorkState {
	t.Helper()
	if event.SpatialID >= goav1.EncoderWebRTCMaxSpatialLayers {
		t.Fatalf("payload %d event %d spatial id %d exceeds WebRTC maximum %d", payloadIndex, eventIndex, event.SpatialID, goav1.EncoderWebRTCMaxSpatialLayers)
	}
	return &h.states[event.SpatialID]
}

func (h *publicRTCLayerPoolDecodeHarness) runEvents(t *testing.T, payloadIndex int, events []goav1.DecoderEvent) {
	t.Helper()
	for eventIndex, event := range events {
		if event.Kind == goav1.DecoderEventSequenceHeader {
			h.sequence = event.SequenceHeader
			h.haveSequence = true
		} else if !h.haveSequence {
			if seq, ok := h.stream.SequenceHeader(); ok {
				h.sequence = seq
				h.haveSequence = true
			}
		}

		framePool := publicRTCLayerDecodeFramePool(t, &h.layerPool, h.sequence, event)
		size, err := goav1.DecoderFrameWorkResidualEventScratchLen(h.sequence, event, h.workers, h.planSpans, h.planJobs, h.planBatches)
		if err != nil {
			t.Fatalf("payload %d event %d scratch len: %v", payloadIndex, eventIndex, err)
		}
		scratch := publicRTCLayerDecodeEventScratch(size)

		var batchRunner goav1.DecoderFrameWorkBatchResidualRunner
		var batchRunnerPtr *goav1.DecoderFrameWorkBatchResidualRunner
		if size.Runner.Workers != 0 {
			batchRunner, err = goav1.BindDecoderFrameWorkBatchResidualRunner(size.Runner, scratch.Runner)
			if err != nil {
				t.Fatalf("payload %d event %d bind residual runner: %v", payloadIndex, eventIndex, err)
			}
			batchRunnerPtr = &batchRunner
		}

		var sideData goav1.DecoderFrameWorkSideData
		var sideDataPtr *goav1.DecoderFrameWorkSideData
		if publicRTCLayerDecodeEventNeedsSideData(event) {
			sideData, err = goav1.BindDecoderFrameWorkSideData(h.sequence, event.FrameSize, event.CDEF, event.Restoration, scratch.SideData)
			if err != nil {
				t.Fatalf("payload %d event %d bind side data: %v", payloadIndex, eventIndex, err)
			}
			sideDataPtr = &sideData
		}

		globalSurface := func(local int) int {
			if framePool == nil {
				return -1
			}
			return goav1.DecoderLayerPoolGlobalSurfaceID(&h.layerPool, framePool, local)
		}
		state := h.stateForEvent(t, payloadIndex, eventIndex, event)
		result, err := goav1.RunDecoderFrameWorkEventWithResidualRunner(goav1.DecoderFrameWorkResidualEventRequest{
			State:             state,
			Refs:              &h.refs,
			FramePool:         framePool,
			Sequence:          h.sequence,
			Event:             event,
			Align:             64,
			ReferenceSurfaces: h.referenceSurfaces,
			ReferenceFrames:   h.referenceFrames,
			Workers:           h.workers,
			Spans:             scratch.Spans,
			Jobs:              scratch.Jobs,
			Batches:           scratch.Batches,
			Releases:          h.releases,
			WorkerPool:        h.workerPool,
			Runner:            batchRunnerPtr,
			SideData:          sideDataPtr,
			PostRunner:        &h.postFilter,
			Stats:             &h.stats,
			External: goav1.DecoderFrameWorkExternalReferenceRuntime{
				Provider:      h.adapter,
				GlobalSurface: globalSurface,
				Releaser:      h.adapter,
				FrameContexts: &h.frameContexts,
			},
		})
		if err != nil {
			t.Fatalf("payload %d event %d run kind=%v type=%v newSeq=%v T%d S%d frameType=%v refresh=%#x surfaceRefs=%+v framePoolNil=%v stateSurface=%d frameSize=%+v: %v",
				payloadIndex, eventIndex, event.Kind, event.Type, event.NewCodedVideoSequence, event.TemporalID, event.SpatialID,
				event.FrameHeader.FrameType, event.FrameSize.RefreshFrameFlags, h.refs, framePool == nil, state.Surface, event.FrameSize, err)
		}
		h.assertReferencedSurfacesResolvable(t, payloadIndex, eventIndex, event)
		if goav1.DecoderEventOutputsFrame(event) {
			if result.Output == nil {
				t.Fatalf("payload %d event %d output frame is nil", payloadIndex, eventIndex)
			}
			h.outputs = append(h.outputs, result.Output)
		}
	}
}

func (h *publicRTCLayerPoolDecodeHarness) assertReferencedSurfacesResolvable(t *testing.T, payloadIndex, eventIndex int, event goav1.DecoderEvent) {
	t.Helper()
	for slot := 0; slot < goav1.RefFrames; slot++ {
		surface, ok := h.refs.ReferenceSlot(slot)
		if !ok {
			continue
		}
		if _, err := h.adapter.FrameSurface(surface); err != nil {
			t.Fatalf("payload %d event %d kind=%v T%d S%d left ref slot %d at unresolved surface %d: %v", payloadIndex, eventIndex, event.Kind, event.TemporalID, event.SpatialID, slot, surface, err)
		}
	}
}

func publicRTCLayerDecodeFramePool(t *testing.T, pool *goav1.FrameLayerPool, sequence goav1.SequenceHeader, event goav1.DecoderEvent) *goav1.FramePool {
	t.Helper()
	switch event.Kind {
	case goav1.DecoderEventFrameHeader, goav1.DecoderEventFrame, goav1.DecoderEventTileGroup:
		format, err := goav1.FrameCodedFormatFromHeaders(sequence, event.FrameSize, 64)
		if err != nil {
			t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
		}
		framePool, err := pool.SubPool(format)
		if err != nil {
			t.Fatalf("layer SubPool: %v", err)
		}
		return framePool
	default:
		return nil
	}
}

func publicRTCLayerDecodeEventNeedsSideData(event goav1.DecoderEvent) bool {
	switch event.Kind {
	case goav1.DecoderEventFrameHeader, goav1.DecoderEventFrame, goav1.DecoderEventTileGroup:
		return true
	default:
		return false
	}
}

func publicRTCLayerDecodeEventScratch(size goav1.DecoderFrameWorkResidualEventScratchSize) goav1.DecoderFrameWorkResidualEventScratch {
	return goav1.DecoderFrameWorkResidualEventScratch{
		Runner: goav1.DecoderFrameWorkBatchResidualRunnerScratch{
			States:                  make([]goav1.TileDecodeState, size.Runner.Workers),
			Storages:                make([]goav1.DecoderFrameWorkTileResidualCDFStorage, size.Runner.Workers),
			TileScratch:             make([]goav1.DecoderFrameWorkTileResidualScratch, size.Runner.Workers),
			RestorationRequests:     make([]goav1.DecoderFrameWorkTileRestorationRequest, size.Runner.RestorationRequests),
			PredictionScratch:       make([]goav1.DecoderFrameWorkPredictionScratch, size.Runner.Workers),
			InterPredictionScratch:  make([]goav1.DecoderFrameWorkInterPredictionScratch, size.Runner.Workers),
			Stats:                   make([]goav1.DecoderFrameWorkTileResidualStats, size.Runner.Workers),
			Int32Scratch:            make([]int32, size.Runner.Int32Scratch),
			ResidualScratch:         make([]int16, size.Runner.ResidualScratch),
			LoopContextAboveScratch: make([]goav1.TileBlockLoopRootAboveContext, size.Runner.LoopContextAbove),
		},
		SideData: publicDecoderFrameWorkSideDataScratch(size.SideData),
		Spans:    make([]goav1.TileSpan, size.Plan.SpanCount),
		Jobs:     make([]goav1.TileJob, size.Plan.JobCount),
		Batches:  make([]goav1.TileBatch, size.Plan.BatchCount),
	}
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
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
		f := goav1.I420Frame{
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
	dec, err := goav1.NewDecoder(tus)
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
	mk := func(boxX int) goav1.I420Frame {
		f := goav1.I420Frame{
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
	frames := []goav1.I420Frame{
		mk(-1),
		mk(64),
		mk(-1),
	}
	encode := func(goldenInterval int) [][]byte {
		enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	var video goav1.VideoEncoder
	if _, err := video.Encode(goav1.I420Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder Encode returned nil error")
	}
	if _, err := video.EncodeI422(goav1.I422Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeI422 returned nil error")
	}
	if _, err := video.EncodeI444(goav1.I444Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeI444 returned nil error")
	}
	if _, err := video.EncodeI400(goav1.I400Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeI400 returned nil error")
	}
	if _, err := video.EncodeNV12(goav1.NV12Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeNV12 returned nil error")
	}
	if _, err := video.EncodeNV21(goav1.NV21Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder EncodeNV21 returned nil error")
	}
	video.SetDecisionStatsEnabled(true)
	video.ResetDecisionStats()
	if got := video.DecisionStats(); got != (goav1.EncoderDecisionStats{}) {
		t.Fatalf("zero VideoEncoder stats=%+v", got)
	}
	if got := video.QIndex(); got != 0 {
		t.Fatalf("zero VideoEncoder QIndex=%d", got)
	}
	if got := video.Reconstruction(); got.Width != 0 || got.Height != 0 {
		t.Fatalf("zero VideoEncoder reconstruction=%+v", got)
	}

	var rtc goav1.RTCEncoder
	if _, err := rtc.Encode(goav1.I420Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder Encode returned nil error")
	}
	if _, err := rtc.EncodeI422(goav1.I422Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI422 returned nil error")
	}
	if _, err := rtc.EncodeI444(goav1.I444Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI444 returned nil error")
	}
	if _, err := rtc.EncodeI400(goav1.I400Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI400 returned nil error")
	}
	if _, err := rtc.EncodeNV12(goav1.NV12Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV12 returned nil error")
	}
	if _, err := rtc.EncodeNV21(goav1.NV21Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV21 returned nil error")
	}
	if _, err := rtc.EncodeNV12Picture(goav1.NV12Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV12Picture returned nil error")
	}
	if _, err := rtc.EncodeNV21Picture(goav1.NV21Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeNV21Picture returned nil error")
	}
	if _, err := rtc.EncodeI422Picture(goav1.I422Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI422Picture returned nil error")
	}
	if _, err := rtc.EncodeI444Picture(goav1.I444Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI444Picture returned nil error")
	}
	if _, err := rtc.EncodeI400Picture(goav1.I400Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder EncodeI400Picture returned nil error")
	}
}

func TestPublicEncoderRejectsInvalidI420Frames(t *testing.T) {
	const w, h = 64, 64
	invalid := goav1.I420Frame{
		Y:            make([]byte, w*h-1),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
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

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
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
