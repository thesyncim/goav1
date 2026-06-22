package goav1_test

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

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
		publicRTCApplyControlBitrates(&controlChange, targetKbps+53)
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
					for packetIndex, payload := range framePayloads {
						rtpPayloads = append(rtpPayloads, payload)
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
			gotLayeredLowMetadata := decodePublicLayeredDecoderLowOverheadMetadata(t, lowOverheads...)
			gotLayeredQueuedMetadata := decodePublicLayeredDecoderRTPPayloadMetadata(t, rtpPayloads...)
			gotLayeredLiveMetadata := decodePublicLayeredLiveDecoderRTPPayloadMetadata(t, rtpPayloads, rtpPayloads...)
			if len(got) != len(want) || len(got) != len(lowOverheads) {
				t.Fatalf("decoded frames got=%d want=%d low-overheads=%d", len(got), len(want), len(lowOverheads))
			}
			if len(gotLayeredLow) != len(want) || len(gotLayeredQueued) != len(want) || len(gotLayeredLive) != len(want) {
				t.Fatalf("layered decoded frames low=%d queued=%d live=%d want=%d", len(gotLayeredLow), len(gotLayeredQueued), len(gotLayeredLive), len(want))
			}
			assertPublicLayeredFrameMetadata(t, gotLayeredLowMetadata, wantMetadata)
			assertPublicLayeredFrameMetadata(t, gotLayeredQueuedMetadata, wantMetadata)
			assertPublicLayeredFrameMetadata(t, gotLayeredLiveMetadata, wantMetadata)
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
	delta, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 1), false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	deltaPayloads := publicRTCPictureRTPPayloads(t, delta, limits)
	key2, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, 2), true)
	if err != nil {
		t.Fatalf("key2 EncodePicture: %v", err)
	}
	key2LowOverheads := publicRTCPictureFramesInOrder(key2)
	key2Payloads := publicRTCPictureRTPPayloads(t, key2, limits)
	if len(deltaPayloads) == 0 || len(key2Payloads) == 0 {
		t.Fatalf("payloads delta=%d key2=%d", len(deltaPayloads), len(key2Payloads))
	}
	probePayloads := append(append([][]byte(nil), key0Payloads...), key2Payloads...)

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
	if got.SpatialLayerCount != spatialLayers {
		t.Fatalf("config spatial layers=%d want %d", got.SpatialLayerCount, spatialLayers)
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
	if got.MaxFramerate != want.MaxFramerate {
		t.Fatalf("config fps=%+v want %+v", got.MaxFramerate, want.MaxFramerate)
	}
	if got.RateControl != want.RateControl || got.Quantizer != want.Quantizer {
		t.Fatalf("config rate control=%d q=%d want %d q=%d", got.RateControl, got.Quantizer, want.RateControl, want.Quantizer)
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
	payloadSlices := make([][]byte, packetCount)
	for i := range packetCount {
		span := spans[i]
		payloadSlices[i] = rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		header, _, err := goav1.ParseRTPAggregationHeader(payloadSlices[i])
		if err != nil {
			t.Fatalf("packet %d aggregation header S%d: %v", i, frame.SpatialID, err)
		}
		if header.StartsNewCodedVideoSequence != (frame.CodedKeyframe && i == 0) {
			t.Fatalf("packet %d N=%v want %v frame=%+v", i, header.StartsNewCodedVideoSequence, frame.CodedKeyframe && i == 0, frame)
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
