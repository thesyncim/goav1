package encoder_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestEncodeKeyframeDecodeMatchesRecon is the non-lossless end-to-end gate:
// the decoder's output must equal the ENCODER'S OWN RECONSTRUCTION bit-for-bit
// (proving headers, mode/tx_type/coefficient coding, contexts, and the
// dequant+inverse recon loop all agree with the decoder), while PSNR against
// the source stays sane for the quantizer strength.
func TestEncodeKeyframeDecodeMatchesRecon(t *testing.T) {
	cases := []struct {
		w, h    int
		qIndex  uint8
		minPSNR float64
	}{
		{64, 64, 50, 30},
		{96, 96, 50, 30},
		{176, 144, 100, 25},
		{128, 64, 200, 18},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dx%d-q%d", tc.w, tc.h, tc.qIndex), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.w)*7919 + int64(tc.qIndex)))
			cw, ch := tc.w/2, tc.h/2
			src := encoder.SourceFrame420{
				Y:            make([]byte, tc.w*tc.h),
				U:            make([]byte, cw*ch),
				V:            make([]byte, cw*ch),
				YStride:      tc.w,
				ChromaStride: cw,
				Width:        tc.w,
				Height:       tc.h,
			}
			// Smooth gradient plus mild noise: realistic DC-predictable content.
			for y := range tc.h {
				for x := range tc.w {
					src.Y[y*tc.w+x] = uint8((128 + x + y/2 + rng.Intn(12)) & 0xff)
				}
			}
			for y := range ch {
				for x := range cw {
					src.U[y*cw+x] = uint8(120 + x/4 + rng.Intn(6))
					src.V[y*cw+x] = uint8(110 + y/4 + rng.Intn(6))
				}
			}

			tu, recon, err := encoder.EncodeKeyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			t.Logf("encoded TU: %d bytes (%.2f bpp)", len(tu), float64(len(tu)*8)/float64(tc.w*tc.h))

			dec, err := goav1.NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 1 {
				t.Fatalf("decoded %d frames, want 1", len(frames))
			}
			f := frames[0]
			comparePlane(t, "Y", f.Y, recon.Y, tc.w, tc.h, tc.w)
			comparePlane(t, "U", f.U, recon.U, cw, ch, cw)
			comparePlane(t, "V", f.V, recon.V, cw, ch, cw)

			psnr := planePSNR(src.Y, recon.Y)
			t.Logf("luma PSNR(src, recon) = %.2f dB", psnr)
			if psnr < tc.minPSNR {
				t.Fatalf("luma PSNR %.2f dB below sanity floor %.2f dB", psnr, tc.minPSNR)
			}
		})
	}
}

func TestEncodeMonochromeKeyframeDecodeMatchesRecon(t *testing.T) {
	cases := []struct {
		w, h    int
		qIndex  uint8
		minPSNR float64
	}{
		{64, 64, 48, 30},
		{96, 96, 72, 28},
		{176, 144, 112, 24},
		{128, 64, 180, 19},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dx%d-q%d", tc.w, tc.h, tc.qIndex), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x600d + int64(tc.w)*7919 + int64(tc.qIndex)))
			src := encoder.SourceFrameMono{
				Y:       make([]byte, tc.w*tc.h),
				YStride: tc.w,
				Width:   tc.w,
				Height:  tc.h,
			}
			for y := range tc.h {
				for x := range tc.w {
					src.Y[y*tc.w+x] = uint8((96 + x*2 + y + rng.Intn(18)) & 0xff)
				}
			}

			tu, recon, err := encoder.EncodeMonochromeKeyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			t.Logf("encoded mono TU: %d bytes (%.2f bpp)", len(tu), float64(len(tu)*8)/float64(tc.w*tc.h))

			seq, prefix, _ := parseKeyframeSequenceAndSize(t, tu)
			if !seq.ColorConfig.MonoChrome || seq.ColorConfig.BitDepth != 8 {
				t.Fatalf("sequence color=%+v, want 8-bit monochrome", seq.ColorConfig)
			}
			if !prefix.ShowFrame || prefix.FrameType != parser.FrameTypeKey {
				t.Fatalf("prefix=%+v, want shown keyframe", prefix)
			}

			dec, err := goav1.NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 1 {
				t.Fatalf("decoded %d frames, want 1", len(frames))
			}
			f := frames[0]
			if !f.Format.MonoChrome || f.Format.BitDepth != 8 {
				t.Fatalf("decoded format=%+v, want 8-bit monochrome", f.Format)
			}
			comparePlane(t, "Y", f.Y, recon.Y, tc.w, tc.h, tc.w)
			compareAbsentPlane(t, "U", f.U)
			compareAbsentPlane(t, "V", f.V)

			psnr := planePSNR(src.Y, recon.Y)
			t.Logf("mono luma PSNR(src, recon) = %.2f dB", psnr)
			if psnr < tc.minPSNR {
				t.Fatalf("luma PSNR %.2f dB below sanity floor %.2f dB", psnr, tc.minPSNR)
			}
		})
	}
}

func TestEncodeHighBitDepthMonochromeKeyframeDecodeMatchesRecon(t *testing.T) {
	cases := []struct {
		name     string
		w, h     int
		bitDepth uint8
		qIndex   uint8
	}{
		{name: "10bit-q72", w: 64, h: 64, bitDepth: 10, qIndex: 72},
		{name: "12bit-q104", w: 96, h: 64, bitDepth: 12, qIndex: 104},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			maxSample := uint16((1 << tc.bitDepth) - 1)
			src := encoder.SourceFrameMono16{
				Y:        make([]uint16, tc.w*tc.h),
				YStride:  tc.w,
				Width:    tc.w,
				Height:   tc.h,
				BitDepth: tc.bitDepth,
			}
			for y := range tc.h {
				for x := range tc.w {
					src.Y[y*tc.w+x] = uint16((41 + x*23 + y*31 + (x*y)%251) & int(maxSample))
				}
			}

			tu, recon, err := encoder.EncodeHighBitDepthMonochromeKeyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			seq, prefix, _ := parseKeyframeSequenceAndSize(t, tu)
			wantProfile := uint8(0)
			if tc.bitDepth == 12 {
				wantProfile = 2
			}
			if seq.SeqProfile != wantProfile ||
				!seq.ColorConfig.MonoChrome ||
				seq.ColorConfig.BitDepth != tc.bitDepth {
				t.Fatalf("sequence profile=%d color=%+v want profile=%d %d-bit monochrome", seq.SeqProfile, seq.ColorConfig, wantProfile, tc.bitDepth)
			}
			if !prefix.ShowFrame || prefix.FrameType != parser.FrameTypeKey {
				t.Fatalf("prefix=%+v, want shown keyframe", prefix)
			}

			dec, err := goav1.NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 1 {
				t.Fatalf("decoded %d frames, want 1", len(frames))
			}
			f := frames[0]
			if !f.Format.MonoChrome || f.Format.BitDepth != tc.bitDepth || f.Layout.BytesPerSample != 2 {
				t.Fatalf("decoded format=%+v bytes=%d, want %d-bit monochrome", f.Format, f.Layout.BytesPerSample, tc.bitDepth)
			}
			wantY := appendHighBitDepthMonoRaw(nil, recon)
			gotY := appendFramePlaneRaw(nil, f.Y, f.Layout.BytesPerSample)
			if string(gotY) != string(wantY) {
				t.Fatalf("decoded luma differs from reconstruction")
			}
			compareAbsentPlane(t, "U", f.U)
			compareAbsentPlane(t, "V", f.V)
		})
	}
}

func TestEncodeKeyframeWithSequenceMaxDecodeMatchesRecon(t *testing.T) {
	const (
		w, h       = 64, 48
		maxW, maxH = 128, 96
		qIndex     = 72
	)
	cw, ch := w/2, h/2
	src := encoder.SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := range h {
		for x := range w {
			src.Y[y*w+x] = uint8(36 + 3*x + 5*y + x*y)
		}
	}
	for i := range src.U {
		src.U[i] = uint8(112 + i%17)
		src.V[i] = uint8(140 - i%19)
	}

	tu, recon, err := encoder.EncodeKeyframeWithSequenceMax(src, qIndex, maxW, maxH)
	if err != nil {
		t.Fatalf("EncodeKeyframeWithSequenceMax: %v", err)
	}
	seq, prefix, size := parseKeyframeSequenceAndSize(t, tu)
	if seq.MaxFrameWidth != maxW || seq.MaxFrameHeight != maxH {
		t.Fatalf("sequence max=%dx%d want %dx%d", seq.MaxFrameWidth, seq.MaxFrameHeight, maxW, maxH)
	}
	if !prefix.FrameSizeOverride || size.UpscaledWidth != w || size.Height != h {
		t.Fatalf("prefix=%+v size=%+v", prefix, size)
	}

	dec, err := goav1.NewDecoder([][]byte{tu})
	if err != nil {
		t.Fatalf("new decoder: %v", err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("decoded %d frames, want 1", len(frames))
	}
	comparePlane(t, "Y", frames[0].Y, recon.Y, w, h, w)
	comparePlane(t, "U", frames[0].U, recon.U, cw, ch, cw)
	comparePlane(t, "V", frames[0].V, recon.V, cw, ch, cw)
}

func appendHighBitDepthMonoRaw(dst []byte, frame encoder.SourceFrameMono16) []byte {
	for y := range frame.Height {
		row := frame.Y[y*frame.YStride : y*frame.YStride+frame.Width]
		for _, sample := range row {
			dst = append(dst, byte(sample), byte(sample>>8))
		}
	}
	return dst
}

func appendFramePlaneRaw(dst []byte, plane goav1.FramePlane, bytesPerSample int) []byte {
	if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
		return dst
	}
	rowBytes := plane.Width * bytesPerSample
	for y := range plane.Height {
		row := plane.Pix[y*plane.Stride : y*plane.Stride+rowBytes]
		dst = append(dst, row...)
	}
	return dst
}

func parseKeyframeSequenceAndSize(t *testing.T, tu []byte) (parser.SequenceHeader, parser.FrameHeaderPrefix, parser.FrameSize) {
	t.Helper()
	var seq parser.SequenceHeader
	var haveSeq bool
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse OBU: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeSequenceHeader:
			seq, err = parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			haveSeq = true
		case obu.TypeFrameHeader:
			if !haveSeq {
				t.Fatal("frame header before sequence header")
			}
			prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			size, err := parser.ParseIntraFrameSize(unit.Payload, seq, prefix, unit.Header.TemporalID, unit.Header.SpatialID)
			if err != nil {
				t.Fatalf("ParseIntraFrameSize: %v", err)
			}
			return seq, prefix, size
		}
	}
	t.Fatal("missing frame header")
	return parser.SequenceHeader{}, parser.FrameHeaderPrefix{}, parser.FrameSize{}
}

func planePSNR(a, b []byte) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sse float64
	for i := range a {
		d := float64(int(a[i]) - int(b[i]))
		sse += d * d
	}
	if sse == 0 {
		return math.Inf(1)
	}
	mse := sse / float64(len(a))
	return 10 * math.Log10(255*255/mse)
}
