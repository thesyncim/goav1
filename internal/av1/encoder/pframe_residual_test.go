package encoder_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodePFrameDecodeMatchesRecon is the temporal-compression gate: a KEY +
// residual-P sequence (the second source differing from the first by a small
// brightness drift plus noise) must decode with frame 1 equal to the encoder's
// P-frame reconstruction bit for bit, with sane fidelity against the second
// source and a P-frame substantially smaller than the keyframe — evidence the
// temporal prediction is doing the work.
func TestEncodePFrameDecodeMatchesRecon(t *testing.T) {
	sizes := []struct{ w, h int }{{64, 64}, {128, 128}}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(sz.w)*131 + int64(sz.h)))
			cw, ch := sz.w/2, sz.h/2
			newFrame := func() encoder.SourceFrame420 {
				return encoder.SourceFrame420{
					Y:            make([]byte, sz.w*sz.h),
					U:            make([]byte, cw*ch),
					V:            make([]byte, cw*ch),
					YStride:      sz.w,
					ChromaStride: cw,
					Width:        sz.w,
					Height:       sz.h,
				}
			}
			src1 := newFrame()
			for y := range sz.h {
				for x := range sz.w {
					src1.Y[y*sz.w+x] = uint8((100 + x + y/2 + rng.Intn(10)) & 0xff)
				}
			}
			for i := range src1.U {
				src1.U[i] = uint8(118 + rng.Intn(8))
				src1.V[i] = uint8(106 + rng.Intn(8))
			}
			// Frame 2: frame 1 content SHIFTED by an even full-pel motion
			// (+4, +2) with a small drift — motion estimation must lock onto
			// the global shift and leave near-zero residuals.
			const shiftX, shiftY = 4, 2
			src2 := newFrame()
			for y := range sz.h {
				for x := range sz.w {
					sx, sy := x-shiftX, y-shiftY
					if sx < 0 {
						sx = 0
					}
					if sy < 0 {
						sy = 0
					}
					src2.Y[y*sz.w+x] = uint8(min(255, int(src1.Y[sy*sz.w+sx])+1))
				}
			}
			for y := range ch {
				for x := range cw {
					sx, sy := x-shiftX/2, y-shiftY/2
					if sx < 0 {
						sx = 0
					}
					if sy < 0 {
						sy = 0
					}
					src2.U[y*cw+x] = src1.U[sy*cw+sx]
					src2.V[y*cw+x] = src1.V[sy*cw+sx]
				}
			}

			const qIndex = 50
			keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, qIndex)
			if err != nil {
				t.Fatalf("encode keyframe: %v", err)
			}
			pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, qIndex)
			if err != nil {
				t.Fatalf("encode p-frame: %v", err)
			}
			t.Logf("key TU %d bytes, P TU %d bytes", len(keyTU), len(pTU))
			// Size evidence only on the larger frame: the tiny 64x64 gradient
			// keyframe is already near-free, while the shifted P pays MV costs
			// and codes fresh edge content.
			if sz.w >= 128 && len(pTU) >= len(keyTU) {
				t.Fatalf("P TU %d bytes not smaller than key TU %d bytes", len(pTU), len(keyTU))
			}

			dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(frames))
			}
			f := frames[1]
			comparePlane(t, "Y", f.Y, pRecon.Y, sz.w, sz.h, sz.w)
			comparePlane(t, "U", f.U, pRecon.U, cw, ch, cw)
			comparePlane(t, "V", f.V, pRecon.V, cw, ch, cw)

			psnr := planePSNR(src2.Y, pRecon.Y)
			t.Logf("P-frame luma PSNR(src2, recon) = %.2f dB", psnr)
			if psnr < 30 {
				t.Fatalf("P-frame luma PSNR %.2f dB below sanity floor", psnr)
			}
		})
	}
}

func TestEncodeLosslessPFrameDecodeMatchesSource(t *testing.T) {
	const w, h = 96, 64
	cw, ch := w/2, h/2
	makeFrame := func(seed int) encoder.SourceFrame420 {
		rng := rand.New(rand.NewSource(int64(seed)))
		f := encoder.SourceFrame420{
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
				f.Y[y*w+x] = uint8((x*3 + y*5 + rng.Intn(37)) & 0xff)
			}
		}
		for y := range ch {
			for x := range cw {
				f.U[y*cw+x] = uint8((96 + x*7 + y*3 + rng.Intn(19)) & 0xff)
				f.V[y*cw+x] = uint8((144 + x*5 + y*11 + rng.Intn(23)) & 0xff)
			}
		}
		return f
	}
	src1 := makeFrame(11)
	src2 := makeFrame(29)

	keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, 72)
	if err != nil {
		t.Fatalf("encode keyframe: %v", err)
	}
	pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, 0)
	if err != nil {
		t.Fatalf("encode lossless p-frame: %v", err)
	}
	if !bytes.Equal(pRecon.Y, src2.Y) || !bytes.Equal(pRecon.U, src2.U) || !bytes.Equal(pRecon.V, src2.V) {
		t.Fatal("lossless P-frame reconstruction differs from source")
	}

	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatalf("new decoder: %v", err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("decoded %d frames, want 2", len(frames))
	}
	f := frames[1]
	comparePlane(t, "decoded Y", f.Y, src2.Y, w, h, w)
	comparePlane(t, "decoded U", f.U, src2.U, cw, ch, cw)
	comparePlane(t, "decoded V", f.V, src2.V, cw, ch, cw)
	if bytes.Equal(pTU, keyTU) {
		t.Fatal("lossless P-frame unexpectedly matched keyframe bytes")
	}
}

func TestEncodeLosslessMonochromePFrameDecodeMatchesSource(t *testing.T) {
	const w, h = 128, 96
	makeFrame := func(seed int) encoder.SourceFrameMono {
		rng := rand.New(rand.NewSource(int64(seed)))
		f := encoder.SourceFrameMono{
			Y:       make([]byte, w*h),
			YStride: w,
			Width:   w,
			Height:  h,
		}
		for y := range h {
			for x := range w {
				f.Y[y*w+x] = uint8((x*5 + y*7 + rng.Intn(43)) & 0xff)
			}
		}
		return f
	}
	src1 := makeFrame(41)
	src2 := makeFrame(83)

	keyTU, keyRecon, err := encoder.EncodeMonochromeKeyframe(src1, 72)
	if err != nil {
		t.Fatalf("encode monochrome keyframe: %v", err)
	}
	pTU, pRecon, err := encoder.EncodeMonochromePFrame(src2, keyRecon, 0)
	if err != nil {
		t.Fatalf("encode lossless monochrome p-frame: %v", err)
	}
	if !bytes.Equal(pRecon.Y, src2.Y) {
		t.Fatal("lossless monochrome P-frame reconstruction differs from source")
	}

	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatalf("new decoder: %v", err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("decoded %d frames, want 2", len(frames))
	}
	f := frames[1]
	if !f.Format.MonoChrome || f.Format.BitDepth != 8 {
		t.Fatalf("decoded format=%+v, want 8-bit monochrome", f.Format)
	}
	compareAbsentPlane(t, "decoded U", f.U)
	compareAbsentPlane(t, "decoded V", f.V)
	comparePlane(t, "decoded Y", f.Y, src2.Y, w, h, w)
	if bytes.Equal(pTU, keyTU) {
		t.Fatal("lossless monochrome P-frame unexpectedly matched keyframe bytes")
	}
}

func TestEncodeMonochromePFrameDecodeMatchesRecon(t *testing.T) {
	const w, h = 128, 96
	src1 := encoder.SourceFrameMono{
		Y:       make([]byte, w*h),
		YStride: w,
		Width:   w,
		Height:  h,
	}
	rng := rand.New(rand.NewSource(0x1400))
	for y := range h {
		for x := range w {
			src1.Y[y*w+x] = uint8((72 + x*2 + y + rng.Intn(18)) & 0xff)
		}
	}

	const shiftX, shiftY = 4, 2
	src2 := encoder.SourceFrameMono{
		Y:       make([]byte, w*h),
		YStride: w,
		Width:   w,
		Height:  h,
	}
	for y := range h {
		for x := range w {
			sx, sy := x-shiftX, y-shiftY
			if sx < 0 {
				sx = 0
			}
			if sy < 0 {
				sy = 0
			}
			src2.Y[y*w+x] = uint8(min(255, int(src1.Y[sy*w+sx])+1))
		}
	}

	const qIndex = 72
	keyTU, keyRecon, err := encoder.EncodeMonochromeKeyframe(src1, qIndex)
	if err != nil {
		t.Fatalf("encode monochrome keyframe: %v", err)
	}
	pTU, pRecon, err := encoder.EncodeMonochromePFrame(src2, keyRecon, qIndex)
	if err != nil {
		t.Fatalf("encode monochrome p-frame: %v", err)
	}
	t.Logf("mono key TU %d bytes, mono P TU %d bytes", len(keyTU), len(pTU))

	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatalf("new decoder: %v", err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("decoded %d frames, want 2", len(frames))
	}
	for i, f := range frames {
		if !f.Format.MonoChrome || f.Format.BitDepth != 8 {
			t.Fatalf("frame %d format=%+v, want 8-bit monochrome", i, f.Format)
		}
		compareAbsentPlane(t, fmt.Sprintf("frame %d U", i), f.U)
		compareAbsentPlane(t, fmt.Sprintf("frame %d V", i), f.V)
	}
	comparePlane(t, "P Y", frames[1].Y, pRecon.Y, w, h, pRecon.YStride)

	psnr := planePSNR(src2.Y, pRecon.Y)
	t.Logf("mono P-frame luma PSNR(src2, recon) = %.2f dB", psnr)
	if psnr < 30 {
		t.Fatalf("mono P-frame luma PSNR %.2f dB below sanity floor", psnr)
	}
}

func TestEncodeHighBitDepthMonochromePFrameDecodeMatchesRecon(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		qIndex   uint8
	}{
		{name: "10bit-lossless", bitDepth: 10, qIndex: 0},
		{name: "10bit", bitDepth: 10, qIndex: 32},
		{name: "12bit-lossless", bitDepth: 12, qIndex: 0},
		{name: "12bit", bitDepth: 12, qIndex: 48},
	}
	const w, h = 64, 64
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			maxSample := uint16((1 << tc.bitDepth) - 1)
			src1 := encoder.SourceFrameMono16{
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

			var (
				keyTU    []byte
				keyRecon encoder.SourceFrameMono16
				err      error
			)
			if tc.qIndex == 0 {
				keyTU, err = encoder.EncodeLosslessHighBitDepthMonochromeKeyframe(src1)
				if err != nil {
					t.Fatalf("encode lossless high-bit-depth monochrome keyframe: %v", err)
				}
				keyRecon = encoder.SourceFrameMono16{
					Y:        append([]uint16(nil), src1.Y...),
					YStride:  src1.YStride,
					Width:    src1.Width,
					Height:   src1.Height,
					BitDepth: src1.BitDepth,
				}
			} else {
				keyTU, keyRecon, err = encoder.EncodeHighBitDepthMonochromeKeyframe(src1, tc.qIndex)
				if err != nil {
					t.Fatalf("encode high-bit-depth monochrome keyframe: %v", err)
				}
			}
			src2 := encoder.SourceFrameMono16{
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

			pTU, pRecon, err := encoder.EncodeHighBitDepthMonochromePFrame(src2, keyRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("encode high-bit-depth monochrome p-frame: %v", err)
			}
			t.Logf("high-bit-depth mono key TU %d bytes, P TU %d bytes", len(keyTU), len(pTU))
			if mono16Equal(pRecon, keyRecon) {
				t.Fatal("P-frame reconstruction unexpectedly equals the reference; residual path was not exercised")
			}
			if tc.qIndex == 0 && !mono16Equal(pRecon, src2) {
				t.Fatal("lossless high-bit-depth monochrome P-frame reconstruction differs from source")
			}

			dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(frames))
			}
			for i, f := range frames {
				if !f.Format.MonoChrome || f.Format.BitDepth != tc.bitDepth || f.Layout.BytesPerSample != 2 {
					t.Fatalf("frame %d format=%+v bytes=%d, want %d-bit monochrome", i, f.Format, f.Layout.BytesPerSample, tc.bitDepth)
				}
				compareAbsentPlane(t, fmt.Sprintf("frame %d U", i), f.U)
				compareAbsentPlane(t, fmt.Sprintf("frame %d V", i), f.V)
			}
			gotY := appendFramePlaneRaw(nil, frames[1].Y, frames[1].Layout.BytesPerSample)
			wantY := appendHighBitDepthMonoRaw(nil, pRecon)
			if string(gotY) != string(wantY) {
				t.Fatal("decoded high-bit-depth P-frame luma differs from reconstruction")
			}
		})
	}
}

func TestEncodeHighBitDepth420PFrameDecodeMatchesRecon(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		qIndex   uint8
	}{
		{name: "10bit-lossless", bitDepth: 10, qIndex: 0},
		{name: "10bit", bitDepth: 10, qIndex: 32},
		{name: "12bit-lossless", bitDepth: 12, qIndex: 0},
		{name: "12bit", bitDepth: 12, qIndex: 48},
	}
	const w, h = 64, 64
	cw, ch := w/2, h/2
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			maxSample := uint16((1 << tc.bitDepth) - 1)
			src1 := encoder.SourceFrame42016{
				Y:            make([]uint16, w*h),
				U:            make([]uint16, cw*ch),
				V:            make([]uint16, cw*ch),
				YStride:      w,
				ChromaStride: cw,
				Width:        w,
				Height:       h,
				BitDepth:     tc.bitDepth,
			}
			for y := range h {
				for x := range w {
					src1.Y[y*w+x] = uint16((41 + x*19 + y*31 + (x*y)%211) & int(maxSample))
				}
			}
			for y := range ch {
				for x := range cw {
					off := y*cw + x
					src1.U[off] = uint16((113 + x*23 + y*17 + (x*y)%127) & int(maxSample))
					src1.V[off] = uint16((197 + x*11 + y*29 + (x*y)%149) & int(maxSample))
				}
			}

			keyTU, keyRecon, err := encoder.EncodeHighBitDepth420Keyframe(src1, tc.qIndex)
			if err != nil {
				t.Fatalf("encode high-bit-depth 4:2:0 keyframe: %v", err)
			}
			src2 := encoder.SourceFrame42016{
				Y:            make([]uint16, w*h),
				U:            make([]uint16, cw*ch),
				V:            make([]uint16, cw*ch),
				YStride:      w,
				ChromaStride: cw,
				Width:        w,
				Height:       h,
				BitDepth:     tc.bitDepth,
			}
			for y := range h {
				for x := range w {
					base := int(keyRecon.Y[y*keyRecon.YStride+x])
					delta := ((x*5 + y*3) % 81) - 40
					src2.Y[y*w+x] = clampUint16(base+delta, maxSample)
				}
			}
			for y := range ch {
				for x := range cw {
					off := y*cw + x
					u := int(keyRecon.U[y*keyRecon.ChromaStride+x]) + ((x*7+y*5)%53 - 26)
					v := int(keyRecon.V[y*keyRecon.ChromaStride+x]) + ((x*3+y*11)%59 - 29)
					src2.U[off] = clampUint16(u, maxSample)
					src2.V[off] = clampUint16(v, maxSample)
				}
			}

			pTU, pRecon, err := encoder.EncodeHighBitDepth420PFrame(src2, keyRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("encode high-bit-depth 4:2:0 p-frame: %v", err)
			}
			t.Logf("high-bit-depth 4:2:0 key TU %d bytes, P TU %d bytes", len(keyTU), len(pTU))
			if frame42016Equal(pRecon, keyRecon) {
				t.Fatal("P-frame reconstruction unexpectedly equals the reference; residual path was not exercised")
			}
			if tc.qIndex == 0 && !frame42016Equal(pRecon, src2) {
				t.Fatal("lossless high-bit-depth 4:2:0 P-frame reconstruction differs from source")
			}

			dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(frames))
			}
			for i, f := range frames {
				if f.Format.MonoChrome || f.Format.BitDepth != tc.bitDepth || f.Layout.BytesPerSample != 2 ||
					!f.Format.SubsamplingX || !f.Format.SubsamplingY {
					t.Fatalf("frame %d format=%+v bytes=%d, want %d-bit 4:2:0", i, f.Format, f.Layout.BytesPerSample, tc.bitDepth)
				}
			}
			got := appendFramePlaneRaw(nil, frames[1].Y, frames[1].Layout.BytesPerSample)
			got = appendFramePlaneRaw(got, frames[1].U, frames[1].Layout.BytesPerSample)
			got = appendFramePlaneRaw(got, frames[1].V, frames[1].Layout.BytesPerSample)
			want := appendHighBitDepth420Raw(nil, pRecon)
			if !bytes.Equal(got, want) {
				t.Fatal("decoded high-bit-depth 4:2:0 P-frame differs from reconstruction")
			}
		})
	}
}

func clampUint16(v int, max uint16) uint16 {
	if v < 0 {
		return 0
	}
	if v > int(max) {
		return max
	}
	return uint16(v)
}

func mono16Equal(a, b encoder.SourceFrameMono16) bool {
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

func frame42016Equal(a, b encoder.SourceFrame42016) bool {
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

func appendHighBitDepth420Raw(dst []byte, frame encoder.SourceFrame42016) []byte {
	dst = appendHighBitDepth420PlaneRaw(dst, frame.Y, frame.YStride, frame.Width, frame.Height)
	cw, ch := frame.Width/2, frame.Height/2
	dst = appendHighBitDepth420PlaneRaw(dst, frame.U, frame.ChromaStride, cw, ch)
	return appendHighBitDepth420PlaneRaw(dst, frame.V, frame.ChromaStride, cw, ch)
}

func appendHighBitDepth420PlaneRaw(dst []byte, samples []uint16, stride, width, height int) []byte {
	for y := range height {
		row := samples[y*stride : y*stride+width]
		for _, sample := range row {
			dst = append(dst, byte(sample), byte(sample>>8))
		}
	}
	return dst
}
