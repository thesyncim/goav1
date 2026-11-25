package encoder_test

import (
	"bytes"
	"fmt"
	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	avframe "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestLazyNonReferenceCDEFReconMatchesDecode: non-reference pictures (the
// top temporal layer) defer their CDEF pass until Recon(). Recon() of every
// frame must still equal the decoder's output, and skipping Recon() on some
// frames must leave the bitstream and every other frame's reconstruction
// unchanged (the deferred pass never feeds prediction).
func TestLazyNonReferenceCDEFReconMatchesDecode(t *testing.T) {
	const w, h, frames = 256, 192, 13
	cw, ch := w/2, h/2
	makeFrame := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f.Y[y*w+x] = uint8((x*3 + y*5 + (x*y)%37 + n*4) & 0xff)
			}
		}
		for i := range f.U {
			f.U[i] = uint8(90 + i%50)
			f.V[i] = uint8(160 - i%40)
		}
		return f
	}
	// Pin the searched pictures to a strong single strength so the
	// non-reference pictures inherit a nonzero reference-derived one.
	encoder.SetCDEFDecisionTestHook(func(bits *uint8, y, uv *[encoder.MaxCDEFStrengths]uint8, fbIdx []uint8) {
		*bits = 0
		*y = [encoder.MaxCDEFStrengths]uint8{62}
		*uv = [encoder.MaxCDEFStrengths]uint8{60}
		clear(fbIdx)
	})
	t.Cleanup(func() { encoder.SetCDEFDecisionTestHook(nil) })
	for _, threads := range []int{1, 4} {
		t.Run(fmt.Sprintf("threads%d", threads), func(t *testing.T) {
			run := func(every int) ([][]byte, map[int]encoder.SourceFrame420) {
				enc, err := encoder.NewVideoEncoder(w, h, 120)
				if err != nil {
					t.Fatal(err)
				}
				defer enc.Close()
				enc.SetMaxThreads(threads)
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatal(err)
				}
				var tus [][]byte
				recons := map[int]encoder.SourceFrame420{}
				for i := range frames {
					tu, _, err := enc.Encode(makeFrame(i), false)
					if err != nil {
						t.Fatal(err)
					}
					tus = append(tus, append([]byte(nil), tu...))
					if i%every == 0 {
						recons[i] = cloneFrame(enc.Recon())
					}
				}
				return tus, recons
			}
			tus, recons := run(1)
			tusSparse, reconsSparse := run(3)
			for i := range tus {
				if !bytes.Equal(tus[i], tusSparse[i]) {
					t.Fatalf("frame %d bitstream depends on Recon() calls", i)
				}
			}
			for i, r := range reconsSparse {
				if !bytes.Equal(r.Y, recons[i].Y) || !bytes.Equal(r.U, recons[i].U) || !bytes.Equal(r.V, recons[i].V) {
					t.Fatalf("frame %d reconstruction depends on earlier Recon() calls", i)
				}
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
					t.Fatalf("decode: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range batch {
					comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
					comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
					comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
					i++
				}
			}
			if i != frames {
				t.Fatalf("decoded %d frames, want %d", i, frames)
			}
		})
	}
}

// TestCDEFSearchMultiStrengthSignalsBitExact forces every searched picture
// (the keyframe and the base-layer inter frames) to signal four CDEF
// strengths with per-64x64 indices, so the tile symbol-log replay splices
// cdef_idx literals into every coded filter block. Every decoded frame must
// equal the encoder's reconstruction, which applied the same indices, for
// both the single-thread fused walk and the multi-thread wavefront.
func TestCDEFSearchMultiStrengthSignalsBitExact(t *testing.T) {
	const w, h = 256, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(7))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(40 + (i%w)/3 + rng.Intn(90))
	}
	makeFrame := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = uint8(100 + (i % cw))
			f.V[i] = uint8(150 - (i % cw))
		}
		sx, sy := 20+n*5, 30+n*3
		for y := sy; y < sy+40 && y < h; y++ {
			for x := sx; x < sx+40 && x < w; x++ {
				f.Y[y*w+x] = uint8(200 + (x+y)%40)
			}
		}
		return f
	}
	forced := 0
	encoder.SetCDEFDecisionTestHook(func(bits *uint8, y, uv *[encoder.MaxCDEFStrengths]uint8, fbIdx []uint8) {
		*bits = 2
		*y = [encoder.MaxCDEFStrengths]uint8{62, 0, 28, 7}
		*uv = [encoder.MaxCDEFStrengths]uint8{60, 2, 0, 33}
		for i := range fbIdx {
			fbIdx[i] = uint8((i*3 + 1) % 4)
		}
		forced++
	})
	t.Cleanup(func() { encoder.SetCDEFDecisionTestHook(nil) })
	// Screen content runs cdef level 7, where the middle layer searches
	// with search_best_ref_fs: the only non-base picture that records.
	for _, cfg := range []struct {
		threads int
		screen  bool
	}{{1, false}, {4, false}, {1, true}, {4, true}} {
		threads := cfg.threads
		t.Run(fmt.Sprintf("threads%d_screen%v", threads, cfg.screen), func(t *testing.T) {
			enc, err := encoder.NewVideoEncoder(w, h, 60)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.screen {
				if err := enc.SetContentHint(encoder.ContentScreen); err != nil {
					t.Fatal(err)
				}
			}
			t.Cleanup(func() { _ = enc.Close() })
			enc.SetMaxThreads(threads)
			if err := enc.SetTemporalLayers(3); err != nil {
				t.Fatal(err)
			}
			const frames = 9
			tus := make([][]byte, 0, frames)
			recons := make([]encoder.SourceFrame420, 0, frames)
			before := forced
			for i := range frames {
				tu, _, err := enc.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("encode frame %d: %v", i, err)
				}
				tus = append(tus, append([]byte(nil), tu...))
				recons = append(recons, cloneFrame(enc.Recon()))
			}
			if forced == before {
				t.Fatal("no picture ran the CDEF search")
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
					t.Fatalf("decode: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range batch {
					comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
					comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
					comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
					i++
				}
			}
			if i != frames {
				t.Fatalf("decoded %d frames, want %d", i, frames)
			}
		})
	}
}

// TestEncodedStreamDecodesInReferenceDecoders is the independent oracle: the
// encoder's output must decode in libaom's aomdec and dav1d to exactly the
// encoder reconstruction. Ordinary test runs keep the reference decoders
// optional; the WebRTC parity gates require both.
func TestEncodedStreamDecodesInReferenceDecoders(t *testing.T) {
	decoders := referenceDecodersForTest(t)
	aomdec := decoders.aomdec
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(11))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(64 + rng.Intn(56))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 117
			f.V[i] = 135
		}
		sx, sy := 10+t*4, 14+t*2
		for y := sy; y < sy+20 && y < h; y++ {
			for x := sx; x < sx+20 && x < w; x++ {
				f.Y[y*w+x] = 215
			}
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	const frames = 6
	stream := ivf.AppendFileHeader(nil, w, h, 30, 1, frames)
	wantYUV := make([]byte, 0, frames*(w*h+2*cw*ch))
	for i := range frames {
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		stream = ivf.AppendFrame(stream, tu, uint64(i))
		r := enc.Recon()
		wantYUV = append(wantYUV, r.Y...)
		wantYUV = append(wantYUV, r.U...)
		wantYUV = append(wantYUV, r.V...)
	}

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, "stream.ivf")
	if err := os.WriteFile(ivfPath, stream, 0o644); err != nil {
		t.Fatal(err)
	}

	check := func(name, bin string, args ...string) {
		outPath := filepath.Join(dir, name+".yuv")
		cmd := exec.Command(bin, append(args, ivfPath)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s failed: %v\n%s", name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("read %s output: %v", name, err)
		}
		if len(got) != len(wantYUV) {
			t.Fatalf("%s output %d bytes, want %d", name, len(got), len(wantYUV))
		}
		if !bytes.Equal(got, wantYUV) {
			for i := range got {
				if got[i] != wantYUV[i] {
					t.Fatalf("%s output differs first at byte %d: got %d want %d", name, i, got[i], wantYUV[i])
				}
			}
		}
		t.Logf("%s: %d frames bit-exact", name, frames)
	}

	check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
	if dav1d := decoders.dav1d; dav1d != "" {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedMonochromeKeyframeDecodesInReferenceDecoders(t *testing.T) {
	decoders := referenceDecodersForTest(t)
	aomdec := decoders.aomdec
	const w, h = 128, 96
	rng := rand.New(rand.NewSource(0x51ed))
	src := encoder.SourceFrameMono{
		Y:       make([]byte, w*h),
		YStride: w,
		Width:   w,
		Height:  h,
	}
	for y := range h {
		for x := range w {
			src.Y[y*w+x] = uint8((x*5 + y*3 + rng.Intn(48)) & 0xff)
		}
	}
	tu, err := encoder.EncodeLosslessMonochromeKeyframe(src)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 1)
	stream = ivf.AppendFrame(stream, tu, 0)

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, "mono.ivf")
	if err := os.WriteFile(ivfPath, stream, 0o644); err != nil {
		t.Fatal(err)
	}

	check := func(name, bin string, args ...string) {
		outPath := filepath.Join(dir, name+".yuv")
		cmd := exec.Command(bin, append(args, ivfPath)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s failed: %v\n%s", name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("read %s output: %v", name, err)
		}
		yLen := len(src.Y)
		i420Len := yLen + 2*(w/2)*(h/2)
		if len(got) != yLen && len(got) != i420Len {
			t.Fatalf("%s output %d bytes, want monochrome Y-only %d or yuv420 %d", name, len(got), yLen, i420Len)
		}
		if !bytes.Equal(got[:yLen], src.Y) {
			for i := range src.Y {
				if got[i] != src.Y[i] {
					t.Fatalf("%s luma differs first at byte %d: got %d want %d", name, i, got[i], src.Y[i])
				}
			}
		}
		t.Logf("%s: monochrome keyframe luma bit-exact (%d raw bytes)", name, len(got))
	}

	check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
	if dav1d := decoders.dav1d; dav1d != "" {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedMonochromeLossyKeyframeDecodesInReferenceDecoders(t *testing.T) {
	decoders := referenceDecodersForTest(t)
	aomdec := decoders.aomdec
	const w, h = 160, 96
	rng := rand.New(rand.NewSource(0x1a55))
	src := encoder.SourceFrameMono{
		Y:       make([]byte, w*h),
		YStride: w,
		Width:   w,
		Height:  h,
	}
	for y := range h {
		for x := range w {
			src.Y[y*w+x] = uint8((80 + x*3 + y*2 + rng.Intn(32)) & 0xff)
		}
	}
	tu, recon, err := encoder.EncodeMonochromeKeyframe(src, 96)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 1)
	stream = ivf.AppendFrame(stream, tu, 0)

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, "mono-lossy.ivf")
	if err := os.WriteFile(ivfPath, stream, 0o644); err != nil {
		t.Fatal(err)
	}

	check := func(name, bin string, args ...string) {
		outPath := filepath.Join(dir, name+".yuv")
		cmd := exec.Command(bin, append(args, ivfPath)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s failed: %v\n%s", name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("read %s output: %v", name, err)
		}
		yLen := len(recon.Y)
		i420Len := yLen + 2*(w/2)*(h/2)
		if len(got) != yLen && len(got) != i420Len {
			t.Fatalf("%s output %d bytes, want monochrome Y-only %d or yuv420 %d", name, len(got), yLen, i420Len)
		}
		if !bytes.Equal(got[:yLen], recon.Y) {
			for i := range recon.Y {
				if got[i] != recon.Y[i] {
					t.Fatalf("%s luma differs first at byte %d: got %d want %d", name, i, got[i], recon.Y[i])
				}
			}
		}
		t.Logf("%s: monochrome lossy keyframe luma matches reconstruction (%d raw bytes)", name, len(got))
	}

	check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
	if dav1d := decoders.dav1d; dav1d != "" {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedMonochromePFrameDecodesInReferenceDecoders(t *testing.T) {
	decoders := referenceDecodersForTest(t)
	aomdec := decoders.aomdec
	const w, h = 128, 96
	src1 := encoder.SourceFrameMono{
		Y:       make([]byte, w*h),
		YStride: w,
		Width:   w,
		Height:  h,
	}
	rng := rand.New(rand.NewSource(0x4400))
	for y := range h {
		for x := range w {
			src1.Y[y*w+x] = uint8((84 + x*3 + y*2 + rng.Intn(24)) & 0xff)
		}
	}
	src2 := encoder.SourceFrameMono{
		Y:       make([]byte, w*h),
		YStride: w,
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
			src2.Y[y*w+x] = uint8(min(255, int(src1.Y[sy*w+sx])+2))
		}
	}

	keyTU, keyRecon, err := encoder.EncodeMonochromeKeyframe(src1, 80)
	if err != nil {
		t.Fatalf("encode keyframe: %v", err)
	}

	for _, tc := range []struct {
		name   string
		qIndex uint8
	}{
		{name: "lossless", qIndex: 0},
		{name: "lossy", qIndex: 80},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pTU, pRecon, err := encoder.EncodeMonochromePFrame(src2, keyRecon, tc.qIndex)
			if err != nil {
				t.Fatalf("encode p-frame: %v", err)
			}
			if tc.qIndex == 0 && !bytes.Equal(pRecon.Y, src2.Y) {
				t.Fatal("lossless monochrome P-frame reconstruction differs from source")
			}
			stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 2)
			stream = ivf.AppendFrame(stream, keyTU, 0)
			stream = ivf.AppendFrame(stream, pTU, 1)

			dir := t.TempDir()
			ivfPath := filepath.Join(dir, "mono-p.ivf")
			if err := os.WriteFile(ivfPath, stream, 0o644); err != nil {
				t.Fatal(err)
			}
			wantY := [][]byte{keyRecon.Y, pRecon.Y}
			check := func(name, bin string, args ...string) {
				outPath := filepath.Join(dir, name+".yuv")
				cmd := exec.Command(bin, append(args, ivfPath)...)
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Fatalf("%s failed: %v\n%s", name, err, out)
				}
				got, err := os.ReadFile(outPath)
				if err != nil {
					t.Fatalf("read %s output: %v", name, err)
				}
				yLen := w * h
				i420Len := yLen + 2*(w/2)*(h/2)
				frameLen := 0
				switch len(got) {
				case len(wantY) * yLen:
					frameLen = yLen
				case len(wantY) * i420Len:
					frameLen = i420Len
				default:
					t.Fatalf("%s output %d bytes, want %d Y-only or %d yuv420 bytes", name, len(got), len(wantY)*yLen, len(wantY)*i420Len)
				}
				for frame, want := range wantY {
					frameY := got[frame*frameLen : frame*frameLen+yLen]
					if !bytes.Equal(frameY, want) {
						for i := range want {
							if frameY[i] != want[i] {
								t.Fatalf("%s frame %d luma differs first at byte %d: got %d want %d", name, frame, i, frameY[i], want[i])
							}
						}
					}
				}
				t.Logf("%s: monochrome key+P q=%d luma matches reconstruction (%d raw bytes)", name, tc.qIndex, len(got))
			}

			check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
			if dav1d := decoders.dav1d; dav1d != "" {
				check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
			}
		})
	}
}

// encodeKeyframe codes src as a forced keyframe on a fresh KeyframeEncoder.
// The reconstruction stays valid for the rest of the test.
func encodeKeyframe(t testing.TB, src encoder.SourceFrame420, qIndex uint8) ([]byte, encoder.SourceFrame420, error) {
	t.Helper()
	e, err := encoder.NewKeyframeEncoder(src.Width, src.Height, qIndex)
	if err != nil {
		return nil, encoder.SourceFrame420{}, err
	}
	t.Cleanup(func() { _ = e.Close() })
	return e.Encode(src)
}

// encodeSequence codes srcs on one fixed-qindex VideoEncoder, the first as a
// forced keyframe, and returns each temporal unit, a copy of each
// reconstruction and whether each frame was coded as a keyframe.
func encodeSequence(t testing.TB, qIndex uint8, srcs ...encoder.SourceFrame420) ([][]byte, []encoder.SourceFrame420, []bool) {
	t.Helper()
	e, err := encoder.NewVideoEncoder(srcs[0].Width, srcs[0].Height, qIndex)
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer e.Close()
	tus := make([][]byte, len(srcs))
	recons := make([]encoder.SourceFrame420, len(srcs))
	keys := make([]bool, len(srcs))
	for i, src := range srcs {
		tu, key, err := e.Encode(src, i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus[i], recons[i], keys[i] = bytes.Clone(tu), cloneFrame(e.Recon()), key
	}
	return tus, recons, keys
}

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

			tu, recon, err := encodeKeyframe(t, src, tc.qIndex)
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

	enc, err := encoder.NewVideoEncoder(w, h, qIndex)
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer enc.Close()
	tu, err := enc.EncodeKeyWithSequenceMax(src, maxW, maxH)
	if err != nil {
		t.Fatalf("EncodeKeyWithSequenceMax: %v", err)
	}
	recon := enc.Recon()
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
		case obu.TypeFrameHeader, obu.TypeFrame:
			if !haveSeq {
				t.Fatal("frame header before sequence header")
			}
			prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			size, err := parser.ParseFrameSize(unit.Payload, seq, prefix, nil, unit.Header.TemporalID, unit.Header.SpatialID)
			if err != nil {
				t.Fatalf("ParseFrameSize: %v", err)
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

// TestKeyframeDirectionalModes codes striped content as a keyframe: it must
// decode bit-exact and pass aomdec. Camera-content keyframes code DC_PRED
// only, as SVT-AV1's rtc presets above M11 do (intra_level
// MAX_INTRA_LEVEL - 1), so the size bound checks that the partition
// decision still follows the stripes (whole-frame 8x8 DC coding cost
// 6.2 KB); the V/H search runs on screen content.
func TestKeyframeDirectionalModes(t *testing.T) {
	const w, h = 256, 256
	cw, ch := w/2, h/2
	mk := func(vertical bool) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for y := range h {
			for x := range w {
				p := x
				if !vertical {
					p = y
				}
				v := uint8(60)
				if (p/4)%2 == 1 {
					v = 200
				}
				f.Y[y*w+x] = v
			}
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	for _, tc := range []struct {
		name     string
		vertical bool
	}{{"vertical-stripes", true}, {"horizontal-stripes", false}} {
		t.Run(tc.name, func(t *testing.T) {
			src := mk(tc.vertical)
			tu, recon, err := encodeKeyframe(t, src, 60)
			if err != nil {
				t.Fatal(err)
			}
			psnr := planePSNR(src.Y, recon.Y)
			t.Logf("%s key: %d bytes, PSNR %.2f", tc.name, len(tu), psnr)
			if len(tu) > 4000 {
				t.Fatalf("key %d bytes: partition not following the stripes (8x8 DC cost ~6.2KB)", len(tu))
			}
			if psnr < 48 {
				t.Fatalf("PSNR %.2f below floor", psnr)
			}
			dec, err := goav1.NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			comparePlane(t, "Y", frames[0].Y, recon.Y, w, h, w)
			comparePlane(t, "U", frames[0].U, recon.U, cw, ch, cw)
			comparePlane(t, "V", frames[0].V, recon.V, cw, ch, cw)

			if aomdec, err := exec.LookPath("aomdec"); err == nil {
				stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 1)
				stream = ivf.AppendFrame(stream, tu, 0)
				dir := t.TempDir()
				p := filepath.Join(dir, "s.ivf")
				os.WriteFile(p, stream, 0o644)
				outPath := filepath.Join(dir, "o.yuv")
				if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
					t.Fatalf("aomdec: %v\n%s", err, out)
				}
				got, _ := os.ReadFile(outPath)
				want := append(append(append([]byte(nil), recon.Y...), recon.U...), recon.V...)
				if string(got) != string(want) {
					t.Fatal("aomdec output differs from reconstruction")
				}
			}
		})
	}
}

// keyframeWavefrontSource is a detailed 4:2:0 picture: smooth gradients,
// noise and hard-edged text-like strokes, so the keyframe exercises the merge
// pricing, the directional search and (as screen content) the palette.
func keyframeWavefrontSource(w, h int, seed int64) encoder.SourceFrame420 {
	rng := rand.New(rand.NewSource(seed))
	f := encoder.SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
	for y := range h {
		for x := range w {
			v := (x*3+y*2)/5 + rng.Intn(24)
			switch {
			case (y/9)%7 == 0 && x%13 < 9:
				v = 20
			case (x/11)%6 == 0 && (y/17)%2 == 0:
				v = 235
			}
			f.Y[y*w+x] = byte(min(max(v, 0), 255))
		}
	}
	for y := range h / 2 {
		for x := range w / 2 {
			f.U[y*w/2+x] = byte(96 + (x/7)%5*12 + rng.Intn(6))
			f.V[y*w/2+x] = byte(150 - (y/5)%4*14 + rng.Intn(6))
		}
	}
	return f
}

// TestKeyframeWavefrontMatchesSerial checks the single-tile keyframe's SB-row
// wavefront (keyframe_split.go) reproduces the serial decision pass byte for
// byte at every thread count and run, for camera and screen content, and
// that the stream decodes to the encoder's reconstruction.
func TestKeyframeWavefrontMatchesSerial(t *testing.T) {
	const w, h = 448, 320
	for _, content := range []encoder.ContentHint{encoder.ContentCamera, encoder.ContentScreen} {
		var want [][]byte
		for _, threads := range []int{1, 2, 4, 8, 3} {
			for run := 0; run < 2; run++ {
				enc, err := encoder.NewVideoEncoder(w, h, 110)
				if err != nil {
					t.Fatal(err)
				}
				if err := enc.SetContentHint(content); err != nil {
					t.Fatal(err)
				}
				enc.SetMaxThreads(threads)
				var tus [][]byte
				var recons [][]byte
				for i := range 3 {
					tu, _, err := enc.Encode(keyframeWavefrontSource(w, h, int64(i)), true)
					if err != nil {
						t.Fatalf("content %d threads %d frame %d: %v", content, threads, i, err)
					}
					tus = append(tus, append([]byte(nil), tu...))
					rc := enc.Recon()
					recons = append(recons, append(append(append([]byte(nil), rc.Y...), rc.U...), rc.V...))
				}
				_ = enc.Close()
				if want == nil {
					want = tus
					dec, err := goav1.NewDecoder(tus)
					if err != nil {
						t.Fatal(err)
					}
					i := 0
					for {
						batch, ok, err := dec.DecodeNext()
						if err != nil {
							t.Fatalf("decode: %v", err)
						}
						if !ok {
							break
						}
						for _, f := range batch {
							comparePlane(t, "Y", f.Y, recons[i][:w*h], w, h, w)
							comparePlane(t, "U", f.U, recons[i][w*h:w*h+w*h/4], w/2, h/2, w/2)
							comparePlane(t, "V", f.V, recons[i][w*h+w*h/4:], w/2, h/2, w/2)
							i++
						}
					}
					dec.Close()
					if i != len(tus) {
						t.Fatalf("decoded %d frames, want %d", i, len(tus))
					}
					continue
				}
				for i := range tus {
					if !bytes.Equal(tus[i], want[i]) {
						t.Fatalf("content %d threads %d run %d keyframe %d: %d bytes differ from serial %d bytes", content, threads, run, i, len(tus[i]), len(want[i]))
					}
				}
			}
		}
	}
}

func TestEncodeLosslessMonochromeKeyframeDecodesBitExact(t *testing.T) {
	sizes := []struct{ w, h int }{
		{64, 64},
		{128, 64},
		{64, 128},
		{96, 96},
		{176, 144},
		{200, 120},
		{72, 72},
	}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x400000 + int64(sz.w)*1000 + int64(sz.h)))
			src := encoder.SourceFrameMono{
				Y:       make([]byte, sz.w*sz.h),
				YStride: sz.w,
				Width:   sz.w,
				Height:  sz.h,
			}
			for y := range sz.h {
				for x := range sz.w {
					src.Y[y*sz.w+x] = uint8((x*3 + y*2 + rng.Intn(64)) & 0xff)
				}
			}

			tu, err := encoder.EncodeLosslessMonochromeKeyframe(src)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			t.Logf("encoded monochrome TU: %d bytes", len(tu))

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
			comparePlane(t, "Y", f.Y, src.Y, sz.w, sz.h, sz.w)
			compareAbsentPlane(t, "U", f.U)
			compareAbsentPlane(t, "V", f.V)
		})
	}
}

func comparePlane(t *testing.T, name string, got goav1.FramePlane, want []byte, w, h, wantStride int) {
	t.Helper()
	if got.Width != w || got.Height != h {
		t.Fatalf("%s plane %dx%d, want %dx%d", name, got.Width, got.Height, w, h)
	}
	for row := range h {
		gotRow := got.Pix[row*got.Stride : row*got.Stride+w]
		wantRow := want[row*wantStride : row*wantStride+w]
		if !bytes.Equal(gotRow, wantRow) {
			col := firstDiff(gotRow, wantRow)
			t.Fatalf("%s plane mismatch at (%d,%d): got %d want %d", name, col, row, gotRow[col], wantRow[col])
		}
	}
}

func compareAbsentPlane(t *testing.T, name string, got goav1.FramePlane) {
	t.Helper()
	if got.Width != 0 || got.Height != 0 || got.Stride != 0 || len(got.Pix) != 0 {
		t.Fatalf("%s plane=%+v, want absent", name, got)
	}
}

func firstDiff(a, b []byte) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// loopfilter_realc_recon_test.go is the multi-tile mask-path safety net: it
// encodes complex real content with the deblocking mask apply forced on for the
// multi-tile (32 tile-column) 1080p layout, then decodes the stream and asserts
// every reconstructed frame is byte-identical to the encoder's own Recon(). The
// decoder re-derives the deblock edges from the shared frame map across tile
// boundaries (the edge-list sweep), so any tile-boundary divergence in the
// forced mask apply shows up as a mismatched pixel.

func loadRealCFrames(t *testing.T, maxFrames int) ([]encoder.SourceFrame420, int, int, bool) {
	t.Helper()
	raw, err := os.ReadFile("/tmp/corpus/realC.yuv")
	if err != nil {
		t.Skipf("realC corpus unavailable: %v", err)
		return nil, 0, 0, false
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	nf := len(raw) / fl
	if nf > maxFrames {
		nf = maxFrames
	}
	if nf < 3 {
		t.Skip("realC too short")
		return nil, 0, 0, false
	}
	frames := make([]encoder.SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = encoder.SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	return frames, w, h, true
}

func lfCloneFrame(f encoder.SourceFrame420) encoder.SourceFrame420 {
	c := f
	c.Y = append([]byte(nil), f.Y...)
	c.U = append([]byte(nil), f.U...)
	c.V = append([]byte(nil), f.V...)
	return c
}

// runRealCReconMatch encodes maxFrames of realC with the given encoder
// configuration and asserts decode == recon for every frame.
func runRealCReconMatch(t *testing.T, maxThreads int) {
	frames, w, h, ok := loadRealCFrames(t, 8)
	if !ok {
		return
	}
	cw, ch := w/2, h/2

	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 5_000_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if maxThreads > 0 {
		enc.SetMaxThreads(maxThreads)
	}

	tus := make([][]byte, 0, len(frames))
	recons := make([]encoder.SourceFrame420, 0, len(frames))
	for i := range frames {
		tu, _, err := enc.Encode(frames[i], i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, lfCloneFrame(enc.Recon()))
	}
	if err := enc.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
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
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			if i >= len(frames) {
				t.Fatalf("decoded more than %d frames", len(frames))
			}
			comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
			comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != len(frames) {
		t.Fatalf("decoded %d frames, want %d", i, len(frames))
	}
}

// TestRealCMaskDefaultMultiTile exercises the DEFAULT (multi-tile, 30 tile-column)
// 1080p encoder, which now routes deblocking through the mask apply, and requires
// decode == recon on complex content across superblock and tile-column boundaries.
func TestRealCMaskDefaultMultiTile(t *testing.T) {
	runRealCReconMatch(t, 0)
}

// TestRealCMaskSingleThread is the single-thread (single-tile) latent-bug gate:
// the serial mask apply must match the decode on complex content.
func TestRealCMaskSingleThread(t *testing.T) {
	runRealCReconMatch(t, 1)
}

// TestVideoEncoderFastPan proves the hierarchical coarse search: a 16px/frame
// pan is beyond the +-8px refinement window, so without the quarter-res seeds
// motion compensation collapses to intra coding. With them the pan tracks,
// inter frames stay small, and the whole chain still decodes bit-exact.
func TestVideoEncoderFastPan(t *testing.T) {
	const w, h = 320, 192
	rng := rand.New(rand.NewSource(11))
	wide := make([]byte, (w+256)*h)
	for y := range h {
		for x := 0; x < w+256; x++ {
			wide[y*(w+256)+x] = uint8(70 + (x/6+y/6)%70 + rng.Intn(40))
		}
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		off := t * 16 // 16px/frame pan, double the refinement window
		for y := range h {
			copy(f.Y[y*w:(y+1)*w], wide[y*(w+256)+off:])
		}
		for i := range f.U {
			f.U[i], f.V[i] = 118, 134
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 120)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	var tus [][]byte
	var recons []encoder.SourceFrame420
	var keyBytes, interBytes int
	for i := range 6 {
		tu, key, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if key {
			keyBytes = len(tu)
		} else {
			interBytes += len(tu)
		}
		tus = append(tus, append([]byte(nil), tu...))
		rc := enc.Recon()
		cp := encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
			YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: rc.Width, Height: rc.Height,
		}
		recons = append(recons, cp)
	}
	avgInter := interBytes / 5
	t.Logf("fast pan: key %d bytes, avg inter %d bytes", keyBytes, avgInter)
	// Tracked motion codes each pan frame at a small fraction of the key;
	// untracked motion re-codes the texture as intra at near-key cost.
	if avgInter*3 >= keyBytes {
		t.Fatalf("fast pan not tracked: avg inter %d vs key %d", avgInter, keyBytes)
	}

	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	// Decoded frames alias pooled surfaces that later decodes recycle, so
	// compare each frame as it is produced rather than after DecodeAll.
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			var sse float64
			for y := range h {
				for x := range w {
					d := float64(f.Y.Pix[y*f.Y.Stride+x]) - float64(recons[i].Y[y*w+x])
					sse += d * d
				}
			}
			if sse != 0 {
				t.Fatalf("frame %d decode differs from encoder recon (sse %.0f)", i, sse)
			}
			i++
		}
	}
	if i != len(tus) {
		t.Fatalf("decoded %d frames, want %d", i, len(tus))
	}
}

// TestVideoEncoderMDS0IntraDC pins the light-PD1 DC_PRED arm: after a cut
// from dense noise to a smooth scene the inter residual variance dwarfs the
// source variance, so whole 16x16/32x32 blocks code as DC intra inside the
// inter frame, and the stream must decode to the encoder reconstruction.
// TestPFrameIntraDCSplitMatchesFused covers the split and wavefront paths.
func TestVideoEncoderMDS0IntraDC(t *testing.T) {
	const w, h = 256, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	mk := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 130
		}
		for y := range h {
			for x := range w {
				if n == 0 {
					f.Y[y*w+x] = uint8(40 + rng.Intn(170))
				} else {
					f.Y[y*w+x] = uint8(60 + (x+y+2*n)/4)
				}
			}
		}
		return f
	}
	frames := []encoder.SourceFrame420{mk(0), mk(1), mk(2)}

	run := func(threads int) ([][]byte, []encoder.SourceFrame420, encoder.EncoderDecisionStats) {
		e, err := encoder.NewVideoEncoder(w, h, 120)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = e.Close() })
		e.SetMaxThreads(threads)
		e.SetGoldenInterval(0)
		e.SetDecisionStatsEnabled(true)
		var tus [][]byte
		var recons []encoder.SourceFrame420
		for i, f := range frames {
			tu, _, err := e.Encode(f, false)
			if err != nil {
				t.Fatalf("threads %d frame %d: %v", threads, i, err)
			}
			tus = append(tus, append([]byte(nil), tu...))
			recons = append(recons, cloneFrame(e.Recon()))
		}
		return tus, recons, e.DecisionStats()
	}
	serialTUs, recons, stats := run(1)
	big := stats.IntraBlockSizes[tile.BlockSize16x16] + stats.IntraBlockSizes[tile.BlockSize32x32]
	t.Logf("intra blocks by size: 8x8=%d 16x16=%d 32x32=%d",
		stats.IntraBlockSizes[tile.BlockSize8x8], stats.IntraBlockSizes[tile.BlockSize16x16], stats.IntraBlockSizes[tile.BlockSize32x32])
	if big == 0 {
		t.Fatal("scene cut coded no 16x16/32x32 DC intra blocks")
	}
	dec, err := goav1.NewDecoder(serialTUs)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
			comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != len(frames) {
		t.Fatalf("decoded %d frames, want %d", i, len(frames))
	}
}

// TestScreenContentIntraCandidatesDecodeExact encodes a scrolling text-like
// picture as screen content, where newly revealed rows of hard-edged
// stripes favour SVT's V/H/SMOOTH intra candidates in the inter frames, and
// checks every frame decodes bit-exact against the encoder reconstruction,
// serial and on the SB-row wavefront.
func TestScreenContentIntraCandidatesDecodeExact(t *testing.T) {
	const w, h = 320, 192
	page := make([]byte, w*(h+160))
	for y := range h + 160 {
		for x := range w {
			v := byte(245)
			switch {
			case (y/6)%5 == 0:
				v = 30 // horizontal rules
			case (x/7)%9 == 0 && (y/24)%2 == 0:
				v = 60 // vertical rules
			case (x+y/3)%11 < 2 && (y/12)%3 != 0:
				v = 90
			}
			page[y*w+x] = v
		}
	}
	frame := func(i int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		off := (i * 13) % 160
		copy(f.Y, page[off*w:(off+h)*w])
		for y := range h / 2 {
			for x := range w / 2 {
				f.U[y*w/2+x] = 128 + byte((x/9)%3)*20
				f.V[y*w/2+x] = 128 - byte((y/7)%2)*30
			}
		}
		return f
	}
	for _, threads := range []int{1, 4} {
		enc, err := encoder.NewVideoEncoder(w, h, 120)
		if err != nil {
			t.Fatal(err)
		}
		if err := enc.SetContentHint(encoder.ContentScreen); err != nil {
			t.Fatal(err)
		}
		enc.SetMaxThreads(threads)
		var tus [][]byte
		var recons [][]byte
		for i := range 10 {
			tu, _, err := enc.Encode(frame(i), false)
			if err != nil {
				t.Fatalf("threads %d frame %d: %v", threads, i, err)
			}
			tus = append(tus, append([]byte(nil), tu...))
			rc := enc.Recon()
			recons = append(recons, append(append(append([]byte(nil), rc.Y...), rc.U...), rc.V...))
		}
		_ = enc.Close()
		dec, err := goav1.NewDecoder(tus)
		if err != nil {
			t.Fatal(err)
		}
		i := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("threads %d decode: %v", threads, err)
			}
			if !ok {
				break
			}
			for _, f := range batch {
				comparePlane(t, "Y", f.Y, recons[i][:w*h], w, h, w)
				comparePlane(t, "U", f.U, recons[i][w*h:w*h+w*h/4], w/2, h/2, w/2)
				comparePlane(t, "V", f.V, recons[i][w*h+w*h/4:], w/2, h/2, w/2)
				i++
			}
		}
		dec.Close()
		if i != len(tus) {
			t.Fatalf("threads %d decoded %d frames, want %d", threads, i, len(tus))
		}
	}
}

// TestEncodePFrameSceneChange is the intra-in-P gate: frame 2 cuts to an
// unrelated smooth scene, so motion search fails everywhere and the encoder
// must fall back to DC intra blocks inside the inter frame. Frame 3 then pans
// the new scene, exercising the ref-MV and mode contexts that intra
// neighbors leave behind. Both must decode bit-exact against the encoder
// reconstruction.
func TestEncodePFrameSceneChange(t *testing.T) {
	const w, h = 128, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(11))
	mk := func() encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	// Scene A: dense noise.
	src1 := mk()
	for i := range src1.Y {
		src1.Y[i] = uint8(60 + rng.Intn(160))
	}
	// Scene B: smooth diagonal gradient - uncorrelated with A, trivially DC
	// predictable, so the intra fallback wins decisively.
	src2 := mk()
	for y := range h {
		for x := range w {
			src2.Y[y*w+x] = uint8(40 + (x+y)/2)
		}
	}
	// Frame 3: scene B panned by a full-pel (-4,-4).
	src3 := mk()
	for y := range h {
		for x := range w {
			src3.Y[y*w+x] = src2.Y[max(0, y-4)*w+max(0, x-4)]
		}
	}

	tus, recons, keys := encodeSequence(t, 60, src1, src2, src3)
	keyTU, cutTU, panTU := tus[0], tus[1], tus[2]
	cutRecon := recons[1]
	if keys[1] || keys[2] {
		t.Fatalf("inter frames coded as keyframes: %v", keys)
	}
	t.Logf("key %dB, scene-cut P %dB, pan P %dB", len(keyTU), len(cutTU), len(panTU))
	psnr := planePSNR(src2.Y, cutRecon.Y)
	t.Logf("scene-cut luma PSNR %.2f dB", psnr)
	if psnr < 32 {
		t.Fatalf("scene-cut PSNR %.2f below floor: intra fallback not effective", psnr)
	}
	// The smooth scene is cheaper as intra than as motion residual from
	// uncorrelated noise, so the cut costs less than the keyframe of scene A.
	// SVT-AV1 preset 12's light-PD1 intra is DC_PRED on blocks of 16x16 and
	// up (disallow_8x8), so the gradient pays residual on every block.
	if len(cutTU) >= len(keyTU) {
		t.Fatalf("scene-cut P %dB not below key %dB: intra fallback not engaged", len(cutTU), len(keyTU))
	}

	dec, err := goav1.NewDecoder([][]byte{keyTU, cutTU, panTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i, f := range frames {
		comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
		comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
		comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
	}
}

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

			tus, recons, keys := encodeSequence(t, 50, src1, src2)
			keyTU, pTU, pRecon := tus[0], tus[1], recons[1]
			if keys[1] {
				t.Fatal("second frame coded as a keyframe")
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
			for i, f := range frames {
				comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, sz.w, sz.h, sz.w)
				comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
				comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			}

			psnr := planePSNR(src2.Y, pRecon.Y)
			t.Logf("P-frame luma PSNR(src2, recon) = %.2f dB", psnr)
			if psnr < 30 {
				t.Fatalf("P-frame luma PSNR %.2f dB below sanity floor", psnr)
			}
		})
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

// TestWebRTCStreamInterLayerPredictionDecodeMatchesRecon codes two spatial
// SVC pictures, so the upper layer predicts from the half-size base layer
// through the scaled inter predictor, and checks that every decoded frame
// equals its layer encoder's reconstruction bit for bit.
func TestWebRTCStreamInterLayerPredictionDecodeMatchesRecon(t *testing.T) {
	const topW, topH = 128, 96
	for _, mode := range []encoder.ScalabilityMode{encoder.ScalabilityModeL2T1, encoder.ScalabilityModeL2T2} {
		t.Run(mode.String(), func(t *testing.T) {
			stream, err := encoder.NewWebRTCStreamConfig(encoder.Config{
				Resolution:        encoder.Resolution{Width: topW, Height: topH},
				MaxFramerate:      encoder.Rational{Num: 30, Den: 1},
				MinBitrateKbps:    100,
				MaxBitrateKbps:    900,
				TargetBitrateKbps: 500,
				Scalability:       mode,
			})
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			src := scaledReferenceDecodeBaseFrame(topW, topH)
			var tus [][]byte
			var recons []encoder.SourceFrame420
			for pic := range 2 {
				if pic == 1 {
					src = shiftSource420(src, 2, 1)
				}
				picture, err := stream.EncodePicture(src, pic == 0)
				if err != nil {
					t.Fatalf("EncodePicture %d: %v", pic, err)
				}
				for i := range picture.FrameNum {
					frame := picture.Frames[i]
					tus = append(tus, append([]byte(nil), frame.TU...))
					recons = append(recons, cloneFrame(stream.LayerRecon(frame.Info.SpatialID)))
				}
			}
			frames := decodeLayerPoolLowOverheads(t, tus...)
			if len(frames) != len(recons) {
				t.Fatalf("decoded %d frames, want %d", len(frames), len(recons))
			}
			for i, f := range frames {
				r := recons[i]
				cw, ch := r.Width/2, r.Height/2
				comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, r.Y, r.Width, r.Height, r.YStride)
				comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, r.U, cw, ch, r.ChromaStride)
				comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, r.V, cw, ch, r.ChromaStride)
			}
		})
	}
}

// decodeLayerPoolLowOverheads decodes the temporal units with LayeredDecoder,
// which keeps one surface pool per coded size so the scaled frame can
// reference the smaller one.
func decodeLayerPoolLowOverheads(t *testing.T, payloads ...[]byte) []*goav1.Frame {
	t.Helper()
	dec, err := goav1.NewLayeredDecoder(payloads)
	if err != nil {
		t.Fatalf("NewLayeredDecoder: %v", err)
	}
	t.Cleanup(dec.Close)
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("DecodeAll: %v", err)
	}
	return frames
}

func scaledReferenceDecodeBaseFrame(width, height int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	for y := range height {
		for x := range width {
			f.Y[y*width+x] = byte(47 + 3*x + 5*y + x*y)
		}
	}
	for y := range ch {
		for x := range cw {
			f.U[y*cw+x] = byte(96 + 3*x + 7*y)
			f.V[y*cw+x] = byte(151 + 5*x + 2*y)
		}
	}
	return f
}

// shiftSource420 moves src by (dx, dy) samples, replicating the top and left
// edges.
func shiftSource420(src encoder.SourceFrame420, dx, dy int) encoder.SourceFrame420 {
	dst := cloneFrame(src)
	shiftPlane(dst.Y, src.Y, src.YStride, src.Width, src.Height, dx, dy)
	shiftPlane(dst.U, src.U, src.ChromaStride, src.Width/2, src.Height/2, dx/2, dy/2)
	shiftPlane(dst.V, src.V, src.ChromaStride, src.Width/2, src.Height/2, dx/2, dy/2)
	return dst
}

func shiftPlane(dst, src []byte, stride, w, h, dx, dy int) {
	for y := range h {
		for x := range w {
			dst[y*stride+x] = src[max(0, y-dy)*stride+max(0, x-dx)]
		}
	}
}

// TestEncodePFrameSubpelMotion is the subpel gate: the second frame is the
// first resampled at a half-pixel offset, so only subpel motion vectors can
// predict it well. The decode must match the encoder reconstruction
// bit-for-bit (the encoder predicts through the decoder's own convolve), the
// P-frame must compress far below the keyframe (the refinement found the
// half-pel motion), and aomdec must agree when present.
func TestEncodePFrameSubpelMotion(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	src1, src2 := makeHalfPelMotionFrames(t, w, h)

	tus, recons, keys := encodeSequence(t, 60, src1, src2)
	keyTU, pTU, keyRecon, pRecon := tus[0], tus[1], recons[0], recons[1]
	if keys[1] {
		t.Fatal("half-pel frame coded as a keyframe")
	}
	t.Logf("key %d bytes, half-pel P %d bytes", len(keyTU), len(pTU))
	if len(pTU)*6 >= len(keyTU) {
		t.Fatalf("P %d bytes not well below key %d: subpel motion not engaged?", len(pTU), len(keyTU))
	}

	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i, f := range frames {
		comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
		comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
		comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
	}
	psnr := planePSNR(src2.Y, pRecon.Y)
	t.Logf("half-pel P luma PSNR %.2f dB", psnr)
	if psnr < 33 {
		t.Fatalf("PSNR %.2f below floor", psnr)
	}

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 2)
		stream = ivf.AppendFrame(stream, keyTU, 0)
		stream = ivf.AppendFrame(stream, pTU, 1)
		dir := t.TempDir()
		p := filepath.Join(dir, "s.ivf")
		os.WriteFile(p, stream, 0o644)
		outPath := filepath.Join(dir, "o.yuv")
		if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
			t.Fatalf("aomdec: %v\n%s", err, out)
		}
		got, _ := os.ReadFile(outPath)
		want := append(append(append([]byte(nil), keyRecon.Y...), keyRecon.U...), keyRecon.V...)
		want = append(append(append(want, pRecon.Y...), pRecon.U...), pRecon.V...)
		if string(got) != string(want) {
			t.Fatal("aomdec output differs from reconstruction")
		}
		t.Log("aomdec: half-pel stream bit-exact")
	}
}

func TestVideoEncoderMinEffortSkipsSubpelMotion(t *testing.T) {
	const w, h = 192, 128
	src1, src2 := makeHalfPelMotionFrames(t, w, h)

	encode := func(t *testing.T, effort int8) ([]byte, []byte) {
		t.Helper()
		enc, err := encoder.NewVideoEncoder(w, h, 60)
		if err != nil {
			t.Fatal(err)
		}
		defer enc.Close()
		enc.SetMaxThreads(1)
		if err := enc.SetContentHint(encoder.ContentCamera); err != nil {
			t.Fatalf("SetContentHint: %v", err)
		}
		if err := enc.SetEffortLevel(effort); err != nil {
			t.Fatalf("SetEffortLevel: %v", err)
		}
		keyTU, key, err := enc.Encode(src1, true)
		if err != nil {
			t.Fatalf("Encode key: %v", err)
		}
		if !key {
			t.Fatal("first frame was not key")
		}
		keyCopy := append([]byte(nil), keyTU...)
		pTU, key, err := enc.Encode(src2, false)
		if err != nil {
			t.Fatalf("Encode delta: %v", err)
		}
		if key {
			t.Fatal("second frame unexpectedly became key")
		}
		return keyCopy, append([]byte(nil), pTU...)
	}

	defaultKey, defaultP := encode(t, 0)
	minKey, minP := encode(t, encoder.WebRTCMinEffortLevel)
	if bytes.Equal(defaultP, minP) {
		t.Fatal("min effort produced the same delta frame as default effort")
	}
	if len(minP) <= len(defaultP) {
		t.Fatalf("min effort delta %d bytes, default %d: want faster full-pel path to spend more bits on half-pel motion", len(minP), len(defaultP))
	}

	seq := parseFirstSequenceHeader(t, minKey)
	prefix := parseFirstFramePrefix(t, minP, seq)
	if prefix.AllowScreenContentTools || prefix.ForceIntegerMV {
		t.Fatalf("min effort changed camera content MV signaling: prefix=%+v", prefix)
	}
	defaultLF, _ := parseFirstInterFrameFilters(t, defaultP, seq, w, h)
	// SVT-AV1 preset 12 rtc codes no deblocking (get_dlf_level_rtc).
	if defaultLF.LevelY != [2]uint8{} {
		t.Fatalf("default effort loop filter levels=%+v want disabled", defaultLF.LevelY)
	}
	minLF, minCDEF := parseFirstInterFrameFilters(t, minP, seq, w, h)
	if minLF.LevelY != [2]uint8{} || minLF.LevelU != 0 || minLF.LevelV != 0 {
		t.Fatalf("min effort loop filter=%+v want disabled", minLF)
	}
	if minCDEF.StrengthCount != 1 || minCDEF.YStrength[0] != 0 || minCDEF.UVStrength[0] != 0 {
		t.Fatalf("min effort cdef=%+v want no-op strengths", minCDEF)
	}
	decodeTemporalUnits(t, defaultKey, defaultP)
	decodeTemporalUnits(t, minKey, minP)
}

func makeHalfPelMotionFrames(t *testing.T, w, h int) (encoder.SourceFrame420, encoder.SourceFrame420) {
	t.Helper()
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(31))
	src1 := encoder.SourceFrame420{
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
			src1.Y[y*w+x] = uint8(50 + rng.Intn(150)) // aperiodic texture
		}
	}
	for i := range src1.U {
		src1.U[i] = 120
		src1.V[i] = 130
	}

	// Frame 2: frame 1 shifted by half a pixel horizontally using the codec's
	// own 8-tap half-pel filter, so full-pel-only search cannot match it.
	src2 := src1
	src2.Y = make([]byte, w*h)
	dstP := avframe.Plane{Pix: src2.Y, Stride: w, Width: w, Height: h}
	refP := avframe.Plane{Pix: src1.Y, Stride: w, Width: w, Height: h}
	halfPel := motion.Vector{Col: 4}
	for py := 0; py < h; py += 8 {
		for px := 0; px < w; px += 8 {
			refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, halfPel, false, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstP, refP, 1, 8, px, py, refX, refY, 8, 8, subX, subY, motion.InterpFilters{}); err != nil {
				t.Fatal(err)
			}
		}
	}
	src2.U = append([]byte(nil), src1.U...)
	src2.V = append([]byte(nil), src1.V...)
	return src1, src2
}

func parseFirstSequenceHeader(t *testing.T, tu []byte) parser.SequenceHeader {
	t.Helper()
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type == obu.TypeSequenceHeader {
			seq, err := parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			return seq
		}
	}
	t.Fatal("missing sequence header")
	return parser.SequenceHeader{}
}

func parseFirstFramePrefix(t *testing.T, tu []byte, seq parser.SequenceHeader) parser.FrameHeaderPrefix {
	t.Helper()
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type == obu.TypeFrameHeader || unit.Header.Type == obu.TypeFrame {
			prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			return prefix
		}
	}
	t.Fatal("missing frame header")
	return parser.FrameHeaderPrefix{}
}

func parseFirstInterFrameFilters(t *testing.T, tu []byte, seq parser.SequenceHeader, width, height int) (parser.LoopFilterParams, parser.CDEFParams) {
	t.Helper()
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type != obu.TypeFrameHeader && unit.Header.Type != obu.TypeFrame {
			continue
		}
		prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
		if err != nil {
			t.Fatalf("ParseFrameHeaderPrefix: %v", err)
		}
		refs := parserReferenceStateForFrame(width, height)
		size, err := parser.ParseFrameSize(unit.Payload, seq, prefix, &refs, unit.Header.TemporalID, unit.Header.SpatialID)
		if err != nil {
			t.Fatalf("ParseFrameSize: %v", err)
		}
		tiles, err := parser.ParseTileInfo(unit.Payload, seq, prefix, size)
		if err != nil {
			t.Fatalf("ParseTileInfo: %v", err)
		}
		quant, err := parser.ParseQuantizationParams(unit.Payload, seq, tiles)
		if err != nil {
			t.Fatalf("ParseQuantizationParams: %v", err)
		}
		seg, err := parser.ParseSegmentationParams(unit.Payload, prefix, quant, nil)
		if err != nil {
			t.Fatalf("ParseSegmentationParams: %v", err)
		}
		delta, err := parser.ParseDeltaParams(unit.Payload, size, quant, seg)
		if err != nil {
			t.Fatalf("ParseDeltaParams: %v", err)
		}
		// A chained frame (primary_ref_frame LAST) codes its deltas against the
		// reference's, which this encoder keeps at the defaults.
		prevDeltas := parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{1, 0, 0, 0, -1, 0, -1, -1}}
		lf, err := parser.ParseLoopFilterParams(unit.Payload, seq, prefix, size, seg, delta, &prevDeltas)
		if err != nil {
			t.Fatalf("ParseLoopFilterParams: %v", err)
		}
		cdef, err := parser.ParseCDEFParams(unit.Payload, seq, size, seg, lf)
		if err != nil {
			t.Fatalf("ParseCDEFParams: %v", err)
		}
		return lf, cdef
	}
	t.Fatal("missing frame header")
	return parser.LoopFilterParams{}, parser.CDEFParams{}
}

func parserReferenceStateForFrame(width, height int) parser.ReferenceState {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := range parser.RefFrames {
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          uint32(width),
				UpscaledWidth:       uint32(width),
				Height:              uint32(height),
				RenderWidth:         uint32(width),
				RenderHeight:        uint32(height),
				SuperResDenominator: 8,
			},
		}
	}
	return refs
}

func decodeTemporalUnits(t *testing.T, payloads ...[]byte) {
	t.Helper()
	dec, err := goav1.NewDecoder(payloads)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(frames) != len(payloads) {
		t.Fatalf("decoded %d frames, want %d", len(frames), len(payloads))
	}
}

// TestEncodePFrameVariableBlocks is the variable-block-size gate: global
// motion plus textured content makes the partition decider merge most areas
// into 16x16 blocks while the moving square edge splits to 8x8, and the
// decode must match the encoder reconstruction bit for bit (16x16 luma TXBs,
// 8x8 chroma TXBs, merged-MV ref stacks, cross-SB diagonal context).
func TestEncodePFrameVariableBlocks(t *testing.T) {
	const w, h = 128, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(9))
	src1 := encoder.SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	for i := range src1.Y {
		src1.Y[i] = uint8(80 + rng.Intn(40))
	}
	for i := range src1.U {
		src1.U[i] = 120
		src1.V[i] = 130
	}
	// Frame 2: global +2px motion with low-amplitude noise (16x16 merges with
	// real residual) plus a moving bright square (8x8 splits at its edges).
	src2 := src1
	src2.Y = make([]byte, w*h)
	for y := range h {
		for x := range w {
			sx, sy := max(0, x-2), max(0, y-2)
			src2.Y[y*w+x] = uint8(int(src1.Y[sy*w+sx]) + rng.Intn(13) - 6)
		}
	}
	for y := 60; y < 76; y++ {
		for x := 60; x < 76; x++ {
			src2.Y[y*w+x] = 230
		}
	}
	const qIndex = 60
	tus, recons, keys := encodeSequence(t, qIndex, src1, src2)
	keyTU, pTU, keyRecon, pRecon := tus[0], tus[1], recons[0], recons[1]
	if keys[1] {
		t.Fatal("variable-block frame coded as a keyframe")
	}
	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i, f := range frames {
		comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
		comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
		comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
	}
	psnr := planePSNR(src2.Y, pRecon.Y)
	t.Logf("variable-block P: %d bytes, luma PSNR %.2f dB", len(pTU), psnr)
	if psnr < 30 {
		t.Fatalf("PSNR %.2f below floor", psnr)
	}

	// Second scenario: a clean global (-4,-4) full-pel shift of the keyframe
	// reconstruction itself, so merged blocks predict exactly. Every 16x16
	// child finds the same vector, the 32x32 tier merges, and the P frame
	// collapses to a handful of merged skip blocks - far below the keyframe.
	src3 := src1
	src3.Y = make([]byte, w*h)
	for y := range h {
		for x := range w {
			src3.Y[y*w+x] = keyRecon.Y[max(0, y-4)*w+max(0, x-4)]
		}
	}
	src3.U = make([]byte, cw*ch)
	src3.V = make([]byte, cw*ch)
	for y := range ch {
		for x := range cw {
			src3.U[y*cw+x] = keyRecon.U[max(0, y-2)*cw+max(0, x-2)]
			src3.V[y*cw+x] = keyRecon.V[max(0, y-2)*cw+max(0, x-2)]
		}
	}
	gtus, grecons, gkeys := encodeSequence(t, qIndex, src1, src3)
	gTU, gRecon := gtus[1], grecons[1]
	if gkeys[1] {
		t.Fatal("global-shift frame coded as a keyframe")
	}
	t.Logf("global-shift P: %d bytes vs key %d", len(gTU), len(keyTU))
	// The variance-partitioned inter frame still collapses the clean global
	// shift well below the keyframe while preserving the clamped edge stripes.
	if len(gTU)*4 >= len(keyTU) {
		t.Fatalf("global-shift P %d bytes not far below key %d", len(gTU), len(keyTU))
	}
	dec2, err := goav1.NewDecoder([][]byte{keyTU, gTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec2.Close()
	gframes, err := dec2.DecodeAll()
	if err != nil {
		t.Fatalf("decode global-shift: %v", err)
	}
	comparePlane(t, "gY", gframes[1].Y, gRecon.Y, w, h, w)
	comparePlane(t, "gU", gframes[1].U, gRecon.U, cw, ch, cw)
	comparePlane(t, "gV", gframes[1].V, gRecon.V, cw, ch, cw)
}

const requireWebRTCReferenceDecodersEnv = "GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS"

type referenceDecoderPaths struct {
	aomdec string
	dav1d  string
}

type lookPathFunc func(string) (string, error)

func resolveReferenceDecoders(lookPath lookPathFunc, required bool) (referenceDecoderPaths, string, error) {
	var paths referenceDecoderPaths
	var missing []string
	for _, name := range []string{"aomdec", "dav1d"} {
		path, err := lookPath(name)
		if err != nil || path == "" {
			missing = append(missing, name)
			continue
		}
		if name == "aomdec" {
			paths.aomdec = path
		} else {
			paths.dav1d = path
		}
	}
	if required && len(missing) != 0 {
		return paths, "", fmt.Errorf("%s=1 requires both aomdec and dav1d on PATH; missing: %s", requireWebRTCReferenceDecodersEnv, strings.Join(missing, ", "))
	}
	if !required && paths.aomdec == "" {
		return paths, "aomdec not on PATH; reference decoder checks are optional outside strict WebRTC gates", nil
	}
	return paths, "", nil
}

func referenceDecodersForTest(t *testing.T) referenceDecoderPaths {
	t.Helper()
	paths, skip, err := resolveReferenceDecoders(exec.LookPath, os.Getenv(requireWebRTCReferenceDecodersEnv) == "1")
	if err != nil {
		t.Fatal(err)
	}
	if skip != "" {
		t.Skip(skip)
	}
	t.Logf("reference decoder binaries: aomdec=%s dav1d=%s", paths.aomdec, paths.dav1d)
	return paths
}

func TestResolveReferenceDecoders(t *testing.T) {
	tests := []struct {
		name             string
		available        map[string]string
		wantPaths        referenceDecoderPaths
		wantMissing      []string
		wantOptionalSkip bool
	}{
		{
			name: "both available",
			available: map[string]string{
				"aomdec": "/tools/aomdec",
				"dav1d":  "/tools/dav1d",
			},
			wantPaths: referenceDecoderPaths{aomdec: "/tools/aomdec", dav1d: "/tools/dav1d"},
		},
		{
			name:        "dav1d missing",
			available:   map[string]string{"aomdec": "/tools/aomdec"},
			wantPaths:   referenceDecoderPaths{aomdec: "/tools/aomdec"},
			wantMissing: []string{"dav1d"},
		},
		{
			name:             "aomdec missing but dav1d available",
			available:        map[string]string{"dav1d": "/tools/dav1d"},
			wantPaths:        referenceDecoderPaths{dav1d: "/tools/dav1d"},
			wantMissing:      []string{"aomdec"},
			wantOptionalSkip: true,
		},
		{
			name:             "both missing",
			available:        map[string]string{},
			wantMissing:      []string{"aomdec", "dav1d"},
			wantOptionalSkip: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lookup := func(name string) (string, error) {
				if path, ok := tt.available[name]; ok {
					return path, nil
				}
				return "", exec.ErrNotFound
			}

			for _, required := range []bool{false, true} {
				paths, skip, err := resolveReferenceDecoders(lookup, required)
				if paths != tt.wantPaths {
					t.Fatalf("resolveReferenceDecoders() paths = %#v, want %#v", paths, tt.wantPaths)
				}
				if !required && (skip != "") != tt.wantOptionalSkip {
					t.Fatalf("optional skip reason = %q, want skip=%v", skip, tt.wantOptionalSkip)
				}
				if required && len(tt.wantMissing) == 0 && err != nil {
					t.Fatalf("required decoder resolution failed: %v", err)
				}
				if required && len(tt.wantMissing) != 0 {
					if err == nil {
						t.Fatal("required decoder resolution succeeded with missing tools")
					}
					for _, name := range tt.wantMissing {
						if !strings.Contains(err.Error(), name) {
							t.Errorf("required decoder error %q does not identify missing %s", err, name)
						}
					}
				}
			}
		})
	}
}

// TestVideoEncoderCBRConvergesAndDecodes is the rate-control gate: a 30-frame
// moving sequence under CBR must (a) decode with every frame bit-identical to
// the encoder reconstruction — now with a different qindex per frame, the
// per-frame header/CDF re-initialization is exercised for real — (b) move the
// working qindex (the controller is alive), and (c) land the steady-state
// bitrate within 30% of target.
func TestVideoEncoderCBRConvergesAndDecodes(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(7))

	bg := make([]byte, w*h)
	for y := range h {
		for x := range w {
			bg[y*w+x] = uint8(60 + (x/4+y/4)%64 + rng.Intn(50))
		}
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		// The mover wraps so the clip never turns static: a frozen scene
		// codes as skip at any q, leaving no rate for a controller to spend.
		sx, sy := 8+(t*4)%160, 16+(t*2)%96
		for y := sy; y < sy+24 && y < h; y++ {
			for x := sx; x < sx+24 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}

	const fps = 30
	const targetBPS = 120_000 // 500 bytes/frame at 30 fps
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: targetBPS,
		FramesPerSecond:     fps,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })

	// The boosted keyframe repays its debt over the controller's 24-frame
	// memory; steady state is measured after that horizon.
	const frames = 80
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	qSeen := map[uint8]bool{}
	steadyBits := 0
	const warmup = 40
	for i := range frames {
		qSeen[enc.QIndex()] = true
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if i >= warmup {
			steadyBits += len(tu) * 8
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
	}
	if len(qSeen) < 3 {
		t.Fatalf("rate controller never moved qindex: %v", qSeen)
	}
	steadyAvg := steadyBits / (frames - warmup)
	target := targetBPS / fps
	t.Logf("steady-state avg %d bits/frame (target %d), distinct qindexes %d", steadyAvg, target, len(qSeen))
	if steadyAvg < target*70/100 || steadyAvg > target*130/100 {
		t.Fatalf("steady-state %d bits/frame outside 30%% of target %d", steadyAvg, target)
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
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
			comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}

// TestVideoEncoderGoldenReference is the multi-ref gate: an object occludes a
// textured region and then reveals it. The revealed pixels exist in the
// golden anchor (the keyframe) but not in the previous frame, so the reveal
// frame must get cheaper with golden references enabled, and every frame must
// decode bit-exact against the encoder reconstruction.
func TestVideoEncoderGoldenReference(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	base := make([]byte, w*h)
	for i := range base {
		base[i] = uint8(50 + rng.Intn(170))
	}
	mk := func(boxX int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
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
	frames := []encoder.SourceFrame420{
		mk(-1), // scene only (keyframe / golden anchor)
		mk(64), // box occludes the center
		mk(-1), // box gone: center revealed, identical to the anchor
	}

	encode := func(goldenInterval int) ([][]byte, []encoder.SourceFrame420) {
		enc, err := encoder.NewVideoEncoder(w, h, 60)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = enc.Close() })
		enc.SetGoldenInterval(goldenInterval)
		var tus [][]byte
		var recons []encoder.SourceFrame420
		for i, f := range frames {
			tu, _, err := enc.Encode(f, false)
			if err != nil {
				t.Fatalf("encode frame %d: %v", i, err)
			}
			tus = append(tus, append([]byte(nil), tu...))
			recons = append(recons, cloneFrame(enc.Recon()))
		}
		return tus, recons
	}

	lastOnly, _ := encode(0)
	withGolden, recons := encode(32)
	t.Logf("reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
	// The revealed region must get materially cheaper (well under 60% of the
	// last-only cost; the exact ratio drifts with unrelated recon changes).
	if len(withGolden[2])*5 >= len(lastOnly[2])*3 {
		t.Fatalf("golden reveal %dB not well below last-only %dB: golden reference not engaged", len(withGolden[2]), len(lastOnly[2]))
	}

	dec, err := goav1.NewDecoder(withGolden)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
			comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != len(frames) {
		t.Fatalf("decoded %d frames, want %d", i, len(frames))
	}
}

func TestHighBitDepthVideoEncoderGoldenReference(t *testing.T) {
	t.Run("mono10", func(t *testing.T) {
		const w, h = 128, 96
		const bitDepth = uint8(10)
		maxSample := uint16((1 << bitDepth) - 1)
		rng := rand.New(rand.NewSource(29))
		base := make([]uint16, w*h)
		for i := range base {
			base[i] = uint16(80 + rng.Intn(760))
		}
		mk := func(boxX int) encoder.SourceFrameMono16 {
			f := encoder.SourceFrameMono16{
				Y:        append([]uint16(nil), base...),
				YStride:  w,
				Width:    w,
				Height:   h,
				BitDepth: bitDepth,
			}
			if boxX >= 0 {
				for y := 24; y < 72; y++ {
					for x := boxX; x < boxX+48 && x < w; x++ {
						f.Y[y*w+x] = maxSample - 12
					}
				}
			}
			return f
		}
		frames := []encoder.SourceFrameMono16{mk(-1), mk(40), mk(-1)}
		encode := func(goldenInterval int) ([][]byte, []encoder.SourceFrameMono16) {
			enc, err := encoder.NewHighBitDepthMonochromeVideoEncoder(w, h, bitDepth, 64)
			if err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(goldenInterval)
			var tus [][]byte
			var recons []encoder.SourceFrameMono16
			for i, f := range frames {
				tu, _, err := enc.Encode(f, false)
				if err != nil {
					t.Fatalf("encode frame %d: %v", i, err)
				}
				tus = append(tus, append([]byte(nil), tu...))
				recons = append(recons, cloneMono16Frame(enc.Recon()))
			}
			return tus, recons
		}

		lastOnly, _ := encode(0)
		withGolden, recons := encode(32)
		t.Logf("10-bit mono reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
		if len(withGolden[2])*5 >= len(lastOnly[2])*4 {
			t.Fatalf("10-bit mono golden reveal %dB not below last-only %dB", len(withGolden[2]), len(lastOnly[2]))
		}
		dec, err := goav1.NewDecoder(withGolden)
		if err != nil {
			t.Fatal(err)
		}
		defer dec.Close()
		decoded, err := dec.DecodeAll()
		if err != nil {
			t.Fatal(err)
		}
		if len(decoded) != len(frames) {
			t.Fatalf("decoded %d frames, want %d", len(decoded), len(frames))
		}
		got := appendFramePlaneRaw(nil, decoded[2].Y, decoded[2].Layout.BytesPerSample)
		want := appendHighBitDepthMonoRaw(nil, recons[2])
		if string(got) != string(want) {
			t.Fatal("decoded 10-bit mono reveal differs from reconstruction")
		}
	})

	t.Run("i420-10", func(t *testing.T) {
		const w, h = 128, 96
		const bitDepth = uint8(10)
		cw, ch := w/2, h/2
		maxSample := uint16((1 << bitDepth) - 1)
		rng := rand.New(rand.NewSource(31))
		base := make([]uint16, w*h)
		for i := range base {
			base[i] = uint16(96 + rng.Intn(720))
		}
		mk := func(boxX int) encoder.SourceFrame42016 {
			f := encoder.SourceFrame42016{
				Y:            append([]uint16(nil), base...),
				U:            make([]uint16, cw*ch),
				V:            make([]uint16, cw*ch),
				YStride:      w,
				ChromaStride: cw,
				Width:        w,
				Height:       h,
				BitDepth:     bitDepth,
			}
			for i := range f.U {
				f.U[i] = maxSample / 3
				f.V[i] = maxSample * 2 / 3
			}
			if boxX >= 0 {
				for y := 24; y < 72; y++ {
					for x := boxX; x < boxX+48 && x < w; x++ {
						f.Y[y*w+x] = maxSample - 8
					}
				}
			}
			return f
		}
		frames := []encoder.SourceFrame42016{mk(-1), mk(40), mk(-1)}
		encode := func(goldenInterval int) ([][]byte, []encoder.SourceFrame42016) {
			enc, err := encoder.NewHighBitDepth420VideoEncoderWithColor(w, h, encoder.Profile0, encoder.SequenceColorConfig{BitDepth: bitDepth, SubsamplingX: true, SubsamplingY: true}, 64)
			if err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(goldenInterval)
			var tus [][]byte
			var recons []encoder.SourceFrame42016
			for i, f := range frames {
				tu, _, err := enc.Encode(f, false)
				if err != nil {
					t.Fatalf("encode frame %d: %v", i, err)
				}
				tus = append(tus, append([]byte(nil), tu...))
				recons = append(recons, clone42016Frame(enc.Recon()))
			}
			return tus, recons
		}

		lastOnly, _ := encode(0)
		withGolden, recons := encode(32)
		t.Logf("10-bit 4:2:0 reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
		if len(withGolden[2])*5 >= len(lastOnly[2])*4 {
			t.Fatalf("10-bit 4:2:0 golden reveal %dB not below last-only %dB", len(withGolden[2]), len(lastOnly[2]))
		}
		dec, err := goav1.NewDecoder(withGolden)
		if err != nil {
			t.Fatal(err)
		}
		defer dec.Close()
		decoded, err := dec.DecodeAll()
		if err != nil {
			t.Fatal(err)
		}
		if len(decoded) != len(frames) {
			t.Fatalf("decoded %d frames, want %d", len(decoded), len(frames))
		}
		var got []byte
		got = appendFramePlaneRaw(got, decoded[2].Y, decoded[2].Layout.BytesPerSample)
		got = appendFramePlaneRaw(got, decoded[2].U, decoded[2].Layout.BytesPerSample)
		got = appendFramePlaneRaw(got, decoded[2].V, decoded[2].Layout.BytesPerSample)
		want := appendHighBitDepth420Raw(nil, recons[2])
		if string(got) != string(want) {
			t.Fatal("decoded 10-bit 4:2:0 reveal differs from reconstruction")
		}
	})
}

func TestVideoEncoderCompoundGoldenAverageReference(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	mk := func(seed int64) encoder.SourceFrame420 {
		rng := rand.New(rand.NewSource(seed))
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(16 + rng.Intn(224))
		}
		for i := range f.U {
			f.U[i] = uint8(32 + rng.Intn(192))
			f.V[i] = uint8(32 + rng.Intn(192))
		}
		return f
	}
	blockShift := func(x, y int) (int, int) {
		dx, dy := 2, 2
		if (x/8)&1 != 0 {
			dx = -2
		}
		if (y/8)&1 != 0 {
			dy = -2
		}
		if x+dx < 0 || x+dx+8 > w {
			dx = -dx
		}
		if y+dy < 0 || y+dy+8 > h {
			dy = -dy
		}
		return dx, dy
	}
	avgShifted := func(a, b encoder.SourceFrame420) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, len(a.Y)),
			U:            make([]byte, len(a.U)),
			V:            make([]byte, len(a.V)),
			YStride:      a.YStride,
			ChromaStride: a.ChromaStride,
			Width:        a.Width,
			Height:       a.Height,
		}
		for by := 0; by < h; by += 8 {
			for bx := 0; bx < w; bx += 8 {
				dx, dy := blockShift(bx, by)
				for y := 0; y < 8; y++ {
					dst := (by+y)*w + bx
					aOff := (by+y)*a.YStride + bx
					bOff := (by+dy+y)*b.YStride + bx + dx
					for x := 0; x < 8; x++ {
						f.Y[dst+x] = uint8((int(a.Y[aOff+x]) + int(b.Y[bOff+x]) + 1) >> 1)
					}
				}
				cx, cy := bx/2, by/2
				cdx, cdy := dx/2, dy/2
				for y := 0; y < 4; y++ {
					dst := (cy+y)*cw + cx
					aOff := (cy+y)*a.ChromaStride + cx
					bOff := (cy+cdy+y)*b.ChromaStride + cx + cdx
					for x := 0; x < 4; x++ {
						f.U[dst+x] = uint8((int(a.U[aOff+x]) + int(b.U[bOff+x]) + 1) >> 1)
						f.V[dst+x] = uint8((int(a.V[aOff+x]) + int(b.V[bOff+x]) + 1) >> 1)
					}
				}
			}
		}
		return f
	}

	first, second := mk(101), mk(202)
	withGoldenEnc, err := encoder.NewVideoEncoder(w, h, 72)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = withGoldenEnc.Close() })
	withGoldenEnc.SetGoldenInterval(32)
	tu0, _, err := withGoldenEnc.Encode(first, false)
	if err != nil {
		t.Fatalf("encode key: %v", err)
	}
	tu0 = append([]byte(nil), tu0...)
	recon0 := cloneFrame(withGoldenEnc.Recon())
	tu1, _, err := withGoldenEnc.Encode(second, false)
	if err != nil {
		t.Fatalf("encode last: %v", err)
	}
	tu1 = append([]byte(nil), tu1...)
	recon1 := cloneFrame(withGoldenEnc.Recon())
	average := avgShifted(recon0, recon1)
	tu2, _, err := withGoldenEnc.Encode(average, false)
	if err != nil {
		t.Fatalf("encode compound average: %v", err)
	}
	tu2 = append([]byte(nil), tu2...)
	recon2 := cloneFrame(withGoldenEnc.Recon())
	withGolden := [][]byte{tu0, tu1, tu2}

	lastOnlyEnc, err := encoder.NewVideoEncoder(w, h, 72)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = lastOnlyEnc.Close() })
	lastOnlyEnc.SetGoldenInterval(0)
	var lastOnly [][]byte
	for i, f := range []encoder.SourceFrame420{first, second, average} {
		tu, _, err := lastOnlyEnc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode last-only frame %d: %v", i, err)
		}
		lastOnly = append(lastOnly, append([]byte(nil), tu...))
	}
	t.Logf("average frame: last-only %dB, with compound %dB", len(lastOnly[2]), len(withGolden[2]))
	// The light-PD1 search (md_subpel_me level 8, no filter search) leaves
	// the GOLDEN-anchored compound a smaller margin on this synthetic
	// average; it must still beat LAST alone.
	if len(withGolden[2]) >= len(lastOnly[2]) {
		t.Fatalf("compound average %dB not well below last-only %dB", len(withGolden[2]), len(lastOnly[2]))
	}

	dec, err := goav1.NewDecoder(withGolden)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	recons := []encoder.SourceFrame420{recon0, recon1, recon2}
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
			comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != len(recons) {
		t.Fatalf("decoded %d frames, want %d", i, len(recons))
	}
}

func cloneMono16Frame(f encoder.SourceFrameMono16) encoder.SourceFrameMono16 {
	return encoder.SourceFrameMono16{
		Y:        append([]uint16(nil), f.Y...),
		YStride:  f.YStride,
		Width:    f.Width,
		Height:   f.Height,
		BitDepth: f.BitDepth,
	}
}

func clone42016Frame(f encoder.SourceFrame42016) encoder.SourceFrame42016 {
	return encoder.SourceFrame42016{
		Y:            append([]uint16(nil), f.Y...),
		U:            append([]uint16(nil), f.U...),
		V:            append([]uint16(nil), f.V...),
		YStride:      f.YStride,
		ChromaStride: f.ChromaStride,
		Width:        f.Width,
		Height:       f.Height,
		BitDepth:     f.BitDepth,
	}
}

// TestVideoEncoderL1T2DroppableLayer is the temporal-scalability gate: under
// the L1T2 pattern the full stream must decode with every frame bit-identical
// to its reconstruction, AND the layer-0 substream (all odd frames dropped)
// must decode to exactly the layer-0 reconstructions — proving the T1 frames
// are truly droppable (they refresh no reference state). When aomdec is on
// PATH both streams are verified by the reference decoder too.
func TestVideoEncoderL1T2DroppableLayer(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 121
			f.V[i] = 129
		}
		sx, sy := 8+t*4, 12+t*2
		for y := sy; y < sy+22 && y < h; y++ {
			for x := sx; x < sx+22 && x < w; x++ {
				f.Y[y*w+x] = 218
			}
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}

	const frames = 9
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	tids := make([]uint8, 0, frames)
	for i := range frames {
		tid := enc.TemporalID()
		tu, isKey, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != isKey {
			t.Fatalf("frame %d key=%v", i, isKey)
		}
		wantTID := uint8(0)
		if i%2 == 1 {
			wantTID = 1
		}
		if tid != wantTID {
			t.Fatalf("frame %d tid=%d want %d", i, tid, wantTID)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
		tids = append(tids, tid)
	}

	decodeAndCompare := func(name string, stream [][]byte, want []encoder.SourceFrame420) {
		dec, err := goav1.NewDecoder(stream)
		if err != nil {
			t.Fatal(err)
		}
		defer dec.Close()
		i := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("%s decode: %v", name, err)
			}
			if !ok {
				break
			}
			for _, f := range batch {
				comparePlane(t, fmt.Sprintf("%s frame%d Y", name, i), f.Y, want[i].Y, w, h, w)
				comparePlane(t, fmt.Sprintf("%s frame%d U", name, i), f.U, want[i].U, cw, ch, cw)
				comparePlane(t, fmt.Sprintf("%s frame%d V", name, i), f.V, want[i].V, cw, ch, cw)
				i++
			}
		}
		if i != len(want) {
			t.Fatalf("%s decoded %d frames, want %d", name, i, len(want))
		}
	}

	decodeAndCompare("full", tus, recons)

	// Layer-0 substream: drop every T1 frame.
	var t0TUs [][]byte
	var t0Recons []encoder.SourceFrame420
	for i := range frames {
		if tids[i] == 0 {
			t0TUs = append(t0TUs, tus[i])
			t0Recons = append(t0Recons, recons[i])
		}
	}
	decodeAndCompare("layer0", t0TUs, t0Recons)

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		dir := t.TempDir()
		runRef := func(name string, stream [][]byte, want []encoder.SourceFrame420) {
			ivfData := ivf.AppendFileHeader(nil, w, h, 30, 1, uint32(len(stream)))
			var wantYUV []byte
			for i, tu := range stream {
				ivfData = ivf.AppendFrame(ivfData, tu, uint64(i))
				wantYUV = append(wantYUV, want[i].Y...)
				wantYUV = append(wantYUV, want[i].U...)
				wantYUV = append(wantYUV, want[i].V...)
			}
			p := filepath.Join(dir, name+".ivf")
			os.WriteFile(p, ivfData, 0o644)
			outPath := filepath.Join(dir, name+".yuv")
			if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
				t.Fatalf("aomdec %s: %v\n%s", name, err, out)
			}
			got, err := os.ReadFile(outPath)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(wantYUV) {
				t.Fatalf("aomdec %s output differs from reconstruction", name)
			}
			t.Logf("aomdec %s: %d frames bit-exact", name, len(stream))
		}
		runRef("full", tus, recons)
		runRef("layer0", t0TUs, t0Recons)
	}
}

// TestVideoEncoderL1T3LayerDrops proves the three-layer temporal structure:
// the T0/T2/T1/T2 group pattern, slot 2 carrying the middle layer for its
// trailing T2, and - the WebRTC property the layers exist for - that
// dropping the T2 leaves, or both T1 and T2, still decodes bit-exact
// against the encoder reconstructions of the kept frames.
func TestVideoEncoderL1T3LayerDrops(t *testing.T) {
	const w, h = 192, 128
	rng := rand.New(rand.NewSource(31))
	bg := make([]byte, w*h)
	for y := range h {
		for x := range w {
			bg[y*w+x] = uint8(60 + (x/4+y/4)%64 + rng.Intn(50))
		}
	}
	makeFrame := func(idx int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: append([]byte(nil), bg...), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		// A small mover so every frame codes real residual.
		sx, sy := 8+idx*6, 16+idx*4
		for y := sy; y < sy+16 && y < h; y++ {
			for x := sx; x < sx+16 && x < w; x++ {
				f.Y[y*w+x] = 230
			}
		}
		for i := range f.U {
			f.U[i], f.V[i] = 116, 132
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 110)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	if err := enc.SetTemporalLayers(3); err != nil {
		t.Fatal(err)
	}
	const frames = 9
	var tus [][]byte
	var recons []encoder.SourceFrame420
	var tids []uint8
	for i := range frames {
		tid := enc.TemporalID()
		tu, key, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if key {
			tid = 0
		}
		tids = append(tids, tid)
		tus = append(tus, append([]byte(nil), tu...))
		rc := enc.Recon()
		recons = append(recons, encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
			YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: w, Height: h,
		})
	}
	wantTIDs := []uint8{0, 2, 1, 2, 0, 2, 1, 2, 0}
	for i, tid := range tids {
		if tid != wantTIDs[i] {
			t.Fatalf("frame %d temporal id %d, want %d (got %v)", i, tid, wantTIDs[i], tids)
		}
	}

	decodeSubset := func(name string, keep func(tid uint8) bool) {
		var subTUs [][]byte
		var subRecons []encoder.SourceFrame420
		for i := range tus {
			if keep(tids[i]) {
				subTUs = append(subTUs, tus[i])
				subRecons = append(subRecons, recons[i])
			}
		}
		dec, err := goav1.NewDecoder(subTUs)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		defer dec.Close()
		i := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("%s decode: %v", name, err)
			}
			if !ok {
				break
			}
			for _, f := range batch {
				for y := range h {
					for x := range w {
						if f.Y.Pix[y*f.Y.Stride+x] != subRecons[i].Y[y*w+x] {
							t.Fatalf("%s frame %d differs at (%d,%d)", name, i, x, y)
						}
					}
				}
				i++
			}
		}
		if i != len(subTUs) {
			t.Fatalf("%s decoded %d frames, want %d", name, i, len(subTUs))
		}
	}
	decodeSubset("all-layers", func(uint8) bool { return true })
	decodeSubset("drop-T2", func(tid uint8) bool { return tid < 2 })
	decodeSubset("T0-only", func(tid uint8) bool { return tid == 0 })
}

// TestVideoEncoderOddDimensions proves frames whose dimensions are not
// multiples of eight encode through the padded coded size with render_size
// carrying the true dimensions: the chain decodes bit-exact over the visible
// region and the streams stay legal for the reference decoders (the TUs ride
// the same cross-check corpus shape as every other oracle test).
func TestVideoEncoderOddDimensions(t *testing.T) {
	for _, dims := range [][2]int{{203, 117}, {158, 95}, {101, 62}} {
		w, h := dims[0], dims[1]
		t.Run(fmtDims(w, h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(w*1000 + h)))
			cw, ch := (w+1)/2, (h+1)/2
			makeFrame := func(idx int) encoder.SourceFrame420 {
				f := encoder.SourceFrame420{
					Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
					YStride: w, ChromaStride: cw, Width: w, Height: h,
				}
				for y := range h {
					for x := range w {
						f.Y[y*w+x] = uint8(60 + (x/5+y/5+idx*3)%70 + rng.Intn(20))
					}
				}
				for i := range f.U {
					f.U[i], f.V[i] = 115, 135
				}
				return f
			}
			enc, err := encoder.NewVideoEncoder(w, h, 120)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = enc.Close() })
			var tus [][]byte
			var recons []encoder.SourceFrame420
			for i := range 4 {
				tu, _, err := enc.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("frame %d: %v", i, err)
				}
				tus = append(tus, append([]byte(nil), tu...))
				rc := enc.Recon()
				if rc.Width != w || rc.Height != h {
					t.Fatalf("recon view %dx%d, want %dx%d", rc.Width, rc.Height, w, h)
				}
				recons = append(recons, encoder.SourceFrame420{
					Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
					YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: w, Height: h,
				})
			}
			dec, err := goav1.NewDecoder(tus)
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			codedW, codedH := (w+7)&^7, (h+7)&^7
			i := 0
			for {
				batch, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatalf("decode: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range batch {
					if f.Y.Width != codedW || f.Y.Height != codedH {
						t.Fatalf("decoded %dx%d, want coded %dx%d", f.Y.Width, f.Y.Height, codedW, codedH)
					}
					for y := range h {
						for x := range w {
							if f.Y.Pix[y*f.Y.Stride+x] != recons[i].Y[y*recons[i].YStride+x] {
								t.Fatalf("frame %d differs at (%d,%d)", i, x, y)
							}
						}
					}
					i++
				}
			}
			if i != len(tus) {
				t.Fatalf("decoded %d frames, want %d", i, len(tus))
			}
		})
	}
}

func fmtDims(w, h int) string {
	return fmt.Sprintf("%dx%d", w, h)
}

// pipelineTestFrame builds a deterministic moving-box frame like the L1T2 gate.
func pipelineTestFrame(w, h, t int, rng *rand.Rand, bg []byte) encoder.SourceFrame420 {
	cw, ch := w/2, h/2
	f := encoder.SourceFrame420{
		Y:            append([]byte(nil), bg...),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for i := range f.U {
		f.U[i] = 121
		f.V[i] = 129
	}
	sx, sy := 8+t*4, 12+t*2
	for y := sy; y < sy+22 && y < h; y++ {
		for x := sx; x < sx+22 && x < w; x++ {
			f.Y[y*w+x] = 218
		}
	}
	return f
}

// encodeSerialTUs codes the frame sequence through the default serial path.
func encodeSerialTUs(t *testing.T, newEnc func() *encoder.VideoEncoder, nFrames, w, h int, keyAt map[int]bool) [][]byte {
	t.Helper()
	rng := rand.New(rand.NewSource(17))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	enc := newEnc()
	defer enc.Close()
	out := make([][]byte, 0, nFrames)
	for i := 0; i < nFrames; i++ {
		f := pipelineTestFrame(w, h, i, rng, bg)
		tu, _, err := enc.Encode(f, keyAt[i])
		if err != nil {
			t.Fatalf("serial encode frame %d: %v", i, err)
		}
		out = append(out, append([]byte(nil), tu...))
	}
	return out
}

// encodePipelinedTUs codes the same sequence through EncodeThroughput/Drain.
func encodePipelinedTUs(t *testing.T, newEnc func() *encoder.VideoEncoder, nFrames, w, h int, keyAt map[int]bool) [][]byte {
	t.Helper()
	rng := rand.New(rand.NewSource(17))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	enc := newEnc()
	defer enc.Close()
	if err := enc.SetThroughputPipelining(true); err != nil {
		t.Fatalf("enable pipelining: %v", err)
	}
	out := make([][]byte, 0, nFrames)
	for i := 0; i < nFrames; i++ {
		f := pipelineTestFrame(w, h, i, rng, bg)
		tu, _, produced, err := enc.EncodeThroughput(f, keyAt[i])
		if err != nil {
			t.Fatalf("pipelined encode frame %d: %v", i, err)
		}
		if produced {
			out = append(out, append([]byte(nil), tu...))
		}
	}
	tu, _, produced, err := enc.Drain()
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if produced {
		out = append(out, append([]byte(nil), tu...))
	}
	return out
}

// TestVideoEncoderPipelineByteIdentical is the differential oracle: opt-in
// throughput pipelining must emit byte-identical temporal units to the default
// serial path for the same source sequence, across layer counts, rate-control
// vs fixed-q, golden on/off, and a mid-stream forced keyframe.
func TestVideoEncoderPipelineByteIdentical(t *testing.T) {
	const w, h = 192, 128
	const frames = 17
	keyMid := map[int]bool{10: true}

	cases := []struct {
		name   string
		keyAt  map[int]bool
		newEnc func() *encoder.VideoEncoder
	}{
		{"L1T1-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T2-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T3-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(3); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T2-cbr", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T2-cbr-nogolden", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(0)
			return enc
		}},
		{"L1T2-cbr-multithread", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetMaxThreads(4)
			return enc
		}},
		{"L1T2-overlap-cbr", nil, func() *encoder.VideoEncoder {
			// Golden off + single-tile 8-bit 4:2:0 + no padding:
			// the concurrent leaf/base overlap path.
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(0)
			return enc
		}},
		{"L1T2-overlap-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(0)
			return enc
		}},
		{"L1T2-overlap-mt", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(0)
			enc.SetMaxThreads(4)
			return enc
		}},
		{"L1T2-overlap-keymid", keyMid, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(0)
			return enc
		}},
		{"L1T2-cqp-keymid", keyMid, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			serial := encodeSerialTUs(t, tc.newEnc, frames, w, h, tc.keyAt)
			pipelined := encodePipelinedTUs(t, tc.newEnc, frames, w, h, tc.keyAt)
			if len(serial) != len(pipelined) {
				t.Fatalf("frame count mismatch: serial %d pipelined %d", len(serial), len(pipelined))
			}
			for i := range serial {
				if !bytes.Equal(serial[i], pipelined[i]) {
					t.Fatalf("frame %d differs: serial %d bytes, pipelined %d bytes", i, len(serial[i]), len(pipelined[i]))
				}
			}
		})
	}
}

// TestVideoEncoderPipelineDrainEmpty verifies Drain on an empty pipeline is a
// no-op and that a pipeline can be reused after draining.
func TestVideoEncoderPipelineDrainEmpty(t *testing.T) {
	const w, h = 64, 64
	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if err := enc.SetThroughputPipelining(true); err != nil {
		t.Fatal(err)
	}
	if _, _, produced, err := enc.Drain(); err != nil || produced {
		t.Fatalf("empty drain: produced=%v err=%v", produced, err)
	}
}

// TestVideoEncoderRealResolutions exercises the stream encoder at production
// resolutions that are multiples of 8 but not of 64 (partial superblocks on
// both axes): every frame must decode bit-identical to its reconstruction in
// goav1, and the muxed stream must decode bit-identical in aomdec when
// available.
func TestVideoEncoderRealResolutions(t *testing.T) {
	sizes := []struct{ w, h int }{
		{640, 360}, // nHD: 360 = 5*64 + 40
		{200, 120}, // tiny with partial SBs both axes
	}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			w, h := sz.w, sz.h
			cw, ch := w/2, h/2
			rng := rand.New(rand.NewSource(int64(w)))
			bg := make([]byte, w*h)
			for i := range bg {
				bg[i] = uint8(64 + rng.Intn(56))
			}
			makeFrame := func(t int) encoder.SourceFrame420 {
				f := encoder.SourceFrame420{
					Y:            append([]byte(nil), bg...),
					U:            make([]byte, cw*ch),
					V:            make([]byte, cw*ch),
					YStride:      w,
					ChromaStride: cw,
					Width:        w,
					Height:       h,
				}
				for i := range f.U {
					f.U[i] = 120
					f.V[i] = 128
				}
				sx, sy := 16+t*6, 24+t*4
				for y := sy; y < sy+28 && y < h; y++ {
					for x := sx; x < sx+28 && x < w; x++ {
						f.Y[y*w+x] = 216
					}
				}
				return f
			}

			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = enc.Close() })
			const frames = 5
			tus := make([][]byte, 0, frames)
			recons := make([]encoder.SourceFrame420, 0, frames)
			for i := range frames {
				tu, _, err := enc.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("encode frame %d: %v", i, err)
				}
				tus = append(tus, append([]byte(nil), tu...))
				recons = append(recons, cloneFrame(enc.Recon()))
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
					t.Fatalf("decode: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range batch {
					comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
					comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
					comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
					i++
				}
			}
			if i != frames {
				t.Fatalf("decoded %d frames, want %d", i, frames)
			}

			if aomdec, err := exec.LookPath("aomdec"); err == nil {
				stream := ivf.AppendFileHeader(nil, uint16(w), uint16(h), 30, 1, frames)
				var wantYUV []byte
				for j, tu := range tus {
					stream = ivf.AppendFrame(stream, tu, uint64(j))
					wantYUV = append(wantYUV, recons[j].Y...)
					wantYUV = append(wantYUV, recons[j].U...)
					wantYUV = append(wantYUV, recons[j].V...)
				}
				dir := t.TempDir()
				p := filepath.Join(dir, "s.ivf")
				os.WriteFile(p, stream, 0o644)
				outPath := filepath.Join(dir, "o.yuv")
				if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
					t.Fatalf("aomdec: %v\n%s", err, out)
				}
				got, _ := os.ReadFile(outPath)
				if string(got) != string(wantYUV) {
					t.Fatalf("aomdec output differs from reconstruction")
				}
				t.Logf("aomdec: %d frames bit-exact", frames)
			}
		})
	}
}

// TestVideoEncoderChainDecodesBitExact is the streaming gate: a KEY + 5-P
// chain of a moving scene must decode with EVERY frame bit-identical to the
// encoder's per-frame reconstruction, fidelity above a sanity floor on each
// frame, and the inter frames a fraction of the keyframe — the full encoder
// loop (reference chaining, motion search, skip decisions) working end to end.
func TestVideoEncoderChainDecodesBitExact(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(99))

	// A textured background with a moving bright square: most blocks skip,
	// the square's blocks track motion, and occlusion edges code residuals.
	bg := make([]byte, w*h)
	for y := range h {
		for x := range w {
			bg[y*w+x] = uint8(60 + (x/4+y/4)%64 + rng.Intn(60))
		}
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		// 24x24 bright square moving diagonally 4px right, 2px down per frame.
		sx, sy := 16+t*4, 24+t*2
		for y := sy; y < sy+24 && y < h; y++ {
			for x := sx; x < sx+24 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 60)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	const frames = 6
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	keySize := 0
	interTotal := 0
	for i := range frames {
		tu, isKey, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != isKey {
			t.Fatalf("frame %d keyframe=%v", i, isKey)
		}
		if isKey {
			keySize = len(tu)
		} else {
			interTotal += len(tu)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
	}
	avgInter := interTotal / (frames - 1)
	t.Logf("key %d bytes, avg P %d bytes", keySize, avgInter)
	if avgInter*3 >= keySize {
		t.Fatalf("avg P %d bytes not well below key %d bytes", avgInter, keySize)
	}

	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	// Decoded frames alias pooled surfaces that later decodes recycle, so
	// compare each frame as it is produced rather than after DecodeAll.
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			if i >= frames {
				t.Fatalf("decoded more than %d frames", frames)
			}
			comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
			comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
			psnr := planePSNR(makeFrame(i).Y, recons[i].Y)
			t.Logf("frame %d luma PSNR %.2f dB", i, psnr)
			if psnr < 30 {
				t.Fatalf("frame %d luma PSNR %.2f below floor", i, psnr)
			}
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}

func TestVideoEncoder1080pHotPathAllocs(t *testing.T) {
	const w, h = 1920, 1080
	f0 := makeEncoder1080pFrame(0)
	f1 := makeEncoder1080pFrame(1)

	pEnc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000,
		FramesPerSecond:     60,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pEnc.Close() })
	if err := pEnc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := pEnc.Encode(f0, true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	pAllocs := testing.AllocsPerRun(5, func() {
		tu, key, err := pEnc.Encode(f1, false)
		if err != nil {
			t.Fatal(err)
		}
		if key {
			t.Fatal("steady P-frame was coded as keyframe")
		}
		if len(tu) == 0 {
			t.Fatal("empty P-frame temporal unit")
		}
	})
	if pAllocs != 0 {
		t.Fatalf("1080p steady P-frame allocations=%f want 0", pAllocs)
	}

	panFrames := makeEncoder1080pPanFrames(8)
	panEnc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000,
		FramesPerSecond:     60,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = panEnc.Close() })
	if err := panEnc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := panEnc.Encode(panFrames[0], true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial pan frame was not a keyframe")
	}
	panFrameIndex := 1
	panAllocs := testing.AllocsPerRun(5, func() {
		tu, key, err := panEnc.Encode(panFrames[panFrameIndex], false)
		panFrameIndex++
		if panFrameIndex == len(panFrames) {
			panFrameIndex = 1
		}
		if err != nil {
			t.Fatal(err)
		}
		if key {
			t.Fatal("steady pan P-frame was coded as keyframe")
		}
		if len(tu) == 0 {
			t.Fatal("empty pan P-frame temporal unit")
		}
	})
	if panAllocs != 0 {
		t.Fatalf("1080p pan P-frame allocations=%f want 0", panAllocs)
	}

	keyEnc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000,
		FramesPerSecond:     60,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = keyEnc.Close() })
	if err := keyEnc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := keyEnc.Encode(f0, true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	keyAllocs := testing.AllocsPerRun(5, func() {
		tu, key, err := keyEnc.Encode(f1, true)
		if err != nil {
			t.Fatal(err)
		}
		if !key {
			t.Fatal("forced frame was not a keyframe")
		}
		if len(tu) == 0 {
			t.Fatal("empty keyframe temporal unit")
		}
	})
	if keyAllocs != 0 {
		t.Fatalf("1080p forced keyframe allocations=%f want 0", keyAllocs)
	}
}

func TestKeyframeEncoder1080pHotPathAllocs(t *testing.T) {
	const w, h = 1920, 1080
	f := makeEncoder1080pFrame(0)
	enc, err := encoder.NewKeyframeEncoder(w, h, 80)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	if err := enc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if tu, recon, err := enc.Encode(f); err != nil {
		t.Fatal(err)
	} else if len(tu) == 0 || len(recon.Y) == 0 {
		t.Fatal("empty reusable keyframe output")
	}
	allocs := testing.AllocsPerRun(5, func() {
		tu, recon, err := enc.Encode(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(tu) == 0 || len(recon.Y) == 0 {
			t.Fatal("empty reusable keyframe output")
		}
	})
	if allocs != 0 {
		t.Fatalf("1080p reusable keyframe allocations=%f want 0", allocs)
	}
}

func TestVideoEncoder1080pMultiTileKeyframeWorkerAllocs(t *testing.T) {
	if raceDetectorEnabled {
		// Race instrumentation changes allocation counts. The zero-allocation
		// contract is enforced by the regular test and scripts/check_allocs.sh.
		t.Skip("allocation guard is unreliable under the race detector")
	}
	if os.Getenv("GOAV1_ENCODER_1080P_ALLOC_CHILD") != "1" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestVideoEncoder1080pMultiTileKeyframeWorkerAllocs$")
		cmd.Env = append(os.Environ(), "GOAV1_ENCODER_1080P_ALLOC_CHILD=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("isolated 1080p allocation guard failed: %v\n%s", err, out)
		}
		return
	}

	const w, h = 1920, 1080
	f := makeEncoder1080pFrame(0)
	enc, err := encoder.NewVideoEncoder(w, h, 80)
	if err != nil {
		t.Fatal(err)
	}
	// Explicitly configure four keyframe tiles, even if the machine's default
	// tile count changes. The encoder worker pool handles the three nonzero
	// tiles while the caller encodes tile zero.
	enc.SetMaxThreads(4)
	t.Cleanup(func() { _ = enc.Close() })
	if err := enc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	encode := func() {
		tu, key, err := enc.Encode(f, true)
		if err != nil {
			t.Fatal(err)
		}
		if !key {
			t.Fatal("forced frame was not a keyframe")
		}
		if recon := enc.Recon(); len(tu) == 0 || len(recon.Y) == 0 {
			t.Fatal("empty reusable keyframe output")
		}
	}
	// Exercise real tile-parallel keyframes under the normal scheduler before
	// the serialized allocation measurement below.
	for i := 0; i < 16; i++ {
		encode()
	}
	// The normal-scheduler encodes above exercise the multi-tile worker pool.
	// CI showed varying process-wide MemStats noise (112–4704 B) on macOS,
	// Windows, and Linux while per-encode allocation guards passed.
	// AllocsPerRun temporarily sets GOMAXPROCS=1, which avoids unrelated
	// runtime allocations on other Ps. The worker goroutines still receive tile
	// jobs and their allocations contribute to the process-wide count, but this
	// does not measure simultaneous multi-P allocation behavior.
	encodeBatch := func() {
		for range 16 {
			encode()
		}
	}
	allocs := testing.AllocsPerRun(1, encodeBatch)
	if allocs != 0 {
		t.Fatalf("1080p multi-tile reusable keyframe allocations=%f want 0", allocs)
	}
}

func makeEncoder1080pFrame(tick int) encoder.SourceFrame420 {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := 0; y < h; y++ {
		row := f.Y[y*w : (y+1)*w]
		for x := range row {
			row[x] = uint8(64 + (x/9+y/11+tick)%72)
		}
	}
	sx, sy := 240+tick*4, 360+tick*2
	for y := sy; y < sy+96; y++ {
		for x := sx; x < sx+96; x++ {
			f.Y[y*w+x] = 220
		}
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	return f
}

func makeEncoder1080pPanFrames(count int) []encoder.SourceFrame420 {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(15))
	wide := make([]byte, (w+512)*h)
	for y := range h {
		for x := 0; x < w+512; x++ {
			wide[y*(w+512)+x] = uint8(60 + (x/7+y/9)%70 + rng.Intn(25))
		}
	}
	for y := range h {
		row := wide[y*(w+512) : (y+1)*(w+512)]
		for x := 1; x < len(row)-1; x++ {
			row[x] = uint8((int(row[x-1]) + 2*int(row[x]) + int(row[x+1])) >> 2)
		}
	}
	frames := make([]encoder.SourceFrame420, count)
	for i := range frames {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		off := (i * 4) % 512
		for y := range h {
			copy(f.Y[y*w:(y+1)*w], wide[y*(w+512)+off:])
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		frames[i] = f
	}
	return frames
}

// cloneFrame deep-copies a reconstruction snapshot; the encoder ping-pongs its
// internal recon buffers, so Recon() contents are only stable until the
// next-but-one Encode call.
func cloneFrame(f encoder.SourceFrame420) encoder.SourceFrame420 {
	f.Y = append([]byte(nil), f.Y...)
	f.U = append([]byte(nil), f.U...)
	f.V = append([]byte(nil), f.V...)
	return f
}

// TestVideoEncoderLayeredTilesParity decodes a full-HD two-layer stream and
// compares every frame against the encoder reconstruction. Layered frames
// carry per-layer quantizers, and multi-tile frames split the quantizer
// plumbing per tile column - a tile coded at the wrong layer's quantizer
// reconstructs differently in the decoder, which only a wide, layered,
// multi-tile stream exercises.
func TestVideoEncoderLayeredTilesParity(t *testing.T) {
	const w, h, n = 1920, 1080, 6
	rng := rand.New(rand.NewSource(3))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = byte(40 + rng.Intn(120))
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 6_000_000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	var recons []encoder.SourceFrame420
	for i := 0; i < n; i++ {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		dx := i * 3
		for y := 0; y < h; y++ {
			copy(f.Y[y*w:y*w+w-dx], bg[y*w+dx:])
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 132
		}
		tu, _, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		tus = append(tus, append([]byte(nil), tu...))
		rc := enc.Recon()
		recons = append(recons, encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
		})
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
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if f.Y.Pix[y*f.Y.Stride+x] != recons[i].Y[y*w+x] {
						t.Fatalf("frame %d differs at (%d,%d)", i, x, y)
					}
				}
			}
			i++
		}
	}
	if i != n {
		t.Fatalf("decoded %d frames, want %d", i, n)
	}
}

// TestVideoEncoderTileColumns is the tile-parallel gate: a width that selects
// four tile columns must produce streams whose every frame decodes bit-exact
// against the encoder reconstruction (per-tile entropy coders, CDF resets,
// tile-relative neighbor availability), cross-checked with aomdec when
// available.
func TestVideoEncoderTileColumns(t *testing.T) {
	const w, h = 1280, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(21))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(60 + rng.Intn(80))
	}
	makeFrame := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		sx, sy := 40+n*8, 30+n*6
		for y := sy; y < sy+40 && y < h; y++ {
			for x := sx; x < sx+40 && x < w; x++ {
				f.Y[y*w+x] = 225
			}
		}
		return f
	}
	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	const frames = 5
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	for i := range frames {
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
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
			t.Fatalf("decode frame %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
			comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		stream := ivf.AppendFileHeader(nil, w, h, 30, 1, uint32(frames))
		for n, tu := range tus {
			stream = ivf.AppendFrame(stream, tu, uint64(n))
		}
		dir := t.TempDir()
		p := filepath.Join(dir, "s.ivf")
		os.WriteFile(p, stream, 0o644)
		outPath := filepath.Join(dir, "o.yuv")
		if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
			t.Fatalf("aomdec: %v\n%s", err, out)
		}
		got, _ := os.ReadFile(outPath)
		var want []byte
		for _, r := range recons {
			want = append(want, r.Y...)
			want = append(want, r.U...)
			want = append(want, r.V...)
		}
		if string(got) != string(want) {
			t.Fatal("aomdec output differs from reconstruction")
		}
		t.Log("aomdec: 4-tile stream bit-exact")
	}
}

// TestWebRTCStreamL1T2Metadata is the scalable-packaging gate: an L1T2 WebRTC
// stream must mark odd frames as temporal layer 1 with the correct dependency
// chains (T1 and T0 frames both depending on the latest layer-0 frame, never
// on a T1 frame), produce descriptors for every frame, and emit temporal
// units that decode end to end.
func TestWebRTCStreamL1T2Metadata(t *testing.T) {
	const w, h = 128, 64
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(23))
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(80 + rng.Intn(40))
		}
		for i := range f.U {
			f.U[i] = 119
			f.V[i] = 131
		}
		sx := 8 + t*6
		for y := 12; y < 36; y++ {
			for x := sx; x < sx+16 && x < w; x++ {
				f.Y[y*w+x] = 212
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStreamLayers(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 150_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}

	const frames = 7
	tus := make([][]byte, 0, frames)
	lastT0 := uint64(0)
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		wantTID := uint8(0)
		if i%2 == 1 {
			wantTID = 1
		}
		if ef.Info.TemporalID != wantTID {
			t.Fatalf("frame %d temporal id %d want %d", i, ef.Info.TemporalID, wantTID)
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
		} else {
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != lastT0 {
				t.Fatalf("frame %d depends on %v (num %d), want %d", i, ef.Info.Dependencies, ef.Info.DependencyNum, lastT0)
			}
		}
		if len(ef.Descriptor) == 0 {
			t.Fatalf("frame %d missing descriptor", i)
		}
		if wantTID == 0 {
			lastT0 = ef.Info.FrameID
		}
		tus = append(tus, append([]byte(nil), ef.TU...))
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
	if n != frames {
		t.Fatalf("decoded %d frames, want %d", n, frames)
	}
}

// TestWebRTCStreamL1T3Metadata is the scalable-packaging gate for the
// three-layer pattern: T0/T2/T1/T2, where the trailing T2 must depend on the
// middle layer instead of the older T0 frame.
func TestWebRTCStreamL1T3Metadata(t *testing.T) {
	const w, h = 128, 64
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(31))
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(70 + (i/11+t*3)%70 + rng.Intn(20))
		}
		for i := range f.U {
			f.U[i] = 119
			f.V[i] = 131
		}
		sx := 8 + t*5
		for y := 12; y < 36; y++ {
			for x := sx; x < sx+16 && x < w; x++ {
				f.Y[y*w+x] = 212
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStreamLayers(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 180_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 3)
	if err != nil {
		t.Fatal(err)
	}

	const frames = 9
	wantTIDs := []uint8{0, 2, 1, 2, 0, 2, 1, 2, 0}
	var tus [][]byte
	lastT0 := uint64(0)
	lastT1 := uint64(0)
	haveT1 := false
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if ef.Info.TemporalID != wantTIDs[i] {
			t.Fatalf("frame %d temporal id %d want %d", i, ef.Info.TemporalID, wantTIDs[i])
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
		} else {
			wantDep := lastT0
			if wantTIDs[i] == 2 && i%4 == 3 {
				if !haveT1 {
					t.Fatalf("frame %d is trailing T2 before a middle-layer frame", i)
				}
				wantDep = lastT1
			}
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != wantDep {
				t.Fatalf("frame %d depends on %v (num %d), want %d", i, ef.Info.Dependencies, ef.Info.DependencyNum, wantDep)
			}
		}
		if len(ef.Descriptor) == 0 {
			t.Fatalf("frame %d missing descriptor", i)
		}
		switch wantTIDs[i] {
		case 0:
			lastT0 = ef.Info.FrameID
		case 1:
			lastT1 = ef.Info.FrameID
			haveT1 = true
		}
		tus = append(tus, append([]byte(nil), ef.TU...))
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
	if n != frames {
		t.Fatalf("decoded %d frames, want %d", n, frames)
	}
}

// TestWebRTCStreamPackagesDecodableFrames is the WebRTC integration gate: an
// L1T1 stream must emit, per frame, a decodable temporal unit plus coherent
// dependency metadata — the keyframe carrying the attached structure with no
// dependencies, every delta frame depending on the previous frame ID — and
// the temporal units must decode bit-identical to the encoder reconstruction.
func TestWebRTCStreamPackagesDecodableFrames(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 118
			f.V[i] = 132
		}
		sx, sy := 12+t*4, 20+t*2
		for y := sy; y < sy+20 && y < h; y++ {
			for x := sx; x < sx+20 && x < w; x++ {
				f.Y[y*w+x] = 210
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStreamLayers(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 150_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 1)
	if err != nil {
		t.Fatal(err)
	}

	const frames = 5
	tus := make([][]byte, 0, frames)
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != ef.Keyframe {
			t.Fatalf("frame %d keyframe=%v", i, ef.Keyframe)
		}
		if ef.Info.FrameID != uint64(i) {
			t.Fatalf("frame %d id=%d", i, ef.Info.FrameID)
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
			if len(ef.Descriptor) < 8 {
				t.Fatalf("keyframe descriptor %d bytes: structure not attached?", len(ef.Descriptor))
			}
		} else {
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != uint64(i-1) {
				t.Fatalf("frame %d deps=%v num=%d", i, ef.Info.Dependencies, ef.Info.DependencyNum)
			}
			if len(ef.Descriptor) == 0 {
				t.Fatalf("frame %d missing descriptor", i)
			}
		}
		tus = append(tus, append([]byte(nil), ef.TU...))
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
	if n != frames {
		t.Fatalf("decoded %d frames, want %d", n, frames)
	}
	fmt.Println("webrtc stream ok")
}

func TestWebRTCStreamAcceptedScalabilityModesDecode(t *testing.T) {
	for _, mode := range acceptedWebRTCStreamPixelModes() {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := webRTCDecodeMatrixConfig(mode)
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			stream.SetGoldenInterval(0)

			var tusBySpatial [encoder.WebRTCMaxSpatialLayers][][]byte
			var wantBySpatial [encoder.WebRTCMaxSpatialLayers]int
			var tusInOrder [][]byte
			for i := 0; i < 2; i++ {
				picture, err := stream.EncodePicture(webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), i), false)
				if err != nil {
					t.Fatalf("EncodePicture(%d): %v", i, err)
				}
				if picture.FrameNum == 0 {
					t.Fatalf("EncodePicture(%d) emitted no frames", i)
				}
				collectWebRTCPictureTUs(t, picture, &tusBySpatial, &wantBySpatial, &tusInOrder)
			}

			decodeCollectedWebRTCPictureTUs(t, stream.Config(), tusInOrder, tusBySpatial, wantBySpatial)
		})
	}
}

func TestWebRTCStreamControlCombinationMatrixDecode(t *testing.T) {
	for _, mode := range acceptedWebRTCStreamPixelModes() {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := webRTCDecodeMatrixConfig(mode)
			cfg.MaxFramerate = encoder.Rational{Num: 24, Den: 1}
			cfg.KeyFrameInterval = 4
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			stream.SetGoldenInterval(0)

			var tusBySpatial [encoder.WebRTCMaxSpatialLayers][][]byte
			var wantBySpatial [encoder.WebRTCMaxSpatialLayers]int
			var tusInOrder [][]byte
			var parsedBySpatial [encoder.WebRTCMaxSpatialLayers]webRTCParsedTUState
			var parsedShared webRTCParsedTUState
			encode := func(frameIndex int, forceKey bool) encoder.WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodePicture(webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), frameIndex), forceKey)
				if err != nil {
					t.Fatalf("EncodePicture(%d, force=%v): %v", frameIndex, forceKey, err)
				}
				verifyWebRTCPicturePayloadHeaders(t, picture, stream.Config(), &parsedBySpatial, &parsedShared)
				collectWebRTCPictureTUs(t, picture, &tusBySpatial, &wantBySpatial, &tusInOrder)
				return picture
			}

			key := encode(0, false)
			if !key.Keyframe {
				t.Fatalf("initial picture was not a keyframe: %+v", key)
			}

			controlChange := stream.Config()
			controlChange.MaxFramerate = encoder.Rational{Num: 60, Den: 1}
			controlChange.MinBitrateKbps += 10
			controlChange.MaxBitrateKbps += 250
			controlChange.TargetBitrateKbps += 120
			beforeFrameID := key.Frames[0].Info.FrameID + uint64(key.FrameNum)
			if err := stream.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig fps/bitrate control change: %v", err)
			}
			delta := encode(1, false)
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("control change picture=%+v beforeFrameID=%d", delta, beforeFrameID)
			}
			assertWebRTCPictureRateControl(t, delta, controlChange.RateControl, controlChange.Quantizer)

			toCQP := stream.Config()
			toCQP.RateControl = encoder.RateControlCQP
			toCQP.Quantizer = 37
			beforeFrameID = delta.Frames[0].Info.FrameID + uint64(delta.FrameNum)
			if err := stream.SetConfig(toCQP); err != nil {
				t.Fatalf("SetConfig CQP transition: %v", err)
			}
			cqpDelta := encode(2, false)
			if cqpDelta.Keyframe || cqpDelta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("CQP transition picture=%+v beforeFrameID=%d", cqpDelta, beforeFrameID)
			}
			assertWebRTCPictureRateControl(t, cqpDelta, encoder.RateControlCQP, toCQP.Quantizer)

			toCBR := stream.Config()
			toCBR.RateControl = encoder.RateControlCBR
			toCBR.Quantizer = 0
			toCBR.MaxFramerate = encoder.Rational{Num: 48, Den: 1}
			toCBR.MinBitrateKbps += 15
			toCBR.MaxBitrateKbps += 180
			toCBR.TargetBitrateKbps = toCBR.MinBitrateKbps + (toCBR.MaxBitrateKbps-toCBR.MinBitrateKbps)/2
			beforeFrameID = cqpDelta.Frames[0].Info.FrameID + uint64(cqpDelta.FrameNum)
			if err := stream.SetConfig(toCBR); err != nil {
				t.Fatalf("SetConfig CBR transition: %v", err)
			}
			cbrDelta := encode(3, false)
			if cbrDelta.Keyframe || cbrDelta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("CBR transition picture=%+v beforeFrameID=%d", cbrDelta, beforeFrameID)
			}
			assertWebRTCPictureRateControl(t, cbrDelta, encoder.RateControlCBR, 0)

			intervalChange := stream.Config()
			intervalChange.MaxFramerate = encoder.Rational{Num: 30000, Den: 1001}
			intervalChange.MinBitrateKbps += 5
			intervalChange.MaxBitrateKbps += 100
			intervalChange.TargetBitrateKbps += 40
			intervalChange.KeyFrameInterval = 2
			if err := stream.SetConfig(intervalChange); err != nil {
				t.Fatalf("SetConfig key-interval control change: %v", err)
			}
			intervalKey := encode(4, false)
			if !intervalKey.Keyframe {
				t.Fatalf("key interval did not force a key picture: %+v", intervalKey)
			}
			assertWebRTCPictureRateControl(t, intervalKey, encoder.RateControlCBR, 0)

			forceConfig := stream.Config()
			forceConfig.MaxFramerate = encoder.Rational{Num: 15, Den: 1}
			forceConfig.TargetBitrateKbps = forceConfig.MinBitrateKbps + (forceConfig.MaxBitrateKbps-forceConfig.MinBitrateKbps)/2
			if err := stream.SetConfig(forceConfig); err != nil {
				t.Fatalf("SetConfig pre-force control change: %v", err)
			}
			forced := encode(5, true)
			if !forced.Keyframe {
				t.Fatalf("force key did not produce key picture: %+v", forced)
			}
			assertWebRTCPictureRateControl(t, forced, encoder.RateControlCBR, 0)

			decodeCollectedWebRTCPictureTUs(t, stream.Config(), tusInOrder, tusBySpatial, wantBySpatial)
		})
	}
}

func assertWebRTCPictureRateControl(t *testing.T, picture encoder.WebRTCEncodedPicture, want encoder.RateControlMode, wantQuantizer uint8) {
	t.Helper()
	for frameIndex := uint8(0); frameIndex < picture.FrameNum; frameIndex++ {
		var settings encoder.FrameEncodeSettings
		var ok bool
		switch {
		case picture.Unit.Key:
			if frameIndex < picture.Unit.KeyUnit.FrameNum {
				settings, ok = picture.Unit.KeyUnit.Frames[frameIndex], true
			}
		case picture.Unit.Delta:
			if frameIndex < picture.Unit.DeltaUnit.FrameNum {
				settings, ok = picture.Unit.DeltaUnit.Frames[frameIndex], true
			}
		}
		if !ok {
			t.Fatalf("picture frame %d missing temporal-unit settings: %+v", frameIndex, picture.Unit)
		}
		if settings.RateControl != want || settings.Quantizer != wantQuantizer {
			t.Fatalf("picture frame %d rate control=%d q=%d want %d/%d settings=%+v",
				frameIndex, settings.RateControl, settings.Quantizer, want, wantQuantizer, settings)
		}
	}
}

func TestWebRTCStream1080pHotPathAllocs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		mode       encoder.ScalabilityMode
		speed      int8
		maxThreads int32
	}{
		{name: encoder.ScalabilityModeL1T3.String(), mode: encoder.ScalabilityModeL1T3},
		{name: encoder.ScalabilityModeS3T3.String(), mode: encoder.ScalabilityModeS3T3},
		{name: "L1T3-single-thread-min-effort", mode: encoder.ScalabilityModeL1T3, speed: encoder.WebRTCMinEffortLevel, maxThreads: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := webRTC1080pAllocConfig(tc.mode)
			cfg.Speed = tc.speed
			cfg.MaxThreads = tc.maxThreads
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", tc.name, err)
			}
			t.Cleanup(func() { _ = stream.Close() })

			frames := [4]encoder.SourceFrame420{
				makeEncoder1080pFrame(0),
				makeEncoder1080pFrame(1),
				makeEncoder1080pFrame(2),
				makeEncoder1080pFrame(3),
			}
			if err := stream.Prewarm(); err != nil {
				t.Fatalf("Prewarm(%s): %v", tc.name, err)
			}
			for i := 0; i < 12; i++ {
				forceKey := i == 0 || i == 6
				picture, err := stream.EncodePicture(frames[i&3], forceKey)
				if err != nil {
					t.Fatalf("warm EncodePicture(%s, %d): %v", tc.name, i, err)
				}
				if picture.FrameNum == 0 {
					t.Fatalf("warm EncodePicture(%s, %d) emitted no frames", tc.name, i)
				}
			}

			frameIndex := 0
			pAllocs := testing.AllocsPerRun(3, func() {
				picture, err := stream.EncodePicture(frames[frameIndex&3], false)
				frameIndex++
				if err != nil {
					t.Fatal(err)
				}
				if picture.Keyframe {
					t.Fatal("steady WebRTC picture was coded as keyframe")
				}
				if picture.FrameNum == 0 {
					t.Fatal("empty steady WebRTC picture")
				}
			})
			if pAllocs != 0 {
				t.Fatalf("1080p WebRTC %s steady picture allocations=%f want 0", tc.name, pAllocs)
			}

			if _, err := stream.EncodePicture(frames[0], true); err != nil {
				t.Fatalf("forced-key warm EncodePicture(%s): %v", tc.name, err)
			}
			keyAllocs := testing.AllocsPerRun(3, func() {
				picture, err := stream.EncodePicture(frames[1], true)
				if err != nil {
					t.Fatal(err)
				}
				if !picture.Keyframe {
					t.Fatal("forced WebRTC picture was not coded as keyframe")
				}
				if picture.FrameNum == 0 {
					t.Fatal("empty forced-key WebRTC picture")
				}
			})
			if keyAllocs != 0 {
				t.Fatalf("1080p WebRTC %s forced-key picture allocations=%f want 0", tc.name, keyAllocs)
			}
		})
	}
}

// acceptedWebRTCStreamPixelModes lists every scalability mode the stream
// accepts, including the *_KEY_SHIFT modes libwebrtc does not export.
func acceptedWebRTCStreamPixelModes() []encoder.ScalabilityMode {
	return []encoder.ScalabilityMode{
		encoder.ScalabilityModeL1T1, encoder.ScalabilityModeL1T2, encoder.ScalabilityModeL1T3,
		encoder.ScalabilityModeL2T1, encoder.ScalabilityModeL2T1h, encoder.ScalabilityModeL2T1_KEY,
		encoder.ScalabilityModeL2T2, encoder.ScalabilityModeL2T2h, encoder.ScalabilityModeL2T2_KEY,
		encoder.ScalabilityModeL2T2_KEY_SHIFT, encoder.ScalabilityModeL2T3, encoder.ScalabilityModeL2T3h,
		encoder.ScalabilityModeL2T3_KEY, encoder.ScalabilityModeL2T3_KEY_SHIFT,
		encoder.ScalabilityModeL3T1, encoder.ScalabilityModeL3T1h, encoder.ScalabilityModeL3T1_KEY,
		encoder.ScalabilityModeL3T2, encoder.ScalabilityModeL3T2h, encoder.ScalabilityModeL3T2_KEY,
		encoder.ScalabilityModeL3T2_KEY_SHIFT, encoder.ScalabilityModeL3T3, encoder.ScalabilityModeL3T3h,
		encoder.ScalabilityModeL3T3_KEY, encoder.ScalabilityModeL3T3_KEY_SHIFT,
		encoder.ScalabilityModeS2T1, encoder.ScalabilityModeS2T1h, encoder.ScalabilityModeS2T2,
		encoder.ScalabilityModeS2T2h, encoder.ScalabilityModeS2T3, encoder.ScalabilityModeS2T3h,
		encoder.ScalabilityModeS3T1, encoder.ScalabilityModeS3T1h, encoder.ScalabilityModeS3T2,
		encoder.ScalabilityModeS3T2h, encoder.ScalabilityModeS3T3, encoder.ScalabilityModeS3T3h,
	}
}

func webRTC1080pAllocConfig(mode encoder.ScalabilityMode) encoder.Config {
	return encoder.Config{
		Resolution:        encoder.Resolution{Width: 1920, Height: 1080},
		MaxFramerate:      encoder.Rational{Num: 60, Den: 1},
		MinBitrateKbps:    800,
		MaxBitrateKbps:    8_000,
		TargetBitrateKbps: 4_000,
		Scalability:       mode,
		RateControl:       encoder.RateControlCBR,
	}
}

func webRTCDecodeMatrixConfig(mode encoder.ScalabilityMode) encoder.Config {
	spatial, _, _, _ := mode.Layers()
	resolution := encoder.Resolution{Width: 192, Height: 128}
	switch spatial {
	case 2:
		resolution = encoder.Resolution{Width: 640, Height: 360}
	case 3:
		resolution = encoder.Resolution{Width: 1008, Height: 576}
	}
	return encoder.Config{
		Resolution:        resolution,
		MaxFramerate:      encoder.Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       mode,
	}
}

func collectWebRTCPictureTUs(t *testing.T, picture encoder.WebRTCEncodedPicture, tusBySpatial *[encoder.WebRTCMaxSpatialLayers][][]byte, wantBySpatial *[encoder.WebRTCMaxSpatialLayers]int, tusInOrder *[][]byte) {
	t.Helper()
	if picture.FrameNum == 0 {
		t.Fatal("picture emitted no frames")
	}
	for frame := uint8(0); frame < picture.FrameNum; frame++ {
		spatialID := picture.Frames[frame].Info.SpatialID
		if spatialID >= encoder.WebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", frame, spatialID)
		}
		tu := append([]byte(nil), picture.Frames[frame].TU...)
		tusBySpatial[spatialID] = append(tusBySpatial[spatialID], tu)
		wantBySpatial[spatialID]++
		*tusInOrder = append(*tusInOrder, tu)
	}
}

func decodeCollectedWebRTCPictureTUs(t *testing.T, config encoder.Config, tusInOrder [][]byte, tusBySpatial [encoder.WebRTCMaxSpatialLayers][][]byte, wantBySpatial [encoder.WebRTCMaxSpatialLayers]int) {
	t.Helper()
	if webRTCSharedReferenceSlotMode(config) {
		frames := decodeLayerPoolLowOverheads(t, tusInOrder...)
		want := 0
		for _, count := range wantBySpatial {
			want += count
		}
		if len(frames) != want {
			t.Fatalf("shared SVC decoded %d frames, want %d", len(frames), want)
		}
		return
	}
	for spatialID, tus := range tusBySpatial {
		if len(tus) == 0 {
			continue
		}
		dec, err := goav1.NewDecoder(tus)
		if err != nil {
			t.Fatalf("spatial %d NewDecoder: %v", spatialID, err)
		}
		gotFrames := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				dec.Close()
				t.Fatalf("spatial %d decode: %v", spatialID, err)
			}
			if !ok {
				break
			}
			gotFrames += len(batch)
		}
		dec.Close()
		if gotFrames != wantBySpatial[spatialID] {
			t.Fatalf("spatial %d decoded %d frames, want %d", spatialID, gotFrames, wantBySpatial[spatialID])
		}
	}
}

func webRTCSharedReferenceSlotMode(config encoder.Config) bool {
	return config.SpatialLayerCount > 1 &&
		!config.Scalability.IsSimulcast()
}

type webRTCParsedTUState struct {
	seq     parser.SequenceHeader
	haveSeq bool
	refs    parser.ReferenceState
}

func verifyWebRTCPicturePayloadHeaders(t *testing.T, picture encoder.WebRTCEncodedPicture, config encoder.Config, states *[encoder.WebRTCMaxSpatialLayers]webRTCParsedTUState, shared *webRTCParsedTUState) {
	t.Helper()
	for frame := uint8(0); frame < picture.FrameNum; frame++ {
		encoded := picture.Frames[frame]
		spatialID := encoded.Info.SpatialID
		if spatialID >= encoder.WebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", frame, spatialID)
		}
		want := config.SpatialLayers[spatialID].Resolution
		state := &states[spatialID]
		if webRTCSharedReferenceSlotMode(config) {
			state = shared
		}
		state.verify(t, encoded, want)
	}
}

func (s *webRTCParsedTUState) verify(t *testing.T, frame encoder.WebRTCEncodedFrame, want encoder.Resolution) {
	t.Helper()
	it := obu.NewLowOverheadIterator(frame.TU)
	seenFrameHeader := false
	seenTileGroup := false
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
			seq, err := parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			s.seq = seq
			s.haveSeq = true
		case obu.TypeFrameHeader, obu.TypeFrame:
			if unit.Header.TemporalID != frame.Info.TemporalID || unit.Header.SpatialID != frame.Info.SpatialID {
				t.Fatalf("frame header extension T%d/S%d want T%d/S%d", unit.Header.TemporalID, unit.Header.SpatialID, frame.Info.TemporalID, frame.Info.SpatialID)
			}
			size := s.parseAndRefreshFrameHeader(t, unit.Payload, unit.Header.TemporalID, unit.Header.SpatialID)
			wantCodedWidth := (uint32(want.Width) + 7) &^ 7
			wantCodedHeight := (uint32(want.Height) + 7) &^ 7
			if size.UpscaledWidth != wantCodedWidth || size.Height != wantCodedHeight ||
				size.RenderWidth != uint32(want.Width) || size.RenderHeight != uint32(want.Height) {
				t.Fatalf("payload coded=%dx%d render=%dx%d want coded=%dx%d render=%dx%d",
					size.UpscaledWidth, size.Height, size.RenderWidth, size.RenderHeight,
					wantCodedWidth, wantCodedHeight, want.Width, want.Height)
			}
			seenFrameHeader = true
			// An OBU_FRAME carries the tile group too.
			seenTileGroup = seenTileGroup || unit.Header.Type == obu.TypeFrame
		case obu.TypeTileGroup:
			if unit.Header.TemporalID != frame.Info.TemporalID || unit.Header.SpatialID != frame.Info.SpatialID {
				t.Fatalf("tile group extension T%d/S%d want T%d/S%d", unit.Header.TemporalID, unit.Header.SpatialID, frame.Info.TemporalID, frame.Info.SpatialID)
			}
			seenTileGroup = true
		}
	}
	if !seenFrameHeader || !seenTileGroup {
		t.Fatalf("missing frame header=%v tile group=%v", seenFrameHeader, seenTileGroup)
	}
}

func (s *webRTCParsedTUState) parseAndRefreshFrameHeader(t *testing.T, payload []byte, temporalID uint8, spatialID uint8) parser.FrameSize {
	t.Helper()
	if !s.haveSeq {
		t.Fatal("frame header before sequence header")
	}
	prefix, err := parser.ParseFrameHeaderPrefix(payload, s.seq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	var size parser.FrameSize
	if prefix.UsesIntraFrameSizePath() {
		size, err = parser.ParseFrameSize(payload, s.seq, prefix, nil, temporalID, spatialID)
	} else {
		size, err = parser.ParseFrameSize(payload, s.seq, prefix, &s.refs, temporalID, spatialID)
	}
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	s.refreshReferences(prefix, size)
	return size
}

func (s *webRTCParsedTUState) refreshReferences(prefix parser.FrameHeaderPrefix, size parser.FrameSize) {
	if size.RefreshFrameFlags == 0 {
		return
	}
	ref := parser.ReferenceFrame{
		Valid:        true,
		OrderHint:    prefix.OrderHint,
		GlobalMotion: parser.DefaultGlobalMotionParams(),
		Size:         size,
	}
	for i := uint8(0); i < parser.RefFrames; i++ {
		if size.RefreshFrameFlags&(1<<i) != 0 {
			s.refs.Frames[i] = ref
		}
	}
}

func webRTCDecodeMatrixFrame(width int, height int, n int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	for y := 0; y < height; y++ {
		row := y * width
		for x := 0; x < width; x++ {
			f.Y[row+x] = uint8(48 + (x+n*7)%80 + (y+n*3)%40)
		}
	}
	for i := range f.U {
		f.U[i] = uint8(116 + n)
		f.V[i] = uint8(132 - n)
	}
	return f
}
