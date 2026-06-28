package encoder_test

import (
	"bytes"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// TestEncodedStreamDecodesInReferenceDecoders is the independent oracle: the
// encoder's output must decode in libaom's aomdec (and dav1d when present) to
// exactly the encoder reconstruction. This closes the residual risk of a
// blind spot shared between the goav1 encoder and decoder.
func TestEncodedStreamDecodesInReferenceDecoders(t *testing.T) {
	aomdec, err := exec.LookPath("aomdec")
	if err != nil {
		t.Skip("aomdec not on PATH")
	}
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
	if dav1d, err := exec.LookPath("dav1d"); err == nil {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedLosslessPFrameDecodesInReferenceDecoders(t *testing.T) {
	aomdec, err := exec.LookPath("aomdec")
	if err != nil {
		t.Skip("aomdec not on PATH")
	}
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
				f.Y[y*w+x] = uint8((x*5 + y*3 + rng.Intn(41)) & 0xff)
			}
		}
		for y := range ch {
			for x := range cw {
				f.U[y*cw+x] = uint8((84 + x*9 + y*5 + rng.Intn(17)) & 0xff)
				f.V[y*cw+x] = uint8((168 + x*3 + y*7 + rng.Intn(29)) & 0xff)
			}
		}
		return f
	}
	src1 := makeFrame(101)
	src2 := makeFrame(202)
	keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, 72)
	if err != nil {
		t.Fatalf("encode keyframe: %v", err)
	}
	pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, 0)
	if err != nil {
		t.Fatalf("encode lossless p-frame: %v", err)
	}

	stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 2)
	stream = ivf.AppendFrame(stream, keyTU, 0)
	stream = ivf.AppendFrame(stream, pTU, 1)
	wantYUV := make([]byte, 0, 2*(w*h+2*cw*ch))
	wantYUV = append(wantYUV, keyRecon.Y...)
	wantYUV = append(wantYUV, keyRecon.U...)
	wantYUV = append(wantYUV, keyRecon.V...)
	wantYUV = append(wantYUV, pRecon.Y...)
	wantYUV = append(wantYUV, pRecon.U...)
	wantYUV = append(wantYUV, pRecon.V...)

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, "lossless-p.ivf")
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
		if !bytes.Equal(got, wantYUV) {
			for i := range min(len(got), len(wantYUV)) {
				if got[i] != wantYUV[i] {
					t.Fatalf("%s output differs first at byte %d: got %d want %d", name, i, got[i], wantYUV[i])
				}
			}
			t.Fatalf("%s output %d bytes, want %d", name, len(got), len(wantYUV))
		}
		t.Logf("%s: qindex-0 P-frame stream bit-exact", name)
	}
	check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
	if dav1d, err := exec.LookPath("dav1d"); err == nil {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedMonochromeKeyframeDecodesInReferenceDecoders(t *testing.T) {
	aomdec, err := exec.LookPath("aomdec")
	if err != nil {
		t.Skip("aomdec not on PATH")
	}
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
	if dav1d, err := exec.LookPath("dav1d"); err == nil {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedMonochromeLossyKeyframeDecodesInReferenceDecoders(t *testing.T) {
	aomdec, err := exec.LookPath("aomdec")
	if err != nil {
		t.Skip("aomdec not on PATH")
	}
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
	if dav1d, err := exec.LookPath("dav1d"); err == nil {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}

func TestEncodedMonochromePFrameDecodesInReferenceDecoders(t *testing.T) {
	aomdec, err := exec.LookPath("aomdec")
	if err != nil {
		t.Skip("aomdec not on PATH")
	}
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
	pTU, pRecon, err := encoder.EncodeMonochromePFrame(src2, keyRecon, 80)
	if err != nil {
		t.Fatalf("encode p-frame: %v", err)
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
		t.Logf("%s: monochrome key+P luma matches reconstruction (%d raw bytes)", name, len(got))
	}

	check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
	if dav1d, err := exec.LookPath("dav1d"); err == nil {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}
