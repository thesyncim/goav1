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
